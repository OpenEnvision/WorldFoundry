import torch.nn
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
# lazy import of pytorchltr and allrank to avoid unnecessary dependencies

def get_resnet(args):
    # Get a ResNet model based on the provided arguments.
    try:
        dropout = float(args.dropout)
    except Exception:
        dropout = 0.0

    if args.backbone_config == 'resnet101':
        resnet = models.resnet101(pretrained=True)
    else:
        resnet = models.resnet50(pretrained=True)

    if args.no_dense_layer:
        resnet.fc = nn.Linear(2048, 1, bias=True)
    else:
        output_layer = nn.Sequential(
            nn.Linear(2048, 1000, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1000, 1, bias=True),
        )
        resnet.fc = output_layer
    return resnet


def get_loss_function(name):
    # assert name in ['point', 'pairLogist', 'pairLambda', 'listMLE']
    if name == 'point':
        loss_f = nn.MSELoss()
    elif name == 'pairLogist':
        from pytorchltr.loss import PairwiseLogisticLoss
        loss_f = PairwiseLogisticLoss()
    elif name == 'pairLambda':
        from pytorchltr.loss import LambdaNDCGLoss1
        loss_f = LambdaNDCGLoss1()
    elif name == 'listMLE':
        try:
            from allrank.models.losses import listMLE
        except Exception:
            print('listMLE not found; please clone from https://github.com/allegro/allRank/tree/master/allrank')
        loss_f = listMLE
    else:
        print('------------------------')
        raise Exception('unknown loss name')
    return loss_f


class LPIPS(nn.Module):
    def __init__(self):
        import lpips
        super(LPIPS, self).__init__()
        self.dist = lpips.LPIPS(net='alex')

    def forward(self, x, y):
        # images must be in range [-1, 1]
        dist = self.dist(2 * x - 1, 2 * y - 1)
        return dist.view(-1)


class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
        self.dist = torch.nn.MSELoss(reduction='none')

    def forward(self, x, y):
        # b*c*H*W
        dist = torch.sum(self.dist(x, y), dim=[1, 2, 3])
        return dist


class SSIM_(nn.Module):
    def __init__(self):
        from pytorch_msssim import ssim, SSIM
        super(SSIM_, self).__init__()
        self.dist = SSIM(data_range=1, size_average=False)

    def forward(self, x, y):
        dist = self.dist(x, y)
        return dist


class GramLoss(torch.nn.Module):
    def __init__(self):
        super(GramLoss, self).__init__()
        # content_layers_default = ['conv_4']
        style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        self.style_layers = []

        blocks = []
        cnn = models.vgg19(pretrained=True).features.eval()
        i = 0  # increment every time we see a conv
        j = 0 # index of current layer
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            blocks.append(layer)
            if name in style_layers_default:
                self.style_layers.append(j)

            j += 1
            if name == 'conv_5':
                break # later layer won't be used

        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, x, y):
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in self.style_layers:
                norm = x.shape[1]*x.shape[2]*x.shape[3]
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1) /norm
                gram_y = act_y @ act_y.permute(0, 2, 1) /norm
                cur_loss = torch.sum(F.mse_loss(gram_x, gram_y, reduction='none'), dim=[1, 2])/(x.shape[0])/(x.shape[0])
                try:
                    gram_loss += cur_loss
                except Exception:
                    gram_loss = cur_loss
        return gram_loss


class ContentLoss(torch.nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        content_layers_default = ['conv_4']
        # style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        self.content_layers = []

        blocks = []
        cnn = models.vgg19(pretrained=True).features.eval()
        i = 0  # increment every time we see a conv
        j = 0 # index of current layer
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            blocks.append(layer)
            if name in content_layers_default:
                self.content_layers.append(j)

            j += 1
            if name == 'conv_5':
                break # later layer won't be used

        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, x, y):
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)

            if i in self.content_layers:
                norm = x.shape[1]*x.shape[2]*x.shape[3]
                cur_loss = torch.sum(F.mse_loss(x, y, reduction='none'), dim=[1, 2, 3])/norm
                try:
                    content_loss += cur_loss
                except Exception:
                    content_loss = cur_loss

        return content_loss

