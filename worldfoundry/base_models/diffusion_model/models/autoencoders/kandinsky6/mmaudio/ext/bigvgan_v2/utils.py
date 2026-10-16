# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
# License text: THIRD-PARTY-NOTICES, BigVGAN HiFi-GAN LICENSE_1.

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)
