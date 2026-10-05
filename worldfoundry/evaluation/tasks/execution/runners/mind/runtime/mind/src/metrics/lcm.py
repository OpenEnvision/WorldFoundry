import torch
import lpips
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def load_models(device):
    lpips_metric = lpips.LPIPS(net='alex', spatial=False).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=[1, 2, 3]).to(device)
    return lpips_metric, ssim_metric, psnr_metric

def lcm_metric(pred, gt,
               lpips_metric, ssim_metric, psnr_metric, batch_size=100, device='cuda:0'):
    f, c, h, w = pred.size()
    assert(torch.all(pred >= 0.0) and torch.all(pred <= 1.0))
    result_dict = {'length': f}

    with torch.no_grad():
        mse_list = []
        lpips_list = []
        psnr_list = []
        ssim_list = []

        for i in range(0, f, batch_size):
            # batch移到GPU
            pred_batch = pred[i:i+batch_size].to(device)
            gt_batch = gt[i:i+batch_size].to(device)

            diff = (pred_batch - gt_batch) ** 2
            mse_list.extend(diff.reshape(len(pred_batch), -1).mean(dim=1).cpu().tolist())

            # LPIPS 输入需要 [-1, 1] 范围
            lpips_batch = lpips_metric((pred_batch * 2 - 1), (gt_batch * 2 - 1)).cpu().tolist()
            lpips_batch = [item[0][0][0] for item in lpips_batch]
            lpips_list.extend(lpips_batch)

            psnr_list.extend(psnr_metric(pred_batch, gt_batch).cpu().tolist())
            ssim_list.extend(ssim_metric(pred_batch, gt_batch).cpu().tolist())

            del pred_batch, gt_batch, diff
            torch.cuda.empty_cache()

        result_dict['mse'] = mse_list
        result_dict['avg_mse'] = sum(mse_list) / len(mse_list)

        result_dict['psnr'] = psnr_list
        result_dict['avg_psnr'] = sum(psnr_list) / len(psnr_list)

        result_dict['ssim'] = ssim_list
        result_dict['avg_ssim'] = sum(ssim_list) / len(ssim_list)

        result_dict['lpips'] = lpips_list
        result_dict['avg_lpips'] = sum(lpips_list) / len(lpips_list)

    return result_dict

def merge_lcm_results(result_dict, new_result):
    """合并两个lcm结果字典"""
    if result_dict is None:
        return new_result
    if new_result is None:
        return result_dict

    result_dict['mse'].extend(new_result['mse'])
    result_dict['psnr'].extend(new_result['psnr'])
    result_dict['ssim'].extend(new_result['ssim'])
    result_dict['lpips'].extend(new_result['lpips'])
    result_dict['length'] += new_result['length']
    result_dict['avg_mse'] = sum(result_dict['mse']) / len(result_dict['mse'])
    result_dict['avg_psnr'] = sum(result_dict['psnr']) / len(result_dict['psnr'])
    result_dict['avg_ssim'] = sum(result_dict['ssim']) / len(result_dict['ssim'])
    result_dict['avg_lpips'] = sum(result_dict['lpips']) / len(result_dict['lpips'])
    return result_dict