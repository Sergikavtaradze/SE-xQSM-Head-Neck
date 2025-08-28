import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import nibabel as nib
import sys
import argparse
from collections import OrderedDict

# Add parent and training directories to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../training')))

from TrainingDataLoadHN import QSMDataSet
from xQSM import xQSM
from utils import psnr, ssim, zero_padding, zero_removing, compute_xsim

def evaluate_model(model, val_loader, device, save_dir='./qualitative_results'):
    results = {}
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for inputs, targets, names, input_affine, target_affine in tqdm(val_loader, desc='Evaluating'):
            inputs = inputs.squeeze(0)  # In case we are padding
            targets = targets.squeeze(0).squeeze(0)  # PSNR/SSIM calculations
            
            # Pad input if needed
            pad_info = None
            if any(dim % 8 != 0 for dim in inputs.shape[1:]):  # Check D, H, W
                inputs_padded, pad_info = zero_padding(inputs, factor=8)
            else:
                inputs_padded = inputs

            outputs_padded = model(inputs_padded.unsqueeze(0).to(device))
            
            # Remove padding if applied
            if pad_info is not None:
                outputs = zero_removing(outputs_padded.squeeze(0).squeeze(0), pad_info)  # Remove batch/channel dims
            else:
                outputs = outputs_padded.squeeze(0).squeeze(0)

            # Calculate metrics
            ssim_values = []
            psnr_val = psnr(outputs.cpu().numpy(), targets.cpu().numpy())

            for slice_idx in range(outputs.shape[-1]):  # iterate through slices (assuming D,H,W format)
                output_slice = outputs[..., slice_idx].unsqueeze(0).unsqueeze(0)  # add batch and channel dims
                target_slice = targets[..., slice_idx].unsqueeze(0).unsqueeze(0)
                
                # Only calculate SSIM for slices with non-zero values
                if torch.sum(target_slice) > 0:
                    ssim_val = ssim(output_slice, target_slice)
                    ssim_values.append(ssim_val.item())
            
            # Average SSIM over all valid slices
            mean_ssim = np.mean(ssim_values) if ssim_values else 0.0
            
            # Calculate xSIM
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            xsim_val, ssim_xsim_val, _ = compute_xsim(outputs_np, targets_np)
            
            # Save per-subject metrics
            subj_name = names[0]
            results[subj_name] = {'psnr': float(psnr_val), 'ssim': float(mean_ssim), 'ssim_xsim': float(ssim_xsim_val), 'xsim': float(xsim_val)}
            print(f"Subject {subj_name}: PSNR={psnr_val:.3f}, SSIM={mean_ssim:.4f}, SSIM_xSIM={ssim_xsim_val:.4f}, xSIM={xsim_val:.4f}")

            # Save as NIfTI
            nib.save(nib.Nifti1Image(inputs.squeeze().cpu().numpy(), input_affine.squeeze()), os.path.join(save_dir, f"{subj_name}_input.nii.gz"))
            nib.save(nib.Nifti1Image(targets.squeeze().cpu().numpy(), target_affine.squeeze()), os.path.join(save_dir, f"{subj_name}_target.nii.gz"))
            nib.save(nib.Nifti1Image(outputs.squeeze().cpu().numpy(), input_affine.squeeze()), os.path.join(save_dir, f"{subj_name}_output.nii.gz"))

    # Write per-subject metrics and overall averages to a .txt file
    metrics_path = os.path.join(save_dir, "metrics.txt")
    subjects = sorted(results.keys())
    avg_psnr = float(np.mean([results[s]['psnr'] for s in subjects])) if subjects else 0.0
    avg_ssim = float(np.mean([results[s]['ssim'] for s in subjects])) if subjects else 0.0
    avg_ssim_xsim = float(np.mean([results[s]['ssim_xsim'] for s in subjects])) if subjects else 0.0
    avg_xsim = float(np.mean([results[s]['xsim'] for s in subjects])) if subjects else 0.0

    with open(metrics_path, 'w') as f:
        f.write("Subject\tPSNR\tSSIM\tSSIM_xSIM\txSIM\n")
        for s in subjects:
            f.write(f"{s}\t{results[s]['psnr']:.3f}\t{results[s]['ssim']:.4f}\t{results[s]['ssim_xsim']:.4f}\t{results[s]['xsim']:.4f}\n")
        f.write(f"\nAverage\t{avg_psnr:.3f}\t{avg_ssim:.4f}\t{avg_ssim_xsim:.4f}\t{avg_xsim:.4f}\n")

    print(f"Saved metrics to: {metrics_path}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate xQSM or xQSM+SE on validation set')
    parser.add_argument('--data_directory', type=str, required=True, help='Path to validation data directory')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to model weights (.pth)')
    parser.add_argument('--save_dir', type=str, required=True, help='Folder name for saving qualitative results')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--encoding_depth', type=int, default=2, help='Encoding depth for xQSM')
    parser.add_argument('--ini_chNo', type=int, default=64, help='Initial number of channels for xQSM')
    parser.add_argument('--split', type=str, required=True, help='Split for evaluation')
    parser.add_argument('--squeeze_exc', action='store_true', help='Enable Squeeze-and-Excitation (SE) layers')
    
    args = parser.parse_args()

    print('Evaluation configuration:')
    for k, v in vars(args).items():
        print(f'  {k}: {v}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dataset = QSMDataSet(root=args.data_directory, split_type=args.split, include_noise=False, affine=True)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"This is the squeeze exc: {args.squeeze_exc}")
    model = xQSM(EncodingDepth=args.encoding_depth, ini_chNo=args.ini_chNo, use_se=args.squeeze_exc)

    state_dict = torch.load(args.weights_path, map_location=device, weights_only=True)

    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = OrderedDict((k.replace('module.', '', 1), v) for k, v in state_dict.items())

    model.load_state_dict(state_dict)  # keep strict=True default
    model.to(device)
    model.eval()

    results = evaluate_model(model, val_loader, device, save_dir=args.save_dir)
    

if __name__ == '__main__':
    main() 

# python run_val_eval.py --data_directory "/Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_data" --weights_path "/Users/sirbucks/Documents/xQSM/2025-Summer-Research/xQSM/ckpt/Aug7/ckpt/Jul31_bs32_ep100_lr1e-4_ps48_xQSM_NoFreeze/xQSM_TransferLearning_Best.pth"

# python run_val_eval.py --data_directory "/Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_data" --weights_path "/Users/sirbucks/Documents/2025-Summer-Research/xQSM/ckpt/synthetic_Aug20/ckpt/xQSM_synthetic_bs32_ep100_lr4e-4_ps48_SE/xQSM_Synthetic_Best.pth" --save_dir "./xQSM_synthetic_bs32_ep100_lr4e-4_ps48_SE"

# python run_val_eval.py --data_directory "/Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_data" --weights_path "/Users/sirbucks/Documents/xQSM/2025-Summer-Research/xQSM/ckpt/synthetic_Aug20/ckpt/xQSM_synthetic_bs32_ep100_lr4e-4_ps48/xQSM_Synthetic_Best.pth" --save_dir "./xQSM_synthetic_bs32_ep100_lr4e-4_ps48" --ini_chNo 32

# python run_val_eval.py --data_directory "/Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_data" --weights_path "/Users/sirbucks/Documents/xQSM/2025-Summer-Research/xQSM/ckpt/synthetic_Aug20/ckpt/xQSM_synthetic_bs32_ep100_lr4e-4_ps48_SE/xQSM_Synthetic_Best.pth" --save_dir "./xQSM_synthetic_bs32_ep100_lr4e-4_ps48_SE" --ini_chNo 32 --squeeze_exc

# python run_val_eval.py --data_directory "/Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_data" --weights_path "/Users/sirbucks/Documents/xQSM/2025-Summer-Research/xQSM/ckpt/Aug8_11/Aug11_bs32_ep100_lr4e-4_ps48_xQSM_SE/xQSM_TransferLearning_Best.pth" --save_dir "./Inference_Results/Aug11_bs32_ep100_lr4e-4_ps48_xQSM_SE" --ini_chNo 64 --squeeze_exc