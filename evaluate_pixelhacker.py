"""
Evaluation script for PixelHacker predictions
Adapted from your original CelebAMaskedDataset evaluation code
Compatible with your existing metric computation logic
"""

import torch
from torchmetrics import StructuralSimilarityIndexMeasure
import lpips
from pytorch_fid import fid_score
import os
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
import shutil
import json


class PixelHackerDataset(Dataset):
    """
    Dataset for loading PixelHacker predictions and ground truth
    Designed to match your CelebAMaskedDataset interface
    
    Args:
        pred_dir: Directory containing PixelHacker predictions (./outputs/)
        gt_dir: Directory containing ground truth original images
        mask_dir: Optional directory containing masks used for inference
        transform: Torchvision transforms
    """
    def __init__(self, pred_dir, gt_dir, mask_dir=None, transform=None):
        self.pred_dir = pred_dir
        self.gt_dir = gt_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get all prediction files
        self.pred_files = sorted([
            f for f in os.listdir(pred_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        # Verify matching ground truth files exist
        missing_files = []
        for pred_file in self.pred_files:
            gt_path = os.path.join(gt_dir, pred_file)
            if not os.path.exists(gt_path):
                missing_files.append(pred_file)
        
        if missing_files:
            print(f"Warning: {len(missing_files)} prediction files have no matching ground truth")
            print(f"First few missing: {missing_files[:5]}")
            # Remove files without ground truth
            self.pred_files = [f for f in self.pred_files if f not in missing_files]
        
        if len(self.pred_files) == 0:
            raise ValueError(f"No valid prediction-ground truth pairs found!")
        
        print(f"Found {len(self.pred_files)} valid image pairs")
    
    def __len__(self):
        return len(self.pred_files)
    
    def __getitem__(self, idx):
        filename = self.pred_files[idx]
        
        # Load prediction (inpainted result)
        pred_path = os.path.join(self.pred_dir, filename)
        pred_img = Image.open(pred_path).convert('RGB')
        
        # Load ground truth
        gt_path = os.path.join(self.gt_dir, filename)
        gt_img = Image.open(gt_path).convert('RGB')
        
        # Load mask if available
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, filename)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                # Convert PixelHacker mask (255=inpaint) to your format (1=keep, 0=masked)
                mask_array = np.array(mask, dtype=np.float32) / 255.0
                mask_array = 1.0 - mask_array  # Invert
                mask = Image.fromarray((mask_array * 255).astype(np.uint8))
            else:
                # Create dummy mask if file doesn't exist
                mask = Image.new('L', gt_img.size, 255)
        else:
            # No mask directory provided - create dummy mask
            mask = Image.new('L', gt_img.size, 255)
        
        # Apply transforms
        if self.transform:
            pred_img = self.transform(pred_img)
            gt_img = self.transform(gt_img)
            mask = transforms.ToTensor()(mask)
        
        # Return in your expected format: (masked_img, mask, gt_img)
        # Here, we return prediction as "masked_img" since that's what gets evaluated
        return pred_img, mask, gt_img


def compute_psnr(pred, gt, eps=1e-8):
    """
    Compute PSNR between prediction and ground truth
    
    Args:
        pred: Predicted images in [0,1], shape (B,3,H,W)
        gt: Ground truth images in [0,1], shape (B,3,H,W)
        eps: Small constant to avoid division by zero
        
    Returns:
        Average PSNR across batch
    """
    mse = F.mse_loss(pred, gt, reduction='none')
    mse = mse.mean(dim=[1, 2, 3])  # Per-sample MSE
    psnr = 10 * torch.log10(1.0 / (mse + eps))
    return psnr.mean().item()


def evaluate_pixelhacker(
    pred_dir,
    gt_dir,
    mask_dir=None,
    batch_size=16,
    img_size=(256, 256),
    dataset_name="FFHQ",
    save_temp_for_fid=True,
    cleanup_temp=True
):
    """
    Evaluate PixelHacker predictions using FID, LPIPS, SSIM, and PSNR
    Matches your evaluate_model_with_latent function signature and logic
    
    Args:
        pred_dir: Directory containing PixelHacker predictions
        gt_dir: Directory containing ground truth images
        mask_dir: Optional directory containing masks
        batch_size: Batch size for evaluation
        img_size: Target image size (H, W)
        dataset_name: Name of dataset (FFHQ, CelebA-HQ, etc.)
        save_temp_for_fid: Whether to save temporary images for FID computation
        cleanup_temp: Whether to cleanup temporary directories after FID
        
    Returns:
        Dictionary containing evaluation metrics
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print(f"Evaluating PixelHacker on {dataset_name}")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Predictions: {pred_dir}")
    print(f"Ground truth: {gt_dir}")
    if mask_dir:
        print(f"Masks: {mask_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"{'='*70}\n")
    
    # Verify directories exist
    if not os.path.exists(pred_dir):
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not os.path.exists(gt_dir):
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
    
    # ------------------------------
    # Dataset
    # ------------------------------
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    
    test_dataset = PixelHackerDataset(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        mask_dir=mask_dir,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )
    
    # ------------------------------
    # Metrics
    # ------------------------------
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Temporary directories for FID computation
    if save_temp_for_fid:
        pred_fid_dir = f"pred_tmp_pixelhacker_{dataset_name.lower().replace('-', '_')}"
        gt_fid_dir = f"gt_tmp_pixelhacker_{dataset_name.lower().replace('-', '_')}"
        os.makedirs(pred_fid_dir, exist_ok=True)
        os.makedirs(gt_fid_dir, exist_ok=True)
    
    # Storage for metrics
    lpips_scores = []
    ssim_scores = []
    psnr_scores = []
    idx_counter = 0
    
    # ------------------------------
    # Inference Loop
    # ------------------------------
    print("Computing LPIPS, SSIM, and PSNR metrics...")
    with torch.no_grad():
        for pred_img, mask, gt_img in tqdm(
            test_loader, desc=f"Evaluating PixelHacker | {dataset_name}"
        ):
            pred_img = pred_img.to(device)
            gt_img = gt_img.to(device)
            
            # Ensure predictions are in valid range [0, 1]
            pred = pred_img.clamp(0.0, 1.0)
            
            # -------- LPIPS --------
            # LPIPS expects images in range [-1, 1]
            pred_lp = pred * 2 - 1
            gt_lp = gt_img * 2 - 1
            lp = lpips_fn(pred_lp, gt_lp)
            
            # Handle different output shapes
            if lp.dim() > 1:
                lp = lp.view(lp.size(0), -1).mean(dim=1)
            lpips_scores.extend(lp.cpu().tolist())
            
            # -------- SSIM --------
            ssim_val = float(ssim_fn(pred, gt_img))
            ssim_scores.append(ssim_val)
            
            # -------- PSNR --------
            psnr_val = compute_psnr(pred, gt_img)
            psnr_scores.append(psnr_val)
            
            # -------- Save for FID --------
            if save_temp_for_fid:
                for i in range(pred.size(0)):
                    save_image(pred[i].cpu(), f"{pred_fid_dir}/{idx_counter}.png")
                    save_image(gt_img[i].cpu(), f"{gt_fid_dir}/{idx_counter}.png")
                    idx_counter += 1
    
    # ------------------------------
    # FID Computation
    # ------------------------------
    if save_temp_for_fid:
        print("\nComputing FID score...")
        try:
            fid_value = fid_score.calculate_fid_given_paths(
                [pred_fid_dir, gt_fid_dir],
                batch_size=32,
                device=device,
                dims=2048
            )
        except Exception as e:
            print(f"Warning: FID computation failed: {e}")
            fid_value = float('nan')
        
        # Cleanup temporary directories
        if cleanup_temp:
            print("Cleaning up temporary FID directories...")
            shutil.rmtree(pred_fid_dir, ignore_errors=True)
            shutil.rmtree(gt_fid_dir, ignore_errors=True)
    else:
        fid_value = float('nan')
    
    # ------------------------------
    # Aggregate Results
    # ------------------------------
    results = {
        "model": "PixelHacker",
        "dataset": dataset_name,
        "num_images": len(test_dataset),
        "FID": float(fid_value),
        "LPIPS": float(np.mean(lpips_scores)),
        "LPIPS_std": float(np.std(lpips_scores)),
        "SSIM": float(np.mean(ssim_scores)),
        "SSIM_std": float(np.std(ssim_scores)),
        "PSNR": float(np.mean(psnr_scores)),
        "PSNR_std": float(np.std(psnr_scores)),
    }
    
    # ------------------------------
    # Print Results
    # ------------------------------
    print(f"\n{'='*70}")
    print(f"=== Results | PixelHacker | {dataset_name} ===")
    print(f"{'='*70}")
    print(f"Number of images evaluated: {results['num_images']}")
    print(f"{'-'*70}")
    print(f"FID       : {results['FID']:.4f}")
    print(f"LPIPS     : {results['LPIPS']:.4f} ± {results['LPIPS_std']:.4f}")
    print(f"SSIM      : {results['SSIM']:.4f} ± {results['SSIM_std']:.4f}")
    print(f"PSNR      : {results['PSNR']:.4f} ± {results['PSNR_std']:.4f}")
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate PixelHacker predictions using standard metrics'
    )
    parser.add_argument(
        '--pred_dir',
        type=str,
        default='./outputs',
        help='Directory containing PixelHacker predictions (default: ./outputs)'
    )
    parser.add_argument(
        '--gt_dir',
        type=str,
        required=True,
        help='Directory containing ground truth images'
    )
    parser.add_argument(
        '--mask_dir',
        type=str,
        default=None,
        help='Optional directory containing masks used for inference'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='FFHQ',
        choices=['FFHQ', 'CelebA-HQ', 'Places2'],
        help='Name of the dataset being evaluated'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for evaluation (default: 16)'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        nargs=2,
        default=[256, 256],
        help='Image size as (height width) (default: 256 256)'
    )
    parser.add_argument(
        '--no_fid',
        action='store_true',
        help='Skip FID computation (faster but less complete evaluation)'
    )
    parser.add_argument(
        '--keep_temp',
        action='store_true',
        help='Keep temporary FID directories after computation'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Optional output JSON file for results (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_pixelhacker(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        mask_dir=args.mask_dir,
        batch_size=args.batch_size,
        img_size=tuple(args.img_size),
        dataset_name=args.dataset_name,
        save_temp_for_fid=not args.no_fid,
        cleanup_temp=not args.keep_temp
    )
    
    # Save results to JSON file
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = f"pixelhacker_{args.dataset_name.lower().replace('-', '_')}_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    # Example usage:
    # 
    # For FFHQ (256x256):
    # python evaluate_pixelhacker.py \
    #     --pred_dir ./outputs \
    #     --gt_dir /path/to/ffhq/original/images \
    #     --dataset_name FFHQ \
    #     --batch_size 16 \
    #     --img_size 256 256
    #
    # For CelebA-HQ (512x512):
    # python evaluate_pixelhacker.py \
    #     --pred_dir ./outputs \
    #     --gt_dir /path/to/celebahq/original/images \
    #     --dataset_name CelebA-HQ \
    #     --batch_size 8 \
    #     --img_size 512 512
    
    main()
