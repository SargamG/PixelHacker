"""
Generate masks for PixelHacker inference
Creates masks compatible with PixelHacker format (255 = inpaint, 0 = keep)
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse


def create_random_rectangular_mask(size, mask_ratio_range=(0.2, 0.5)):
    """
    Create random rectangular mask for PixelHacker
    
    Args:
        size: (H, W) tuple
        mask_ratio_range: (min, max) ratio of area to mask
        
    Returns:
        PIL Image (0 = keep, 255 = inpaint)
    """
    H, W = size
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # Random mask dimensions
    mask_h_ratio = np.random.uniform(*mask_ratio_range)
    mask_w_ratio = np.random.uniform(*mask_ratio_range)
    
    mask_h = int(H * mask_h_ratio)
    mask_w = int(W * mask_w_ratio)
    
    # Random position
    y = np.random.randint(0, H - mask_h + 1)
    x = np.random.randint(0, W - mask_w + 1)
    
    # Set masked region to 255 (white = inpaint this area)
    mask[y:y+mask_h, x:x+mask_w] = 255
    
    return Image.fromarray(mask)


def create_center_mask(size, mask_ratio=0.4):
    """
    Create center-cropped mask
    
    Args:
        size: (H, W) tuple
        mask_ratio: Ratio of image to mask (centered)
        
    Returns:
        PIL Image (0 = keep, 255 = inpaint)
    """
    H, W = size
    mask = np.zeros((H, W), dtype=np.uint8)
    
    mask_h = int(H * mask_ratio)
    mask_w = int(W * mask_ratio)
    
    y = (H - mask_h) // 2
    x = (W - mask_w) // 2
    
    mask[y:y+mask_h, x:x+mask_w] = 255
    
    return Image.fromarray(mask)


def create_random_brush_mask(size, mask_ratio_range=(0.2, 0.5), num_strokes=10):
    """
    Create random brush stroke mask
    
    Args:
        size: (H, W) tuple
        mask_ratio_range: Target mask ratio range
        num_strokes: Number of brush strokes
        
    Returns:
        PIL Image (0 = keep, 255 = inpaint)
    """
    H, W = size
    mask = np.zeros((H, W), dtype=np.uint8)
    
    target_ratio = np.random.uniform(*mask_ratio_range)
    current_ratio = 0
    
    attempts = 0
    max_attempts = num_strokes * 2
    
    while current_ratio < target_ratio and attempts < max_attempts:
        # Random stroke parameters
        thickness = np.random.randint(10, 40)
        start_point = (np.random.randint(0, W), np.random.randint(0, H))
        
        # Create random path
        for _ in range(np.random.randint(5, 20)):
            end_point = (
                np.clip(start_point[0] + np.random.randint(-100, 100), 0, W-1),
                np.clip(start_point[1] + np.random.randint(-100, 100), 0, H-1)
            )
            
            # Draw line on mask
            from PIL import ImageDraw
            temp_img = Image.fromarray(mask)
            draw = ImageDraw.Draw(temp_img)
            draw.line([start_point, end_point], fill=255, width=thickness)
            mask = np.array(temp_img)
            
            start_point = end_point
        
        current_ratio = np.sum(mask > 0) / (H * W)
        attempts += 1
    
    return Image.fromarray(mask)


def convert_celeba_mask_to_pixelhacker(mask_img):
    """
    Convert CelebA-style mask (1=keep, 0=masked) to PixelHacker format (0=keep, 255=inpaint)
    
    Args:
        mask_img: PIL Image or numpy array
        
    Returns:
        PIL Image in PixelHacker format
    """
    if isinstance(mask_img, Image.Image):
        mask = np.array(mask_img.convert('L'))
    else:
        mask = mask_img
    
    # Normalize to 0-1 range
    mask = mask.astype(np.float32) / 255.0
    
    # Invert: 1->0 (keep), 0->255 (inpaint)
    mask = (1 - mask) * 255
    
    return Image.fromarray(mask.astype(np.uint8))


def generate_masks_for_pixelhacker(
    img_dir,
    mask_dir,
    mask_type='random_rectangular',
    mask_ratio_range=(0.2, 0.5),
    file_extensions=('.png', '.jpg', '.jpeg')
):
    """
    Generate masks for PixelHacker inference
    
    Args:
        img_dir: Directory containing images
        mask_dir: Output directory for masks
        mask_type: Type of mask ('random_rectangular', 'center', 'brush')
        mask_ratio_range: (min, max) ratio of area to mask
        file_extensions: Valid image file extensions
    """
    img_path = Path(img_dir)
    mask_path = Path(mask_dir)
    mask_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in file_extensions:
        image_files.extend(list(img_path.glob(f'*{ext}')))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {img_dir}")
    
    print(f"Found {len(image_files)} images")
    print(f"Mask type: {mask_type}")
    print(f"Mask ratio range: {mask_ratio_range}")
    print(f"Output directory: {mask_dir}")
    print(f"Mask format: PixelHacker (0=keep, 255=inpaint)")
    print()
    
    # Generate masks
    for img_file in tqdm(image_files, desc="Generating masks"):
        # Load image to get size
        try:
            img = Image.open(img_file).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            continue
        
        # Generate mask based on type
        if mask_type == 'random_rectangular':
            mask = create_random_rectangular_mask(
                size=img.size[::-1],  # (H, W)
                mask_ratio_range=mask_ratio_range
            )
        elif mask_type == 'center':
            mask = create_center_mask(
                size=img.size[::-1],
                mask_ratio=mask_ratio_range[0]
            )
        elif mask_type == 'brush':
            mask = create_random_brush_mask(
                size=img.size[::-1],
                mask_ratio_range=mask_ratio_range
            )
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
        
        # Save mask with same filename as image
        mask_file = mask_path / img_file.name
        mask.save(mask_file)
    
    print(f"\nMasks generated successfully!")
    print(f"Saved to: {mask_dir}")
    print(f"\nNext steps:")
    print(f"1. Copy images to PixelHacker/imgs/")
    print(f"2. Copy masks to PixelHacker/masks/")
    print(f"3. Run: python infer_pixelhacker.py --config config/PixelHacker_sdvae_f8d4.yaml --weight weight/ft_ffhq/diffusion_pytorch_model.bin")


def main():
    parser = argparse.ArgumentParser(
        description='Generate masks for PixelHacker inference'
    )
    parser.add_argument(
        '--img_dir',
        type=str,
        required=True,
        help='Directory containing input images'
    )
    parser.add_argument(
        '--mask_dir',
        type=str,
        default='./masks',
        help='Output directory for masks (default: ./masks)'
    )
    parser.add_argument(
        '--mask_type',
        type=str,
        default='random_rectangular',
        choices=['random_rectangular', 'center', 'brush'],
        help='Type of mask to generate'
    )
    parser.add_argument(
        '--min_ratio',
        type=float,
        default=0.2,
        help='Minimum mask ratio (default: 0.2)'
    )
    parser.add_argument(
        '--max_ratio',
        type=float,
        default=0.5,
        help='Maximum mask ratio (default: 0.5)'
    )
    parser.add_argument(
        '--copy_to_pixelhacker',
        type=str,
        default=None,
        help='If provided, automatically copy images and masks to PixelHacker directory'
    )
    
    args = parser.parse_args()
    
    # Generate masks
    generate_masks_for_pixelhacker(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        mask_type=args.mask_type,
        mask_ratio_range=(args.min_ratio, args.max_ratio)
    )
    
    # Optionally copy to PixelHacker directory
    if args.copy_to_pixelhacker:
        import shutil
        pixelhacker_imgs = os.path.join(args.copy_to_pixelhacker, 'imgs')
        pixelhacker_masks = os.path.join(args.copy_to_pixelhacker, 'masks')
        
        os.makedirs(pixelhacker_imgs, exist_ok=True)
        os.makedirs(pixelhacker_masks, exist_ok=True)
        
        print(f"\nCopying files to PixelHacker directory...")
        
        # Copy images
        for img_file in Path(args.img_dir).glob('*'):
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                shutil.copy(img_file, os.path.join(pixelhacker_imgs, img_file.name))
        
        # Copy masks
        for mask_file in Path(args.mask_dir).glob('*'):
            if mask_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                shutil.copy(mask_file, os.path.join(pixelhacker_masks, mask_file.name))
        
        print(f"Files copied to: {args.copy_to_pixelhacker}")


if __name__ == "__main__":
    # Example usage:
    # python generate_pixelhacker_masks.py \
    #     --img_dir /path/to/ffhq/images \
    #     --mask_dir ./masks_output \
    #     --mask_type random_rectangular \
    #     --min_ratio 0.2 \
    #     --max_ratio 0.5
    
    main()
