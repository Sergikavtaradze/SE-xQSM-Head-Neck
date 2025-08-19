#!/usr/bin/env python3
"""
Script to convert .mat files to NIfTI format
"""

import os
import numpy as np
import scipy.io as sio
import nibabel as nib
from pathlib import Path
import argparse
from tqdm import tqdm

def examine_mat_structure(mat_file_path):
    """
    Examine the structure of a .mat file to understand its contents
    """
    try:
        mat_data = sio.loadmat(mat_file_path)
        print(f"Examining {mat_file_path}")
        print("Keys in .mat file:")
        for key in mat_data.keys():
            if not key.startswith('__'):  # Skip metadata keys
                data = mat_data[key]
                print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
                if hasattr(data, 'min') and hasattr(data, 'max'):
                    print(f"    min={data.min():.6f}, max={data.max():.6f}")
        return mat_data
    except Exception as e:
        print(f"Error loading {mat_file_path}: {e}")
        return None

def convert_mat_to_nifti(mat_file_path, output_dir, data_key=None, affine=None):
    """
    Convert a single .mat file to NIfTI format
    
    Args:
        mat_file_path: Path to the .mat file
        output_dir: Directory to save the .nii files
        data_key: Key to extract from .mat file (if None, will try to auto-detect)
        affine: Affine transformation matrix (if None, will use identity)
    """
    try:
        # Load .mat file
        mat_data = sio.loadmat(mat_file_path)
        
        # Get the file name without extension
        file_name = Path(mat_file_path).stem
        
        # If data_key not specified, try to find the main data field
        if data_key is None:
            # Find non-metadata keys
            data_keys = [key for key in mat_data.keys() if not key.startswith('__')]
            if len(data_keys) == 1:
                data_key = data_keys[0]
            else:
                # Try common field names for volume data
                common_keys = ['vol', 'volume', 'data', 'field', 'image']
                for key in common_keys:
                    if key in mat_data:
                        data_key = key
                        break
                
                if data_key is None and data_keys:
                    # Just use the first available key
                    data_key = data_keys[0]
        
        if data_key is None or data_key not in mat_data:
            print(f"Could not find valid data key in {mat_file_path}")
            return False
        
        # Extract the volume data
        volume_data = mat_data[data_key]
        
        # Ensure the data is 3D
        if volume_data.ndim != 3:
            print(f"Warning: {mat_file_path} contains {volume_data.ndim}D data, expected 3D")
            if volume_data.ndim > 3:
                # Take the first 3D slice if it's 4D
                volume_data = volume_data[:, :, :, 0] if volume_data.shape[-1] == 1 else volume_data[:, :, :, volume_data.shape[-1]//2]
            elif volume_data.ndim < 3:
                print(f"Skipping {mat_file_path}: data is {volume_data.ndim}D")
                return False
        
        # Convert to float32
        volume_data = volume_data.astype(np.float32)
        
        # Create affine matrix if not provided (identity with 1mm voxels)
        if affine is None:
            affine = np.eye(4)
        
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(volume_data, affine)
        
        # Save as .nii.gz file
        output_path = os.path.join(output_dir, f"{file_name}.nii.gz")
        nib.save(nifti_img, output_path)
        
        return True
        
    except Exception as e:
        print(f"Error converting {mat_file_path}: {e}")
        return False

def convert_all_mat_files(input_dir, output_dir, data_key=None):
    """
    Convert all .mat files in a directory to NIfTI format
    
    Args:
        input_dir: Directory containing .mat files
        output_dir: Directory to save .nii files
        data_key: Key to extract from .mat files (if None, auto-detect)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .mat files
    mat_files = list(Path(input_dir).glob("*.mat"))
    
    if not mat_files:
        print(f"No .mat files found in {input_dir}")
        return
    
    print(f"Found {len(mat_files)} .mat files to convert")
    
    # If data_key not specified, examine the first file to determine the structure
    if data_key is None:
        print("Auto-detecting data structure from first file...")
        first_file = mat_files[0]
        mat_data = examine_mat_structure(first_file)
        if mat_data:
            data_keys = [key for key in mat_data.keys() if not key.startswith('__')]
            if len(data_keys) == 1:
                data_key = data_keys[0]
                print(f"Using data key: {data_key}")
            else:
                print(f"Multiple data keys found: {data_keys}")
                # Let the conversion function handle auto-detection
    
    # Convert all files with progress bar
    successful_conversions = 0
    failed_conversions = 0
    
    for mat_file in tqdm(mat_files, desc="Converting files"):
        success = convert_mat_to_nifti(mat_file, output_dir, data_key)
        if success:
            successful_conversions += 1
        else:
            failed_conversions += 1
    
    print(f"\nConversion complete!")
    print(f"Successfully converted: {successful_conversions} files")
    print(f"Failed conversions: {failed_conversions} files")

def main():
    parser = argparse.ArgumentParser(description="Convert .mat files to NIfTI format")
    parser.add_argument("--input_dir", "-i", default="simulated_volumes_1000", 
                        help="Directory containing .mat files")
    parser.add_argument("--output_dir", "-o", default="simulated_volumes_1000_nifti",
                        help="Directory to save .nii files")
    parser.add_argument("--data_key", "-k", default=None,
                        help="Key to extract from .mat files (auto-detect if not specified)")
    parser.add_argument("--examine_only", action="store_true",
                        help="Only examine the structure of the first .mat file")
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    if args.examine_only:
        # Just examine the first .mat file
        mat_files = list(Path(input_dir).glob("*.mat"))
        if mat_files:
            examine_mat_structure(mat_files[0])
        else:
            print(f"No .mat files found in {input_dir}")
    else:
        # Convert all files
        convert_all_mat_files(input_dir, output_dir, args.data_key)

if __name__ == "__main__":
    main()


#### Usage Examples ####

# # Convert all files (auto-detect data field)
# python mat_to_NiFti.py --input_dir simulated_volumes_1000 --output_dir output_folder

# # Specify a specific data field
# python mat_to_NiFti.py --input_dir input_folder --output_dir output_folder --data_key vol_field

# # Just examine file structure without converting
# python mat_to_NiFti.py --input_dir input_folder --examine_only