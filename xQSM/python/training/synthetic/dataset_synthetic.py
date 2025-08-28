"""
PyTorch Dataset and DataLoader for xQSM Synthetic Data Training
Adapted from DeepQSM dataset code for use with xQSM architecture and synthetic data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
import random
import os
import glob
from typing import Tuple, Optional, List, Union
from pathlib import Path


class xQSMSyntheticDataset(Dataset):
    """Dataset for xQSM synthetic data training using patch extraction"""
    
    def __init__(
        self,
        data_directory: str,
        split_type: str = 'train',
        patch_size: Union[int, Tuple[int, int, int]] = 48,
        patches_per_volume: int = 100,
        train_ratio: float = 0.8,
        transform=None,
        preload_data: bool = False,
        normalize: bool = True,
        include_noise: bool = True,
    ):
        """
        Args:
            data_directory: Directory containing the synthetic data
            split_type: 'train', 'val', or 'test'
            patch_size: Size of cubic patches to extract (int or tuple)
            patches_per_volume: Number of patches to extract per volume per epoch
            train_ratio: Ratio of data to use for training (rest for validation)
            transform: Optional transform to apply to patches
            preload_data: Whether to load all data into memory
            normalize: Whether to normalize the data
            include_noise: Whether to add noise augmentation during training
        """
        self.data_directory = data_directory
        self.split_type = split_type
        self.patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.patches_per_volume = patches_per_volume
        self.train_ratio = train_ratio
        self.transform = transform
        self.preload_data = preload_data
        self.normalize = normalize
        self.include_noise = include_noise
        
        # Noise parameters (consistent with xQSM training)
        self.Prob = torch.tensor(0.8)  # 20% probability to add noise
        self.SNRs = torch.tensor([50, 40, 20, 10, 5])  # Noise SNRs
        
        # Find and load data files
        self.files = []
        self.vol_shapes = []
        self._scan_data_directory()
        
        # Split data into train/val
        self._create_data_split()
        
        # Preload data if requested
        if self.preload_data:
            self.input_data, self.target_data = self._load_all_volumes()
        else:
            self.input_data = None
            self.target_data = None
        
        # Calculate total patches for the epoch
        self.total_patches = len(self.split_files) * self.patches_per_volume
    
    def _scan_data_directory(self):
        """Scan directory for synthetic data files"""
        data_path = Path(self.data_directory)
        
        # Look for input and target directories or patterns
        input_pattern = data_path / "vol_field*.nii.gz"
        target_pattern = data_path / "vol_susc*.nii.gz"
        
        input_files = sorted(glob.glob(str(input_pattern)))
        target_files = sorted(glob.glob(str(target_pattern)))
        
        assert len(input_files) == len(target_files), \
            f"Mismatch: {len(input_files)} input files vs {len(target_files)} target files"
        
        for input_file, target_file in zip(input_files, target_files):
            # Load one file to get shape
            nii_data = nib.load(input_file)
            shape = nii_data.shape
            
            input_name = Path(input_file).stem
            target_name = Path(target_file).stem
            
            self.files.append({
                "input": input_file,
                "target": target_file,
                "name": f"{input_name}_{target_name}",
                "shape": shape
            })
            self.vol_shapes.append(shape)
        
        print(f"Found {len(self.files)} volume pairs in {self.data_directory}")
        if not self.files:
            raise RuntimeError(f"No data files found in {self.data_directory}")
    
    def _create_data_split(self):
        """Split data into train/validation sets"""
        num_files = len(self.files)
        num_train = int(num_files * self.train_ratio)
        
        # Create consistent splits
        indices = list(range(num_files))
        random.seed(42)  # For reproducible splits
        random.shuffle(indices)
        
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        
        if self.split_type == 'train':
            self.split_files = [self.files[i] for i in train_indices]
            self.split_shapes = [self.vol_shapes[i] for i in train_indices]
        elif self.split_type == 'val':
            self.split_files = [self.files[i] for i in val_indices]
            self.split_shapes = [self.vol_shapes[i] for i in val_indices]
        else:
            # Use all data for test
            self.split_files = self.files
            self.split_shapes = self.vol_shapes
        
        print(f"Split {self.split_type}: {len(self.split_files)} volumes")
    
    def _load_volume(self, volume_path: str) -> np.ndarray:
        """Load a single volume from file path"""
        return nib.load(volume_path).get_fdata(dtype=np.float32) #training is fp32

    def _load_all_volumes(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load all volumes into memory"""
        print(f"Loading all {len(self.split_files)} volumes into memory...")
        input_data = []
        target_data = []
        
        for i, file_info in enumerate(self.split_files):
            input_vol = self._load_volume(file_info["input"])
            target_vol = self._load_volume(file_info["target"])
            
            if self.normalize:
                input_vol = self._normalize_volume(input_vol)
                target_vol = self._normalize_volume(target_vol)
            
            input_data.append(input_vol)
            target_data.append(target_vol)
        
        print(f"Loaded {len(input_data)} volume pairs")
        return input_data, target_data
    
    # def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
    #     """Normalize volume to [-1, 1] range"""
    #     vol_min = volume.min()
    #     vol_max = volume.max()
    #     if vol_max > vol_min:
    #         volume = 2 * (volume - vol_min) / (vol_max - vol_min) - 1
    #     return volume
    
    def _extract_random_patch(
        self, 
        input_volume: np.ndarray, 
        target_volume: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a random patch from volumes - Same as DeepQSM implementation"""
        x_max, y_max, z_max = input_volume.shape
        
        # Ensure patch fits in volume
        if x_max < self.patch_size[0] or y_max < self.patch_size[1] or z_max < self.patch_size[2]:
            # Pad volume if it's smaller than patch size
            pad_x = max(0, self.patch_size[0] - x_max)
            pad_y = max(0, self.patch_size[1] - y_max) 
            pad_z = max(0, self.patch_size[2] - z_max)
            
            input_volume = np.pad(input_volume, 
                                ((0, pad_x), (0, pad_y), (0, pad_z)), 
                                mode='constant', constant_values=0)
            target_volume = np.pad(target_volume,
                                 ((0, pad_x), (0, pad_y), (0, pad_z)),
                                 mode='constant', constant_values=0)
            x_max, y_max, z_max = input_volume.shape
        
        # Random starting points
        start_x = random.randint(0, x_max - self.patch_size[0])
        start_y = random.randint(0, y_max - self.patch_size[1])
        start_z = random.randint(0, z_max - self.patch_size[2])
        
        # Extract patches
        input_patch = input_volume[
            start_x:start_x + self.patch_size[0],
            start_y:start_y + self.patch_size[1],
            start_z:start_z + self.patch_size[2]
        ]
        
        target_patch = target_volume[
            start_x:start_x + self.patch_size[0],
            start_y:start_y + self.patch_size[1],
            start_z:start_z + self.patch_size[2]
        ]
        
        return input_patch, target_patch
    
    def __len__(self) -> int:
        return self.total_patches
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        # Determine which volume this patch comes from
        volume_idx = idx // self.patches_per_volume
        volume_idx = volume_idx % len(self.split_files)  # Handle overflow
        
        # Get file info
        file_info = self.split_files[volume_idx]
        name = file_info["name"]
        
        # Load volume data
        if self.input_data is not None:
            input_volume = self.input_data[volume_idx]
            target_volume = self.target_data[volume_idx]
        else:
            input_volume = self._load_volume(file_info["input"])
            target_volume = self._load_volume(file_info["target"])
            
            if self.normalize:
                input_volume = self._normalize_volume(input_volume)
                target_volume = self._normalize_volume(target_volume)
        
        # Extract random patch - same method as DeepQSM
        input_patch, target_patch = self._extract_random_patch(input_volume, target_volume)
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_patch).float()
        target_tensor = torch.from_numpy(target_patch).float()
        
        # Add noise augmentation if enabled and training
        if self.include_noise and self.split_type == 'train':
            tmp = torch.rand(1)
            if tmp > self.Prob:
                tmp_mask = input_tensor != 0
                tmp_idx = torch.randint(5, (1,1))
                tmp_SNR = self.SNRs[tmp_idx]
                input_tensor = AddNoise(input_tensor, tmp_SNR)
        
        # Add channel dimension (consistent with xQSM training)
        input_tensor = input_tensor.unsqueeze(0)  # [1, H, W, D]
        target_tensor = target_tensor.unsqueeze(0)  # [1, H, W, D]
        
        # Apply transforms if provided
        if self.transform is not None:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)
        
        return input_tensor, target_tensor, name


def AddNoise(ins, SNR):
    """Add noise to input tensor based on SNR - Same as xQSM implementation"""
    sigPower = SigPower(ins)
    noisePower = sigPower / SNR
    noise = torch.sqrt(noisePower.float()) * torch.randn(ins.size()).float()
    return ins + noise


def SigPower(ins):
    """Calculate signal power - Same as xQSM implementation"""
    ll = torch.numel(ins)
    tmp1 = torch.sum(ins ** 2)
    return torch.div(tmp1, ll)


def create_xqsm_synthetic_dataloaders(
    data_directory: str,
    patch_size: Union[int, Tuple[int, int, int]] = 48,
    patches_per_volume: int = 100,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    normalize: bool = True,
    include_noise: bool = True,
    persistent_workers: bool = True,            # NEW
    prefetch_factor: int = 4,                   # NEW
    preload_data: bool = True,                  # NEW (passed into datasets)
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    
    train_dataset = xQSMSyntheticDataset(
        data_directory=data_directory,
        split_type='train',
        patch_size=patch_size,
        patches_per_volume=patches_per_volume,
        train_ratio=train_ratio,
        normalize=normalize,
        include_noise=include_noise,
        preload_data=preload_data,              # NEW
        **dataset_kwargs
    )
    
    val_dataset = xQSMSyntheticDataset(
        data_directory=data_directory,
        split_type='val',
        patch_size=patch_size,
        patches_per_volume=patches_per_volume // 2,
        train_ratio=train_ratio,
        normalize=normalize,
        include_noise=False,
        preload_data=preload_data,              # NEW
        **dataset_kwargs
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=persistent_workers,  # NEW
        prefetch_factor=prefetch_factor         # NEW
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=persistent_workers,  # NEW
        prefetch_factor=prefetch_factor         # NEW
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset with synthetic data
    print("Testing xQSM Synthetic Dataset...")
    
    # Example usage - update path to your converted synthetic data
    data_dir = "/Users/sirbucks/Documents/xQSM/2025-Summer-Research/simulated_volumes_1000_nifti"
    
    try:
        # Create dataloaders
        train_loader, val_loader = create_xqsm_synthetic_dataloaders(
            data_directory=data_dir,
            patch_size=48,
            patches_per_volume=50,
            batch_size=4,
            num_workers=0,
            train_ratio=0.8
        )
        
        print(f"Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        # Test loading a batch
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        print(f"Train batch shapes: Input {train_batch[0].shape}, Target {train_batch[1].shape}")
        print(f"Val batch shapes: Input {val_batch[0].shape}, Target {val_batch[1].shape}")
        print("Dataset test completed successfully!")
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
        print("Make sure the data directory path is correct and contains .nii.gz files")
