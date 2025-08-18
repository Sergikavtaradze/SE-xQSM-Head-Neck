"""
PyTorch Dataset and DataLoader for QSM Deep Learning
Handles patch creation and data loading for QSM reconstruction
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
import random
from typing import Tuple, Optional, List, Union


class QSMPatchDataset(Dataset):
    """Dataset for QSM patch-based training"""
    
    def __init__(
        self,
        input_volumes: Union[List[str], List[np.ndarray]],
        target_volumes: Union[List[str], List[np.ndarray]],
        patch_size: int = 64,
        patches_per_volume: int = 100,
        transform=None,
        preload_data: bool = True,
        normalize: bool = True,
        data_type: str = "simulation"  # "simulation" or "brain"
    ):
        """
        Args:
            input_volumes: List of file paths or numpy arrays for input volumes
            target_volumes: List of file paths or numpy arrays for target volumes  
            patch_size: Size of cubic patches to extract
            patches_per_volume: Number of patches to extract per volume per epoch
            transform: Optional transform to apply to patches
            preload_data: Whether to load all data into memory
            normalize: Whether to normalize the data
            data_type: Type of data ("simulation" or "brain")
        """
        self.input_volumes = input_volumes
        self.target_volumes = target_volumes
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.transform = transform
        self.normalize = normalize
        self.data_type = data_type
        
        assert len(input_volumes) == len(target_volumes), \
            "Number of input and target volumes must match"
        
        self.num_volumes = len(input_volumes)
        self.total_patches = self.num_volumes * patches_per_volume
        
        # Preload data if requested
        if preload_data:
            self.input_data, self.target_data = self._load_all_volumes()
        else:
            self.input_data = None
            self.target_data = None
    
    def _load_volume(self, volume_path: Union[str, np.ndarray]) -> np.ndarray:
        """Load a single volume from file path or return array directly"""
        if isinstance(volume_path, str):
            if volume_path.endswith('.nii') or volume_path.endswith('.nii.gz'):
                return nib.load(volume_path).get_fdata()
            else:
                return np.load(volume_path)
        else:
            return volume_path
    
    def _load_all_volumes(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load all volumes into memory"""
        print("Loading all volumes into memory...")
        input_data = []
        target_data = []
        
        for i in range(self.num_volumes):
            input_vol = self._load_volume(self.input_volumes[i])
            target_vol = self._load_volume(self.target_volumes[i])
            
            if self.normalize:
                input_vol = self._normalize_volume(input_vol)
                target_vol = self._normalize_volume(target_vol)
            
            input_data.append(input_vol)
            target_data.append(target_vol)
            
        print(f"Loaded {len(input_data)} volume pairs")
        return input_data, target_data
    
    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume to [-1, 1] range"""
        vol_min = volume.min()
        vol_max = volume.max()
        if vol_max > vol_min:
            volume = 2 * (volume - vol_min) / (vol_max - vol_min) - 1
        return volume
    
    def _extract_random_patch(
        self, 
        input_volume: np.ndarray, 
        target_volume: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a random patch from volumes"""
        x_max, y_max, z_max = input_volume.shape
        
        # Ensure patch fits in volume
        if x_max < self.patch_size or y_max < self.patch_size or z_max < self.patch_size:
            # Pad volume if it's smaller than patch size
            pad_x = max(0, self.patch_size - x_max)
            pad_y = max(0, self.patch_size - y_max) 
            pad_z = max(0, self.patch_size - z_max)
            
            input_volume = np.pad(input_volume, 
                                ((0, pad_x), (0, pad_y), (0, pad_z)), 
                                mode='constant', constant_values=0)
            target_volume = np.pad(target_volume,
                                 ((0, pad_x), (0, pad_y), (0, pad_z)),
                                 mode='constant', constant_values=0)
            x_max, y_max, z_max = input_volume.shape
        
        # Random starting points
        start_x = random.randint(0, x_max - self.patch_size)
        start_y = random.randint(0, y_max - self.patch_size)
        start_z = random.randint(0, z_max - self.patch_size)
        
        # Extract patches
        input_patch = input_volume[
            start_x:start_x + self.patch_size,
            start_y:start_y + self.patch_size,
            start_z:start_z + self.patch_size
        ]
        
        target_patch = target_volume[
            start_x:start_x + self.patch_size,
            start_y:start_y + self.patch_size,
            start_z:start_z + self.patch_size
        ]
        
        return input_patch, target_patch
    
    def __len__(self) -> int:
        return self.total_patches
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Determine which volume this patch comes from
        volume_idx = idx // self.patches_per_volume
        
        # Load volume data
        if self.input_data is not None:
            input_volume = self.input_data[volume_idx]
            target_volume = self.target_data[volume_idx]
        else:
            input_volume = self._load_volume(self.input_volumes[volume_idx])
            target_volume = self._load_volume(self.target_volumes[volume_idx])
            
            if self.normalize:
                input_volume = self._normalize_volume(input_volume)
                target_volume = self._normalize_volume(target_volume)
        
        # Extract random patch
        input_patch, target_patch = self._extract_random_patch(input_volume, target_volume)
        
        # Convert to tensors and add channel dimension
        input_tensor = torch.from_numpy(input_patch).float().unsqueeze(0)  # [1, H, W, D]
        target_tensor = torch.from_numpy(target_patch).float().unsqueeze(0)  # [1, H, W, D]
        
        # Apply transforms if provided
        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)
        
        return input_tensor, target_tensor


class QSMPrecomputedPatchDataset(Dataset):
    """Dataset for precomputed QSM patches"""
    
    def __init__(
        self,
        input_patches: np.ndarray,
        target_patches: np.ndarray,
        transform=None,
        normalize: bool = True
    ):
        """
        Args:
            input_patches: Precomputed input patches [N, H, W, D]
            target_patches: Precomputed target patches [N, H, W, D]
            transform: Optional transform to apply
            normalize: Whether to normalize the data
        """
        assert input_patches.shape == target_patches.shape, \
            "Input and target patches must have the same shape"
        
        self.input_patches = input_patches
        self.target_patches = target_patches
        self.transform = transform
        
        if normalize:
            self.input_patches = self._normalize_patches(self.input_patches)
            self.target_patches = self._normalize_patches(self.target_patches)
    
    def _normalize_patches(self, patches: np.ndarray) -> np.ndarray:
        """Normalize patches to [-1, 1] range"""
        # Normalize each patch individually
        normalized = np.zeros_like(patches)
        for i in range(patches.shape[0]):
            patch = patches[i]
            patch_min = patch.min()
            patch_max = patch.max()
            if patch_max > patch_min:
                normalized[i] = 2 * (patch - patch_min) / (patch_max - patch_min) - 1
            else:
                normalized[i] = patch
        return normalized
    
    def __len__(self) -> int:
        return len(self.input_patches)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_patch = self.input_patches[idx]
        target_patch = self.target_patches[idx]
        
        # Convert to tensors and add channel dimension
        input_tensor = torch.from_numpy(input_patch).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_patch).float().unsqueeze(0)
        
        # Apply transforms if provided
        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)
        
        return input_tensor, target_tensor


def create_qsm_dataloaders(
    train_input_volumes: List[Union[str, np.ndarray]],
    train_target_volumes: List[Union[str, np.ndarray]],
    val_input_volumes: Optional[List[Union[str, np.ndarray]]] = None,
    val_target_volumes: Optional[List[Union[str, np.ndarray]]] = None,
    patch_size: int = 64,
    patches_per_volume: int = 100,
    batch_size: int = 8,
    num_workers: int = 4,
    normalize: bool = True,
    **dataset_kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation DataLoaders for QSM training
    
    Args:
        train_input_volumes: Training input volumes
        train_target_volumes: Training target volumes
        val_input_volumes: Validation input volumes (optional)
        val_target_volumes: Validation target volumes (optional)
        patch_size: Size of patches to extract
        patches_per_volume: Number of patches per volume per epoch
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        normalize: Whether to normalize data
        **dataset_kwargs: Additional arguments for Dataset
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    # Create training dataset
    train_dataset = QSMPatchDataset(
        input_volumes=train_input_volumes,
        target_volumes=train_target_volumes,
        patch_size=patch_size,
        patches_per_volume=patches_per_volume,
        normalize=normalize,
        **dataset_kwargs
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Create validation dataset if provided
    val_loader = None
    if val_input_volumes is not None and val_target_volumes is not None:
        val_dataset = QSMPatchDataset(
            input_volumes=val_input_volumes,
            target_volumes=val_target_volumes,
            patch_size=patch_size,
            patches_per_volume=patches_per_volume // 2,  # Fewer patches for validation
            normalize=normalize,
            **dataset_kwargs
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    return train_loader, val_loader


def create_patch_arrays_from_volumes(
    input_volumes: List[Union[str, np.ndarray]],
    target_volumes: List[Union[str, np.ndarray]],
    patch_size: int = 64,
    patches_per_volume: int = 500,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create precomputed patch arrays from full volumes
    Similar to the original notebook implementation
    
    Args:
        input_volumes: List of input volume paths or arrays
        target_volumes: List of target volume paths or arrays
        patch_size: Size of cubic patches
        patches_per_volume: Number of patches to extract per volume
        normalize: Whether to normalize data
    
    Returns:
        Tuple of (input_patches, target_patches) arrays
    """
    
    # Calculate total number of patches
    total_patches = len(input_volumes) * patches_per_volume
    
    # Initialize patch arrays
    input_patches = np.zeros((total_patches, patch_size, patch_size, patch_size))
    target_patches = np.zeros((total_patches, patch_size, patch_size, patch_size))
    
    patch_idx = 0
    
    for vol_idx, (input_vol_path, target_vol_path) in enumerate(zip(input_volumes, target_volumes)):
        print(f"Processing volume {vol_idx + 1}/{len(input_volumes)}")
        
        # Load volumes
        if isinstance(input_vol_path, str):
            if input_vol_path.endswith('.nii') or input_vol_path.endswith('.nii.gz'):
                input_vol = nib.load(input_vol_path).get_fdata()
            else:
                input_vol = np.load(input_vol_path)
        else:
            input_vol = input_vol_path
            
        if isinstance(target_vol_path, str):
            if target_vol_path.endswith('.nii') or target_vol_path.endswith('.nii.gz'):
                target_vol = nib.load(target_vol_path).get_fdata()
            else:
                target_vol = np.load(target_vol_path)
        else:
            target_vol = target_vol_path
        
        # Normalize if requested
        if normalize:
            input_vol = _normalize_volume(input_vol)
            target_vol = _normalize_volume(target_vol)
        
        # Extract patches
        for p in range(patches_per_volume):
            input_patch, target_patch = _extract_random_patch(input_vol, target_vol, patch_size)
            input_patches[patch_idx] = input_patch
            target_patches[patch_idx] = target_patch
            patch_idx += 1
    
    return input_patches, target_patches


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Normalize volume to [-1, 1] range"""
    vol_min = volume.min()
    vol_max = volume.max()
    if vol_max > vol_min:
        volume = 2 * (volume - vol_min) / (vol_max - vol_min) - 1
    return volume


def _extract_random_patch(
    input_volume: np.ndarray, 
    target_volume: np.ndarray, 
    patch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a random patch from volumes"""
    x_max, y_max, z_max = input_volume.shape
    
    # Random starting points
    start_x = random.randint(0, max(0, x_max - patch_size))
    start_y = random.randint(0, max(0, y_max - patch_size))
    start_z = random.randint(0, max(0, z_max - patch_size))
    
    # Extract patches
    input_patch = input_volume[
        start_x:start_x + patch_size,
        start_y:start_y + patch_size,
        start_z:start_z + patch_size
    ]
    
    target_patch = target_volume[
        start_x:start_x + patch_size,
        start_y:start_y + patch_size,
        start_z:start_z + patch_size
    ]
    
    return input_patch, target_patch


if __name__ == "__main__":
    # Test the dataset with dummy data
    print("Testing QSM Dataset...")
    
    # Create dummy volumes
    dummy_input = np.random.randn(128, 128, 128)
    dummy_target = np.random.randn(128, 128, 128)
    
    # Test with precomputed patches
    input_patches, target_patches = create_patch_arrays_from_volumes(
        [dummy_input], [dummy_target], 
        patch_size=64, patches_per_volume=10
    )
    
    print(f"Created patches - Input: {input_patches.shape}, Target: {target_patches.shape}")
    
    # Test dataset
    dataset = QSMPrecomputedPatchDataset(input_patches, target_patches)
    print(f"Dataset length: {len(dataset)}")
    
    # Test dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx}: Input shape {inputs.shape}, Target shape {targets.shape}")
        if batch_idx == 2:  # Test a few batches
            break
    
    print("Dataset test completed successfully!")
