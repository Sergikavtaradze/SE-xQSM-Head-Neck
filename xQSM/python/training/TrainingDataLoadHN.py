# Make sure to update the DATA_DIRECTORY to your own path, when testing this code
import os
import numpy as np
import random
import torch
import nibabel as nib
from torch.utils import data
import glob
from functools import wraps


class QSMDataSet(data.Dataset):
    train_subjects=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08']
    val_subjects=['sub-09', 'sub-10']
    test_subjects=None
    brain_train_subjects=['sub-01', 'sub-02', 'sub-03', 'sub-05', 'sub-06', 'sub-07']           # sub 04 is missing brain mask data
    brain_val_subjects=['sub-08', 'sub-09']                                                     # sub 10 is missing brain mask data
    brain_test_subjects=None
    # Specific for dataset of 10 subjects each with 6 repetition
    # 80/20 split
    train_num_patches = 48*125
    val_num_patches = 12*125
    
    # We have 56 Volumes for the 8 subjects
    # We use 56*20 patches per epoch (1120), could use less
    def __init__(self, input_root = './', target_root=None, split_type='train', transform=None, include_noise=True, patch_size=(32, 32, 32), 
                 stride=(38, 38, 38), num_random_patches_per_vol=42, brain_only=False, affine=False,
                 input_suffix=None, target_suffixes=None):
        super(QSMDataSet, self).__init__()
        # Support separate roots for inputs and targets. If target_root is None, fall back to input_root
        self.input_root = input_root
        self.target_root = target_root if target_root is not None else input_root
        # Optional overrides for suffix patterns
        self.input_suffix = input_suffix
        self.target_suffixes = target_suffixes
        self.split_type = split_type
        self.transform = transform
        self.include_noise = include_noise
        self.patch_size = self.patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, (float, int)) else patch_size
        self.stride = stride
        self.num_random_patches_per_vol = num_random_patches_per_vol
        self.brain_only = brain_only
        self.affine = affine
        # Instead of using 1 volume per subject/repetition
        # We use way more patches as we are not using full volumes per epoch
        # This will be used to make the epoch size consistent
        self.num_patches = self.val_num_patches if split_type == 'val' else self.train_num_patches

        # Noise parameters (same as original)
        self.Prob = torch.tensor(0.8)   ## 20% (1 - 0.8) probability to add noise
        self.SNRs = torch.tensor([50, 40, 20, 10, 5])  # Noise SNRs
        
        # Find all available data files
        self.files = []
        self.vol_shapes = []
        self._scan_data_directory()

        self.fixed_patches = []
        self.random_patch_meta = []
        self._generate_patch_info()

    @property
    def current_subjects(self):
        """Property decorator to get current split subjects"""
        if self.brain_only:
            if self.split_type == 'train':
                return self.brain_train_subjects
            elif self.split_type == 'val':
                return self.brain_val_subjects
            elif self.split_type == 'test':
                return self.brain_test_subjects if self.brain_test_subjects else []
            else:
                raise ValueError(f"Unknown split_type: {self.split_type}")
        else:
            if self.split_type == 'train':
                return self.train_subjects
            elif self.split_type == 'val':
                return self.val_subjects
            elif self.split_type == 'test':
                return self.test_subjects if self.test_subjects else []
            else:
                raise ValueError(f"Unknown split_type: {self.split_type}")

    @classmethod
    def get_train_subjects(self, cls):
        """Class method decorator to get training subjects"""
        return cls.train_subjects if not self.brain_only else cls.brain_train_subjects

    @classmethod
    def get_val_subjects(self, cls):
        """Class method decorator to get validation subjects"""
        return cls.val_subjects if not self.brain_only else cls.brain_test_subjects

    @classmethod
    def get_test_subjects(self, cls):
        """Class method decorator to get test subjects"""
        return cls.test_subjects if cls.test_subjects else []

    @classmethod
    def get_all_split_subjects(self, cls):
        """Class method to get all subjects organized by split"""
        return {
            'train': cls.train_subjects,
            'val': cls.val_subjects,
            'test': cls.test_subjects if cls.test_subjects else []
        } if not self.brain_only else {
            'train': cls.brain_train_subjects,
            'val': cls.brain_val_subjects,
            'test': cls.brain_test_subjects if cls.brain_test_subjects else []
        }

    def _scan_data_directory(self):
        """Scan the data directory for current split subjects only"""
        current_subjects = self.current_subjects
        
        if not current_subjects:
            return
        print(f"These are the current subject {current_subjects}")
        
        # Determine file patterns based on overrides or defaults
        if self.input_suffix is not None:
            input_suffix = self.input_suffix
        else:
            if self.brain_only:
                input_suffix = "_unwrapped-SEGUE_mask-nfe_bfr-PDF_localfield_masked_cropped.nii.gz"
            else:
                input_suffix = "_unwrapped-SEGUE_mask-nfe_bfr-PDF_localfield.nii.gz"

        if self.target_suffixes is not None:
            target_suffixes = self.target_suffixes if isinstance(self.target_suffixes, (list, tuple)) else [self.target_suffixes]
        else:
            if self.brain_only:
                target_suffixes = ["_unwrapped-SEGUE_mask-nfe_bfr-PDF_susc-autoNDI_Chimap_masked_cropped.nii.gz"]
            else:
                # Default to consensus chi-map if using default non-brain data
                target_suffixes = ["_ConsensusMean.nii.gz"]
        
        # Process all subjects with the determined patterns
        for subject in current_subjects:
            # Create pattern for this specific subject
           
            input_pattern = os.path.join(self.input_root, f"{subject}/ses-*/qsm/*{input_suffix}")
            input_files = glob.glob(input_pattern)
            
            for input_file in input_files:
                # Extract subject and session from filename
                filename = os.path.basename(input_file)
                # Parse sub-XX_ses-XX from filename
                parts = filename.split('_')
                if len(parts) >= 2:
                    sub_id = parts[0]  # sub-XX
                    ses_id = parts[1]  # ses-XX
                    
                    # Construct and verify one or more target files for this input
                    for target_suffix in target_suffixes:
                        target_filename = f"{sub_id}_{ses_id}{target_suffix}"
                        target_file = os.path.join(self.target_root, f"{sub_id}/{ses_id}/qsm/{target_filename}")
                        
                        # Allow either .nii or .nii.gz on disk regardless of suffix form
                        candidate_targets = [target_file]
                        if target_file.endswith('.nii'):
                            candidate_targets.append(target_file + '.gz')
                        elif target_file.endswith('.nii.gz'):
                            candidate_targets.append(target_file[:-3])
                        found_target = next((t for t in candidate_targets if os.path.exists(t)), None)
                        
                        # Check if both files exist
                        if os.path.exists(input_file) and found_target is not None:
                            # Input shape used later for patchwise training
                            input_shape = nib.load(input_file).shape
                            self.vol_shapes.append(input_shape)
                            self.files.append({
                                "input": input_file,
                                "target": found_target,
                                "subject": sub_id,
                                "session": ses_id,
                                "name": f"{sub_id}_{ses_id}"
                            })
    def _generate_patch_info(self):
        # Generate fixed patches
        for vol_idx, _ in enumerate(self.files):
            d, h, w = self.vol_shapes[vol_idx]
            pd, ph, pw = self.patch_size
            sd, sh, sw = self.stride
            
            i = 0
            while i + pd <= d:
                j = 0
                while j + ph <= h:
                    k = 0
                    while k + pw <= w:
                        self.fixed_patches.append({
                            "vol_idx": vol_idx,
                            "coords": (i, j, k)
                        })
                        k += sw
                    j += sh
                i += sd
        
        # Generate metadata for random patches (coordinates generated in __getitem__)
        for vol_idx in range(len(self.files)):
            for _ in range(self.num_random_patches_per_vol):
                self.random_patch_meta.append({
                    "vol_idx": vol_idx
                })

    def __len__(self):
        #print(len(self.files))
        #print(self.num_patches)
        #return self.num_patches # Consistent epoch size
        if self.affine:
            return len(self.files)
        else:
            #return len(self.random_patch_meta)
            return len(self.fixed_patches) + len(self.random_patch_meta)
    
    def __getitem__(self, index):
        #print(f'This is the length {self.__len__()}')
        #print(f'this is the index: {index}')
        
        # When doing inference we are not loading patches
        # We need to index directly into the class parameter self.files
        if not self.affine: 
            if index < len(self.fixed_patches):
                # Fixed patch case
                #print(f'this is the patch info: {self.fixed_patches[index]}')
                patch_info = self.fixed_patches[index]
                vol_idx = patch_info["vol_idx"]
                #print(f'this is the fixed volume index: {vol_idx}')
                i, j, k = patch_info["coords"]
                pd, ph, pw = self.patch_size # Use for patch extraction later
            else:
                # Random patch case
                random_idx = index - len(self.fixed_patches)
                patch_info = self.random_patch_meta[random_idx]
                #print(f'this is the random patch info: {patch_info}')
                vol_idx = patch_info["vol_idx"]
                #print(f'this is the random volume index: {vol_idx}')
                # Generate random coordinates dynamically
                d, h, w = self.vol_shapes[vol_idx]
                pd, ph, pw = self.patch_size
                i = random.randint(0, d - pd)
                j = random.randint(0, h - ph)
                k = random.randint(0, w - pw)
            
            name = self.files[vol_idx]["name"]
            
            # Load NIfTI files (handles .gz compression automatically)
            # Could implement caching here? Not sure if it's worth it
            # I have plenty of time but not plenty of RAM
            input_nii = nib.load(self.files[vol_idx]["input"])
            target_nii = nib.load(self.files[vol_idx]["target"])
        else:
            name = self.files[index]["name"]
            
            # Load NIfTI files (handles .gz compression automatically)
            # Could implement caching here? Not sure if it's worth it
            # I have plenty of time but not plenty of RAM
            input_nii = nib.load(self.files[index]["input"])
            target_nii = nib.load(self.files[index]["target"])

            # Get affine matrices
            input_affine = input_nii.affine
            target_affine = target_nii.affine
        
        # Get data arrays
        input_data = input_nii.get_fdata()  # get_fdata() over get_data()
        target_data = target_nii.get_fdata()
        
        # Convert to numpy arrays and then to torch tensors
        input_data = np.array(input_data, dtype=np.float32)
        target_data = np.array(target_data, dtype=np.float32)
        
        input_tensor = torch.from_numpy(input_data).float()
        target_tensor = torch.from_numpy(target_data).float()
        
        # Have to place this before the tensors are divided into patches
        if self.affine is True and self.split_type == 'val' or self.split_type == 'test' or self.split_type == 'train':
            # Be careful, Usually if the split type is train you wouldn't be evaluating it, but we are for ROI_averages calculations
            return input_tensor.unsqueeze(0), target_tensor.unsqueeze(0), name, input_affine, target_affine
        else:
            ######################
            # Patchwise training #
            ######################        

            # If caching volumes have to use .clone(), since we don't want the cached volumes to be modified with the subsequent noise addition
            input_tensor = input_tensor[i:i+pd, j:j+ph, k:k+pw] # .clone()
            target_tensor = target_tensor[i:i+pd, j:j+ph, k:k+pw] # .clone()

            #print(f'this is the input tensor: {input_tensor.shape}')
            #print(f'this is the target tensor: {target_tensor.shape}')

            # Add noise augmentation (optional)
            if self.include_noise:
                tmp = torch.rand(1)
                if tmp > self.Prob:
                    tmp_idx = torch.randint(5, (1,1))
                    tmp_SNR = self.SNRs[tmp_idx]
                    input_tensor = AddNoise(input_tensor, tmp_SNR)
            
            # Add channel dimension (batch_size will be added by DataLoader)
            input_tensor = torch.unsqueeze(input_tensor, 0)
            target_tensor = torch.unsqueeze(target_tensor, 0)
            return input_tensor, target_tensor, name

    def get_file_info(self, index):
        """Get file information for a specific index"""
        return self.files[index]

    def get_all_subjects(self):
        """Get list of all unique subjects"""
        return list(set([f["subject"] for f in self.files]))

    def get_all_sessions(self):
        """Get list of all unique sessions"""
        return list(set([f["session"] for f in self.files]))

def AddNoise(ins, SNR):
    """Add noise to input tensor based on SNR"""
    sigPower = SigPower(ins)
    noisePower = sigPower / SNR
    noise = torch.sqrt(noisePower.float()) * torch.randn(ins.size()).float()
    return ins + noise


def SigPower(ins):
    """Calculate signal power"""
    ll = torch.numel(ins)
    tmp1 = torch.sum(ins ** 2)
    return torch.div(tmp1, ll)


# Test the dataloader
if __name__ == '__main__':
    # Update this path to your QSM data directory
    INPUT_ROOT = '/Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_data'
    TARGET_ROOT = '/Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_consensus'
    BATCH_SIZE = 2
    print(INPUT_ROOT)
    try:
        INPUT_ROOT = '/Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_data'
        TARGET_ROOT = '/Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_consensus'
        BATCH_SIZE = 2
        print(f"This is the input root {INPUT_ROOT}")
        print(f"This is the target root {TARGET_ROOT}")
        # Create subset datasets
        train_dataset = QSMDataSet(input_root=INPUT_ROOT, target_root=TARGET_ROOT, split_type='train', brain_only=False)
        val_dataset = QSMDataSet(input_root=INPUT_ROOT, target_root=TARGET_ROOT, split_type='val', brain_only=False)
        
        # Create dataloaders
        trainloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valloader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"Dataset created: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        # Test loading one batch from each split
        train_batch = next(iter(trainloader))
        val_batch = next(iter(valloader))
        
        print(f"Train batch: {train_batch[0].shape}, Val batch: {val_batch[0].shape}")
        print("Dataset loading test successful!")
                
    except ValueError as e:
        print(f"Error: {e}")
