"""
Training script for QSM Deep Learning Models
PyTorch implementation converted from TensorFlow/Keras
"""

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import time
import os
import pickle
import logging
from torch.utils.data import DataLoader
import argparse
import warnings
import numpy as np

from models import get_model, count_parameters
from dataset import create_qsm_dataloaders

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def load_pretrained_weights(model, pretrained_path):
    """
    Load pretrained weights into the model
    
    Args:
        model: QSM model instance
        pretrained_path: Path to pretrained weights file
    """
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            # Load the state dict
            pretrained_dict = torch.load(pretrained_path, map_location='cpu')
            
            # Handle DataParallel wrapper if present
            if 'module.' in list(pretrained_dict.keys())[0]:
                # Remove 'module.' prefix from keys
                pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
            
            # Load the weights
            model.load_state_dict(pretrained_dict, strict=False)
            print(f"Loaded pretrained weights from: {pretrained_path}")
            
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Using random initialization...")
    else:
        print("Using random initialization...")
    
    return model

def validate_model(model, val_loader, criterion, device):
    """
    Validate the model on validation set
    
    Args:
        model: QSM model
        val_loader: Validation dataloader
        criterion: Loss function
        device: Device to run on
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            if len(batch_data) == 3:
                inputs, targets, _ = batch_data  # Handle case with 3 elements
            else:
                inputs, targets = batch_data     # Handle case with 2 elements
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def SaveNet(model, epoch, snapshot_path='./checkpoints', ckpt_folder=None, best_loss=None):
    """
    Save network checkpoints
    
    Args:
        model: Model to save
        epoch: Current epoch
        snapshot_path: Directory to save checkpoints
        ckpt_folder: Specific checkpoint folder name
        best_loss: Best validation loss (if this is the best model)
    """
    path = os.path.join(snapshot_path, ckpt_folder)
    os.makedirs(path, exist_ok=True)
    
    # Always save latest checkpoint
    latest_path = os.path.join(path, 'QSM_PyTorch_Latest.pth')
    torch.save(model.state_dict(), latest_path)
    
    # Save best model if specified
    if best_loss is not None:
        best_path = os.path.join(path, 'QSM_PyTorch_Best.pth')
        torch.save(model.state_dict(), best_path)

class QSMTrainer:
    """Trainer class for QSM models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str,
        log_dir: str,
        save_period: int = 20
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.save_period = save_period
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('QSMTrainer')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = os.path.join(self.log_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch:04d}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with validation loss: {self.best_val_loss:.6f}")
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint and return starting epoch"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        start_epoch = checkpoint['epoch'] + 1
        self.logger.info(f"Resumed training from epoch {start_epoch}")
        return start_epoch
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            if len(batch_data) == 3:
                inputs, targets, _ = batch_data  # Handle case with 3 elements
            else:
                inputs, targets = batch_data     # Handle case with 2 elements
                
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log batch progress
            if batch_idx % 10 == 0:
                print(f"Training batch {batch_idx+1} of {num_batches}")
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Validate the model"""
        if self.val_loader is None:
            return 0.0
        
        return validate_model(self.model, self.val_loader, self.criterion, self.device)
    
    def train(self, num_epochs: int, start_epoch: int = 0):
        """Main training loop"""
        self.logger.info(f"Starting training for {num_epochs} epochs from epoch {start_epoch}")
        self.logger.info(f"Model parameters: {count_parameters(self.model):,}")
        
        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            if self.val_loader is not None:
                self.val_losses.append(val_loss)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch results
            log_msg = f'Epoch {epoch:04d}, Train Loss: {train_loss:.6f}'
            if self.val_loader is not None:
                log_msg += f', Val Loss: {val_loss:.6f}'
            log_msg += f', Time: {epoch_time:.2f}s'
            self.logger.info(log_msg)
            
            # Check if this is the best model
            is_best = False
            if self.val_loader is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
            
            # Save checkpoint
            if (epoch + 1) % self.save_period == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        self.logger.info("Training completed!")
        
        # Save final training history
        history_path = os.path.join(self.log_dir, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump({
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'epochs': num_epochs
            }, f)


def TrainQSMNetwork(data_directory, pretrained_path=None, model_type='model1', 
                   LR=0.001, batch_size=32, epochs=50, patch_size=64, useGPU=True, 
                   snapshot_path='./checkpoints', ckpt_folder=None):
    """
    Train QSM model with PyTorch implementation
    
    Args:
        data_directory: Path to QSM data
        pretrained_path: Path to pretrained model weights
        model_type: Model architecture ('model1' or 'model2')
        LR: Learning rate
        batch_size: Batch size
        epochs: Number of epochs
        patch_size: Patch size
        useGPU: Whether to use GPU
        snapshot_path: Directory to save checkpoints
        ckpt_folder: Folder to save checkpoints
    """
    print('='*80)
    print('QSM DEEP LEARNING TRAINING - PYTORCH IMPLEMENTATION')
    print('='*80)
    
    # Data Loading - Create dummy data for now (replace with actual data loading)
    print("Note: Using dummy data for demonstration - replace with actual data loading")
    dummy_input = np.random.randn(10, 128, 128, 128)
    dummy_target = np.random.randn(10, 128, 128, 128)
    
    # Create dataloaders
    train_loader, val_loader = create_qsm_dataloaders(
        [dummy_input[i] for i in range(8)],  # Train volumes
        [dummy_target[i] for i in range(8)],
        [dummy_input[i] for i in range(8, 10)],  # Val volumes
        [dummy_target[i] for i in range(8, 10)],
        patch_size=patch_size,
        patches_per_volume=100,
        batch_size=batch_size,
        num_workers=4
    )
    
    print(f"Training batch length: {len(train_loader)}")
    print(f"Validation batch length: {len(val_loader)}")

    # Create model
    model = get_model(model_type)
    
    # Load pretrained weights if available
    if pretrained_path:
        model = load_pretrained_weights(model, pretrained_path)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Setup: {trainable_params:,}/{total_params:,} trainable parameters ({trainable_params/total_params*100:.1f}%)")
    
    # Set model to training mode
    model.train()

    criterion = nn.MSELoss(reduction='sum')

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = LS.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
    
    # Track best validation loss
    best_val_loss = float('inf')
    best_epoch = 0
    
    ## start the timer. 
    time_start = time.time()
    
    # Device selection
    if useGPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
        model.to(device)
        print(f"Using GPU: {torch.cuda.device_count()} devices")
    else:
        device = torch.device("cpu")
        model.to(device)
        print("Using CPU")

    # Create checkpoint directory
    full_checkpoint_path = os.path.join(snapshot_path, ckpt_folder)
    if os.path.exists(full_checkpoint_path):
        raise FileExistsError(f"Checkpoint folder already exists: {full_checkpoint_path}. Please use a different folder name.")
    else:
        os.makedirs(full_checkpoint_path, exist_ok=True)
        print(f"Created checkpoint folder: {full_checkpoint_path}")

    print(f"Training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        for i, batch_data in enumerate(train_loader):
            if len(batch_data) == 3:
                inputs, targets, _ = batch_data
            else:
                inputs, targets = batch_data
                
            print(f"Training batch {i+1} of {len(train_loader)}")
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # zero the gradient buffers 
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # backward pass
            loss.backward()                  
            optimizer.step()
            
            # Accumulate loss
            epoch_train_loss += loss.item()
            num_train_batches += 1
        
        # Calculate average training loss
        avg_train_loss = epoch_train_loss / num_train_batches
        
        # Validation phase
        val_loss = validate_model(model, val_loader, criterion, device)
        
        # Learning rate scheduler step
        scheduler.step()
        
        # Calculate timing
        time_end = time.time()
        epoch_time = (time_end - time_start) / epoch
        
        # Print epoch summary with all requested information
        print(f'Epoch [{epoch:3d}/{epochs}] train_loss: {avg_train_loss:.6f} | val_loss: {val_loss:.6f} | best_val_loss: {best_val_loss:.6f} (epoch {best_epoch}) | Time: {epoch_time:.0f}s')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            SaveNet(model, epoch, snapshot_path, ckpt_folder, best_val_loss)
            print(f"New best model! Val: {best_val_loss:.6f} (epoch {best_epoch})")
                    
    # Final summary
    total_time = time.time() - time_start
    print('='*80)
    print('TRAINING COMPLETE')
    print(f'Best validation loss: {best_val_loss:.6f} (achieved at epoch {best_epoch})')
    print(f'Total training time: {total_time:.0f}s ({total_time/60:.1f}min)')
    
    SaveNet(model, epochs, snapshot_path, ckpt_folder)
    print(f"Final model saved to: {snapshot_path}")
    print('='*80)


if __name__ == '__main__':
    # Configuration
    parser = argparse.ArgumentParser()
    
    # Path parameters
    parser.add_argument("--data_directory", required=True, type=str)
    parser.add_argument("--pretrained_path", default=None, type=str)
    parser.add_argument("--snapshot_path", required=True, type=str)
    parser.add_argument("--ckpt_folder", required=True, type=str)

    # Parameters for training
    parser.add_argument("-lr", "--learning_rate", default=0.001, type=float)
    parser.add_argument("-bs", "--batch_size", default=30, type=int)
    parser.add_argument("-ep", "--epochs", default=50, type=int)
    parser.add_argument("-ps", "--patch_size", default=64, type=int)
    parser.add_argument("--use_gpu", action="store_false", help="Default is True, Use GPU for training")
    
    # Model parameters
    parser.add_argument("-mt", "--model_type", default="model1", choices=["model1", "model2"], type=str)
    args = parser.parse_args()

    # Path parameters
    data_directory = args.data_directory
    pretrained_path = args.pretrained_path
    snapshot_path = args.snapshot_path
    ckpt_folder = args.ckpt_folder
    
    # Training parameters
    batch_size = args.batch_size
    patch_size = args.patch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    use_gpu = args.use_gpu
    
    # Model parameters
    model_type = args.model_type

    if epochs != 100:
        warnings.warn("Epochs is not 100. Change the LR_Scheduler for more/less epochs.")
    if epochs == 100:
        print("Starting QSM Deep Learning Training...")
    else:
        warnings.warn("Training parameters are not optimal. Change the parameters to default values for better results.")

    print(f"Data Directory: {data_directory}")
    print(f"Pretrained Weights: {pretrained_path}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Patch Size: {patch_size}")
    print(f"Epochs: {epochs}")
    print(f"Use GPU: {use_gpu}")
    print(f"Model Type: {model_type}")
    
    ## Start QSM training
    TrainQSMNetwork(
        data_directory=data_directory,
        pretrained_path=pretrained_path,
        model_type=model_type,
        LR=learning_rate,
        batch_size=batch_size, 
        epochs=epochs,
        patch_size=patch_size,
        useGPU=use_gpu,
        snapshot_path=snapshot_path,
        ckpt_folder=ckpt_folder
    )
