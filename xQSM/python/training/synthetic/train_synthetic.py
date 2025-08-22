"""
xQSM Training Script for Synthetic Data
Adapted from the original xQSM training code for use with synthetic volumes
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.utils.data as data
import time
import argparse
import os
import sys
from pathlib import Path
import warnings

# Add parent directories to path to import xQSM modules
current_dir = Path(__file__).parent
training_dir = current_dir.parent
python_dir = training_dir.parent
sys.path.append(str(training_dir))
sys.path.append(str(python_dir))

from xQSM import xQSM, weights_init, get_parameter_number
from dataset_synthetic import create_xqsm_synthetic_dataloaders


def SaveNet(Chi_Net, epoch, snapshot_path='./', ckpt_folder=None, is_best=False):
    """Save network checkpoints"""
    print('Saving checkpoint...')
    path = snapshot_path
    if ckpt_folder is not None:
        path = os.path.join(snapshot_path, ckpt_folder)
        os.makedirs(path, exist_ok=True)
    
    if is_best:
        best_path = os.path.join(path, 'xQSM_Synthetic_Best.pth')
        torch.save(Chi_Net.state_dict(), best_path)
        print(f"  → Best model saved: {best_path}")
    else:
        latest_path = os.path.join(path, 'xQSM_Synthetic_Latest.pth')
        epoch_path = os.path.join(path, f'xQSM_Synthetic_{epoch}.pth')
        torch.save(Chi_Net.state_dict(), latest_path)
        torch.save(Chi_Net.state_dict(), epoch_path)
        print(f"  → Checkpoint saved: {epoch_path}")


def TrainNet(
    Chi_Net, 
    data_directory,
    LR=0.001, 
    Batchsize=32, 
    Epoches=100, 
    patch_size=48,
    patches_per_volume=100,
    train_ratio=0.8,
    useGPU=True, 
    snapshot_path='./', 
    ckpt_folder=None,
    resume_from='/cluster/project7/SAMed/xQSM/xQSM/Pretrained_Checkpoints',
    save_every=20,
    use_se=False,
    ini_chNo=64,
    encoding_depth=2
):
    """Train the xQSM network on synthetic data"""
    
    print('Setting up data loaders...')
    train_loader, val_loader = create_xqsm_synthetic_dataloaders(
        data_directory=data_directory,
        patch_size=patch_size,
        patches_per_volume=patches_per_volume,
        batch_size=Batchsize,
        train_ratio=train_ratio,
        num_workers=4,
        normalize=True,
        include_noise=True
    )
    
    print(f'Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}')
    print('Data loader setup complete.')

    print('Setting up training...')
    criterion = nn.MSELoss(reduction='sum')
    optimizer1 = optim.Adam(Chi_Net.parameters(), lr=LR)
    scheduler1 = LS.MultiStepLR(optimizer1, milestones=[50, 80], gamma=0.1)
    
    # Track best validation loss
    best_val_loss = float('inf')
    best_epoch = 0
    start_epoch = 1

    # Initialize weights if not resuming
    if resume_from is None or ini_chNo != 64 or encoding_depth != 2:
        Chi_Net.apply(weights_init)
        print("Initialized network weights")
    else:
        # Load checkpoint
        if os.path.exists(resume_from):
            if use_se:
                checkpoint = torch.load(resume_from, map_location='cpu', strict=False)
            else:
                checkpoint = torch.load(resume_from, map_location='cpu', strict=True)
            Chi_Net.load_state_dict(checkpoint)
            print(f"Resumed from checkpoint: {resume_from}")
        else:
            print(f"Warning: Checkpoint {resume_from} not found. Starting from scratch.")
            Chi_Net.apply(weights_init)

    Chi_Net.train()
    print("Network parameter info:")
    print(get_parameter_number(Chi_Net))
    
    time_start = time.time()
    
    if useGPU:
        if torch.cuda.is_available():
            print(f"{torch.cuda.device_count()} Available GPUs!")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            Chi_Net.to(device)

            for epoch in range(start_epoch, Epoches + 1):
                # Training phase
                Chi_Net.train()
                epoch_train_loss = 0.0
                num_train_batches = 0

                # Save periodic checkpoints
                if epoch % save_every == 0:
                    SaveNet(Chi_Net, epoch, snapshot_path, ckpt_folder, is_best=False)

                print(f'\nEpoch [{epoch:3d}/{Epoches}] Training...')
                
                for i, (lfss, chis, names) in enumerate(train_loader):
                    lfss = lfss.to(device)
                    chis = chis.to(device)
                    
                    # Zero gradients
                    optimizer1.zero_grad()
                    
                    # Forward pass
                    pred_chis = Chi_Net(lfss)
                    
                    # Calculate loss
                    loss1 = criterion(pred_chis, chis)
                    
                    # Backward pass
                    loss1.backward()
                    optimizer1.step()

                    epoch_train_loss += loss1.item()
                    num_train_batches += 1
                    
                    # Print progress every 20 batches
                    if i % 19 == 0:
                        acc_loss1 = loss1.item()
                        time_end = time.time()
                        print(f'  Batch [{i+1:3d}/{len(train_loader)}] Loss: {acc_loss1:.6f} | '
                              f'LR: {optimizer1.param_groups[0]["lr"]:.6f} | '
                              f'Time: {time_end - time_start:.0f}s')
                
                # Validation phase
                Chi_Net.eval()
                total_val_loss = 0.0
                num_val_batches = 0
                
                print(f'Epoch [{epoch:3d}/{Epoches}] Validating...')
                with torch.no_grad():
                    for inputs, targets, _ in val_loader:
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        
                        outputs = Chi_Net(inputs)
                        loss = criterion(outputs, targets)
                        
                        total_val_loss += loss.item()
                        num_val_batches += 1
                
                # Calculate average losses
                avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0.0
                val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0

                # Update learning rate
                scheduler1.step()

                # Calculate timing
                time_end = time.time()
                epoch_time = (time_end - time_start) / epoch
                
                # Print epoch summary
                print(f'Epoch [{epoch:3d}/{Epoches}] Summary:')
                print(f'  Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}')
                print(f'  Best Val Loss: {best_val_loss:.6f} (epoch {best_epoch})')
                print(f'  Time per epoch: {epoch_time:.0f}s | LR: {optimizer1.param_groups[0]["lr"]:.6f}')
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    SaveNet(Chi_Net, epoch, snapshot_path, ckpt_folder, is_best=True)
                    print(f'  *** NEW BEST MODEL! Val Loss: {best_val_loss:.6f} ***')
                
                print('-' * 80)
        else:
            print('No CUDA Device Available!')
            return False
    else:
        print('GPU training disabled. This may be very slow.')
        return False
    
    print('\nTraining Complete!')
    # Save final model
    SaveNet(Chi_Net, Epoches, snapshot_path, ckpt_folder, is_best=False)
    
    print(f'Best validation loss: {best_val_loss:.6f} achieved at epoch {best_epoch}')
    return True


def main():
    parser = argparse.ArgumentParser(description="Train xQSM on synthetic data")
    
    # Training parameters
    parser.add_argument("-lr", "--learning_rate", default=0.0004, type=float, 
                        help="Learning rate (default: 0.0004)")
    parser.add_argument("-bs", "--batch_size", default=32, type=int, 
                        help="Batch size (default: 32)")
    parser.add_argument("-ep", "--epochs", default=100, type=int, 
                        help="Number of epochs (default: 100)")
    parser.add_argument("-ps", "--patch_size", default=48, type=int, 
                        help="Patch size (default: 48)")
    parser.add_argument("-ppv", "--patches_per_volume", default=100, type=int,
                        help="Patches per volume per epoch (default: 100)")
    parser.add_argument("-tr", "--train_ratio", default=0.8, type=float,
                        help="Training ratio (default: 0.8)")
    
    # Model parameters
    parser.add_argument("-ed", "--encoding_depth", default=2, type=int,
                        help="Encoding depth for xQSM (default: 2)")
    parser.add_argument("-ic", "--initial_channels", default=32, type=int,
                        help="Initial number of channels (default: 32)")
    parser.add_argument("--use_se", action="store_true", default=False,
                        help="Use squeeze-and-excitation blocks")
    
    # Data and paths
    parser.add_argument("--data_directory", required=True, type=str,
                        help="Directory containing synthetic data (NIfTI files)")
    parser.add_argument("--snapshot_path", default="./ckpt", type=str,
                        help="Directory to save checkpoints (default: ./ckpt)")
    parser.add_argument("--ckpt_folder", default=None, type=str,
                        help="Subfolder name for checkpoints (optional)")
    parser.add_argument("--resume_from", default=None, type=str,
                        help="Path to checkpoint to resume from (optional)")
    
    # Training options
    parser.add_argument("--use_gpu", action="store_true", default=True,
                        help="Use GPU for training (default: True)")
    parser.add_argument("--save_every", default=20, type=int,
                        help="Save checkpoint every N epochs (default: 20)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_directory):
        print(f"Error: Data directory {args.data_directory} does not exist")
        return
    
    # Create checkpoint folder name if not provided
    if args.ckpt_folder is None:
        args.ckpt_folder = f"xQSM_synthetic_bs{args.batch_size}_ep{args.epochs}_lr{args.learning_rate}_ps{args.patch_size}"
    
    print("=" * 80)
    print("xQSM Synthetic Data Training")
    print("=" * 80)
    print(f"Data directory: {args.data_directory}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Patch size: {args.patch_size}")
    print(f"Patches per volume: {args.patches_per_volume}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Encoding depth: {args.encoding_depth}")
    print(f"Initial channels: {args.initial_channels}")
    print(f"Use SE blocks: {args.use_se}")
    print(f"Checkpoint folder: {args.ckpt_folder}")
    print("=" * 80)

    if args.encoding_depth != 2:
        warnings.warn("Encoding depth is not 2. Frozen encoding layers will have random initialization and no learning rate.")
    elif args.initial_channels != 64:
        warnings.warn("Initial number of channels is not 64. Frozen encoding layers will have random initialization and no learning rate.")
    else:
        warnings.warn("Training parameters are not optimal. Change the parameters to default values for better results.")

    # Create xQSM model
    Chi_Net = xQSM(
        EncodingDepth=args.encoding_depth, 
        ini_chNo=args.initial_channels,
        use_se=args.use_se
    )
    
    # Train network
    success = TrainNet(
        Chi_Net,
        data_directory=args.data_directory,
        LR=args.learning_rate,
        Batchsize=args.batch_size,
        Epoches=args.epochs,
        patch_size=args.patch_size,
        patches_per_volume=args.patches_per_volume,
        train_ratio=args.train_ratio,
        useGPU=args.use_gpu,
        snapshot_path=args.snapshot_path,
        ckpt_folder=args.ckpt_folder,
        resume_from=args.resume_from,
        save_every=args.save_every,
        use_se=args.use_se,
        ini_chNo=args.initial_channels
    )
    
    if success:
        print("Training completed successfully!")
    else:
        print("Training failed!")


if __name__ == '__main__':
    main()
