################### Transfer Learning Training for xQSM #####################
#########  Network Training with Frozen Encoding Layers #################### 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import time
import os
from xQSM import * 
from TrainingDataLoadHN import QSMDataSet
from torch.utils import data
from torch.amp import autocast, GradScaler # adding mixed precision training
import argparse
import warnings

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def freeze_encoding_layers(model):
    """
    Freeze the encoding layers of the xQSM model for transfer learning
    
    Args:
        model: xQSM model instance
    """
    # Freeze input octave layer
    for param in model.InputOct.parameters():
        param.requires_grad = False
    
    # Freeze all encoding convolution layers
    for i, encode_conv in enumerate(model.EncodeConvs):
        for param in encode_conv.parameters():
            param.requires_grad = False
    
    # Count trainable vs total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"Transfer Learning Setup: {trainable_params:,}/{total_params:,} trainable parameters ({trainable_params/total_params*100:.1f}%)")
    
    return model

def load_pretrained_weights(model, pretrained_path):
    """
    Load pretrained weights into the model
    
    Args:
        model: xQSM model instance
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
            model.load_state_dict(pretrained_dict, strict=True)
            print(f"Loaded pretrained weights from: {pretrained_path}")
            
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Using random initialization...")
            model.apply(weights_init)
    else:
        print("Using random initialization...")
        model.apply(weights_init)
    
    return model

def validate_model(model, val_loader, criterion, device):
    """
    Validate the model on validation set
    
    Args:
        model: xQSM model
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
        for inputs, targets, names in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def SaveNet(model, epoch, snapshot_path='./transfer_learning_checkpoints', ckpt_folder=None, best_loss=None):
    """
    Save network checkpoints
    
    Args:
        model: Model to save
        epoch: Current epoch
        snapshot_path: Directory to save checkpoints
        ckpt_folder: Folder for checkpoints
        best_loss: Best validation loss (if this is the best model)
    """
    if ckpt_folder:
        snapshot_path = os.path.join(snapshot_path, ckpt_folder)
        os.makedirs(snapshot_path, exist_ok=True) 
    else:
        os.makedirs(snapshot_path, exist_ok=True)
    
    # Always save latest checkpoint
    latest_path = os.path.join(snapshot_path, 'xQSM_TransferLearning_Latest.pth')
    torch.save(model.state_dict(), latest_path)
    
    # Save epoch-specific checkpoint
    # epoch_path = os.path.join(snapshot_path, f'xQSM_TransferLearning_epoch_{epoch}.pth')
    # torch.save(model.state_dict(), epoch_path)
    
    # Save best model if specified
    if best_loss is not None:
        best_path = os.path.join(snapshot_path, 'xQSM_TransferLearning_Best.pth')
        torch.save(model.state_dict(), best_path)

def TrainTransferLearning(data_directory, pretrained_path=None, encoding_depth=2, ini_chNo=64, 
                          LR=0.001, batch_size=32, Epoches=50, useGPU=True, 
                          snapshot_path='./transfer_learning_checkpoints', ckpt_folder=None):
    """
    Train xQSM model with transfer learning approach
    
    Args:
        data_directory: Path to head and neck QSM data
        pretrained_path: Path to pretrained model weights
        encoding_depth: Depth of encoding layers
        ini_chNo: Initial number of channels
        LR: Learning rate
        Batchsize: Batch size
        Epoches: Number of epochs
        useGPU: Whether to use GPU
        snapshot_path: Directory to save checkpoints
        ckpt_folder: Folder for checkpoints
    """
    print('='*80)
    print('TRANSFER LEARNING TRAINING FOR HEAD AND NECK QSM')
    print('='*80)
    
    # Data Loading
    train_dataset = QSMDataSet(data_directory, split_type='train')
    val_dataset = QSMDataSet(data_directory, split_type='val')
        
    print(f'Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples')
    
    # Create dataloaders
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Create model
    Chi_Net = xQSM(EncodingDepth=encoding_depth, ini_chNo=ini_chNo)
    
    # Load pretrained weights if available
    Chi_Net = load_pretrained_weights(Chi_Net, pretrained_path)
    
    # Freeze encoding layers
    Chi_Net = freeze_encoding_layers(Chi_Net)
    
    # Set model to training mode
    Chi_Net.train()

    criterion = nn.MSELoss(reduction='mean')

    # Only optimize unfrozen layers
    trainable_params = [p for p in Chi_Net.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=LR)

    scheduler = LS.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
    
    # Track best validation loss
    best_val_loss = float('inf')
    best_epoch = 0
    
    ## start the timer. 
    time_start = time.time()
    
    # Device selection
    if useGPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
        Chi_Net = nn.DataParallel(Chi_Net)
        Chi_Net.to(device)
        print(f"Using GPU: {torch.cuda.device_count()} devices")
    else:
        device = torch.device("cpu")
        Chi_Net.to(device)
        print("Using CPU")

    print(f"Training for {Epoches} epochs...")
    
    for epoch in range(1, Epoches + 1):
        # Training phase
        Chi_Net.train()
        epoch_train_loss = 0.0
        num_train_batches = 0
        
        scaler = GradScaler('cuda')

        for i, (inputs, targets, names) in enumerate(train_loader):
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # zero the gradient buffers 
            optimizer.zero_grad()
            
            # forward pass with autocast
            with autocast('cuda'):                             
                outputs = Chi_Net(inputs)
                loss = criterion(outputs, targets)
            
            # backward pass
            scaler.scale(loss).backward()                  
            scaler.step(optimizer)
            scaler.update()
            
            # Accumulate loss
            epoch_train_loss += loss.item()
            num_train_batches += 1
        
        # Calculate average training loss (for printing)
        avg_train_loss = epoch_train_loss / num_train_batches
        
        # Validation phase
        val_loss = validate_model(Chi_Net, val_loader, criterion, device)
        
        # Learning rate scheduler step
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        
        # Calculate timing
        time_end = time.time()
        epoch_time = (time_end - time_start) / epoch
        
        # Print epoch summary with all requested information
        print(f'Epoch [{epoch:3d}/{Epoches}] train_loss: {avg_train_loss:.6f} | val_loss: {val_loss:.6f} | best_val_loss: {best_val_loss:.6f} (epoch {best_epoch}) | Time: {epoch_time:.0f}s')
        
        # Save checkpoints periodically
        if epoch % 10 == 0:
            print(f"  → Checkpoint saved (epoch {epoch})")
            SaveNet(Chi_Net, epoch, snapshot_path, ckpt_folder=ckpt_folder)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            SaveNet(Chi_Net, epoch, snapshot_path, ckpt_folder=ckpt_folder, best_val_loss=best_val_loss)
            print(f"New best model! Val: {best_val_loss:.6f} (epoch {best_epoch})")
                    
    # Final summary
    total_time = time.time() - time_start
    print('='*80)
    print('TRAINING COMPLETE')
    print(f'Best validation loss: {best_val_loss:.6f} (achieved at epoch {best_epoch})')
    print(f'Total training time: {total_time:.0f}s ({total_time/60:.1f}min)')
    
    SaveNet(Chi_Net, Epoches, snapshot_path, ckpt_folder=ckpt_folder)
    print(f"Final model saved to: {snapshot_path}")
    print('='*80)
    
if __name__ == '__main__':
    # Configuration
    parser = argparse.ArgumentParser()
    
    # Path parameters
    parser.add_argument("--data_directory", required=True, type=str)
    parser.add_argument("--pretrained_path", required=True, type=str)
    parser.add_argument("--snapshot_path", required=True, type=str)
    parser.add_argument("--ckpt_folder", required=True, type=str)

    # Parameters for training
    parser.add_argument("-lr", "--learning_rate", default=4e-4, type=float)
    parser.add_argument("-bs", "--batch_size", default=32, type=int)
    parser.add_argument("-ep", "--epochs", default=100, type=int)
    parser.add_argument("-ps", "--patch_size", default=(32, 32, 32), type=int)
    parser.add_argument("--use_gpu", action="store_false", help="Default is True, Use GPU for training,")
    
    # Architecture parameters
    parser.add_argument("-ed", "--encoding_depth", default=2, type=int)
    parser.add_argument("-ic", "--ini_chNo", default=64, type=int)
    parser.add_argument("-se", "--squeeze_exc", action="store_true", help="Default is False (i.e. do not use squeeze and excitation blocks)")
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
    
    # Architecture parameters
    encoding_depth = args.encoding_depth
    ini_chNo = args.ini_chNo
    use_se = args.squeeze_exc

    if encoding_depth != 2:
        warnings.warn("Encoding depth is not 2. Frozen encoding layers will have random initialization and no learning rate.")
    if ini_chNo != 64:
        warnings.warn("Initial number of channels is not 64. Frozen encoding layers will have random initialization and no learning rate.")
    if epochs != 100:
        warnings.warn("Epochs is not 100. Change the LR_Scheduler for more/less epochs.")
    if encoding_depth == 2 and ini_chNo == 64 and epochs == 100:
        print("Starting Transfer Learning Training...")
    else:
        warnings.warn("Training parameters are not optimal. Change the parameters for better results.")

    print(f"Data Directory: {data_directory}")
    print(f"Pretrained Weights: {pretrained_path}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"epochs: {epochs}")
    print(f"Patch Size: {patch_size}")
    print(f"Use GPU: {use_gpu}")
    print(f"Use Squeeze and Excitation: {use_se}")
        
    ## Start transfer learning training
    TrainTransferLearning(
        data_directory=data_directory,
        pretrained_path=pretrained_path,
        encoding_depth=encoding_depth,
        ini_chNo=ini_chNo,
        LR=learning_rate,
        batch_size=batch_size, 
        patch_size=patch_size,
        Epoches=epochs,
        useGPU=use_gpu,
        snapshot_path=snapshot_path,
        ckpt_folder=ckpt_folder,
        use_se=use_se
    ) 