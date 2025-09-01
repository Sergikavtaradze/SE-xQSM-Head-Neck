################### train AutoBCS framework #####################
#########  Network Training #################### 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.utils.data as data
import time
import argparse
from xQSM import * 
from TrainingDataLoadHN import * 

#########  Section 1: DataSet Load #############
def DataLoad(batch_size, data_directory, patch_size, split_type):

    dataset = QSMDataSet(data_directory, patch_size=patch_size, split_type=split_type, brain_only=True)
    print('dataLength: %d' % dataset.__len__())
    trainloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return trainloader

def SaveNet(Chi_Net, epo, snapshot_path='./', ckpt_folder=None, enSave=False):
    print('save results')
    path = snapshot_path
    if ckpt_folder is not None:
        import os
        path = os.path.join(snapshot_path, ckpt_folder)
        os.makedirs(path, exist_ok=True)
    latest_path = os.path.join(path, 'ChiNet_Latest.pth')
    epoch_path = os.path.join(path, f'ChiNet_{epo}.pth')
    if enSave:
        pass
    else:
        torch.save(Chi_Net.state_dict(), latest_path)
        torch.save(Chi_Net.state_dict(), epoch_path)

def TrainNet(Chi_Net, LR=0.001, Batchsize=32, Epoches=100, patch_size=(32, 32, 32), useGPU=True, data_directory='..', snapshot_path='./', ckpt_folder=None):
    print('DataLoader setting begins')
    train_loader = DataLoad(Batchsize, data_directory, patch_size, split_type='train')
    val_loader = DataLoad(Batchsize, data_directory, patch_size, split_type='val')
    print('Dataloader settting end')

    print('Training Begins')
    criterion = nn.MSELoss(reduction='sum')

    optimizer1 = optim.Adam(Chi_Net.parameters(), lr=LR)

    scheduler1 = LS.MultiStepLR(optimizer1, milestones=[50, 80], gamma=0.1)
    
    # Track best validation loss
    best_val_loss = float('inf')
    best_epoch = 0

    Chi_Net.apply(weights_init)
    Chi_Net.train()

    print(Chi_Net.state_dict)
    print(get_parameter_number(Chi_Net))
    
    time_start = time.time()
    if useGPU:
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "Available GPUs!")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            Chi_Net.to(device)

            for epoch in range(1, Epoches + 1):
                Chi_Net.train()
                epoch_train_loss = 0.0
                num_train_batches = 0

                if epoch % 20 == 0:
                    SaveNet(Chi_Net, epoch, snapshot_path, ckpt_folder, enSave=False)

                for i, data in enumerate(train_loader):
                    lfss, chis, name = data
                    lfss = lfss.to(device)
                    chis = chis.to(device)
                    ## zero the gradient buffers 
                    optimizer1.zero_grad()
                    ## forward: 
                    pred_chis = Chi_Net(lfss)
                    ## loss
                    loss1 = criterion(pred_chis, chis)
                    ## backward
                    loss1.backward()
                    ##
                    optimizer1.step()

                    epoch_train_loss += loss1.item()
                    num_train_batches += 1
                    ## print statistical information 
                    ## print every 20 mini-batch size
                    if i % 19 == 0:
                        acc_loss1 = loss1.item()
                        time_end = time.time()
                        print('Outside: Epoch : %d, batch: %d, Loss1: %f, lr1: %f,  used time: %d s' %
                              (epoch, i + 1, acc_loss1, optimizer1.param_groups[0]['lr'], time_end - time_start))
                
        
                Chi_Net.eval()
                total_loss = 0.0
                num_batches = 0
                
                with torch.no_grad():
                    for inputs, targets, _ in val_loader:
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        
                        outputs = Chi_Net(inputs)
                        loss = criterion(outputs, targets)
                        
                        total_loss += loss.item()
                        num_batches += 1
                
                # Calculate average training loss for printing
                avg_train_loss = epoch_train_loss / num_train_batches

                val_loss = total_loss / num_batches if num_batches > 0 else 0.0

                scheduler1.step()

                time_end = time.time()
                epoch_time = (time_end - time_start) / epoch
                
                # Print epoch summary with all requested information
                print(f'Epoch [{epoch:3d}/{Epoches}] train_loss: {avg_train_loss:.6f} | val_loss: {val_loss:.6f} | best_val_loss: {best_val_loss:.6f} (epoch {best_epoch}) | Time: {epoch_time:.0f}s')
                
                # Save checkpoints periodically
                # if epoch % 10 == 0:
                #     print(f"  → Checkpoint saved (epoch {epoch})")
                #     SaveNet(Chi_Net, epoch, snapshot_path)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    SaveNet(Chi_Net, epoch, snapshot_path, ckpt_folder, best_val_loss)
                    print(f"New best model! Val: {best_val_loss:.6f} (epoch {best_epoch})")
        else:
            print('No Cuda Device!')
            quit()
    print('Training Ends')
    SaveNet(Chi_Net, Epoches, snapshot_path, ckpt_folder)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", "--learning_rate", default=0.001, type=float)
    parser.add_argument("-bs", "--batch_size", default=32, type=int)
    parser.add_argument("-ep", "--epochs", default=100, type=int)
    parser.add_argument("-ps", "--patch_size", default=(32, 32, 32), type=tuple, help="Patch size for training")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="Use GPU for training (default: True)")
    parser.add_argument("--data_directory", default="..", type=str, help="Directory for training data")
    parser.add_argument("--snapshot_path", default="./", type=str, help="Directory to save checkpoints")
    parser.add_argument("--ckpt_folder", default=None, type=str, help="Subfolder for checkpoints (optional)")
    args = parser.parse_args()

    Chi_Net = xQSM(2, 64)

    ## train network
    TrainNet(
        Chi_Net,
        LR=args.learning_rate,
        Batchsize=args.batch_size,
        Epoches=args.epochs,
        patch_size=args.patch_size,
        useGPU=args.use_gpu,
        data_directory=args.data_directory,
        snapshot_path=args.snapshot_path,
        ckpt_folder=args.ckpt_folder
    )

if __name__ == '__main__':
    main()

# python Train_Original.py --data_directory /Users/sirbucks/Documents/xQSM/2025-Summer-Research/QSM_data