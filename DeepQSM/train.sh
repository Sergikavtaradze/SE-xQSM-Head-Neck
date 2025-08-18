#!/bin/bash -l
 
#specify the required resources (optional - uncomment for cluster use)
#$ -l tmem=48G
#$ -l gpu=1
#$ -l gpu_type=a6000|P100|V100
 
# Set the job name, output file paths (optional - uncomment for cluster use)
#$ -N Train_QSM_PyTorch
#$ -o ./job_info
#$ -e ./job_info
#$ -wd ./

# Training script for QSM Deep Learning Models
# PyTorch implementation adapted from xQSM training structure

# Default parameters
MODEL_TYPE="model1"
DATA_DIR=""
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=4e-4
PATCH_SIZE=64
SNAPSHOT_PATH="./checkpoints"
CKPT_FOLDER=""
PRETRAINED_PATH=""

# Activate the virtual environment
# Initialize Conda (uncomment and modify path as needed)
##################################################################################
# Need to change the command to the correct path for the conda installation #
##################################################################################
# eval "$(/SAN/medic/CARES/mobarak/venvs/anaconda3/bin/conda shell.bash hook)"
# conda activate qsm_pytorch

########################################################
## CUDA Environment Setup (uncomment if needed)
########################################################

# Add CUDA binary directories to PATH - enables system to find and execute CUDA tools (nvidia-smi, nvcc, etc.)
# export PATH=/share/apps/cuda-11.8/bin:/usr/local/cuda-11.8/bin:${PATH}

# Set runtime library path - tells system where to find CUDA shared libraries during program execution
# This includes both shared (/share/apps) and local (/usr/local) CUDA installations
# export LD_LIBRARY_PATH=/share/apps/cuda-11.8/lib64:/usr/local/cuda-11.8/lib:/lib64:${LD_LIBRARY_PATH}

# Set CUDA include directory - specifies location of CUDA header files
# Used during compilation of CUDA programs (if needed)
# export CUDA_INC_DIR=/share/apps/cuda-11.8/include

# Set compile-time library path - tells compiler where to find libraries during linking
# Similar to LD_LIBRARY_PATH but used at build time instead of runtime
# export LIBRARY_PATH=/share/apps/cuda-11.8/lib64:/usr/local/cuda-11.8/lib:/lib64:${LIBRARY_PATH}

########################################################
## Finding available GPUs
########################################################

nvidia-smi

# Get the first available GPU ID
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2 < 100 {print $1}' | head -n 1| tr -d ',')

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "No available GPU found. Using CPU..."
    GPU_FLAG=""
else
    echo "Using GPU: $CUDA_VISIBLE_DEVICES"
    GPU_FLAG=""
fi

# Add CUDA blocking for better error reporting
export CUDA_LAUNCH_BLOCKING=1

# Build training command
CMD="python3 train.py"
CMD="$CMD --data_directory \"$DATA_DIR\""
CMD="$CMD --snapshot_path \"$SNAPSHOT_PATH\""
CMD="$CMD --ckpt_folder \"$CKPT_FOLDER\""
CMD="$CMD -mt $MODEL_TYPE"
CMD="$CMD -bs $BATCH_SIZE"
CMD="$CMD -ep $EPOCHS"
CMD="$CMD -lr $LEARNING_RATE"
CMD="$CMD -ps $PATCH_SIZE"

python3 train.py -bs 32 -ep 100 -lr 4e-4 -ps 48 -mt model1 \
--data_directory "/cluster/project7/SAMed/xQSM/2025-Summer-Research/QSM_data" \
--snapshot_path "/cluster/project7/SAMed/xQSM/2025-Summer-Research/DeepQSM/ckpt/" \
--ckpt_folder "Aug18_bs32_ep100_lr4e-4_ps48_model1" 
# --pretrained_path "/cluster/project7/SAMed/xQSM/2025-Summer-Research/xQSM/Pretrained_Checkpoints/xQSM_invivo.pth" 
