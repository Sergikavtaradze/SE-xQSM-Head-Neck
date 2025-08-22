#!/bin/bash -l
 
#specify the required resources
#$ -l tmem=48G
#$ -l gpu=1
#$ -l gpu_type=a6000|a100|a100_80|a100_dgx|rtx6000|rtx8000
 
# Set the job name, output file paths
#$ -N Aug20_xQSM_Synthetic_Train_lr4e-4_ps48_ep100_bs80_SE
#$ -o /cluster/project7/SAMed/xQSM/xQSM/python/training/synthetic/job_info
#$ -e /cluster/project7/SAMed/xQSM/xQSM/python/training/synthetic/job_info
#$ -wd /home/mobislam
 
# Activate the virtual environment
# Initialize Conda
# source /share/apps/source_files/python/python-3.9.16.source
# Activate the specific environment

##################################################################################
# Need to change the command to the correct path for the conda installation #
##################################################################################
eval "$(/SAN/medic/CARES/mobarak/venvs/anaconda3/bin/conda shell.bash hook)"
conda activate 3DSAM-adapter

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

# Navigate to the directory containing the training script
cd /cluster/project7/SAMed/xQSM/xQSM/python/training/synthetic/

python3 train_synthetic.py \
--data_directory "/cluster/project7/SAMed/xQSM/simulated_volumes_nifti" \
--snapshot_path "/cluster/project7/SAMed/xQSM/xQSM/python/training/synthetic/ckpt/" \
--ckpt_folder "xQSM_synthetic_bs80_ep100_lr4e-4_ps48_SE" \
--learning_rate 0.0004 \
--batch_size 80 \
--epochs 100 \
--patch_size 48 \
--initial_channels 64 \
--encoding_depth 2 \
--use_se