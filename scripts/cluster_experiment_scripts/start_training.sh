#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
##SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=12000  # memory in Mb

export DATASET=$1

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}
export TMP_DIR=/disk/scratch/${STUDENT_ID}

# Comet

export USE_COMET=1
export COMET_API_KEY=ZVxnfYIYLUY5bQHYtnfZOnHjE

# rsync dataset

mkdir -p ${TMP_DIR}/dataset/
rsync -ua /home/${STUDENT_ID}/data/datasets/${DATASET} ${TMP_DIR}/datasets/

export DATASET_DIR=${TMP_DIR}/datasets/${DATASET}
export MODEL_DIR=/home/${STUDENT_ID}/data/models

# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd /home/${STUDENT_ID}/Real-Time-Voice-Cloning/
export PYTHONPATH=.
python -W ignore fastspeech2/train.py --experiment-name $2

