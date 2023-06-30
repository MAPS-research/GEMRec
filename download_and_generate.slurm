#!/bin/bash

#SBATCH --job-name=download_and_generate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1

# Singularity path
ext3_path=/scratch/$USER/webui_env/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/webui_env/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/ModelCoffer
python3 download_and_generate.py -p all > logs/download_and_generate.log
"


