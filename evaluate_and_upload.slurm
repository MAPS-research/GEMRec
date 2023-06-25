#!/bin/bash

#SBATCH --job-name=evaluate_and_upload
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128GB
#SBATCH --time=6:00:00


ext3_path=/scratch/$USER/webui_env/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/webui_env/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif

# start running
singularity exec \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/ModelCoffer
python3 evaluate_and_upload.py > logs/evaluate_and_upload.log
"