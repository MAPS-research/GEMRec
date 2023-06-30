#!/bin/bash

#SBATCH --job-name=parallel_download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=48:00:00

# Singularity path
ext3_path=/scratch/$USER/webui_env/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/webui_env/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif

# start running
singularity exec \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/ModelCoffer
python3 download_and_generate.py -pd 5 > logs/bins5.log 2>&1 &
python3 download_and_generate.py -pd 6 > logs/bins6.log 2>&1 &
python3 download_and_generate.py -pd 7 > logs/bins7.log 2>&1 &
python3 download_and_generate.py -pd 8 > logs/bins8.log 2>&1 &
python3 download_and_generate.py -pd 9 > logs/bins9.log 2>&1 &
python3 download_and_generate.py -pd 10 11 > logs/bins10to11.log 2>&1 &
python3 download_and_generate.py -pd 12 13 > logs/bins12to13.log 2>&1 &
wait
"