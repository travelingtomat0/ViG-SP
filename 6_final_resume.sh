#!/bin/bash

SBATCH --mail-type=END                           # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
SBATCH --mail-user=tomap@student.ethz.ch
SBATCH --output=/itet-stor/tomap/net_scratch/log/%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
SBATCH --error=/itet-stor/tomap/net_scratch/log/%j.err  # where to store error messages

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=/scratch/tomap/run_folder/
mkdir -p /scratch/tomap/run_folder/
#rm -r /scratch/mateodi/run_folder/*
if [[ ! -d ${TMPDIR} ]]; then
    echo 'Failed to access directory' >&2
    exit 1
fi
trap "exit 1" HUP INT TERM
#trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

echo "Activate Conda Environment"
# Binary or script to execute
eval "$(/itet-stor/tomap/net_scratch/conda/bin/conda shell.bash hook)"
conda activate pytcu11_clone

#rsync -a /scratch_net/tikgpu07/mateodi/imagenet-100/ /scratch/mateodi/imagenet-100/
#rsync -a /itet-stor/tomap/superpixel_gnns/imagenet-1k/ /scratch/tomap/imagenet-1k

# Check if MODELPATH and NUMWORKERS arguments are provided
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <TMPDIR> <NUMWORKERS>"
    exit 1
fi

MODELPATH="$1"
NUMWORKERS="$2"

echo "Run Python Script"

python -m torch.distributed.launch --nproc_per_node="$NUMWORKERS" ~/vig_pytorch/train_copy.py /scratch/tomap/imagenet-1k/ILSVRC/Data/CLS-LOC/ --model sp_vig --no-resume-opt --sched cosine --epochs 125 --cooldown-epochs 15 --segments 196 --compactness 10 --opt adamw -j "$NUMWORKERS" --num-classes 1000 --warmup-lr 1e-6 --mixup 0 --cutmix 0 --model-ema --model-ema-decay 0.99996 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 20 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 6e-5 --weight-decay .05 --drop 0 --drop-path .1 -b 22 --output /itet-stor/tomap/superpixel_gnns/outputs/sp_model/ --resume "$MODELPATH"
#--model-ema --no-resume-opt --lr 3e-4 --warmup-lr 1e-6
# --sched cosine
# /scratch/tomap/imagenet-1k/ILSVRC/Data/CLS-LOC/

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
