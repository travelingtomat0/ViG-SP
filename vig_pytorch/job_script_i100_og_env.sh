#!/bin/bash

#SBATCH --mail-type=ALL                           # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/home/mateodi/log/test%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/home/mateodi/log/test%j.err  # where to store error messages

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
mkdir /scratch/mateodi/run_folder/$SLURM_JOB_ID/
TMPDIR=/scratch/mateodi/run_folder/$SLURM_JOB_ID/
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

# Binary or script to execute
python -m torch.distributed.launch --nproc_per_node=1 /itet-stor/mateodi/net_scratch/mod_vig_pytorch/vig_pytorch/train.py /scratch/mateodi/imagenet-100/ --model sp_vig --sched cosine --epochs 2 --opt adamw -j 1 --num-classes 100 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --model-ema --model-ema-decay 0.99996 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 20 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 2e-3 --weight-decay .05 --drop 0 --drop-path .1 -b 16 --output /itet-stor/mateodi/superpixel_gnns/outputs/

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
