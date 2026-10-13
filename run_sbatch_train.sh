#!/bin/bash
#SBATCH --job-name=train_dpccn_ecapa_teacher
#SBATCH --partition=pdefault
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:3
#SBATCH --mem=144G
#SBATCH --time=12:00:00
#SBATCH --output=/home/sidcs/wesep/slurm_run_logs/%x-%j.out
#SBATCH --error=/home/sidcs/wesep/slurm_run_logs/%x-%j.err

# ---- 1. Unpack the Python env into RAM ----
ENV=/dev/shm/$USER/env_$SLURM_JOB_ID
mkdir -p $ENV
tar -xzf $HOME/mtse.tar.gz -C $ENV
export PATH=$ENV/mtse/bin:$PATH
export PYTHONUNBUFFERED=1

# ---- 2. Mount Turbo ----
TURBO=/tmp/sidcs/turbo
mkdir -p $TURBO
fusermount -u $TURBO 2>/dev/null
sshfs -o IdentityFile=$HOME/.ssh/id_greatlakes,BatchMode=yes,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
  sidcs@greatlakes-xfer.arc-ts.umich.edu:/nfs/turbo/coe-profdj $TURBO

# ---- 3. Clean up both when the job ends ----
trap 'rm -rf $ENV; fusermount -u $TURBO 2>/dev/null' EXIT

# ---- 4. Checks ----
ls $TURBO/sidcs_backup/datasets/LibriMix > /dev/null || { echo "Turbo mount failed"; exit 1; }
echo "python: $(which python)"
python -c "import torch; print('torch:', torch.__file__)"


cd /home/sidcs/wesep
srun python train_dpcnn.py