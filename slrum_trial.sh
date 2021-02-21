#!/bin/bash
#SBATCH --account=def-drod1901
#SBATCH --time=5-0:0:0
#SBATCH --cpus-per-task=32 
#SBATCH --mem=128G
OUTDIR=~/project/out/$SLURM_JOB_ID
mkdir -p $OUTDIR
cd $SLURM_TMPDIR

module load python/3.7

virtualenv --no-download $SLURM_TMPDIR/env  # SLURM_TMPDIR is on the compute node

source $SLURM_TMPDIR/env/bin/activate


pip install scipy
pip install pandas
pip install matplotlib
pip install seaborn
pip install Brian2
pip install scikit_plot
pip install Nni
pip install scikit_learn
pip install scikitplot

cd ..

git clone https://github.com/nikhil-garg/EMG_exp.git
cd EMG_exp

python experiment_exploration.py --log_file_path $OUTDIR

