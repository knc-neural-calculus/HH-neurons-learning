#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=GEN_VOLT
#SBATCH --mem-per-cpu=1g
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --account=forger1
#SBATCH --partition=standard
#SBATCH --mail-type=NONE
srun ./gen_volt $1 $2
