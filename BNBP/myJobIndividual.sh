#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1g
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --account=forger1
#SBATCH --partition=standard
#SBATCH --mail-type=NONE
for (( c=$1; c<=$2; c++ ))
do
	srun ./hh_psweep 8 $c $3
done
