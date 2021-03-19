#!/bin/bash
#SBATCH --job-name=HH
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1g
#SBATCH --time=8:00:00
#SBATCH --account=forger1
#SBATCH --partition=standard
#SBATCH --mail-type=NONE
FILES=psweep/config/$1*
arr=()
i=0
for f in $FILES
do
	arr[$i]=$f
	i=$(($i+1))
done

srun="srun --exclusive -N1 -n1"

parallel="parallel -N 1 --delay .2 -j $2 --joblog parallel_joblog --resume"
echo $parallel

$parallel "$srun ./hh_psweep arg1:{1}" ::: arr
