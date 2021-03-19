"""launches batch jobs using the slurm HPC system

## usage
    run_batch_jobs.py COMMAND

## COMMANDS
COMMAND is one of the following:

- `multi`
    `run_multi()` -- launch slurm jobs with multiple threads and multiple parameter sets/runs per job

- `indiv`
    `run_indiv()` -- launch slurm jobs with one set of parameters per job
"""


import sys
from os import listdir, system
from os.path import isfile, join
import glob
# import subprocess

def run_multi(run_ID : str, n_per_job : int, path : str = "psweep/config/", nthreads : int = 1):
    """launch slurm jobs with multiple threads and multiple parameter sets/runs per job
    
    ### Parameters:
     - `run_ID : str`   
       config file pattern
     - `n_per_job : int`   
       number of runs per job
     - `path : str`   
       where to look for config files
       (defaults to `"psweep/config/"`)
     - `nthreads : int`   
       number of threads per job
       (defaults to `1`)
    """

    # configfiles = [f for f in listdir(path) if isfile(join(path, f)) and f ]
    configfiles = glob.glob(path + run_ID + '_*')
    n = len(configfiles)

    if n % n_per_job != 0:
        Warning(f'n_per_job ({n_per_job}) does not divide the number of config files ({n}). things might be buggy')
        str_cont = input('press enter to continue, or Ctrl+C to exit')
        if str_cont in 'exit 0 n N no No'.split(' '):
            exit(1)

    for i in range(0, n, n_per_job):
        cmd = "sbatch --job-name=HH_" + run_ID + " myJobIndividual.sh "
        cmd += " " + str(i)
        cmd += " " + str(n_per_job + i)
        cmd += " " + run_ID
        print("CALLING " + cmd)
        system(cmd)



def run_indiv(run_ID : str, path : str = "psweep/config/"):
    """launch slurm jobs with one set of parameters per job

    ### Parameters:
     - `run_ID : str`   
       config file pattern
     - `path : str`   
       where to look for config files
       (defaults to `"psweep/config/"`)
    """
    

    configfiles = glob.glob(path + run_ID + '_*')
    n = len(configfiles)

    for i in range(0, n):
        cmd = "sbatch --job-name=HH_" + run_ID + " myJobIndividual.sh "
        cmd += " " + str(i)
        cmd += " " + str(i)
        cmd += " " + run_ID
        print("CALLING " + cmd)
        system(cmd)



if __name__ == '__main__':
    import fire
    fire.Fire({
        'multi' : run_multi,
        'indiv' : run_indiv,
    })

