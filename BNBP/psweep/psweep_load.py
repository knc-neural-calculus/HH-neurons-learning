"""psweep_load.py - reads runs matching `run_ID` and saves them to a pandas dataframe (runs `read_and_save()`)

## usage
    psweep_load.py <flags>

## `read_and_save()`
    ### Parameters:
     - `run_ID : str`
       will look for run folders matching `{run_ID}_{n}`
       (defaults to `''`)
     - `datadir : str`
       folder in which to look for the data
       (defaults to `'../../../psweep_data/'`)
     - `file_save : Optional[str]`
       where to pickle the pandas dataframe into. if no value given, set to `'data_{run_ID}.df'`
       (defaults to `None`)
     - `rem_cols : List[str]`
       columns to remove
       (defaults to `None`)

    ### Modifies
     - saves the dataframe as a pickle file into `file_save`
"""

from os import listdir
from os.path import isdir, join
import os
import sys
import math
import glob

import numpy as np
import pandas as pd
from typing import *

if __name__ == '__main__':
	sys.path.insert(0, "..")
else:
	sys.path.insert(0, "HH-SGD")


from psweep.psweep import *


def fcomp(a,b,delta = 1e-5):
	return abs(a-b) < delta

def read_loss(
		filename : str,
		mode : LossMode = 'abs',
		first_n : int = 5,
		last_n : int = 5,
	) -> float:
	"""read the final loss from a training sample

	### Parameters:
	 - `filename : str`   
	 - `mode : LossMode`   
	   one of 'abs', 'rel'. if 'abs', returns the average loss for the `last_n` timesteps. if 'rel', returns the ratio between the average loss between `last_n` and `first_n` timesteps
	   (defaults to `'abs'`)
	 - `first_n : int`   
	   only used if loss mode is 'rel'
	   (defaults to `5`)
	 - `last_n : int`   
	   steps to average the loss over
	   (defaults to `5`)
	
	### Returns:
	 - `float` 
	   computed average loss from `filename`
	"""
		
	try:
		data = np.genfromtxt(filename, delimiter=',')
	except ValueError:
		return float('nan')

	try:
		data = data[:,:-1]
	except:
		return float('nan')
	
	avg_final_loss = np.average(data[-last_n:])

	if mode == 'abs':
		return avg_final_loss
	elif mode == 'rel':
		# loss_first = np.average(np.amin(np.average(data[:first_n], axis = 1)))
		loss_first = np.average(data[:first_n])
		return avg_final_loss / loss_first


def read_percent(filename) -> float:
	try:
		pair = open(filename, "r").readlines()[-1]
		return float(pair.split('\t')[-1])
		# tab_idx = pair.find('\t')
		# return float(pair.substr(pair+1))
	except:
		return np.nan
	


def read_config(
		filename : str,
		keys_map : Dict[str,str] = CONFIG_KEYS_MAP,
		default : Dict[str, Any] = CONSTS_DEFAULT,
	) -> Dict[str, Any]:
	"""reads config from `filename`
	
	### Parameters:
	 - `filename : str`   
	 - `keys_map : Dict[str,str]`   
	   [unused]
	   (defaults to `psweep.CONFIG_KEYS_MAP`)
	 - `default : Dict[str, Any]`   
	   used for casting types
	   (defaults to `psweep.CONSTS_DEFAULT`)
	
	### Returns:
	 - `Dict[str, Any]` 
	   dict containing params
	"""
	
	cfg : Dict[str, Any] = {}

	# read the actual config file into a dict
	with open(filename, 'r') as fin:
		for line in fin:
			# expected line format:
			# <key> = <value>
			# should work w/ any whitespace
			key, _, val = line.split()

			cfg[key] = val

	#	# map keys if needed
	#	if keys_map is not None:
	#		cfg = {
	#			keys_map[k] : cfg[k]
	#			for k in cfg
	#		}

	# change type
	if default is not None:
		cfg = {
			k : type(default[k])(cfg[k])
			for k in cfg		
		}

	# read ID from filename
	# cfg['CONFIG_ID'] = filename[
	# 	- ( LEN_ID + 1 + len('config.txt') ) 
	# 	: - ( len('config.txt') + 1 )
	# ]

	return cfg



def read_single_folder(directory : str, keys_map : Dict[str,str] = CONFIG_KEYS_MAP) -> Dict[str, Any]:
	"""reads all loss types for a given run
	### Parameters:
	 - `directory : str`   
	   directory where the run is
	 - `keys_map : Dict[str,str]`   
	   passed to `read_config`
	   (defaults to `CONFIG_KEYS_MAP`)
	
	### Returns:
	 - `Dict[str, Any]` 
	   [description]
	"""
	data = read_config(directory + 'config.txt', keys_map = keys_map)

	for c in LOSS_TYPES:
		data[c] = read_loss(directory + 'loss.txt', LOSS_TYPES[c])

	if ENABLE_TESTING_DATA:
		data['TEST_ACCURACY'] = read_percent(directory + 'percent0.txt')
	
	# print('\t%s' % str(data))		
	return data



def read_all_data(
		datadir : str = '../../../psweep_data/', 
		rem_cols : List[str] = None, 
		run_ID : str = '',
	) -> pd.DataFrame:
	"""gets data from many different runs (matching `run_ID`) and puts them into a dataframe
	
	### Parameters:
	 - `datadir : str`   
	   looks in this directory for directories of the form '{run_ID}_*', passes them to `read_single_folder`
	   (defaults to `'../../../psweep_data/'`)
	 - `rem_cols : List[str]`   
	   columns to remove from the dataframe
	   (defaults to `None`)
	 - `run_ID : str`   
	   used to match directories
	   (defaults to `''`)
	
	### Returns:
	 - `pd.DataFrame` 
	   each row in the table is a run, with a column for each hyperparameter, as well as the loss and accuracy
	"""
	# get directories
	# dirnames = [join(datadir, f) + '/' for f in listdir(datadir) if isdir(join(datadir, f)) and f[0] == sys.argv[1][0]]
	dirnames = [ x + '/' for x in glob.glob(datadir + run_ID + '_*') if isdir(x) ]
	
	n_total_dir = len(dirnames)
	print('> directories found:\t%d\n' % n_total_dir)

	# get columns
	cols = CONSTS_DEFAULT_KEYS + [ s for s in LOSS_TYPES ]
	if ENABLE_TESTING_DATA:
		cols = cols + ['TEST_ACCURACY']

	# read in data
	data = []
	i = 0
	for d in dirnames:
		print('\t read: \t%d\t/\t%d' % (i,n_total_dir), end='\r')
		data.append(read_single_folder(d))
		i = i + 1

	print('\n\n> directories read in:\t%d' % len(data))

	# bulk write to dataframe
	# df = pd.DataFrame( columns = cols )
	# df = df.append(data, ignore_index = True)
	# df = df.append(data)
	df = pd.DataFrame(data)

	# remove some columns with junk
	if rem_cols is not None:
		for r in rem_cols:
			del df[r]

	return df	


def read_and_save(run_ID = '', datadir = '../../../psweep_data/', file_save : Optional[str] = None, rem_cols : List[str] = None) -> pd.DataFrame:
	"""reads runs matching `run_ID` and saves them to a pandas dataframe
	
	### Parameters:
	 - `run_ID : str`   
	   will look for run folders matching `{run_ID}_{n}`
	   (defaults to `''`)
	 - `datadir : str`   
	   folder in which to look for the data
	   (defaults to `'../../../psweep_data/'`)
	 - `file_save : Optional[str]`   
	   where to pickle the pandas dataframe into. if no value given, set to `'data_{run_ID}.df'`
	   (defaults to `None`)
	 - `rem_cols : List[str]`   
	   columns to remove
	   (defaults to `None`)

	### Modifies
	 - saves the dataframe as a pickle file into `file_save`
	"""
	if file_save is None:
		file_save = f'data_{run_ID}.df'

	df = read_all_data(datadir, rem_cols, run_ID)
	df.to_pickle(file_save)


if __name__ == "__main__":
	import fire
	fire.Fire(read_and_save)










def main_OLD(argv = sys.argv):
	datadir = None

	if len(argv) > 1:
		run_ID = argv[1]
	else:
		run_ID = ''

	if len(argv) > 2:
		datadir = argv[2]
	else:
		datadir = '../../../psweep_data/'

	filename = 'data_%s.df' % run_ID
	
	print('filename=\t%s\ndatadir=\t%s\nrun_ID=\t%s' % (filename, datadir, run_ID))

	read_and_save(filename, datadir, rem_cols = None, run_ID = run_ID)
