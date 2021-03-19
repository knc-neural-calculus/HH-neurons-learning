"""
## usage:
    plot_raster.py COMMAND

## COMMANDS
    COMMAND is one of the following:

     dir -- `plot_dir_vts()`
       plots both layer voltage traces for a given dir

     vt -- `plot_vt()`
       plots a voltage trace from a file
"""

import os
import sys
from typing import *

import numpy as np
from nptyping import NDArray
import matplotlib.pyplot as plt


def load_voltage_traces(filename : str):
	"""loads voltage traces from either a numpy or csv file
	
	if numpy, extension should be .npy or .npz
	csv or tsv expect csv-style format with appropriate delim
	
	### Parameters:
	 - `filename : str`   
	   filename
	
	### Returns:
	 - `NDArray` 
	   numpy array with data
	"""
	filetype = filename.split('.')[-1]
	data = None
	if filetype in ['npy', 'npz']:
		data = np.load(filename)['arr_0']
	elif filetype == 'csv':
		data = np.genfromtxt(filename, delimiter=',').T
	elif filetype == 'tsv':
		data = np.genfromtxt(filename, delimiter='\t').T


	data = data[:-1]
	# data = data[:,00000:100000]
	# data = data[:,1400000:]
	
	return data


def spiketrain_from_vts(vts : NDArray, threshold : float = 20.0, maxlength : float = 30.0) -> List[float]:
	"""turns a voltage trace into a spiketrain"""
	# output = np.where(voltage_trace > 20.0, 1, 0)
	print(vts.shape)
	
	output = []
	for x in vts:
		output.append(np.where(x > 20.0)[0])

	return output


def plot_vt(filename_vt : str, file_save : Optional[str] = None, show : bool = False):
	"""plots a voltage trace from a file
	
	### Parameters:
	 - `filename_vt : str`   
	   voltage trace file
	 - `file_save : Optional[str]`   
	   where to save image. if None, not saved
	   (defaults to `None`)
	 - `show : bool`   
	   whether to show the image
	   (defaults to `False`)
	"""

	vts = load_voltage_traces(filename_vt)
	data = spiketrain_from_vts(vts)

	# plot voltage trace
	for x in vts:
		plt.plot(x)

	# plot raster, offset
	for i,x in enumerate(data):
		plt.plot(
			x,
			np.full(
				len(x),
				2 * i - ( 2 * len(data) + 50 ),
			),
			'b.'
		)

	if file_save is not None:
		plt.savefig(file_save)

	if show:
		plt.show()

def plot_dir_vts(dirname : str):
	"""plots both layer voltage traces for a given dir
	
	### Parameters:
	 - `dirname : str`   
	"""
	run_cfg = dirname.split('/')[-1]

	plot_vt(dirname + 'X1_voltages.csv', run_cfg)
	plot_vt(dirname + 'X2_voltages.csv', run_cfg)
	

if __name__ == "__main__":
	import fire
	fire.Fire({
		'dir' : plot_dir_vts,
		'vt' : plot_vt,
	})