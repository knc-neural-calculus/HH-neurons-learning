import os
import sys
import math
from typing import *
from types import FunctionType

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("..")
sys.path.append("HH-SGD")

from psweep.psweep import *


def fcomp(a,b,delta = 1e-5):
	return abs(a-b) < delta


def load_pickled(filename = 'data.df'):
	return pd.read_pickle(filename)



class FloatSet(set):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def add_float(self, myfloat : float, delta = 1e-5):
		if all( (not fcomp(x, myfloat, delta)) for x in self ):
			self.add(myfloat)
	

def compute_ranges(
		df,
		cols_ignore = REMOVE_COLS + ['CONFIG_ID', 'DIRNAME'] + [ k for k in LOSS_TYPES ],
		typemap = TYPE_MAP,
	):
	"""
	takes in a dataframe, returns a 2-tuple of dicts:
	(variable_ranges, consts)
	"""
	
	# get all unique elements in each column
	ranges = {}
	for col in df.columns:
		if col not in cols_ignore:
			is_floatCol = isinstance( typemap[col], float)
			if is_floatCol:
				ranges[col] = FloatSet()
			else:
				ranges[col] = set()
			
			for x in df[col]:
				if is_floatCol:
					ranges[col].add_float(x)
				else:
					ranges[col].add(x)

	var_ranges : Dict[str, List[float]] = {}
	consts : Dict[str, float] = {}

	for k in ranges:
		if len(ranges[k]) == 1:
			# not variable
			consts[k] = ranges[k].pop()
		else:
			# variable
			var_ranges[k] = sorted(list(ranges[k]))

	return (var_ranges, consts)





def save_sorted_many(df, basepath = ''):
	pd.set_option('display.max_rows', None)
	pd.set_option('display.max_columns', 50)
	pd.set_option('display.width', None)
	pd.set_option('display.max_colwidth', 500)

	with open(basepath + 'loss_rel.txt', 'w') as fout:
		print(df.sort_values('LOSS_REL'), file=fout)

	with open(basepath + 'loss_abs.txt', 'w') as fout:
		print(df.sort_values('LOSS_ABS'), file=fout)

	with open(basepath + 'test_acc.txt', 'w') as fout:
		print(df.sort_values('TEST_ACCURACY'), file=fout)

def save_sorted(df : pd.DataFrame, path : str = '', sort_col : str = 'TEST_ACCURACY'):
	with open(path, 'w') as fout:
		print(df.sort_values(sort_col), file=fout)


def plot_pair_heatmap(
		data,
		X, Y,
		others,
		ranges = None,
		# defaults = None,
		accuracy_mode = 'LOSS_REL',
	):

	if ranges is None:
		ranges = compute_ranges(data)[0]

	print(ranges)
	
	# if defaults is None:
	# 	defaults = CONSTS_DEFAULT


	# * trimming input dataframe

	# for everything in `others`, delete all dataframe elements that dont match
	for k in others:
		data = data.loc[ fcomp(data[k], others[k]) ]

	# dont need config ID or any of the default values
	del data['CONFIG_ID']

	for ot in others:
		del data[ot]

	# only need 1 type of loss
	for lt in LOSS_TYPES:
		if lt != accuracy_mode:
			del data[lt]
	
	data_im = np.full(
		( len(ranges[Y]), len(ranges[X]) ),
		np.nan,
		dtype=np.float
	)

	print(data)

	for i,x in enumerate(ranges[X]):
		for j,y in enumerate(ranges[Y]):
			temp = data.loc[
				fcomp(data[X], x)
				& fcomp(data[Y], y)
			][accuracy_mode]

			if temp.empty:
				data_im[j,i] = np.nan
			else:
				data_im[j,i] = temp.to_list()[0]			
			
	# * chart stuff

	x_vals = ranges[X]
	y_vals = ranges[Y]

	fig, ax = plt.subplots()
	im = ax.imshow(data_im, cmap="inferno")
	cbar = plt.colorbar(im)
	cbar.ax.set_ylabel(accuracy_mode, rotation=-90, va="bottom")

	# We want to show all ticks...
	ax.set_xticks(np.arange(len(x_vals)))
	ax.set_yticks(np.arange(len(y_vals)))

	# ... and label them with the respective list entries
	ax.set_xticklabels([str(round(x_vals[i], 2)) if (i == len(x_vals) - 1 or i % 5 == 0) else "" for i in range(len(x_vals))])
	ax.set_yticklabels([str(round(y_vals[i], 2)) if (i == len(y_vals) - 1 or i % 5 == 0) else "" for i in range(len(y_vals))])

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	# for i in range(len(x_vals)):
	# 	for j in range(len(y_vals)):
	# 		text = ax.text(j, i, data_im[i, j],
	# 					ha="center", va="center", color="w")

	b, t = plt.ylim() # discover the values for bottom and top
	b += 0.5 # Add 0.5 to the bottom
	t -= 0.5 # Subtract 0.5 from the top
	plt.ylim(b, t) # update the ylim(bottom, top) values

	ax.set_title("Test Accuracy Varying Inhibition Scalars")
	fig.tight_layout()

	ax.set_xlabel(r"$\lambda_1$")
	ax.set_ylabel(r"$\lambda_2$").set_rotation(0)

	plt.tight_layout()
	plt.show()
	plt.savefig('heatmap.png')



def main(argv = sys.argv):
	if len(argv) > 1:
		filename = argv[1]
	else:
		filename = 'data.df'

	mode = 'hm'
	if len(argv) > 2:
		mode = argv[2]

	df = None
	if os.path.isfile(filename):
		df = pd.read_pickle(filename)
	else:
		from psweep.psweep_load import read_and_save
		df = read_and_save(filename)

	mx = 0.0
	mx_i = 0
	for i, v in enumerate(df['TEST_ACCURACY']):
		if v > mx:
			mx = v 
			mx_i = i 

	print(df)
	print(mx, df['DIRNAME'][mx_i])

	if mode == 'hm':
		plot_pair_heatmap(
			df, 'LF_OUT', 'LF_HIDDEN',
			others = {},
			accuracy_mode = 'TEST_ACCURACY',
		)
	elif mode == 'table':
		save_sorted(df, os.path.dirname(filename) + '/')
	elif mode == 'cr':
		print(compute_ranges(df))


def dynamic_load(filename : str, verbose : bool = True):
	df = None
	if os.path.isfile(filename):
		df = pd.read_pickle(filename)
	else:
		from psweep.psweep_load import read_and_save
		df = read_and_save(filename)

	if verbose:
		# REVIEW: what exactly does this do?
		mx = 0.0
		mx_i = 0
		for i, v in enumerate(df['TEST_ACCURACY']):
			if v > mx:
				mx = v 
				mx_i = i 

		print(df)
		print(mx, df['DIRNAME'][mx_i])

	return df


def main_hm(filename : str = 'data.df', X : str = 'LF_OUT', Y : str = 'LF_HIDDEN', accuracy_mode : str = 'TEST_ACCURACY'):
	"""plots a heatmap of the accuracy (`accuracy_mode`) dependent on `X,Y`
	
	# Parameters:
	 - `filename : str`   
	   pandas dataframe pickle file to look for. if not found, will try to generate and save it
	   (defaults to `'data.df'`)
	 - `X : str`   
	   X axis variable for the heatmap. make sure that the specified dataframe file actually has this as a variable
	   (defaults to `'LF_OUT'`)
	 - `Y : str`   
	   Y axis variable for the heatmap. make sure that the specified dataframe file actually has this as a variable
	   (defaults to `'LF_HIDDEN'`)
	 - `accuracy_mode : str`   
	   the accuracy measure to plot on the heatmap
	   (defaults to `'TEST_ACCURACY'`)
	"""
	df = dynamic_load(filename)

	plot_pair_heatmap(
		data = df, 
		X = X, Y = Y,
		others = {},
		accuracy_mode = accuracy_mode,
	)


def main_table(filename : str = 'data.df', sort_col : str = 'TEST_ACCURACY', save_path = None):
	"""outputs a table of the data from the given dataframe, sorted by `sort_col`
	
	# Parameters:
	 - `filename : str`   
	   pandas dataframe pickle file to look for. if not found, will try to generate and save it
	   (defaults to `'data.df'`)
	 - `sort_col : str`   
	   [description]
	   (defaults to `'TEST_ACCURACY'`)
	 - `save_path : str`
	   text file where the table should be saved. if `None`, saves to 
	   	`filename.split('.')[:-1] + '_' + sort_col + '.txt'`
	   (defaults to `None`)
	"""
	if save_path is None:
		save_path = filename.split('.')[:-1] + '_' + sort_col + '.txt'

	df : pd.DataFrame = dynamic_load(filename)
	save_sorted(df, path = save_path, sort_col = sort_col)



def main_cr(filename : str = 'data.df'):
	"""prints the ranges for the variables in the dataframe
	
	# Parameters:
	 - `filename : str`   
	   pandas dataframe pickle file to look for. if not found, will try to generate and save it
	   (defaults to `'data.df'`)
	"""
	df : pd.DataFrame = dynamic_load(filename)
	print(compute_ranges(df))


MAIN_FUNCS = {
	'heatmap' : main_hm,
	'hm' : main_hm,
	'table' : main_table,
	'compute_ranges' : main_cr,
	'cr' : main_cr,
}

if __name__ == '__main__':
	import fire
	fire.Fire(MAIN_FUNCS)


