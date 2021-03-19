import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt

class LossPlotters(object):

	@staticmethod
	def single(dirname : str, verbose : bool = True, show : bool = True):
		"""plot the loss for a single run. if a file is not found, function will print a warning and do nothing
				
		# Parameters:
		 - `dirname : str`   
		   directory to look for `loss.txt` in
		 - `verbose : bool`   
		   whether to print the average loss
		   (defaults to `True`)
		 - `show : bool`   
		   whether to show the plot upon completion (set to False if wrapped in another script)
		   (defaults to `True`)
		"""

		filename = dirname + 'loss.txt'
		
		# do nothing on error
		if not os.path.isfile(filename):
			warnings.warn(f'invalid directory, could not find\t{filename}')
			return

		data = np.genfromtxt(filename, delimiter=',')
		data = data[:,:-1]

		data_avg = np.average(data, axis = 1)
		
		if verbose:
			print(data_avg)

		plt.plot(np.arange(0,len(data_avg),1), data_avg, label = dirname)	

		if show:
			plt.show()
			

	@staticmethod
	def multi(rootdir : str = '../../data/', verbose : bool = True):
		"""given a directory full of run data, plot the loss for each of those runs
				
		# Parameters:
		 - `rootdir : str`   
		   searches every directory in this directory for loss files
		   (defaults to `'../../data/'`)
		 - `verbose : bool`   
		   whether to print info about which directory is being searched
		   (defaults to `True`)
		"""

		dirnames = os.listdir(rootdir)

		if verbose:
			print('plotting data:')
		
		for d in dirnames:
			print('\t' + d)
			LossPlotters.single(d, verbose = verbose, show = False)

		plt.legend()
		plt.show()



if __name__ == '__main__':
	import fire
	fire.Fire(LossPlotters)




	


	