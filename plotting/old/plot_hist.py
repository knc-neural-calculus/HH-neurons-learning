import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(filename : str):
	all_weights = np.genfromtxt(filename, delimiter=',').flatten()
	
	hist,bins = np.histogram(all_weights)
	
	plt.plot(hist, bins)


def plot_weights_dist(dirname):

	bins = np.linspace(-2.0,2.0, 50)
	
	print('reading files:')
	for n in range(0,100,1):
		filename = dirname + 'W1/weights_W1_e-%d.csv' % n
		print('\t' + filename)
		
		if not os.path.isfile(filename):
			break
		else:
			all_weights = np.genfromtxt(filename, delimiter=',').flatten()
			hist, _ = np.histogram(all_weights,bins)

			plt.plot(bins[:-1], hist, label=('e=%d' % n))

	plt.legend()
	plt.show()
	


if __name__ == "__main__":
	plot_weights_dist(sys.argv[1])
