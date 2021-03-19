import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(dirname):

	filename = dirname + 'loss.txt'
	
	if not os.path.isfile(filename):
		return

	data = np.genfromtxt(filename, delimiter=',')
	data = data[:,:-1]

	data_avg = np.average(data, axis = 1)

	print(data_avg)

	plt.plot(np.arange(0,len(data_avg),1), data_avg, label=dirname)	


if __name__ == '__main__':
	if len(sys.argv) > 1:
		dirnames = sys.argv[1:]
	else:
		dirnames = os.listdir('../../data/')
		dirnames = [ '../../data/' + x + '/' for x in dirnames ]


	print('plotting data:')
	for d in dirnames:
		print('\t' + d)
		plot_loss(d)

	plt.legend()
	plt.show()
	


	