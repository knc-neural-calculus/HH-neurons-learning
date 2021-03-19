import os
import sys
import numpy as np
import math

# DOCUMENT: psweep_loss file (might be deprecated)

def cull_nans(datas):
    it = 0
    for data, filename, culled in datas:
        if culled: 
            continue

        last_loss = np.average(data[:-1])
        if math.isnan(last_loss):
            datas[it] = data, filename, True

        it += 1

def cull_loss(datas, min_tol, max_tol):
        it = 0
        for data, filename, culled in datas:
            if culled: 
                continue

            data_avg1 = np.average(data[:-1])
            data_avg2 = np.average(data[:-2])
            data_avg3 = np.average(data[:-3])
            mn = min([data_avg1, data_avg2, data_avg3])
            mx = max([data_avg1, data_avg2, data_avg3])
            if mn < min_tol or mx > max_tol:
                datas[it] = data, filename, True

            it += 1

def cull_variance(datas, variance_tol):
        it = 0
        for data, filename, culled in datas:
            if culled: 
                continue

            variance = 0
            dprev = data[0,0]
            for d in data[:,0]:
                variance += abs(d - dprev)
                dprev = d

            if variance < variance_tol:
                datas[it] = data, filename, True

            it += 1

if __name__ == '__main__':
	raise NotImplementedError('unclear what this file does, removing functionality for now')
	if len(sys.argv) > 1:
		dirnames = sys.argv[1:]
	else:
		dirnames = os.listdir('../../data/')
		dirnames = [ '../../data/' + x + '/' for x in dirnames ]


	for d in dirnames:
            subdirs = [x[0] for x in os.walk(d)]
            datas = []
            for subdir in subdirs[1:]:
                filename = subdir + '/loss.txt'
                data = np.genfromtxt(filename, delimiter=',', skip_footer=3)
                data = data[:,:-1]
                datas.append((data, filename, False))

            cull_nans(datas)
            cull_variance(datas, 0.1)
            cull_loss(datas, 0.1, 0.15)
            for data, filename, culled in datas:
                if not culled:
                    print(filename)
