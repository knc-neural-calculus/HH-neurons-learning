import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def load_spiketrains(idx):
    spiketrains = []
    with open('../mnist/TEST_POISSON_' + str(idx) + '.txt', "r") as fl:
        for line in fl:
            spiketrains.append([int(e.strip()) for e in line.split(',') if e != '\n'])
    return spiketrains

def plot_raster(spiketrains):
    lambdas = [len(spiketrain) for spiketrain in spiketrains]
    img = np.zeros((len(spiketrains), 1000))
    idx = 0
    for spiketrain in spiketrains:
        for spike in spiketrain:
            for k in range(2):
                for j in range(1):
                    img[idx+k, spike] = np.log(lambdas[idx])
                    if idx + k == len(spiketrains)-1:
                        break 

            for k in range(2):
                for j in range(1):
                    img[idx-k, spike] = np.log(lambdas[idx])
                    if idx - k == 0:
                        break 
        idx += 1

    plt.imshow(img, cmap = 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("poisson_raster.png")

def plot_2d_hist(spiketrains, bins = (20,20)):
    x_pos = []
    y_pos = []
    idx = 0
    for spiketrain in spiketrains:
        for spike in spiketrain:
            x_pos.append(idx % 28)
            y_pos.append(idx / 28)
        idx += 1

    plt.hist2d(x_pos, y_pos, bins = bins)
    plt.savefig("poisson_histogram.png")

if __name__ == "__main__":
    spiketrains = load_spiketrains(2)
    plot_raster(spiketrains)
    plot_2d_hist(spiketrains)
