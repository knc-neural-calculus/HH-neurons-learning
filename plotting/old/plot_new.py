import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def plot_weights_dist(dirname, idx_str):

    bins = np.linspace(-2.0,2.0, 50)
    
    print('reading files:')
    for n in range(0,12000,100):
        filename = dirname + idx_str + '/weights_' + idx_str + '_e-%d.csv' % n
        print('\t' + filename)
        
        if not os.path.isfile(filename):
            break
        else:
            all_weights = np.genfromtxt(filename, delimiter=',').flatten()
            hist, _ = np.histogram(all_weights,bins)

            plt.plot(bins[:-1], hist, label=('e=%d' % n), c=[(n/100) / 120, 0.2, 0.2])


    plt.show()

def plot_weights_neg_percent(dirname, idx_str):
    bins = np.linspace(-2.0,2.0, 50)
    

    x = range(0,12000,100)
    y = []
    print('reading files:')
    for n in range(0,12000,100):
        filename = dirname + idx_str + '/weights_' + idx_str + '_e-%d.csv' % n
        print('\t' + filename)
        
        if not os.path.isfile(filename):
            break
        else:
            all_weights = np.genfromtxt(filename, delimiter=',').flatten()
            n_neg = 0
            for w in all_weights:
                if w < 0:
                    n_neg += 1 
            y.append(100 * n_neg / len(all_weights))
    plt.plot(x, y)
    plt.show()

def heatmap_alt(dirname, idx_str, fig, ax, epoch):
    filename = dirname + idx_str + '/weights_' + idx_str + '_e-%d.csv' % epoch
    W = np.genfromtxt(filename, delimiter=',')

    X = range(0, W.shape[0])
    Y = range(0, W.shape[1])
    
    im = ax.imshow(W, cmap='hot', interpolation='nearest')

    ax.set_title("epoch = %d, idx_str = %s" % (epoch, idx_str))
    cbar = ax.figure.colorbar(im, ax=ax)

def heatmap(dirname, idx_str, fig, ax, epoch, row_idx):
    filename = dirname + idx_str + '/weights_' + idx_str + '_e-%d.csv' % epoch
    W = np.genfromtxt(filename, delimiter=',')
    W = W[row_idx][:-1]
    W = W.reshape((28, 28))

    X = range(0, W.shape[0])
    Y = range(0, W.shape[1])
    
    im = ax.imshow(W, cmap='hot', interpolation='nearest')

#    ax.set_title("e %d, r %d" % (epoch, row_idx))
#    cbar = ax.figure.colorbar(im, ax=ax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    


if __name__ == "__main__":
    epochs = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 11900 ]
    rows = [80 + i for i in range(10)]
    fig, axs = plt.subplots(len(rows), len(epochs))

    for i, r in enumerate(rows):
        for j, e in enumerate(epochs):
            print(i, j)
            heatmap(sys.argv[1], 'W1', fig, axs[i][j], e, r) 

#    fig.tight_layout()
    plt.show()