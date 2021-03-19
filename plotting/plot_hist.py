import os
import sys
from typing import *

import numpy as np
import matplotlib.pyplot as plt


class WeightPlotters(object):
    @staticmethod
    def wgts_hist(
            dirname : str, 
            idx_str : str,
            bins_gen : Tuple[float,float,int] = (-3.0, 2.0, 50),
            sample_range : Tuple[int, int, int] = (0, 3000, 100),
            show : bool = True,
        ):
        """plot the histogram of weights and how they change over time
        
        for n in the sample range, grabs the files '{dirname}{idx_str}/weights_{idx_str}_e-{n}.csv'
        and plots a histogram of weights changing over time
        
        # Parameters:
         - `dirname : str`
           directory to look in
         - `idx_str : str`   
           string corresponding to the run
         - `bins_gen : Tuple[float,float,int]`   
           passed to `np.linspace` to generate bins. fmt is: start, stop, n_bins
           (defaults to `(-3.0, 2.0, 50)`)
         - `sample_range : Tuple[int, int, int]`   
            for which sample counts to plot. fmt is: start, stop, step. make sure the step actually matches the sample counts at which the weights were saved.
           (defaults to `(0, 3000, 100)`)
         - `show : bool`   
           whether to show the plot before returning. Set to false if youre plotting this in something else
           (defaults to `True`)
        """

        fig, ax = plt.subplots(1,1)
        bins = np.linspace(*bins_gen)
        
        print('reading files:')
        for n in range(*sample_range):
            # filename = dirname + idx_str + '/weights_' + idx_str + '_e-%d.csv' % n
            filename = f'{dirname}{idx_str}/weights_{idx_str}_e-{n}.csv'
            print('\t' + filename)
            
            if not os.path.isfile(filename):
                break
            else:
                all_weights = np.genfromtxt(filename, delimiter=',').flatten()
                hist, _ = np.histogram(all_weights,bins)
                ax.set_title(dirname)

                plt.plot(
                    bins[:-1], 
                    hist, 
                    label = f'e={n}',
                    c = [(n/100) / 30, 0.2, 0.2],
                )
        if show:
            plt.show()


    @staticmethod
    def wgts_hist_neg_percent(
            dirname : str, 
            idx_str : str,
            bins_gen : Tuple[float,float,int] = (-3.0, 2.0, 50),
            sample_range : Tuple[int, int, int] = (0, 3000, 100),
            show : bool = True,
        ):
        """shows how the proportion of negative weights changes as more samples are included
        
        # Parameters:
         - `dirname : str`
           directory to look in
         - `idx_str : str`   
           string corresponding to the run
         - `bins_gen : Tuple[float,float,int]`   
           passed to `np.linspace` to generate bins. fmt is: start, stop, n_bins
           (defaults to `(-3.0, 2.0, 50)`)
         - `sample_range : Tuple[int, int, int]`   
            for which sample counts to plot. fmt is: start, stop, step. make sure the step actually matches the sample counts at which the weights were saved.
           (defaults to `(0, 3000, 100)`)
         - `show : bool`   
           whether to show the plot before returning. Set to false if youre plotting this in something else
           (defaults to `True`)
        """
        
        bins = np.linspace(*bins_gen)
        x = []
        y = []
        print('reading files:')
        for n in range(*sample_range):
            x.append(n)
            filename = f'{dirname}{idx_str}/weights_{idx_str}_e-{n}.csv'
            print('\t' + filename)
            
            if not os.path.isfile(filename):
                break
            else:
                all_weights = np.genfromtxt(filename, delimiter=',').flatten()
                n_neg = 0
                for w in all_weights:
                    if w < 0:
                        n_neg += 1 
                y.append(sample_range[2] * n_neg / len(all_weights))

        plt.plot(x, y)
        
        if show:
            plt.show()

    @staticmethod
    def percept(
            dirname : str, 
            idx_str : str,
            sample : int,
            figax_objs : Optional[Tuple[plt.figure, plt.axes]] = None, 
            show : bool = True,
        ):
        """plot the perceptive field for the given run
        
        gets the file using the pattern: 
        '{dirname}{idx_str}/weights_{idx_str}_e-{sample}.csv'

        # Parameters:
         - `dirname : str`   
         - `idx_str : str`   
         - `sample : int`   
           sample count at which to get the weights. make sure the weights were actually saved for this sample count
         - `figax_objs : Optional[Tuple[plt.figure, plt.axes]]`   
           figure and axis on which to plot this. if none given, creates a few figure
           (defaults to `None`)
         - `show : bool`   
           whether to show the figure. set to false if wrapping
           (defaults to `True`)
        """
        
        if figax_objs is None:
            fig, ax = plt.subplots(1,1)
        else:
            fig, ax = figax_objs
        
        filename = f'{dirname}{idx_str}/weights_{idx_str}_e-{sample}.csv'
        W = np.genfromtxt(filename, delimiter=',')

        X = range(0, W.shape[0])
        Y = range(0, W.shape[1])
        
        im = ax.imshow(W, cmap='seismic', interpolation = 'nearest')

        ax.set_title("sample = {sample}, idx_str = {idx_str}")
        cbar = ax.figure.colorbar(im, ax=ax)

        if show:
            plt.show()

    @staticmethod
    def percept_toarr(
            dirname : str, 
            idx_str : str, 
            fig : plt.figure, ax : plt.axes, 
            sample : int, 
            row_idx : int,
        ):
        """used for plotting the big grid of perceptive fields"""
        filename = f'{dirname}{idx_str}/weights_{idx_str}_e-{sample}.csv'
        W = np.genfromtxt(filename, delimiter=',')
        W = W[row_idx][:-1]
        W = W.reshape((28, 28))

        X = range(0, W.shape[0])
        Y = range(0, W.shape[1])
        
        im = ax.imshow(W, cmap='seismic', interpolation='nearest')

    #    ax.set_title("e %d, r %d" % (sample, row_idx))
    #    cbar = ax.figure.colorbar(im, ax=ax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    
    @staticmethod
    def perceptive_fields(
            dirname : str, 
            sample : int = 1900, 
            save : Optional[str] = 'perceptive_fields.png',
            show : bool = True,
        ):
        """plots a grid of perceptive fields
                
        # Parameters:
         - `dirname : str`   
           where to find the data
         - `sample : int`   
           which sample count to make the plot for
           (defaults to `1900`)
         - `save : Optional[str]`   
           where to save the figure. skips if `Non`
           (defaults to `'perceptive_fields.png'`)
         - `show : bool`   
           whether to show the figure
           (defaults to `True`)
        """
        
        fig, axs = plt.subplots(10, 10)
        for r in range(100): 
            print(r)
            WeightPlotters.percept_toarr(dirname, 'W1', fig, axs[r % 10][r // 10], sample, r) 
        plt.subplots_adjust(
                left=0.0, 
                right=0.45, 
                bottom=0.1, 
                top=0.9, 
                wspace=0.0, 
                hspace=0.1)

        #    fig.tight_layout()

        if save is not None:
            plt.savefig(save)
        
        if show:
            plt.show()


if __name__ == "__main__":
    import fire
    fire.Fire(WeightPlotters)


#    fig, axs = plt.subplots(4,1)
#    heatmap(sys.argv[1], 'W2', fig, axs[0], 0)
#    heatmap(sys.argv[1], 'W2', fig, axs[1], 3000)
#    heatmap(sys.argv[1], 'W2', fig, axs[2], 7000)
#    heatmap(sys.argv[1], 'W2', fig, axs[3], 11900)

#    plot_weights_hist(sys.argv[1], "W1")
#    plot_weights_neg_percent(sys.argv[1], "W1")

#    epochs = [0, 500, 1000, 1500, 1900 ]
#    rows = [70 + i for i in range(30)]
#    fig, axs = plt.subplots(len(epochs), len(rows))
#
#    for i, r in enumerate(rows):
#        for j, e in enumerate(epochs):
#            print(i, j)
#            heatmap2(sys.argv[1], 'W1', fig, axs[j][i], e, r) 


