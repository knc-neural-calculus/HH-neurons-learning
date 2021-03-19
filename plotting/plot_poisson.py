from typing import *

import matplotlib.pyplot as plt


class SpiketrainPlotter(object):

    @staticmethod
    def load_spiketrain(
            dir : str = '../HH-SGD/mnist/', 
            idx : int = 10, 
            pattern : str = 'OUT_POISSON_{idx}.txt',
        ) -> List[List[int]]:
        """from a file, load spiketrains into a list of lists
        
        ### Parameters:
        - `dir : str`   
        directory to look for the spiketrains in
        (defaults to `'../HH-SGD/mnist/'`)
        - `idx : int`   
        index of file
        (defaults to `10`)
        - `pattern : str`   
        pattern to get the filename, appended to `dir` to get full path
        (defaults to `'OUT_POISSON_{idx}.txt'`)
        
        ### Returns:
        - `List[List[int]]` 
        output spiketrains, indexed by (pixel,spike_index)
        """

        spiketrains = []
        with open(dir + pattern.format(idx = idx), 'r') as fl:
            for line in fl:
                spiketrains.append([int(e.strip()) for e in line.split(',') if e != '\n'])

        return spiketrains

    @staticmethod
    def raster(
            show : bool = True,
            save : Optional[str] = 'poisson.png',
            **kwargs,
        ):
        """shows a raster plot of the spiketrains
        
        ### Parameters:
        - `show : bool`   
        whether to show the plot
        (defaults to `True`)
        - `save : Optional[str]`   
        where to save the plot. If `None`, skips automatic saving
        (defaults to `'poisson.png'`)
        - `**kwargs`
        passed to `load_spiketrain`, check the documentation for it
        """

        lst_spiketrains = SpiketrainPlotter.load_spiketrain(**kwargs)

        x_pos = []
        y_pos = []
        for i,spiketrain in enumerate(lst_spiketrains):
            for spike in spiketrain:
                x_pos.append(spike)
                y_pos.append(i) 

        plt.scatter(x_pos, y_pos, s = 0.4)
        
        if save is not None:
            plt.savefig(save)

        if show:
            plt.show()
        

    @staticmethod
    def histogram(
            show : bool = True,
            save : Optional[str] = 'poisson_histogram.png',
            **kwargs,
        ):
        """plots a 2-D histogram of the spiketrains for the input layer. should roughly match the base MNIST data
        
        ### Parameters:
        - `show : bool`   
        whether to show the plot
        (defaults to `True`)
        - `save : Optional[str]`   
        where to save the plot. If `None`, skips automatic saving
        (defaults to `'poisson_histogram.png'`)
        - `**kwargs`
        passed to `load_spiketrain`, check the documentation for it
        """

        lst_spiketrains = SpiketrainPlotter.load_spiketrain(**kwargs)

        x_pos = []
        y_pos = []

        for i,spiketrain in enumerate(lst_spiketrains):
            for spike in spiketrain:
                x_pos.append(i+1 % 28)
                y_pos.append(i+1 / 28)
            
        print(len(x_pos))

        plt.hist2d(x_pos, y_pos, bins = (20,20))

        if save is not None:
            plt.savefig(save)
        
        if show:
            plt.show()


if __name__ == '__main__':
    import fire
    fire.Fire(SpiketrainPlotter)