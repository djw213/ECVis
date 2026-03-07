import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class TradeOffVisualisation(ABC):

    def __init__(self):
        self.xlabel = None
        self.ylabel = None
        self.zlabel = None

    
    @abstractmethod
    def plot(self, Y):
        pass


class ScatterPlot(TradeOffVisualisation):

    def __init__(self):
        TradeOffVisualisation.__init__(self)


    def plot(self, Y, colours=None):
        """
        Produce a scatter plot of the solution set Y. If it's a 2-objective or
        3-objective solution set, use a 2- or 3-dimensional scatter plot. If M>3,
        use a pairwise box plot.
        """
        fig = plt.figure()
        N, M = Y.shape

        if colours is None:
            colours = ['k'] * N

        if M in [2,3]:
            if M == 2:
                ax = fig.add_subplot(111)
                ax.scatter(Y[:,0], Y[:,1], c=colours)
                ax.set_aspect('equal', 'box')

            if M == 3:
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(Y[:,0], Y[:,1], Y[:,2], c=colours)

                if not self.zlabel is None:
                    ax.set_zlabel(self.zlabel)            

            if not self.xlabel is None:
                ax.set_xlabel(self.xlabel)
            if not self.ylabel is None:
                ax.set_ylabel(self.ylabel)
        else:

            fig = plt.figure()

            Ymin, Ymax = Y.min(), Y.max()

            for r in range(M):
                for c in range(M):
                    if c >= r:
                        ax = fig.add_subplot(M, M, ((M*r)+c)+1)
                        ax.scatter(Y[:,r], Y[:,c])
                        ax.set_xlim(Ymin, Ymax)
                        ax.set_ylim(Ymin, Ymax)

                    if not c == r:
                        ax.set_xticks([])
                        ax.set_yticks([])