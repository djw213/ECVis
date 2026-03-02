import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class TradeOffVisualisation(ABC):

    def __init__(self):
        pass

    
    @abstractmethod
    def plot(self, Y):
        pass


class ScatterPlot(TradeOffVisualisation):

    def __init__(self):
        TradeOffVisualisation.__init__(self)


    def plot(self, Y):
        pass