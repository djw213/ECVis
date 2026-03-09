import unittest
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as coll
import ecvis.emo as emo


class TestScatterPlot(unittest.TestCase):

    def test2Obj(self):
        # Some test data - sample from the positive quadrant of the unit circle.
        Nsamples = 100
        Y = abs(np.array([(lambda v: v/np.linalg.norm(v))(np.random.normal(size=2)) for _ in range(Nsamples)]))

        # Initialise a scatter plot object.
        plot = emo.ScatterPlot()
        plot.plot(Y)

        # Does it contain correct number of points?
        ax = plt.gca()
        sc_artists = [c for c in ax.collections if isinstance(c, coll.PathCollection)]
        self.assertEqual(1, len(sc_artists), f"Incorrect number of artists {len(sc_artists)} should be 1")

        # Check number of points.
        Npoints = sc_artists[-1].get_offsets().shape[0]
        self.assertEqual(Nsamples, Npoints, f"Incorrect number of points {Npoints} should be {Nsamples}")

        # Check there are no labels.
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        self.assertEqual("", xlabel, f"xlabel should be blank, not {xlabel}")
        self.assertEqual("", ylabel, f"ylabel should be blank, not {ylabel}")

        # Add some labels, re-render, and check again.
        xl_true = "Objective 1"
        yl_true = "Objective 2"
        plot.xlabel = xl_true
        plot.ylabel = yl_true
        plot.plot(Y)
        ax = plt.gca()
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        self.assertEqual(xl_true, xlabel, f"xlabel should be '{xl_true}', not '{xlabel}'")
        self.assertEqual(yl_true, ylabel, f"ylabel should be '{yl_true}', not '{ylabel}'")