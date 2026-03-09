import unittest
import numpy as np
import ecvis.rank as rk


class TestRankCoordinates(unittest.TestCase):

    def testRankCoords(self):
        Rtrue = np.array([
            [1, 3, 4],
            [2, 1, 3],
            [4, 2, 1],
            [3, 4, 2]
        ])

        Y = np.array([
            [0.3, 0.7, 0.9],
            [0.4, 0.1, 0.7],
            [0.6, 0.2, 0.1],
            [0.5, 0.8, 0.3]
        ])

        R = rk.rank_coordinates(Y)

        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                self.assertEqual(Rtrue[i,j], R[i,j], f"element ({i},{j}) has wrong value: {R[i,j]} should be {Rtrue[i,j]}")