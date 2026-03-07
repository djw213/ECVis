import numpy as np
import scipy.stats as st


def rank_coordinates(Y):
    N, M = Y.shape
    R = np.zeros_like(Y)

    for m in range(M):
        R[:,m] = st.rankdata(Y[:,m])

    return R


def best_objective(Y):
    N = Y.shape[0]
    R = rank_coordinates(Y)

    best_obj = R.argmin(axis=1)
    assert best_obj.shape[0] == N, "Incorrect number of solutions in best objective array"
    return best_obj



def average_rank(Y):
    """
    Compute the average rank of the objective vector in the array Y.

    @param A matrix Y in which each row represents a solution's objective vector
    and each column represents an objective.
    @return A Numpy array in which element i is the average rank of solution i.
    """
    N = Y.shape[0]
    R = rank_coordinates(Y)

    r = R.mean(axis=1)
    assert r.shape[0] == N, "Incorrect number of solutions in rank array"
    return r