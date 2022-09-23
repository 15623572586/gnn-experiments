import numpy as np
import pandas as pd
import torch
from scipy.sparse.linalg import eigs
from scipy.stats import stats


def build_graph(input):
    graph_kernels = []
    for batch in range(input.size()[0]):
        w_matrix = np.full((12, 12), float(0.0))
        data = input[batch].numpy()
        # 计算两个序列间的相似度 -- 皮尔逊相关系数
        for i in range(12):
            for j in range(12):
                if i != j:
                    if w_matrix[j][i].any() != 0:
                        w_matrix[i][j] = w_matrix[j][i]
                    else:
                        datai, dataj = data[:, i], data[:, j]
                        pearsonr = stats.pearsonr(datai, dataj)  # 皮尔逊相关系数
                        # if pearsonr[0] >= 0.5:
                        w_matrix[i][j] = pearsonr[0]
                else:
                    w_matrix[i][j] = 1.0
        # Load wighted adjacency matrix W
        wa = weight_matrix(w_matrix)
        # Calculate graph kernel
        la = scaled_laplacian(wa)
        lk = cheb_poly_approx(la, 3, 12)
        # graph_kernel = torch.tensor(lk).type(torch.float32)
        graph_kernels.append(lk)
    return torch.tensor(graph_kernels).type(torch.float32)


def scaled_laplacian(wa):
    """
    Normalized graph Laplacian function.
    :param wa: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    """
    # d -> diagonal degree matrix
    n, d = np.shape(wa)[0], np.sum(wa, axis=1)
    # la -> normalized graph Laplacian
    la = -wa
    la[np.diag_indices_from(la)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                la[i, j] = la[i, j] / np.sqrt(d[i] * d[j])
    lambda_max = eigs(la, k=1, which='LR')[0][0].real
    return np.mat(2 * la / lambda_max - np.identity(n))


def cheb_poly_approx(la, ks, n):
    """
    Chebyshev polynomials approximation function.
    :param la: np.matrix, [n_route, n_route], graph Laplacian.
    :param ks: int, kernel size of spatial convolution.
    :param n: int, size of graph.
    :return: np.ndarray, [n_route, ks * n_route].
    """
    la0, la1 = np.mat(np.identity(n)), np.mat(np.copy(la))

    if ks > 1:
        la_list = [np.copy(la0), np.copy(la1)]
        for i in range(ks - 2):
            la_n = np.mat(2 * la * la1 - la0)
            la_list.append(np.copy(la_n))
            la0, la1 = np.mat(np.copy(la1)), np.mat(np.copy(la_n))
        return np.concatenate(la_list, axis=-1)
    elif ks == 1:
        return np.asarray(la0)
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received {ks}')


def weight_matrix(wa, sigma2=0.6, epsilon=0.3, scaling=True):
    """
    Load weight matrix function.
    :param wa
    :param sigma2: float, scalar of matrix wa.
    :param epsilon: float, thresholds to control the sparsity of matrix wa.
    :param scaling: bool, whether applies numerical scaling on wa.
    :return: np.ndarray, [n_route, n_route].
    """

    # check whether wa is a 0/1 matrix.
    if set(np.unique(wa)) == {0, 1}:
        print('The input graph is a 0/1 matrix, set "scaling" to False.')
        scaling = False

    if scaling:
        n = wa.shape[0]
        wa = wa / 10000.  # change the scaling number if necessary
        wa2, wa_mask = wa * wa, np.ones([n, n]) - np.identity(n)
        return np.exp(-wa2 / sigma2) * (np.exp(-wa2 / sigma2) >= epsilon) * wa_mask
    else:
        return wa
