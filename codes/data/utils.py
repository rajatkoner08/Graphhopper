import scipy.sparse as sp
from scipy.linalg import block_diag
import numpy as np
from collections import namedtuple

fields = ['graph_nodes', 'graph_rowlength', 'graph_seq', 'graph_adj', 'edge_type', 'edge_norm',
          'question_tokens','question_rowlength', 'question_seq', 'gq_bias','pad_keeper']
BatchData = namedtuple('BatchData', fields, defaults=(None,)*len(fields))


def construct_bias(adj):
    if sp.issparse(adj):
        adj = adj.toarray()
    mask = adj==0
    adj[mask] = -1e9
    return adj


def adj_to_bias_1(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


def adj_to_bias(adj):
    """
    >>> arr = np.random.randint(0,2, (5,5))
    >>> out1 = adj_to_bias_1(arr[np.newaxis,:], [5])
    >>> out2 = adj_to_bias(arr)
    >>> np.array_equal(out1[0], out2)
    True
    """
    if sp.issparse(adj):
        adj = adj.toarray()
    mg = np.eye(adj.shape[0])
    mg = adj + mg
    mask = adj > 0.0
    mg[mask] = 1.0
    return -1e9 * (1.0 - mg)

def adj_to_bias_gcn(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).toarray()

def construct_padded_bias(adj, max_num_entities):
    """bias matrix derived from adjacency matrix to support node propagation dynamics
    padding value by default is -1e9
    """
    bias = adj_to_bias(adj)
    if bias.shape[0]>max_num_entities:
        raise ValueError("max num entities not large enough {} {}".format(bias.shape[0], max_num_entities))
    canvas = np.ones((max_num_entities, max_num_entities), bias.dtype)*-1e9
    l = bias.shape[0]
    canvas[:l,:l] = bias
    return canvas

def batch_bias_mat_gcn(arrs):
    return block_diag(*arrs)

def batch_bias_mat(arrs):
    """
    build block diagnal adjacency  matrix for bias_mat
    modified from scipy.linalg.block_diag()
    >>> a = np.eye(3); b=np.ones((2,2))*2
    >>> out = batch_bias_mat([a,b])
    >>> assert np.array_equal(a, out[:3,:3])
    >>> assert np.array_equal(b, out[3:,3:])
    >>> assert np.array_equal(np.ones((3,2))*-1e9,out[:3,3:])
    >>> assert np.array_equal(np.ones((2,3))*-1e9,out[3:,:3])
    >>> out1 = batch_bias_mat([a,a,b])
    >>> assert out1.shape==(8,8)

    """
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(a) for a in arrs]

    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    if bad_args:
        raise ValueError("arguments in the following positions have dimension "
                         "greater than 2: %s" % bad_args)

    shapes = np.array([a.shape for a in arrs])
    out_dtype = np.find_common_type([arr.dtype for arr in arrs], [])
    out = np.ones(np.sum(shapes, axis=0), dtype=out_dtype) * -1e9

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out

def row_normalize(x):
    """
    x -> x'
    >>> arr = np.arange(4).reshape(2,2).astype(np.float32)
    >>> out = row_normalize(arr)
    >>> assert np.array_equal(out, np.array([[0.,1.],[0.4,0.6]],dtype=np.float32))
    """
    rowsum = np.sum(x, axis=-1)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    return x * r_inv[:,np.newaxis]