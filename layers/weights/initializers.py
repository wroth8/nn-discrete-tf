import numpy as np

from scipy.interpolate import interp1d


def map_to_ecdf(w):
    '''
    Assigns each weight to the value of the empirical cdf \in [0,1]. The weights are basically sorted and stored with
    equal distances to their neighbors in this sorted list.
    '''
    if not isinstance(w, np.ndarray):
        w = np.asarray(w)
    if w.size == 0:
        return w.copy()
    elif w.size == 1:
        return np.full_like(w, 0.5)
    else:
        xs, ys = np.unique(w, return_counts=True) # np.unique implicitly sorts and flattens w
        ys = np.asarray(ys, dtype=np.float64) # float64 is important here
        ys = np.cumsum(ys)
        ys /= ys[-1]
        f = interp1d(xs, ys, kind='linear')
        return f(w)


def initialize_probabilities_from_expectation(w_expect, w_values, new_dim=-1, q_init_max=0.95):
    '''
    Initialize weight probabilities from given expected values according to the procedure defined in [1]
    
    [1] W. Roth, G. Schindler, H. Fr{\"o}ning, F. Pernkopf;
        Training Discrete-Valued Neural Networks with Sign Activations Using Weight Distributions; ECML PKDD 2019
    '''
    if isinstance(w_values, np.ndarray):
        assert w_values.ndim == 1
        w_values = w_values.tolist()
    assert np.all(w_values[:-1] < w_values[1:]) # Check w_values is strictly sorted
    
    assert w_values[0] < 0.0 and w_values[-1] > 0.0

    n_w_values = len(w_values)
    assert 1.0 / n_w_values < q_init_max and q_init_max < 1.0
    
    q_init_min = (1.0 - q_init_max) / (n_w_values - 1.0)
    q_values = np.zeros(w_expect.shape + (n_w_values,), dtype=w_expect.dtype)
    slope = [(q_init_max - q_init_min) / (w_1 - w_0) for (w_0, w_1) in zip(w_values[:-1], w_values[1:])]

    # Compute probabilities for the smallest value q(w = w_values[0])
    idx0 = w_expect <= w_values[0]
    idx1 = w_expect > w_values[1]
    idx2 = np.logical_and(w_expect > w_values[0], w_expect <= w_values[1])
    q_values[idx0, 0] = q_init_max
    q_values[idx1, 0] = q_init_min
    q_values[idx2, 0] = q_init_max - slope[0] * (w_expect[idx2] - w_values[0])

    # Compute probabilities for intermediate values q(w = w_values[i]), 0 < i < n_w_values - 1
    for w_idx in range(1, n_w_values - 1):
        idx0 = np.logical_or(w_expect <= w_values[w_idx - 1], w_expect > w_values[w_idx + 1])
        idx1 = np.logical_and(w_expect > w_values[w_idx - 1], w_expect <= w_values[w_idx])
        idx2 = np.logical_and(w_expect > w_values[w_idx], w_expect <= w_values[w_idx + 1])
        q_values[idx0, w_idx] = q_init_min
        q_values[idx1, w_idx] = q_init_min + slope[w_idx - 1] * (w_expect[idx1] - w_values[w_idx - 1])
        q_values[idx2, w_idx] = q_init_max - slope[w_idx] * (w_expect[idx2] - w_values[w_idx])

    # Compute probabilities for the largest value q(w = w_values[-1])
    idx0 = w_expect > w_values[-1]
    idx1 = w_expect <= w_values[-2]
    idx2 = np.logical_and(w_expect > w_values[-2], w_expect <= w_values[-1])
    q_values[idx0, -1] = q_init_max
    q_values[idx1, -1] = q_init_min
    q_values[idx2, -1] = q_init_min + slope[-1] * (w_expect[idx2] - w_values[-2])
    
    if new_dim != -1:
        q_values = np.swapaxes(q_values, -1, new_dim)
    
    return q_values


def initialize_shayer_probabilities_from_expectation(w_expect, w_values=[-1.0, 0.0, 1.0], q_init_max=0.95):
    '''
    Initialize weight probabilities from given expected values according to the procedure defined in [2]. Actually, this
    function generalizes the ideas of [2] to arbitrary binary weights beyond {-1,1} and ternary weights beyond {-1,0,1}.
    Note, however, that for ternary weights the zero weight must be given.
    
    [2] O. Shayer, D. Levi, E. Fetaya:;
        Learning Discrete Weights Using the Local Reparameterization Trick; ICLR 2018
    '''
    if isinstance(w_values, np.ndarray):
        assert w_values.ndim == 1
        w_values = w_values.tolist()
    assert np.all(w_values[:-1] < w_values[1:]) # Check w_values is strictly sorted
    assert len(w_values) in [2, 3] # Only binary or ternary values are allowed
    assert q_init_max >= 0.5 and q_init_max <= 1.0
    
    q_init_min = 1.0 - q_init_max
    
    if len(w_values) == 2:
        # Binary weights
        q_pos = (w_expect - w_values[0]) / (-w_values[0] + w_values[1])
        q_pos = np.clip(q_pos, q_init_min, q_init_max)
        return q_pos
    elif len(w_values) == 3:
        # Ternary weights
        assert w_values[1] == 0.0 # ternary weights => zero weight must be given
        q_zro = np.zeros_like(w_expect)
        # slope: (delta y / delta x) from left point to right point
        slope = [(q_init_max - q_init_min) / (-w_values[0]), (q_init_min - q_init_max) / w_values[2]]
        idx_neg = w_expect < 0.0
        idx_pos = np.logical_not(idx_neg)
        q_zro[idx_neg] = q_init_max + slope[0] * w_expect[idx_neg]
        q_zro[idx_pos] = q_init_max + slope[1] * w_expect[idx_pos]
        q_cond_pos = (w_expect / (1.0 - q_zro) - w_values[0]) / (-w_values[0] + w_values[2])
        # Note: From the paper [1] it is not clear whether clipping is done here or before computing q_cond_pos.
        q_zro = np.clip(q_zro, q_init_min, q_init_max)
        q_cond_pos = np.clip(q_cond_pos, q_init_min, q_init_max)
        return q_zro, q_cond_pos
    else:
        raise Exception('initialize_shayer_probabilities_from_expectation is only defined for binary and ternary weights (not for weight values {})'.format(w_values))
