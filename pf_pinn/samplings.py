import numpy as np
from pyDOE import lhs


def make_flattend_grid_data(spans, nums):
    series = [np.random.uniform(*span, num) for span, num in zip(spans, nums)]
    grid = np.meshgrid(*series)
    flatten = np.vstack([g.flatten() for g in grid]).T
    return flatten


def make_lhs_sampling_data(mins, maxs, num):
    lb = np.array(mins)
    ub = np.array(maxs)
    if not len(lb) == len(ub):
        raise ValueError(f"mins and maxs should have the same length.")
    return lhs(len(lb), int(num)) * (ub - lb) + lb


def make_semi_circle_data(radius, num, center=[0, 0]):
    square = make_lhs_sampling_data(mins=[center[0] - radius, center[1]],
                                    maxs=[center[0] + radius,
                                          center[1] + radius],
                                    num=num)
    semi_circle = square[square[:, 0] ** 2 + square[:, 1] ** 2 <= radius ** 2]
    return semi_circle


def make_uniform_grid_data(mins, maxs, num):
    if not len(mins) == len(maxs) == len(num):
        raise ValueError(f"mins, maxs, num should have the same length.")
    each_col = [np.linspace(mins[i], maxs[i], num[i])[1:-1]
                for i in range(len(mins))]
    return np.stack(np.meshgrid(*each_col), axis=-1).reshape(-1, len(mins))


def make_uniform_grid_data_transition(mins, maxs, num):
    if not len(mins) == len(maxs) == len(num):
        raise ValueError(f"mins, maxs, num should have the same length.")
    each_col = [np.linspace(mins[i], maxs[i], num[i])[1:-1]
                for i in range(len(mins))]
    distances = [(maxs[i] - mins[i]) / (num[i] - 1) for i in range(len(mins))]
    shift = [np.random.uniform(-distances[i], distances[i], 1)
             for i in range(len(distances))]
    shift = np.concatenate(shift, axis=0)
    each_col = [each_col[i] + shift[i] for i in range(len(each_col))]

    # each_col_cliped = [np.clip(each_col[i] + shift[i], mins[i], maxs[i]) for i in range(len(each_col))]
    return np.stack(np.meshgrid(*each_col), axis=-1).reshape(-1, len(mins))
