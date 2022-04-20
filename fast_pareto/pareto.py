from typing import List, Optional

import numpy as np


def _change_directions(
    costs: np.ndarray, larger_is_better_objectives: Optional[List[int]] = None
) -> np.ndarray:
    """
    Determine the pareto front from a provided set of costs.

    Args:
        costs (np.ndarray):
            An array of costs (or objectives).
            The shape is (n_observations, n_objectives).
        larger_is_better_objectives (List[int]):
            The indices of the objectives that are better when the values are larger.
            If None, we consider all objectives are better when they are smaller.

    Returns:
        transformed_costs (np.ndarray):
            An array of costs (or objectives) that are transformed so that
            smaller is better.
            The shape is (n_observations, n_objectives).
    """
    _costs = costs.copy()
    if larger_is_better_objectives is not None:
        _costs[:, larger_is_better_objectives] *= -1

    return _costs


def is_pareto_front(
    costs: np.ndarray, larger_is_better_objectives: Optional[List[int]] = None
) -> np.ndarray:
    """
    Determine the pareto front from a provided set of costs.

    Args:
        costs (np.ndarray):
            An array of costs (or objectives).
            The shape is (n_observations, n_objectives).
        larger_is_better_objectives (List[int]):
            The indices of the objectives that are better when the values are larger.
            If None, we consider all objectives are better when they are smaller.

    Returns:
        mask (np.ndarray):
            The mask of the pareto front.
            Each element is True or False and the shape is (n_observations, ).
    """
    costs = _change_directions(costs, larger_is_better_objectives)
    on_front_indices = np.arange(costs.shape[0])
    (n_observations, _) = costs.shape
    next_index = 0

    while next_index < len(costs):
        nd_mask = np.any(costs < costs[next_index], axis=1)
        nd_mask[next_index] = True
        # Remove dominated points
        on_front_indices, costs = on_front_indices[nd_mask], costs[nd_mask]
        next_index = np.sum(nd_mask[:next_index]) + 1

    mask = np.zeros(n_observations, dtype=np.bool8)
    mask[on_front_indices] = True
    return mask


def nondominated_sort(
    costs: np.ndarray, larger_is_better_objectives: Optional[List[int]] = None
) -> np.ndarray:
    """
    Calculate the non-dominated rank of each observation.

    Args:
        costs (np.ndarray):
            An array of costs (or objectives).
            The shape is (n_observations, n_objectives).
        larger_is_better_objectives (List[int]):
            The indices of the objectives that are better when the values are larger.
            If None, we consider all objectives are better when they are smaller.

    Returns:
        ranks (np.ndarray):
            The non-dominated rank of each observation.
            The shape is (n_observations, ).
            The rank starts from zero and lower rank is better.
    """
    costs = _change_directions(costs)
    ranks = np.zeros(len(costs), dtype=np.int32)
    rank = 0
    indices = np.arange(len(costs))
    while indices.size > 0:
        on_front = is_pareto_front(costs)
        ranks[indices[on_front]] = rank
        # Remove pareto front points
        indices, costs = indices[~on_front], costs[~on_front]
        rank += 1

    return ranks
