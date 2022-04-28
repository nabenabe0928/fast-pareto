from copy import deepcopy
from typing import List, Optional

import numpy as np

import scipy.stats


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
    (_, n_objectives) = costs.shape
    _costs = deepcopy(costs)
    if larger_is_better_objectives is None:
        return _costs

    if (
        max(larger_is_better_objectives) >= n_objectives
        or min(larger_is_better_objectives) < 0
    ):
        raise ValueError(
            "The indices specified in larger_is_better_objectives must be in "
            f"[0, n_objectives(={n_objectives})), but got {larger_is_better_objectives}"
        )

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


def _tie_break(costs: np.ndarray, nd_ranks: np.ndarray) -> np.ndarray:
    """
    Tie-break the non-domination ranks (, but we cannot guarantee no duplications)

    Args:
        costs (np.ndarray):
            An array of costs (or objectives).
            This array must be already sorted and must be copied.
            The shape is (n_observations, n_objectives).
        nd_ranks (np.ndarray):
            The non-dominated rank of each observation.
            The shape is (n_observations, ).
            The rank starts from zero and lower rank is better.

    Returns:
        tie_broken_nd_ranks (np.ndarray):
            The each non-dominated rank will be tie-broken
            so that we can sort identically (but we may get duplications).
            The shape is (n_observations, ) and the array is a permutation of zero to n_observations - 1.

    Reference:
        Paper Title:
            Techniques for Highly Multiobjective Optimisation: Some Nondominated Points are Better than Others
        One sentence summary:
            Average ranking strategy is effective to tie-break in some evolution strategies methods.
        Authors:
            David Come and Joshua Knowles
        URL:
            https://arxiv.org/pdf/0908.3025.pdf
    """
    rank_lb = 0  # the non-domination rank starts from zero
    rank_ub = nd_ranks.max() + 1
    masks: List[List[int]] = [[] for _ in range(rank_lb, rank_ub)]
    for idx, nd_rank in enumerate(nd_ranks):
        masks[nd_rank].append(idx)

    n_checked = 0
    ranks = scipy.stats.rankdata(costs, axis=0)
    # min_ranks_factor plays a role when we tie-break same average ranks
    min_ranks_factor = np.min(ranks, axis=-1) / (nd_ranks.size**2 + 1)
    avg_ranks = np.mean(ranks, axis=-1) + min_ranks_factor
    tie_broken_nd_ranks = np.zeros_like(nd_ranks, dtype=np.int32)

    for nd_rank in range(rank_lb, rank_ub):
        mask = masks[nd_rank]
        tie_break_ranks = scipy.stats.rankdata(avg_ranks[mask]).astype(np.int32)
        # -1 to start tie_broken_nd_ranks from zero
        tie_broken_nd_ranks[mask] = tie_break_ranks + n_checked - 1
        n_checked += len(mask)

    return tie_broken_nd_ranks


def nondominated_rank(
    costs: np.ndarray,
    larger_is_better_objectives: Optional[List[int]] = None,
    tie_break: bool = False,
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
        tie_break (bool):
            Whether we apply tie-break or not.

    Returns:
        ranks (np.ndarray):
            IF not tie_break:
                The non-dominated rank of each observation.
                The shape is (n_observations, ).
                The rank starts from zero and lower rank is better.
            else:
                The each non-dominated rank will be tie-broken
                so that we can sort identically.
                The shape is (n_observations, ) and the array is a permutation of zero to n_observations - 1.
    """
    costs = _change_directions(costs, larger_is_better_objectives)
    cached_costs = deepcopy(costs)
    ranks = np.zeros(len(costs), dtype=np.int32)
    rank = 0
    indices = np.arange(len(costs))
    while indices.size > 0:
        on_front = is_pareto_front(costs)
        ranks[indices[on_front]] = rank
        # Remove pareto front points
        indices, costs = indices[~on_front], costs[~on_front]
        rank += 1

    return ranks if not tie_break else _tie_break(cached_costs, ranks)
