from copy import deepcopy
from typing import List, Literal, Optional, Tuple

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
        larger_is_better_objectives (Optional[List[int]]):
            The indices of the objectives that are better when the values are larger.
            If None, we consider all objectives are better when they are smaller.

    Returns:
        transformed_costs (np.ndarray):
            An array of costs (or objectives) that are transformed so that
            smaller is better.
            The shape is (n_observations, n_objectives).
    """
    n_objectives = costs.shape[-1]
    _costs = deepcopy(costs)
    if larger_is_better_objectives is None or len(larger_is_better_objectives) == 0:
        return _costs

    if (
        max(larger_is_better_objectives) >= n_objectives
        or min(larger_is_better_objectives) < 0
    ):
        raise ValueError(
            "The indices specified in larger_is_better_objectives must be in "
            f"[0, n_objectives(={n_objectives})), but got {larger_is_better_objectives}"
        )

    _costs[..., larger_is_better_objectives] *= -1
    return _costs


def _get_ordered_costs_and_order_inv(
    costs: np.ndarray,
    ordered: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    (n_observations, _) = costs.shape
    if not ordered:
        order = np.lexsort([costs[:, 1], costs[:, 0]])
        order_inv = np.zeros_like(order)
        order_inv[order] = np.arange(n_observations)
        ordered_costs = costs[order]
    else:
        ordered_costs = costs
        order_inv = np.arange(n_observations)

    return ordered_costs, order_inv


def is_pareto_front2d(
    costs: np.ndarray,
    larger_is_better_objectives: Optional[List[int]] = None,
    filter_duplication: bool = False,
    ordered: bool = False,
) -> np.ndarray:
    """
    Determine the pareto front from a provided set of 2d costs.

    Args:
        costs (np.ndarray):
            An array of costs (or objectives).
            The shape must be (n_observations, 2).
        larger_is_better_objectives (Optional[List[int]]):
            The indices of the objectives that are better when the values are larger.
            If None, we consider all objectives are better when they are smaller.
        filter_duplication (bool):
            If True, duplications will be False from the second occurence.
            True actually speeds up the runtime.
        ordered (bool):
            Whether the costs are already sorted by costs[:, 0].

    Returns:
        mask (np.ndarray):
            The mask of the pareto front.
            Each element is True or False and the shape is (n_observations, ).
    """
    (n_observations, n_objectives) = costs.shape
    if n_objectives != 2:
        raise ValueError(f"n_objectives must be 2, but got {n_objectives}")

    costs = _change_directions(costs, larger_is_better_objectives)
    ordered_costs, order_inv = _get_ordered_costs_and_order_inv(costs, ordered=ordered)

    min_costs_y = np.minimum.accumulate(ordered_costs[:, 1])
    min_mask = min_costs_y == ordered_costs[:, 1]
    new_min_mask = min_costs_y[1:] < min_costs_y[:-1]

    on_front = np.ones(n_observations, dtype=np.bool8)
    if filter_duplication:
        on_front[1:] = min_mask[1:] & new_min_mask
    else:
        same_mask = ordered_costs[1:, 0] == ordered_costs[:-1, 0]
        on_front[1:] = min_mask[1:] & (same_mask | new_min_mask)

    return on_front[order_inv]


def is_pareto_front(
    costs: np.ndarray,
    larger_is_better_objectives: Optional[List[int]] = None,
    filter_duplication: bool = False,
) -> np.ndarray:
    """
    Determine the pareto front from a provided set of costs.
    The time complexity is O(N (log N)^(M - 2)) for M > 3
    and O(N log N) for M = 2, 3 where
    N is n_observations and M is n_objectives. (Kung's algorithm)

    Args:
        costs (np.ndarray):
            An array of costs (or objectives).
            The shape is (n_observations, n_objectives).
        larger_is_better_objectives (Optional[List[int]]):
            The indices of the objectives that are better when the values are larger.
            If None, we consider all objectives are better when they are smaller.
        filter_duplication (bool):
            If True, duplications will be False from the second occurence.
            True actually speeds up the runtime.

    Returns:
        mask (np.ndarray):
            The mask of the pareto front.
            Each element is True or False and the shape is (n_observations, ).

    NOTE:
        f dominates g if and only if:
            1. f[i] <= g[i] for all i, and
            2. f[i] < g[i] for some i
        ==> g is not dominated by f if and only if:
            1. f[i] > g[i] for some i, or
            2. f[i] == g[i] for all i

        If we filter all observations by only the condition 1,
        we might miss the observations that satisfy the condition 2.
        However, we already know that those observations do not satisfy the condition 1.
        It implies that the summation of all costs cannot be larger than those of pareto solutions.
        We use this fact to speedup.
    """
    (n_observations, _) = costs.shape
    total_costs = np.sum(costs, axis=-1)  # shape = (n_observations, )
    costs = _change_directions(costs, larger_is_better_objectives)
    on_front_indices = np.arange(n_observations)
    next_index = 0

    while next_index < len(costs):
        nd_mask = np.any(costs < costs[next_index], axis=1)
        nd_mask[next_index] = True
        # Remove dominated points
        on_front_indices, costs = on_front_indices[nd_mask], costs[nd_mask]
        next_index = np.sum(nd_mask[:next_index]) + 1

    mask = np.zeros(n_observations, dtype=np.bool8)
    mask[on_front_indices] = True

    if not filter_duplication:
        missed_pareto_idx = np.arange(n_observations)[~mask][
            np.any(total_costs[mask][:, np.newaxis] == total_costs[~mask], axis=0)
        ]
        mask[missed_pareto_idx] = True

    return mask


def _compute_rank_based_crowding_distance(ranks: np.ndarray) -> np.ndarray:
    (n_observations, n_obj) = ranks.shape
    order = np.argsort(ranks, axis=0)
    order_inv = np.zeros_like(order[:, 0], dtype=np.int32)
    dists = np.zeros(n_observations)
    for m in range(n_obj):
        sorted_ranks = ranks[:, m][order[:, m]]
        order_inv[order[:, m]] = np.arange(n_observations)
        scale = sorted_ranks[-1] - sorted_ranks[0]
        crowding_dists = (
            np.hstack([np.inf, sorted_ranks[2:] - sorted_ranks[:-2], np.inf]) / scale
        )
        dists += crowding_dists[order_inv]

    # crowding dist is better when it is larger
    return scipy.stats.rankdata(-dists).astype(np.int32)


def _tie_break_by_method(
    nd_masks: List[List[int]],
    ranks: np.ndarray,
    avg_ranks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Tie-break the non-domination ranks (, but we cannot guarantee no duplications)

    Args:
        nd_masks (List[List[int]]):
            The indices of observations in each non-domination rank.
        ranks (np.ndarray):
            The ranks of each objective in each observation.
            The shape is (n_observations, n_obj).
        avg_ranks (Optional[np.ndarray]):
            The average rank + small deviation by the best rank of objectives in each observation.
            The shape is (n_observations, ).

    Returns:
        tie_broken_nd_ranks (np.ndarray):
            The each non-dominated rank will be tie-broken
            so that we can sort identically (but we may get duplications).
            The shape is (n_observations, ) and the array is a permutation of zero to n_observations - 1.

    Reference for avg_rank:
        Paper Title:
            Techniques for Highly Multiobjective Optimisation: Some Nondominated Points are Better than Others
        One sentence summary:
            Average ranking strategy is effective to tie-break in some evolution strategies methods.
        Authors:
            David Come and Joshua Knowles
        URL:
            https://arxiv.org/pdf/0908.3025.pdf

    Reference for crowding distance:
        Paper Title:
            A fast and elitist multiobjective genetic algorithm: NSGA-II
        One sentence summary:
            Consider the proximity to neighbors.
        Authors:
            K. Deb et al.
        URL:
            http://vision.ucsd.edu/~sagarwal/nsga2.pdf
    """
    n_checked = 0
    (size, _) = ranks.shape
    tie_broken_nd_ranks = np.zeros(size, dtype=np.int32)

    for mask in nd_masks:
        if avg_ranks is not None:
            tie_break_ranks = scipy.stats.rankdata(avg_ranks[mask]).astype(np.int32)
        else:  # Use crowding distance
            tie_break_ranks = _compute_rank_based_crowding_distance(ranks=ranks[mask])

        # -1 to start tie_broken_nd_ranks from zero
        tie_broken_nd_ranks[mask] = tie_break_ranks + n_checked - 1
        n_checked += len(mask)

    return tie_broken_nd_ranks


def _tie_break(
    costs: np.ndarray,
    nd_ranks: np.ndarray,
    tie_break: Literal["crowding_distance", "avg_rank"],
) -> np.ndarray:
    methods = ["crowding_distance", "avg_rank"]
    ranks = scipy.stats.rankdata(costs, axis=0)
    masks: List[List[int]] = [[] for _ in range(nd_ranks.max() + 1)]
    for idx, nd_rank in enumerate(nd_ranks):
        masks[nd_rank].append(idx)

    if tie_break == methods[0]:
        return _tie_break_by_method(nd_masks=masks, ranks=ranks)
    elif tie_break == methods[1]:
        # min_ranks_factor plays a role when we tie-break same average ranks
        min_ranks_factor = np.min(ranks, axis=-1) / (nd_ranks.size**2 + 1)
        avg_ranks = np.mean(ranks, axis=-1) + min_ranks_factor
        return _tie_break_by_method(nd_masks=masks, ranks=ranks, avg_ranks=avg_ranks)
    else:
        raise ValueError(f"tie_break method must be in {methods}, but got {tie_break}")


def _compute_nondominated_rank_by_sorted_costs(
    costs: np.ndarray,
    filter_duplication: bool,
) -> np.ndarray:
    (n_observations, n_obj) = costs.shape

    if n_obj == 1:
        return scipy.stats.rankdata(costs[:, 0], method="dense") - 1

    ranks = np.zeros(n_observations, dtype=np.int32)
    rank = 0
    indices = np.arange(n_observations)
    ordered_costs, order_inv = _get_ordered_costs_and_order_inv(
        costs=costs, ordered=False
    )
    while indices.size > 0:
        if n_obj == 2:
            on_front = is_pareto_front2d(
                ordered_costs, filter_duplication=filter_duplication, ordered=True
            )
        else:
            on_front = is_pareto_front(
                ordered_costs, filter_duplication=filter_duplication
            )

        ranks[indices[on_front]] = rank
        # Remove pareto front points
        indices, ordered_costs = indices[~on_front], ordered_costs[~on_front]
        rank += 1

    return ranks[order_inv]


def nondominated_rank(
    costs: np.ndarray,
    larger_is_better_objectives: Optional[List[int]] = None,
    tie_break: Optional[Literal["crowding_distance", "avg_rank"]] = None,
    filter_duplication: bool = True,
) -> np.ndarray:
    """
    Calculate the non-dominated rank of each observation.

    Args:
        costs (np.ndarray):
            An array of costs (or objectives).
            The shape is (n_observations, n_objectives).
        larger_is_better_objectives (Optional[List[int]]):
            The indices of the objectives that are better when the values are larger.
            If None, we consider all objectives are better when they are smaller.
        tie_break (bool):
            Whether we apply tie-break or not.
        filter_duplication (bool):
            If True, duplications will be less prioritized in the sorting from the second occurence.
            True actually speeds up the runtime.

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
    (_, n_obj) = costs.shape
    costs = _change_directions(costs, larger_is_better_objectives)

    cached_costs = deepcopy(costs)
    ranks = _compute_nondominated_rank_by_sorted_costs(
        costs=costs, filter_duplication=filter_duplication
    )

    if tie_break is None:
        return ranks
    else:
        return _tie_break(cached_costs, ranks, tie_break)
