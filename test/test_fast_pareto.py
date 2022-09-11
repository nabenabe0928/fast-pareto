import pytest
import unittest

import numpy as np

from scipy.stats import rankdata

from fast_pareto.pareto import (
    _change_directions,
    _compute_rank_based_crowding_distance,
    _tie_break,
    is_pareto_front,
    nondominated_rank,
)


def naive_pareto_front(costs: np.ndarray) -> np.ndarray:
    size = len(costs)
    is_front = np.ones(size, dtype=np.bool8)
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            cond1 = np.all(costs[j] <= costs[i])
            cond2 = np.any(costs[j] < costs[i])
            if cond1 and cond2:
                is_front[i] = False
                break

    return is_front


def test_compute_rank_based_crowding_distance() -> None:
    n_obj, n_observations = 4, 100
    costs = np.random.random((n_observations, n_obj))
    all_ranks = rankdata(costs, axis=0)
    ranks = all_ranks[is_pareto_front(costs)]

    n_targets = ranks.shape[0]
    sol = _compute_rank_based_crowding_distance(ranks)
    dists = np.zeros(n_targets)
    for m in range(n_obj):
        order = np.argsort(ranks[:, m])
        sorted_ranks = ranks[:, m][order]
        scale = sorted_ranks[-1] - sorted_ranks[0]

        for i, idx in enumerate(order):
            if i == 0:
                dists[idx] += np.inf
            elif i == n_targets - 1:
                dists[idx] += np.inf
            else:
                dists[idx] += (sorted_ranks[i + 1] - sorted_ranks[i - 1]) / scale

    assert np.allclose(rankdata(-dists).astype(np.int32), sol)


def test_tie_break() -> None:
    costs = np.random.random((30, 2))
    nd_ranks = nondominated_rank(costs)
    with pytest.raises(ValueError):
        _tie_break(costs, nd_ranks, tie_break="dummy")  # type: ignore

    sol = _tie_break(costs, nd_ranks, tie_break="crowding_distance")
    assert sol.shape == (30,)
    sol = _tie_break(costs, nd_ranks, tie_break="avg_rank")
    assert sol.shape == (30,)


def test_change_directions() -> None:
    costs = np.random.normal(size=(20, 10))
    new_costs = _change_directions(costs)
    assert np.allclose(costs, new_costs)

    costs = np.random.normal(size=(20, 10))
    larger_is_better_objectives = [1, 3, 4]
    new_costs = _change_directions(
        costs, larger_is_better_objectives=larger_is_better_objectives
    )
    for i in range(len(costs)):
        for idx in larger_is_better_objectives:
            costs[i][idx] *= -1
    assert np.allclose(costs, new_costs)

    with pytest.raises(ValueError):
        costs = np.random.normal(size=(20, 10))
        larger_is_better_objectives = [20]
        _change_directions(
            costs, larger_is_better_objectives=larger_is_better_objectives
        )

    with pytest.raises(ValueError):
        costs = np.random.normal(size=(20, 10))
        larger_is_better_objectives = [-1]
        _change_directions(
            costs, larger_is_better_objectives=larger_is_better_objectives
        )


def test_pareto_front() -> None:
    for _ in range(3):
        costs = np.random.normal(size=(100, 3))
        assert np.allclose(is_pareto_front(costs), naive_pareto_front(costs))

        costs = np.random.normal(size=(100, 3))
        new_costs = costs.copy()
        new_costs[:, 0] *= -1
        assert np.allclose(
            is_pareto_front(costs, larger_is_better_objectives=[0]),
            naive_pareto_front(new_costs),
        )


def test_with_same_values() -> None:
    costs = np.array([[1, 1], [1, 1], [1, 2], [2, 1], [0, 1.5], [1.5, 0], [0, 1.5]])
    pf = is_pareto_front(costs=costs, filter_duplication=False)
    assert np.allclose(pf, np.array([True, True, False, False, True, True, True]))
    pf = is_pareto_front(costs=costs, filter_duplication=True)
    assert np.allclose(pf, np.array([True, False, False, False, True, True, False]))
    pf = nondominated_rank(costs, tie_break=None, filter_duplication=False)
    assert np.allclose(pf, np.array([0, 0, 1, 1, 0, 0, 0]))
    pf = nondominated_rank(costs, tie_break=None, filter_duplication=True)
    assert np.allclose(pf, np.array([0, 1, 2, 2, 0, 0, 1]))

    costs = np.array([[1, 1], [1, 1], [1, 2], [2, 1], [1, 1], [0, 1.5], [0, 1.5]])
    pf = nondominated_rank(costs, tie_break=None, filter_duplication=False)
    assert np.allclose(pf, np.array([0, 0, 1, 1, 0, 0, 0]))
    pf = nondominated_rank(costs, tie_break=None, filter_duplication=True)
    assert np.allclose(pf, np.array([0, 1, 3, 3, 2, 0, 1]))


def test_no_change_in_costs_pareto_front() -> None:
    costs = np.random.normal(size=(100, 3))
    ans = costs.copy()
    is_pareto_front(costs, larger_is_better_objectives=[0])
    assert np.allclose(costs, ans)


def test_nondominated_rank() -> None:
    for _ in range(3):
        costs = np.random.normal(size=(100, 1))
        ranks = np.argsort(np.argsort(costs.flatten()))
        assert np.allclose(nondominated_rank(costs), ranks)

        costs = np.random.normal(size=(100, 1))
        new_costs = costs.copy()
        new_costs[:, 0] *= -1
        ranks = np.argsort(np.argsort(new_costs.flatten()))
        assert np.allclose(
            nondominated_rank(costs, larger_is_better_objectives=[0]), ranks
        )


def test_no_change_in_costs_nondominated_rank() -> None:
    costs = np.random.normal(size=(100, 3))
    ans = costs.copy()
    nondominated_rank(costs, larger_is_better_objectives=[0])
    assert np.allclose(costs, ans)


def test_nondominated_rank_with_tie_break() -> None:
    for _ in range(3):
        costs = np.random.normal(size=(100, 1))
        nd_ranks = nondominated_rank(costs)
        for method in ["avg_rank", "crowding_distance"]:
            ranks_with_tie_break = nondominated_rank(costs, tie_break=method)  # type: ignore
            rank_min = nd_ranks.min()
            rank_max = nd_ranks.max()

            head, tail = 0, 0
            for rank in range(rank_min, rank_max + 1):
                mask = nd_ranks == rank
                targets = nd_ranks[mask]
                tail += targets.size
                assert np.all(head <= ranks_with_tie_break[mask] <= tail - 1)
                head = tail


if __name__ == "__main__":
    unittest.main()
