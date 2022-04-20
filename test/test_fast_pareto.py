import pytest
import unittest

import numpy as np

from fast_pareto.pareto import (
    _change_directions,
    is_pareto_front,
    nondominated_sort
)


def test_change_directions():
    costs = np.random.normal(size=(20, 10))
    new_costs = _change_directions(costs)
    assert np.allclose(costs, new_costs)

    costs = np.random.normal(size=(20, 10))
    larger_is_better_objectives = [1, 3, 4]
    new_costs = _change_directions(costs, larger_is_better_objectives=larger_is_better_objectives)
    for i in range(len(costs)):
        for idx in larger_is_better_objectives:
            costs[i][idx] *= -1
    assert np.allclose(costs, new_costs)

    with pytest.raises(ValueError):
        costs = np.random.normal(size=(20, 10))
        larger_is_better_objectives = [20]
        _change_directions(costs, larger_is_better_objectives=larger_is_better_objectives)

    with pytest.raises(ValueError):
        costs = np.random.normal(size=(20, 10))
        larger_is_better_objectives = [-1]
        _change_directions(costs, larger_is_better_objectives=larger_is_better_objectives)


if __name__ == "__main__":
    unittest.main()
