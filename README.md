# Fast non-dominated sort library

[![Build Status](https://github.com/nabenabe0928/fast_pareto/workflows/Functionality%20test/badge.svg?branch=main)](https://github.com/nabenabe0928/fast_pareto)
[![codecov](https://codecov.io/gh/nabenabe0928/fast_pareto/branch/main/graph/badge.svg?token=ZBJJ77IHI4)](https://codecov.io/gh/nabenabe0928/fast_pareto)

This library is solely for non-dominated search and to find pareto optimal solutions.
There are only two functions in this library.
Note that all objective 

## Setup

```shell
$ pip install fast-pareto
```

## Examples

```python
from fast_pareto import is_pareto_front, nondominated_rank
import numpy as np


is_pareto_front(np.array([[0, 1], [1, 0], [1, 1]]))
>>> array([True, True, False])

is_pareto_front(np.array([[0, -1], [1, 0], [1, -1]]), larger_is_better_objectives=[1])
>>> array([True, True, False])

nondominated_rank(np.array([[2], [1], [0]]))
>>> array([2, 1, 0], dtype=int32)

nondominated_rank(np.array([[2], [1], [0]]), larger_is_better_objectives=[0])
>>> array([0, 1, 2], dtype=int32)
```

## is_pareto_front

This function determines the pareto front from a provided set of costs.
The arguments are a numpy array with the shape of `(n_observations, n_objectives)` and a list of indices that show which objectives are "larger is better".
If `None` is provided, we consider all objectives should be minimized.
This function returns the true/false mask with the shape of `(n_observations, )`.
True means the corresponding observation is on the pareto front given a set of solutions.

## nondominated_rank

This function calculates the non-dominated rank of each observation.
The arguments are a numpy array with the shape of `(n_observations, n_objectives)` and a list of indices that show which objectives are "larger is better".
If `None` is provided, we consider all objectives should be minimized.
This function returns the non-dominated rank with the shape of `(n_observations, )`.
The non-dominated rank is better when it is smaller.
For this implementation, we return zero when those observations are on the pareto front.

You can see the examples of [the results obtained by this module](example/example_visualizations.ipynb) below.
<table>
    <tr>
        <td><img src="figs/nd-rank-gauss.png" alt=""></td>
        <td><img src="figs/nd-rank-inv.png" alt=""></td>
    </tr>
</table>

Note that we added the tie-breaking feature using average ranks in v0.0.5
and when you specify `tie_break=True`, this function returns the ranks of each observation
with tie-breaking.
For example, when we have non-domination ranks of `[0, 0, 1, 1]` with `tie_break=False`,
then `tie_break=True` tries to differentiate those values to be such as `[1, 0, 2, 3]`.
When using this feature, we will not get non-domination ranks anymore,
but if the rank for the i-th observation `r[i]` and that for the j-th observation `r[j]`
have the relationship of `r[i] < r[j]`, it is guaranteed that the non-domination rank
of the i-th observation is lower or equal to that of the j-th observation.

## Benchmarking
### Test code

```python
for n_points in [100, 1000, 10000]:
    for n_costs in [1, 5, 10, 50]:
        print(f"n_points={n_points}, n_costs={n_costs}")
        %time nondominated_rank(np.random.normal(size=(n_points, n_costs)))
        print("\n")
```

## Results

```shell
n_points=100, n_costs=1
CPU times: user 10.7 ms, sys: 0 ns, total: 10.7 ms
Wall time: 10.2 ms


n_points=100, n_costs=5
CPU times: user 3.29 ms, sys: 0 ns, total: 3.29 ms
Wall time: 3.3 ms


n_points=100, n_costs=10
CPU times: user 3 ms, sys: 0 ns, total: 3 ms
Wall time: 3 ms


n_points=100, n_costs=50
CPU times: user 3.57 ms, sys: 0 ns, total: 3.57 ms
Wall time: 3.57 ms


n_points=1000, n_costs=1
CPU times: user 105 ms, sys: 0 ns, total: 105 ms
Wall time: 105 ms


n_points=1000, n_costs=5
CPU times: user 37.8 ms, sys: 0 ns, total: 37.8 ms
Wall time: 37.8 ms


n_points=1000, n_costs=10
CPU times: user 53.5 ms, sys: 0 ns, total: 53.5 ms
Wall time: 53.5 ms


n_points=1000, n_costs=50
CPU times: user 90.4 ms, sys: 0 ns, total: 90.4 ms
Wall time: 90.4 ms


n_points=10000, n_costs=1
CPU times: user 3.36 s, sys: 0 ns, total: 3.36 s
Wall time: 3.36 s


n_points=10000, n_costs=5
CPU times: user 1.22 s, sys: 0 ns, total: 1.22 s
Wall time: 1.22 s


n_points=10000, n_costs=10
CPU times: user 2.72 s, sys: 0 ns, total: 2.72 s
Wall time: 2.72 s


n_points=10000, n_costs=50
CPU times: user 14.4 s, sys: 0 ns, total: 14.4 s
Wall time: 14.4 s
```

## Appendix

To supplement the knowledge, I note the definition of non-dominated rank.
Suppose we would like to minimize the multiobjective function <img src="https://render.githubusercontent.com/render/math?math=f: \mathbb{R}^D \rightarrow \mathbb{R}^M">. <img src="https://render.githubusercontent.com/render/math?math=f(\boldsymbol{x})$ is said to dominate $f(\boldsymbol{x}^\prime)"> if and only if <img src="https://render.githubusercontent.com/render/math?math=\forall i \in [1, M], f_i(\boldsymbol{x}) \leq f_i(\boldsymbol{x}^\prime)"> and <img src="https://render.githubusercontent.com/render/math?math=\exists i \in [1, M], f_i(\boldsymbol{x}) < f_i(\boldsymbol{x}^\prime)">.

When there is no such observation that dominates <img src="https://render.githubusercontent.com/render/math?math=f(\boldsymbol{x})">, <img src="https://render.githubusercontent.com/render/math?math=f(\boldsymbol{x})"> is said to be pareto optimal and the non-domination rank of <img src="https://render.githubusercontent.com/render/math?math=f(\boldsymbol{x})"> is defined as 1 (but in our code, we define it as zero).
Furthermore, <img src="https://render.githubusercontent.com/render/math?math=f(\boldsymbol{x}^\prime)"> is said to be the non-domination rank of <img src="https://render.githubusercontent.com/render/math?math=n"> when <img src="https://render.githubusercontent.com/render/math?math=f(\boldsymbol{x}^\prime)"> is the pareto optimal in a set such that it excludes observations with the non-domination rank of <img src="https://render.githubusercontent.com/render/math?math=n - 1"> or lower.
