# Fast non-dominated search library

This library is solely for non-dominated search and to find pareto optimal solutions.
There are only two functions in this library.
Note that all objective 

## Setup

```shell
$ pip install fast_pareto
```

## Examples

```python

>>> []
```

## is_pareto_front

This function determines the pareto front from a provided set of costs.
The arguments are a numpy array with the shape of `(n_observations, n_objectives)` and a list of indices that show which objectives are "larger is better".
If `None` is provided, we consider all objectives should be minimized.
This function returns the true/false mask with the shape of `(n_observations, )`.
True means the corresponding observation is on the pareto front given a set of solutions.

## nondominated_sort

This function calculates the non-dominated rank of each observation.
The arguments are a numpy array with the shape of `(n_observations, n_objectives)` and a list of indices that show which objectives are "larger is better".
If `None` is provided, we consider all objectives should be minimized.
This function returns the non-dominated rank with the shape of `(n_observations, )`.
The non-dominated rank is better when it is smaller.
For this implementation, we return zero when those observations are on the pareto front.

## Benchmarking
### Test code

```python
for n_points in [100, 1000, 10000]:
    for n_costs in [1, 5, 10, 50]:
        print(f"n_points={n_points}, n_costs={n_costs}")
        %time nondominated_sort(np.random.normal(size=(n_points, n_costs)))
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
