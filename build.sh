rm -r build/ dist/ fast_pareto.egg-info/

python setup.py bdist_wheel
twine upload --repository pypi dist/*
