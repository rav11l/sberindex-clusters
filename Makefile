.PHONY: data all test figures

data:
	python -m sclusters.cli data --download

all:
	python -m sclusters.cli all

figures:
	python -m sclusters.cli figures

test:
	pytest -q tests
