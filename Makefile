# Shell to use with Make
SHELL := /bin/bash

# Setup flags
FLAGS=

# Set important Paths
PROJECT := dsmlt
LOCALPATH := $(CURDIR)/$(PROJECT)

# Default targets
.DEFAULT: clean lint install setup cov-report

# Install all packages need for development
.PHONY: install
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -Ue .

# Setup the package from source
.PHONY: setup
setup:
	python setup.py install

# Clean build files
.PHONY: clean
clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -f `find . -type f -name '.*~' `
	rm -f `find . -type f -name '@*' `
	rm -f `find . -type f -name '#*#' `
	rm -f `find . -type f -name '*.orig' `
	rm -f `find . -type f -name '*.rej' `
	rm -f .coverage
	rm -rf coverage
	rm -rf build
	rm -rf htmlcov
	rm -rf dist
	rm -rf $(PROJECT).egg-info
	rm -rf .eggs
	rm -rf site
	rm -rf classes_$(PROJECT).png
	rm -rf packages_$(PROJECT).png
	rm -rf docs/_build

# Format source code
.PHONY: format
format:
	black .
	autoflake -r --in-place \
		--remove-unused-variables \
		--remove-all-unused-imports \
		--remove-duplicate-keys .

.PHONY:checkrst
checkrst:
	python setup.py check --restructuredtext

# Check pep8 rules
.PHONY: lint flake
lint flake: checkrst
	 flake8 --show-source dsmlt/ tests/

# Targets for testing
.PHONY: test
test: lint
	py.test -s $(FLAGS) ./tests/

# Targets for testing verbose
.PHONY: vtest
vtest: lint
	py.test -s -v $(FLAGS) ./tests/

# Make tests with coverage
.PHONY: cov cover coverage
cov cover coverage: lint checkrst
	py.test -s -v --cov dsmlt $(FLAGS) ./tests

# Make coverage report
.PHONY: cov-report cover-report coverage-report
cov-report cover-report coverage-report: lint checkrst
	py.test -s -v --cov-report term --cov-report html --cov dsmlt $(FLAGS) ./tests
	@echo "open file://`pwd`/htmlcov/index.html"
