# Shell to use with Make
SHELL := /bin/bash

# Set important Paths
PROJECT := dsmlt
LOCALPATH := $(CURDIR)/$(PROJECT)

# Export targets not associated with files
.PHONY: clean flake test vtest coverage install setup

# Clean build files
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

# Check pep8 rules
flake:
	 flake8 dsmlt/ tests/

# Targets for testing
test: flake
	py.test -s $(FLAGS) ./tests/

# Targets for testing verbose
vtest: flake
	py.test -s -v $(FLAGS) ./tests/

# Make coverage report
cov cover coverage: flake
	py.test -s -v --cov-report term --cov-report html --cov dsmlt ./tests
	@echo "open file://`pwd`/htmlcov/index.html"

# Install the package from source
install:
	python setup.py install

# Setup packages need for development
setup:
	pip install -r requirements-dev.txt
