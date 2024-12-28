SHELL:=/bin/bash
VIRTUAL_ENV=.venv
PYTHON=${VIRTUAL_ENV}/bin/python3
PROJECT=gemma_template


DEFAULT_GOAL: help
.PHONY: help all install lint readme docs

# Colors for echos
bold = \033[0;32m\033[1m***
end_bold = *** \033[0m\033[0;0m

all: ##@target >> Run all.
	@make install
	@make lint

# And add help text after each target name starting with '\#\#'
# A category can be added with @category
HELP_FUN = \
	%help; \
	while(<>) { push @{$$help{$$2 // 'options'}}, [$$1, $$3] if /^([a-zA-Z\-\$\(]+)\s*:.*\#\#(?:@([a-zA-Z\-\)]+))?\s(.*)$$/ }; \
	print "Usage: make [target]\n\n"; \
	for (sort keys %help) { \
	print "${WHITE}$$_:${RESET}\n"; \
	for (@{$$help{$$_}}) { \
	$$sep = " " x (32 - length $$_->[0]); \
	print "  ${YELLOW}$$_->[0]${RESET}$$sep${GREEN}$$_->[1]${RESET}\n"; \
	}; \
	print "\n"; }

help: ##@target >> Show this help.
	@perl -e '$(HELP_FUN)' $(MAKEFILE_LIST)
	@echo ""
	@echo "Note: to activate the environment in your local shell type:"
	@echo "	$$ source $(VIRTUAL_ENV)/bin/activate"

install: ##@target >> Setup virtualenv and Install dependencies.
	@echo -e "$(bold) Setup virtualenv and install Dependencies $(end_bold)"
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r requirements-dev.txt

lint: ##@target >> Run Lint.
	@echo -e "$(bold) Run Lint $(end_bold)"
	@echo -e "$(bold) Black $(end_bold)"
	@$(PYTHON) -m black $(PROJECT)
	@echo -e "$(bold) iSort $(end_bold)"
	@$(PYTHON) -m isort $(PROJECT)
	@echo -e "$(bold) Flake8 $(end_bold)"
	@$(PYTHON) -m flake8 $(PROJECT)

test:
	tox -p
	rm -rf *.egg-info

readme:
	python setup.py check --restructuredtext --strict && ([ $$? -eq 0 ] && echo "README.rst and CHANGELOG.md ok") || echo "Invalid markup in README.md or CHANGELOG.md!"

docs:
	mkdocs serve
