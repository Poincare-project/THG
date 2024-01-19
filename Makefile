testsuite: reformat
	@echo "Running tests"
	@echo "======"
	@python -m pytest -s -v

reformat:
	@echo "Reformatting code"
	@echo "======"
	@python -m black .
	@python -m isort .

