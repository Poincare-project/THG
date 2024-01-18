testsuite:
	@echo "Running tests"
	@echo "======"
	@python -m pytest -s

reformat:
	@echo "Reformatting code"
	@echo "======"
	@python -m black .
	@python -m isort .

