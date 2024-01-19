testsuite: reformat
	@echo "Running tests"
	@echo "======"
	@python -m pytest -s -q

reformat:
	@echo "Reformatting code"
	@echo "======"
	@python -m black .
	@python -m isort .

