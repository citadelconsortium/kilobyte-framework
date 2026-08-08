.PHONY: test check

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

check:
	PYTHONPATH=src python -m compileall -q src tests
	PYTHONPATH=src python -m unittest discover -s tests -v

