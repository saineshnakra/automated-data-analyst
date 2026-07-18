# Contributing

Thanks for improving Automated Data Analyst.

1. Fork the repository and create a focused branch.
2. Install `requirements-dev.txt` in a Python 3.11+ virtual environment.
3. Run `ruff check .` and `python -m unittest discover -s tests -v`.
4. Open a pull request explaining the user problem, the change, and how it was tested.

Keep analysis deterministic and explainable. Features that upload user rows to an external service or execute generated code are out of scope unless they are isolated, opt-in, and reviewed for security and privacy.
