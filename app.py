"""The ADA browser app, which is in ADA Lab now: https://github.com/saineshnakra/ada-lab.

This file serves it from the ada-lab package that requirements.txt installs, so the
deployment at automated-data-analyst.streamlit.app keeps working until it is pointed
at the ADA Lab repository.
"""

import runpy
from pathlib import Path

import ada_lab

runpy.run_path(str(Path(ada_lab.__file__).with_name("app.py")), run_name="__main__")
