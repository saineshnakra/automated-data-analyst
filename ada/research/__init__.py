"""Paper assets for a research study, built from the data and a ``study.yml``.

    ada init data/measurements.csv     # answer a few questions; writes study.yml
    ada build                          # writes paper-assets/
    ada verify                         # rebuilds and checks every number matches

From Python::

    from ada import research
    research.init("data/measurements.csv", interactive=False)
    built = research.build("study.yml")

``study.yml`` holds every decision a result depends on. ``build`` turns it and
the data into a data audit, a data dictionary, a processing record, tables,
figures with their source data and captions, a methods draft, and a package
another researcher can rebuild everything from. ``provenance.json`` records
where each number came from.
"""

from .build import Build, build
from .custom import figure, figure_size, table
from .interview import init
from .study import Study, StudyError, load

__all__ = ["Build", "Study", "StudyError", "build", "figure", "figure_size", "init", "load", "table"]
