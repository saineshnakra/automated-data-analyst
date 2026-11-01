"""``python -m ada``: the ``ada`` command, for when its script is not on the PATH."""

from .cli import main

raise SystemExit(main())
