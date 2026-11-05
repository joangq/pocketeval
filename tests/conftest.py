"""Exercise flow helpers offline if the optional runtime is not installed."""

from __future__ import annotations

import sys
from importlib.util import find_spec

if find_spec("pocketflow") is None:
    from tests import pocketflow_stub

    sys.modules["pocketflow"] = pocketflow_stub
