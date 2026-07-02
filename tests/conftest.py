"""pytest configuration: make the local ``fixtures`` helper importable from tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
