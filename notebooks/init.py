"""Initializes the environment for the project. Makes the modules in the
root directory importable in the notebooks. Must be imported in every notebook."""

import sys
import os

_project_root = os.path.abspath('../')
if _project_root not in sys.path:
    sys.path.append(_project_root)
