"""
pytest configuration for the BSS PoC test suite.
Adds src/poc/ to sys.path so that 'import bss_poc' resolves correctly
regardless of where pytest is invoked from.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
