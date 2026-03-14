"""
Global constants and type definitions for PDtoolkit.

This module contains package-level constants, type aliases, and common
definitions used across the PDtoolkit package.

Note: The original R file (00_GLOBALS.R) only declared global variables
for R CMD check. In Python, we use this module to define common constants
and type hints that are shared across modules.
"""

from typing import List, Union, Sequence
import numpy as np

# Default special case values (equivalent to R's NA, NaN, Inf, -Inf)
DEFAULT_SPECIAL_CASES: List[Union[float, None]] = [None, np.nan, np.inf, -np.inf]

# Type aliases for common patterns
NumericValue = Union[int, float]
SpecialCaseType = Union[float, None]

# Default thresholds used across the package
DEFAULT_SC_THRESHOLD: float = 0.2
DEFAULT_MIN_PCT_OBS: float = 0.05
DEFAULT_MIN_AVG_RATE: float = 0.01
DEFAULT_P_VALUE: float = 0.05

# Tolerance for floating-point comparisons
FLOAT_TOLERANCE: float = 1e-8

# Default replacement string for special cases
DEFAULT_SC_REPLACEMENT: str = "SC"
