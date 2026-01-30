"""Custom/experimental models added by xjy.

This package is intentionally kept separate from the upstream `models/` package
so you can add new research prototypes without touching upstream model files.

Naming convention:
- Each file my_models/V#.py defines a class `Model` compatible with exp/ training.
- Select via: --model V1 ... --model V10 (after registration in exp/exp_basic.py)
"""

from . import model_1_30
from . import V1, V2, V3, V4, V5, V6, V7, V8, V9, V10

__all__ = [
    "model_1_30",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
]
