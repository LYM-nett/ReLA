try:  # pragma: no cover - optional during pure-Python diagnostics
    from . import data  # register all new datasets
    from . import modeling
except ModuleNotFoundError:
    data = None  # type: ignore
    modeling = None  # type: ignore

# config
from .config import add_maskformer2_config, add_refcoco_config

# dataset loading
try:  # pragma: no cover
    from .data.dataset_mappers.refcoco_mapper import RefCOCOMapper
except ModuleNotFoundError:  # pragma: no cover - Detectron2 absent
    RefCOCOMapper = None  # type: ignore

# models
try:  # pragma: no cover
    from .GRES import GRES
except ModuleNotFoundError:  # pragma: no cover
    GRES = None  # type: ignore

# evaluation
try:  # pragma: no cover
    from .evaluation.refer_evaluation import ReferEvaluator
except ModuleNotFoundError:  # pragma: no cover
    ReferEvaluator = None  # type: ignore
