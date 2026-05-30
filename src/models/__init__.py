from .duala import build_duala_models, predict_embeddings

try:
    from .lightea import LightEAWrapper
    HAS_LIGHTEA = True
except ImportError:
    HAS_LIGHTEA = False

try:
    from .gcn_align import GCNAlignWrapper
    HAS_GCNALIGN = True
except ImportError:
    HAS_GCNALIGN = False
