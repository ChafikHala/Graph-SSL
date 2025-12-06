# Import Hala's GCN
from .base_gnn import BaseGCN

# Import Your GIN
from .gin import GINEncoder

__all__ = ["BaseGCN", "GINEncoder"]