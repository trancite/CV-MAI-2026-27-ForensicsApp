"""Register course functionality here so it appears in the sidebar."""

from .grayscale import GrayscaleTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry
from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwap
from .contrast_operations import ContrastOperations
from .histogram import HistogramVisualization
from .histogram_matching import HistogramMatching
from .masking import Masking
from .masking import Masking
from .convolution import Convolution
from .masking_superposition import  MaskingSuperposition
from .filters import FiltersTool
from .canny_edge import CannyEdge

def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            ChannelSplitTool(),
            ChannelSwap(),
            ContrastOperations(),
            HistogramVisualization(),
            HistogramMatching(),
            Masking(),
            Convolution(),
            MaskingSuperposition(),
            FiltersTool(),
            CannyEdge(),


        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
