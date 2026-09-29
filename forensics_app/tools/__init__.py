"""Register course functionality here so it appears in the sidebar."""

from .grayscale import GrayscaleTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry
from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwap
from .contrast_streching import ContrastStreching
from .histogram import HistogramVisualization
from .histogram_matching import HistogramMatching

def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            ChannelSplitTool(),
            ChannelSwap(),
            ContrastStreching(),
            HistogramVisualization(),
            HistogramMatching()
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
