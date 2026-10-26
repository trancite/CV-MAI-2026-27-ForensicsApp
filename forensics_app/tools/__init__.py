"""Register course functionality here so it appears in the sidebar."""


from .grayscale import GrayscaleTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry
from .set2.channel_split import ChannelSplitTool
from .set2.channel_swap import ChannelSwap
from .set2.contrast_operations import ContrastOperations
from .set2.histogram import HistogramVisualization
from .set2.histogram_matching import HistogramMatching
from .set2.masking import Masking
from .set2.masking_superposition import  MaskingSuperposition
from .set3.convolution import Convolution
from .set3.filters import FiltersTool
from .set3.canny_edge import CannyEdge
from .set3.sharpening import Sharpening
from .set4.cross_correlation import CrossCorrelationTool
from .set4.hog import HOGTool
from .set4.person_detection_hog import PersonDetectionHog
from .set5.corner_detection import CornerDetectionTool
from .set5.descriptor_and_keypoints import DescriptorAndKeypointsTool
from .set5.match_with_second_image import MatchWithSecondImageTool
from .set5.find_closest_image import FindClosestImageTool


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
            Sharpening(),
            HOGTool(),
            CrossCorrelationTool(),
            PersonDetectionHog(),
            CornerDetectionTool(),
            DescriptorAndKeypointsTool(),
            MatchWithSecondImageTool(),
            FindClosestImageTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
