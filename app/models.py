from pydantic import BaseModel
from typing import Tuple, Optional

class ProcessingConfig(BaseModel):
    video_path: str = "../samples_videos/1.mp4"
    min_contour_area: int = 800
    heatmap_alpha: float = 0.5
    show_heatmap: bool = True
    show_boxes: bool = True
    box_color: Tuple[int, int, int] = (0, 255, 0)  # BGR
    colormap: int  # OpenCV colormap
    time_range_start: float = 0.0
    time_range_end: float = -1.0  # -1 means process entire video