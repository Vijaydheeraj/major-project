import os
import re
from typing import Tuple, Optional

def extract_camera_data(video_path: str) -> Tuple[int, Optional[str]]:
    filename = os.path.basename(video_path)
    match = re.search(r'CAM(\d+)', filename)
    camera_number = int(match.group(1)) if match else 0

    time_match = re.search(r'(\d{2}h\d{2}m\d{2}s)', filename)
    time_str = time_match.group(1) if time_match else None

    return camera_number, time_str

