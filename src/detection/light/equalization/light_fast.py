import cv2
import numpy as np
from typing import Any

def enhance_brightness(frame: Any) -> Any:
    """
    Enhances the brightness of a frame using histogram equalization.

    Args:
        frame (Any): Input frame in BGR format (OpenCV image format).

    Returns:
        Any: Enhanced frame in BGR format.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError("The provided frame is not a NumPy array. Please check the source of the image.")

    # Convert the frame to YUV color space
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    # Apply histogram equalization on the Y channel
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])

    # Convert the YUV frame back to BGR color space
    enhanced_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    return enhanced_frame