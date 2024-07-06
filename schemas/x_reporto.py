from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import numpy as np

class XReporto(BaseModel):

    bounding_boxes: Optional[List[List[float]]]  # List of bounding boxes, each represented as [xmin, ymin, xmax, ymax]
    lm_sentences_decoded: Optional[List[str]]  # List of generated sentences
    report_text: Optional[str]  # Path to the saved report text
    detected_classes: Optional[List[int]]  # List of detected classes

    class Config:
        arbitrary_types_allowed = True


