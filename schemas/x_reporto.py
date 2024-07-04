from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import numpy as np

class xReporto(BaseModel):

    bounding_boxes: Optional[List[List[float]]]  # List of bounding boxes, each represented as [xmin, ymin, xmax, ymax]
    lm_sentences_decoded: Optional[List[str]]  # List of generated sentences
    report_text: Optional[str]  # Path to the saved report text

    class Config:
        arbitrary_types_allowed = True


