from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import numpy as np

class HeatMap(BaseModel):

    heatmap: Optional[List[List[List[float]]]]  # Heatmap generated
    labels: Optional[List[int]]  # List of labels corresponding to the heatmap
    confidence: Optional[List[float]]  # Confidence score of the heatmap

    class Config:
        arbitrary_types_allowed = True
