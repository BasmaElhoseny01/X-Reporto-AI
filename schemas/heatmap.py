from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import numpy as np

class HeatMap(BaseModel):

    heatmap: Optional[List[List[List[float]]]]  # Heatmap generated
    labels: Optional[List[int]]  # List of labels corresponding to the heatmap
    confidence: Optional[List[float]]  # Confidence score of the heatmap
    severity: Optional[float]  # Severity score of the heatmap
    report: Optional[str]  # Path to the saved report text

    class Config:
        arbitrary_types_allowed = True
