# define the router for the heatmap
from fastapi import APIRouter, Depends, HTTPException, Query, Path,File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm,SecurityScopes,HTTPBearer, HTTPAuthorizationCredentials,HTTPBasicCredentials
from typing import List, Dict, Any,Union,Optional
import os
import numpy as np
import cv2
import io
import uuid
from src.inference.heat_map_inference import HeatMapInference
from schemas import heatmap as heatmap_schema
# define the router
router = APIRouter(
    tags=["Heatmap"],
    prefix="/heatmap"
)


@router.post("/generate_heatmap")
async def generate_heatmap(
    image: UploadFile = File(...)
) -> heatmap_schema.HeatMap:
    
    # Read the image
    image = await image.read()
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)
    
    # Generate unique filename
    unique_filename = str(uuid.uuid4()) + ".jpg"

    image_path = f"{unique_filename}"

    # Save the image
    cv2.imwrite(image_path, image)

    # Initialize the Inference class
    inference = HeatMapInference()

    # Perform inference
    heatmap, labels, confidence = inference.infer(image_path)

    # delete the image
    os.remove(image_path)

    # convert the heatmap to list of list of list
    heatmap = heatmap.tolist()
    # print(f"heatmap: {heatmap}")
    # print(f"labels: {labels}")
    # print(f"confidence: {confidence}")

    return {
        "heatmap": heatmap,
        "labels": labels,
        "confidence": confidence
    }