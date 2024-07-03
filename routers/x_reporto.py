# define the router for the heatmap
from fastapi import APIRouter, Depends, HTTPException, Query, Path,File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm,SecurityScopes,HTTPBearer, HTTPAuthorizationCredentials,HTTPBasicCredentials
from typing import List, Dict, Any,Union,Optional
import os
import torch
import numpy as np
import cv2
import io

# define the router
router = APIRouter(
    tags=["X-Reporto"],
    prefix="/x_reporto"
)


@router.post("/inference")
async def inference(
    image: UploadFile = File(...)
):
    # Read the image
    image = await image.read()
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)
    
    # Save the image
    cv2.imwrite("temp.jpg", image)
    
    # Return the image
    return FileResponse("temp.jpg")

