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
from src.inference.x_reporto import XReporto
from schemas import x_reporto as x_reporto_schema
import uuid
import time
# define the router
router = APIRouter(
    tags=["X-Reporto"],
    prefix="/x_reporto"
)


@router.post("/report")
async def inference(
    image: UploadFile = File(...)
)-> x_reporto_schema.XReporto:
    # Read the image
    image = await image.read()
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)
    
    # Generate unique filename
    unique_filename = str(uuid.uuid4()) + ".jpg"

    image_path = f"{unique_filename}"  # Assuming you want to save it in a 'temp' folder

    # Save the image
    cv2.imwrite(image_path, image)

    start = time.time()
    # Initialize the Inference class
    inference = XReporto()

    # Perform inference
    bounding_boxes, generated_sentences, report_text = inference.generate_image_report(image_path)
    # delete the image
    os.remove(image_path)
    end = time.time()

    print(f"Time taken to generate report: {end-start}")
    # Return the results
    return {
        "bounding_boxes": bounding_boxes,
        "lm_sentences_decoded": generated_sentences,
        "report_text": report_text
    }

@router.get("/denoise")
async def denoise(
    image: UploadFile = File(...)
)-> StreamingResponse:
    # Read the image
    image = await image.read()
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)
    
    # Generate unique filename
    unique_filename = str(uuid.uuid4()) + ".jpg"

    image_path = f"{unique_filename}"

    # Save the image
    cv2.imwrite(image_path, image)

    # Initialize the Inference class
    inference = XReporto()

    # Perform denoising
    #TODO: Implement the denoise_image method in the Inference class
    denoised_image = inference.denoise_image(image_path) 

    # delete the image
    os.remove(image_path)

    # Return the denoised image
    return StreamingResponse(io.BytesIO(cv2.imencode('.jpg', denoised_image)[1].tobytes()), media_type="image/jpeg")
