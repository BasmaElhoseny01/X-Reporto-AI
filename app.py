# create fastapi app
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from routers import x_reporto
from routers import heatmap, x_reporto
from src.inference.x_reporto import XReporto
from src.inference.heat_map_inference import HeatMapInference

import uvicorn

app = FastAPI()

app.include_router(x_reporto.router)
app.include_router(heatmap.router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Inference class on startup event of the FastAPI app
@app.on_event("startup")
async def startup_event():
    global x_reporto_inference
    x_reporto_inference = XReporto()
    global heatmap_inference
    heatmap_inference = HeatMapInference()

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
