# create fastapi app
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import heatmap, x_reporto
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

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
