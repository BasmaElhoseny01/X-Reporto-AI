# Base Image
# FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# set the working directory in the container
WORKDIR /ai_app


# Install dependencies for OpenCV
RUN apt-get update && apt-get install -y \
libgl1-mesa-glx \
libglib2.0-0

# copy the dependencies file to the working directory
COPY requirements_docker.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy the content of the local src directory to the working directory
ADD src /ai_app/src

# Config File
ADD config.py /ai_app/config.py

# Copy Models
# X-Reporto Models
ADD models/object_detector_best.pth /ai_app/models/object_detector.pth
ADD models/abnormal_classifier_best.pth /ai_app/models/binary_classifier_region_abnormal.pth
ADD models/region_classifier_best.pth /ai_app/models/binary_classifier_selection_region.pth
ADD models/LM_best.pth /ai_app/models/LM.pth

# Heat Map Models
ADD models/heat_map_best.pth /ai_app/models/heat_map_best.pth

# Copy example
# Add datasets/mimic-cxr-jpg/files/p11/p11001469/s54076811/d0d2bd0c-8bc50aa2-a9ab3ca1-cf9c9404-543a10b7.jpg images/d0d2bd0c-8bc50aa2-a9ab3ca1-cf9c9404-543a10b7.jpg
ADD dataset_volume images/

# # Make port 8000 available to the world outside this container
# CMD ["python", "-m", "src.inference.heat_map_inference","/ai_app/images/d0d2bd0c-8bc50aa2-a9ab3ca1-cf9c9404-543a10b7.jpg"]
CMD ["python", "-m", "src.inference.heat_map_inference","/ai_app/images/d0d2bd0c-8bc50aa2-a9ab3ca1-cf9c9404-543a10b7.jpg"]

# docker run --rm -it --gpus all -v /media/basma/basmavolume/Graduation-Project/dataset_volume:/ai_app/image gp_ai:v1 python -m src.inference.heat_map_inference /ai_app/images/bb54dd1c-16e10d5b-6f17da3a-42f38fc2-a4733777.jpg
# docker run --rm -it --gpus all -v /media/basma/basmavolume/Graduation-Project/dataset_volume:/ai_app/images gp_ai:v1 /ai_app/images/bb54dd1c-16e10d5b-6f17da3a-42f38fc2-a4733777.jpg
