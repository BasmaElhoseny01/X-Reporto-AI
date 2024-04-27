from src.x_reporto.models.x_reporto_v1 import XReportoV1

class Inference:
    def __init__(self):

        # Read the model
        x_reporto = XReportoV1()

    # def generate_image_report(self,image_path):
    #     # Input is Image
    #     # Output is Image with bounding box
    #     # Selected Regions / Abnormal Region
    #     # Report

    #     pass


if __name__=="__main__":
    print("Inference")
    # Initialize the Inference class
    inference = Inference()

    # Generate the report
    # inference.generate_image_report("path/to/image.jpg")
    pass