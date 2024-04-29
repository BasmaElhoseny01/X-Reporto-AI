import torch
# from src.object_detector.models.frccn_object_detector_v1 import FrcnnObjectDetectorV1
from src.x_reporto.models.x_reporto_factory import XReporto
def extract_backbone(model_path, save_path):
    """
    Extract the backbone atoms from a PDB file
    """
    # Create the model
    x_reporto_model = XReporto().create_model()
    print(x_reporto_model)

    # Load the model
    # model.load_state_dict(torch.load(model_path))
    # print(torch.load(model_path).keys())

    # print(model.backbone)
    # # model.load_state_dict(torch.load("models/" + str(RUN) + '/' + name + ".pth"))
    # # # Load the model
    # # model = torch.load(model_path)
    # # # Extract the backbone atoms
    # # backbone = model.object_detector.object_detector_module.backbone
    # # # Save the backbone atoms
    # # torch.save(backbone, save_path)

if __name__ =="__main__":
    extract_backbone("models/object_detector_best.pth", "backbone.pth")