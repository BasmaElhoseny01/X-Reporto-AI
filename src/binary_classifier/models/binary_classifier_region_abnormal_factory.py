from src.binary_classifier.models.binary_classifier_region_abnormal_v1 import BinaryClassifierRegionAbnormalV1

class BinaryClassifierRegionAbnormal():
    def __init__(self):
        pass
    
    def create_model(self) -> BinaryClassifierRegionAbnormalV1:
        return BinaryClassifierRegionAbnormalV1()

# model=BinaryClassifierRegionAbnormal().create_model()
# print(model)