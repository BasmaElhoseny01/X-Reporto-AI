from src.binary_classifier.models.binary_classifier_selection_region_v1 import BinaryClassifierSelectionRegionV1

class BinaryClassifierSelectionRegion():
    def __init__(self):
        pass
    
    def create_model(self) -> BinaryClassifierSelectionRegionV1:
        return BinaryClassifierSelectionRegionV1()

# model=BinaryClassifierSelectionRegion().create_model()
# print(model)