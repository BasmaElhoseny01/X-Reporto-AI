from src.x_reporto.models.x_reporto_v1 import XReportoV1

class XReporto():
    def __init__(self):
        pass
    
    def create_model(self) -> XReportoV1:
        return XReportoV1()

model=XReporto().create_model()
print(model)