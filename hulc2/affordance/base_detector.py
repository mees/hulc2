class BaseDetector:
    def __init__(self, *args, **kwargs):
        pass

    def cuda(self):
        pass

    def eval(self):
        pass

    def load_from_checkpoint(self, *args, **kwargs):
        pass

    def predict(self, inputs: dict):
        pass