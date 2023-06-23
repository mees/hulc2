import matplotlib.pyplot as plt
import numpy as np


class BaseDetector:
    def __init__(self, cfg, *args, **kwargs):
        self.n_classes = 1
        cm = plt.get_cmap("jet")
        self._colors = cm(np.linspace(0, 1, self.n_classes))
        self.clusters = {}

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, value):
        self._colors = value

    def predict(self, new_point):
        return 0
