import numpy as np
import cv2
from monai.transforms import MapTransform

class HistogramEqualizationd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            # Předpoklad: img je 2D numpy array (např. [H, W])
            if img.ndim == 2:
                d[key] = cv2.equalizeHist(img.astype(np.uint8))
            else:
                # Použít adaptivní CLAHE na každém kanálu zvlášť
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                d[key] = np.stack([clahe.apply(img[i].astype(np.uint8)) for i in range(img.shape[0])])
        return d
