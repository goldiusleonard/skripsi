from PIL import Image
from skimage import img_as_ubyte
import numpy as np
import cv2

class wbKelvin:
    def __init__(self, temp) -> None:
        self.temp = temp
        self.kelvin_table = {
            1000: (255,56,0),
            1500: (255,109,0),
            2000: (255,137,18),
            2500: (255,161,72),
            3000: (255,180,107),
            3500: (255,196,137),
            4000: (255,209,163),
            4500: (255,219,186),
            5000: (255,228,206),
            5500: (255,236,224),
            6000: (255,243,239),
            6500: (255,249,253),
            7000: (245,243,255),
            7500: (235,238,255),
            8000: (227,233,255),
            8500: (220,229,255),
            9000: (214,225,255),
            9500: (208,222,255),
            10000: (204,219,255)}
    
    def __call__(self, x):
        if not isinstance(x, Image.Image):
            raise RuntimeError("Input is not PIL Image!")
        x = x.convert("RGB")
        r, g, b = self.kelvin_table[self.temp]
        matrix = ( r / 255.0, 0.0, 0.0, 0.0,
               0.0, g / 255.0, 0.0, 0.0,
               0.0, 0.0, b / 255.0, 0.0 )

        return x.convert('RGB', matrix)

class wbPercentile:
    def __init__(self, percentile_value) -> None:
        self.percentile_value = percentile_value
    
    def __call__(self, x):
        if isinstance(x, Image.Image):
            x = np.asarray(x, dtype=np.uint8)
        else:
            raise RuntimeError("Input is not PIL Image!")
        
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        
        x = img_as_ubyte(
            (x*1.0 / np.percentile(x, 
             self.percentile_value, axis=(0, 1))).clip(0, 1))
        
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        return Image.fromarray(x)

class wbMeanCorrection:
    def __init__(self) -> None:
        pass
    
    def __call__(self, x):
        if isinstance(x, Image.Image):
            x = np.asarray(x, dtype=np.uint8)
        else:
            raise RuntimeError("Input is not PIL Image!")
        
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        
        B, G, R = cv2.split(x)
        B_avg = cv2.mean(B)[0]
        G_avg = cv2.mean(G)[0]
        R_avg = cv2.mean(R)[0]

        k = (B_avg + G_avg + R_avg) / 3
        
        if B_avg == 0:
            kb = 0
        else:
            kb = k / B_avg
        
        if G_avg == 0:
            kg = 0
        else:
            kg = k / G_avg
        
        if R_avg == 0:
            kr = 0
        else:    
            kr = k / R_avg

        B = cv2.addWeighted(src1=B, alpha=kb, src2=0, beta=0, gamma=0)
        G = cv2.addWeighted(src1=G, alpha=kg, src2=0, beta=0, gamma=0)
        R = cv2.addWeighted(src1=R, alpha=kr, src2=0, beta=0, gamma=0)

        x = cv2.merge([B,G,R])
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        return Image.fromarray(x)