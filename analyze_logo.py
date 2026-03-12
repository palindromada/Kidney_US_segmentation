import cv2
import numpy as np

img = cv2.imread('/Users/ada/Desktop/Segmentazione rene dalle ultrasound /kidneyUS-main/kidneyUSlogo.png')
h, w = img.shape[:2]

print(f"Dimensioni logo: {w}x{h}")

# Cerca i colori più saturi (colorati, non grigi)
roi = img[h//4:3*h//4, w//4:3*w//4]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Pixel con saturazione significativa
saturated = roi_hsv[:,:,1] > 40
print(f"Pixel colorati: {np.sum(saturated)}")

if np.sum(saturated) > 0:
    saturated_pixels = roi[saturated]
    # Raggruppa per colore
    from collections import Counter
    colors = [(int(p[2])//20*20, int(p[1])//20*20, int(p[0])//20*20) for p in saturated_pixels]
    counter = Counter(colors)
    
    print("\nColori principali (RGB raggruppati):")
    for rgb, count in counter.most_common(10):
        print(f"  RGB{rgb}: {count} pixels")
