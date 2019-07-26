import cv2
import numpy as np
import json

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from scipy.ndimage import label


RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)


def segment_on_dt(a, img, border=None):
    """
    https://stackoverflow.com/questions/11294859/how-to-define-the-markers-for-watershed-in-opencv
    """
    if border is None:
        border = cv2.dilate(img, None, iterations=5)
        border = border - cv2.erode(border, None)
    # else:
    #     border = cv2.dilate(border, None, iterations=2)
    cv2.imshow("border", border)
    
    dt = img - cv2.dilate(border, None, iterations=3)
    _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)

    cv2.imshow("dt", dt)
    cv2.waitKey(0)

    lbl, ncc = label(dt)
    print(ncc)
    lbl = lbl * (255 / (ncc + 1))
    # Completing the markers now. 
    lbl[border == 255] = 255

    lbl = lbl.astype(np.int32)
    cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    return 255 - lbl

# LOAD IMG AND ANNOTATION
img_file = "./data/5.jpg"
annot_file = "./data/annot_5.json"
with open(annot_file, 'r') as f:
    ann = json.load(f)

annots = ann['annotations']
polygons = [s for a in annots for s in a['segmentation']] # get the first sample annot
polygons = [np.array(p).reshape(-1, 2) for p in polygons]

# READ IMAGE
img = cv2.imread(img_file)
image = img.copy()
mask = np.zeros(img.shape[:2], dtype=np.uint8)
out = cv2.fillPoly(img.copy(), polygons, RED)
m_out = cv2.fillPoly(mask.copy(), polygons, 255)
edge_out = cv2.drawContours(mask.copy(), polygons, -1, 255)

img2 = img.copy()
for poly in polygons:
    n = len(poly)
    for i in range(n):
        img2 = cv2.line(img2, tuple(poly[i]), tuple(poly[(i+1)%n]), GREEN)
cv2.imshow("out", out)
cv2.imshow("polygons", img2)
cv2.imshow("edge", edge_out)
cv2.imshow("mout", m_out)

# edge_prob = edge_out.astype(float) / 255
# edge_pixels = edge_out == 255
# edge_prob[edge_pixels] = np.random.uniform(0.6, 0.99, size=edge_pixels.sum())
thresh = edge_out

result = segment_on_dt(img, 255 - edge_out, edge_out)

cv2.imshow("result", result)

result[result != 255] = 0
result = cv2.dilate(result, None)
img[result == 255] = (0, 0, 255)

cv2.imshow("img", img)

cv2.waitKey(0)
