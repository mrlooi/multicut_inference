# import sys
import cv2
import numpy
np  =numpy
from scipy.ndimage import label

def segment_on_dt(a, img):
    border = cv2.dilate(img, None, iterations=5)
    border = border - cv2.erode(border, None)

    cv2.imshow("border", border)

    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
    _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)

    cv2.imshow("dt", dt)
    lbl, ncc = label(dt)
    print(ncc)
    lbl = lbl * (255 / (ncc + 1))
    # Completing the markers now. 
    lbl[border == 255] = 255

    lbl = lbl.astype(numpy.int32)
    cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(numpy.uint8)
    return 255 - lbl

img_file = "./data/7EU8I.jpg"
img = cv2.imread(img_file)

# Pre-processing.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
_, thresh = cv2.threshold(img_gray, 0, 255,
        cv2.THRESH_OTSU)

img_bin = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
        numpy.ones((3, 3), dtype=int))


cv2.imshow("thresh", thresh)
cv2.imshow("img_bin", img_bin)

result = segment_on_dt(img, img_bin)
# cv2.imwrite(sys.argv[2], result)

cv2.imshow("result", result)

result[result != 255] = 0
result = cv2.dilate(result, None)
img[result == 255] = (0, 0, 255)
# cv2.imwrite(sys.argv[3], img)

cv2.imshow("img", img)
cv2.waitKey(0)