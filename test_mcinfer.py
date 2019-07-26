import cv2
import numpy as np
import json

import MCInference

RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)

def logit(p):
    eps = 1e-4
    clip_p = np.clip(p, eps, 1.0 - eps)
    return np.log((1.0-clip_p)/ clip_p)


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
mask = np.zeros(img.shape[:2], dtype=np.uint8)
out = cv2.fillPoly(img.copy(), polygons, RED)
m_out = cv2.fillPoly(mask.copy(), polygons, 255)
edge_out = cv2.drawContours(mask.copy(), polygons, -1, 255)

# Resize
scale = 1.0
scaled_polygons = [p*scale for p in polygons]
mm = cv2.resize(m_out, None, fx=scale, fy=scale)
mm_prob = mm / 255 

# visualize
img2 = img.copy()
for poly in polygons:
    n = len(poly)
    for i in range(n):
        img2 = cv2.line(img2, tuple(poly[i]), tuple(poly[(i+1)%n]), GREEN)
cv2.imshow("out", out)
cv2.imshow("polygons", img2)
cv2.imshow("mout", (mm_prob > 0.5).astype(float))
cv2.waitKey(0)

# generate edge vertex pairs from polygons as 1-D 
h, w = mm.shape[:2]
edges_flattened_px = np.empty((0, 2), dtype=np.int32)
for poly in scaled_polygons:
    n = len(poly)
    poly_rounded = np.round(poly).astype(np.int32)
    p_idx = [px[1] * w + px[0] for px in poly_rounded]
    edges_idx = [[p_idx[i], p_idx[(i+1)%n]] for i in range(n)]
    edges_flattened_px = np.vstack((edges_flattened_px, edges_idx))
total_edges = len(edges_flattened_px)

# generate edges and edge probabilities
edge_prob = np.random.uniform(0.6, 0.99, size=total_edges)
edge_costs = logit(edge_prob)
general_edge_costs = np.hstack((edges_flattened_px, edge_costs[:, None]))


# generate fg and bg unaries as 1-D
num_bg_classes = 1
unaries_fg = np.zeros(h*w, dtype=np.float64)
unaries_bg = np.zeros_like(unaries_fg)
bg_pixels = (mm_prob < 0.5).flatten()
fg_pixels = ~bg_pixels
total_bg = np.sum(bg_pixels)
total_fg = h * w - total_bg

# add noise to fg and bg unaries
fg_px_prob = np.random.uniform(0.6, 0.99, size=total_fg)
bg_px_prob = np.random.uniform(0.6, 0.99, size=total_bg)
unaries_fg[fg_pixels] = fg_px_prob
unaries_fg[bg_pixels] = 1.0 - bg_px_prob
unaries_bg[bg_pixels] = bg_px_prob
unaries_bg[fg_pixels] = 1.0 - fg_px_prob
unaries_bg = logit(unaries_bg)
unaries_fg = logit(unaries_fg)

unaries = np.vstack((unaries_bg, unaries_fg)).T # bg first


# Run MC
class_specific_edge_costs = np.array([[]], np.float64)
solution = np.zeros((h*w, 2), np.int32)

print("Running MC Inference...")
MCInference.infer(unaries.copy(), general_edge_costs.copy(), class_specific_edge_costs, num_bg_classes, solution)
print("Done")

# visualize solution fg pixels 
solution_mask = np.zeros(h*w).astype(np.uint8)
solution_labels = solution[:, 0]
fg_labels = solution_labels > 0
_fg_pixels = solution[:, 1][fg_labels]
solution_mask[_fg_pixels] = 255

final_mask = solution_mask.reshape((h, w))
cv2.imshow("solution", final_mask)
cv2.imshow("gt", mm)
cv2.waitKey(0)
