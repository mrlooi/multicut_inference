import numpy as np

import MCInference


def logit(p):
    eps = 1e-4
    clip_p = np.clip(p, eps, 1.0 - eps)
    return np.log((1.0-clip_p)/ clip_p)


unaries = np.array([[0.0, 0.0, 0.8],
					[0.1, 0.0, 0.9],
					[0.7, 0.2, 0.2],
					[0.7, 0.6, 0.3]],
					np.float64)

unaries = logit(unaries)

general_edge_costs = np.array([	[0, 1, logit(0.8)],
						[0, 2, logit(0.2)],
						[1, 3, logit(0.4)],
						[2, 3, logit(0.1)] ],
						np.float64)

class_specific_edge_costs = np.array([[]], np.float64)

solution = np.zeros((4, 2), np.int32)

MCInference.infer(unaries, general_edge_costs, class_specific_edge_costs, 0, solution)

print(solution)
