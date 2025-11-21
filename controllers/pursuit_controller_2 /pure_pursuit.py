import math

class PurePursuit:
    def __init__(self, L_d):
        self.L_d = L_d

    def _find_goal_point(self, path_points):
        best = None
        best_diff = float('inf')
        for x, y in path_points:
            d = math.hypot(x, y)
            diff = abs(d - self.L_d)
            if diff < best_diff:
                best_diff = diff
                best = (x, y)
        return best

    def update(self, v, path_points):
        gp = self._find_goal_point(path_points)
        if gp is None:
            return 0.0
        xg, yg = gp
        curvature = (2 * yg) / (self.L_d**2)
        return v * curvature
