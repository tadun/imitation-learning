import math

class PurePursuit:
    def __init__(self, L_d):
        #lookahead distance
        self.L_d = L_d
    
    def _find_goal_point(self, path_points):        
        #find point closest to lookahead distance
        best_point = None
        best_diff = float('inf')

        for point in path_points:
            x, y = point
            #calculate distance to point
            dist = math.sqrt(x**2 + y**2)
            
            #Find point with distance closest to L_d
            diff = abs(dist - self.L_d)
            if diff < best_diff:
                best_diff = diff
                best_point = point
        
        return best_point

    def update(self, v, path_points):
        #find target point
        goal_point = self._find_goal_point(path_points)
        
        #safety check - return 0 angular velocity
        if goal_point is None:
            return 0.0  
            
        x_goal, y_goal = goal_point
        
        #calculate the curvature (kappa) required to reach target point (standard formula)
        #y_goal is lateral deviation to the goal point.
        curvature = (2 * y_goal) / (self.L_d**2)
        
        #calculate w
        w_desired = v * curvature
        
        return w_desired