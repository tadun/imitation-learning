import math

def calculate_steering(v, path_points, L_d):
    #find point closest to lookahead distance
    best_point = None
    best_diff = float('inf')

    for point in path_points:
        x, y = point
        #calculate distance to point
        dist = math.hypot(x, y)
        
        #Find point with distance closest to L_d
        diff = abs(dist - L_d)
        if diff < best_diff:
            best_diff = diff
            best_point = point
    
    #safety check - return 0 angular velocity
    if best_point is None:
        return 0.0  
        
    x_goal, y_goal = best_point
    
    #calculate the curvature (kappa) required to reach target point (standard formula)
    #y_goal is lateral deviation to the goal point.
    curvature = (2 * y_goal) / (L_d**2)
    
    #calculate w
    w_desired = v * curvature
    
    return w_desired