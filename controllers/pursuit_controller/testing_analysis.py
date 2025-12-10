import numpy as np
import pandas as pd
import math
from dtw import dtw # library for Dynamic Time Warping
from scipy.spatial.distance import cdist # for fast distance calculations

TARGET_DISTANCE = 0.5 # target following distance from pursuit_controller.py
LATERAL_TOLERANCE_M = 0.05 # allowable error for convergence check


def load_data(log_file_path):
    # using pandas makes working with columns easy
    df = pd.read_csv(log_file_path)
    return df

def calculate_lateral_deviation(df):
    # 1. create coordinate arrays
    teacher_path = df[['t_x', 't_y']].values
    student_path = df[['s_x', 's_y']].values
    
    lateral_errors = []
    
    # iterate through every student pose
    for i in range(len(student_path)):
        s_point = student_path[i]
        
        # calculate euclidean distance from the current student point
        # to ALL points on the teacher's path (cdist is fast)
        distances = cdist([s_point], teacher_path, 'euclidean')
        
        # the lateral error at this step is the shortest distance to the teacher's path
        min_dist_to_path = np.min(distances)
        
        # subtract the target following distance to get the tracking error
        lateral_error = abs(min_dist_to_path - TARGET_DISTANCE)
        lateral_errors.append(lateral_error)
        
    return np.mean(lateral_errors)

def calculate_dtw(df):
    #create coordinate arrays for comparison (Shape: N rows, 2 columns)
    #.T is removed to ensure the shape is (N, 2)
    teacher_path = df[['t_x', 't_y']].values
    student_path = df[['s_x', 's_y']].values
    
    #run DTW calculation
    #pass the (N, 2) arrays directly. NO more .T needed here.
    alignment = dtw(student_path, teacher_path, keep_internals=True)
    
    #Access the confirmed attribute
    return alignment.normalizedDistance

def calculate_heading_error(df):
    #calculate angular difference (t_yaw - s_yaw)
    diff = df['t_yaw'].values - df['s_yaw'].values
    
    # normalize angles to the range (-pi, pi) before taking the mean absolute value
    diff_norm = np.arctan2(np.sin(diff), np.cos(diff))
    
    return np.mean(np.abs(diff_norm))

def calculate_effort(df):

    #Effort = sum(v^2 + w^2)
    effort = (df['v_cmd']**2 + df['w_cmd']**2).sum()
    
    return effort




def run_analysis(log_file_path):
    try:
        data = load_data(log_file_path)
    except FileNotFoundError:
        print(f"error: log file '{log_file_path}' not found. please run the webots simulation first.")
        return
        
    print(f"analyzing {log_file_path} with {len(data)} steps...")
    
    mean_lat_dev = calculate_lateral_deviation(data)
    dtw_dist = calculate_dtw(data)
    heading_error = calculate_heading_error(data)
    total_effort = calculate_effort(data)
    

    print("\n--- EVALUATION RESULTS ---")
    print(f"Mean Lateral Deviation: {mean_lat_dev:.3f} m")
    print(f"Mean Absolute Heading Error: {heading_error:.3f} rad")
    print(f"Dynamic Time Warping Distance: {dtw_dist:.3f}")
    print(f"Total Control Effort (Sum V^2 + W^2): {total_effort:.3f}")


if __name__ == "__main__":
    # assumes the simulation creates this file in the current directory
    run_analysis("test_log.csv")