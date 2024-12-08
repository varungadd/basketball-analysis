# app/video_analysis.py

import cv2
import mediapipe as mp
import numpy as np
import os
from flask import current_app as app
from .models import Video
from . import db
import pandas as pd

# Helper function to calculate angle
def calculate_angle(a, b, c):
    """
    Calculates the angle at point b given three points a, b, and c.
    Args:
        a, b, c: Tuple of (x, y) coordinates.
    Returns:
        angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Function to analyze video
def analyze_video(video_record):
    """
    Analyze the uploaded video and update the Video instance with results.
    """
    input_video_path = os.path.join(app.root_path, 'static', 'uploads', video_record.filename)
    input_video_name = os.path.splitext(video_record.filename)[0]
    output_filename = f"{input_video_name}_overall_analysis.avi"
    output_path = os.path.join(app.root_path, 'static', 'uploads', output_filename)
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize Video Capture
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize counters
    total_frames = 0
    marks = {
        "feet": 0,
        "elbow": 0,
        "jump": 0,
        "follow": 0,
        "trajectory": 0
    }
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract relevant landmarks
            left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
            right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # Calculate Feet Alignment and Distance
            feet_aligned, feet_distance, shoulder_width = calculate_feet_alignment_and_angle(
                left_foot, right_foot, left_shoulder, right_shoulder)
            
            # Calculate Elbow Alignment and Angle
            elbow_aligned, elbow_angle = calculate_elbow_alignment_and_angle(
                left_shoulder, left_elbow, left_wrist)
            
            # Analyze Jump Shot Mechanics
            jump_shot = analyze_jump_shot(left_ankle, right_ankle, left_knee, right_knee, left_hip, right_hip)
            
            # Check Follow Through
            follow_through = check_follow_through(left_wrist, right_wrist, left_elbow, right_elbow, left_shoulder, right_shoulder)
            
            # Evaluate Ball Trajectory
            trajectory = evaluate_ball_trajectory(frame)
            
            # Update Marks
            if feet_aligned:
                marks["feet"] += 1
            if elbow_aligned:
                marks["elbow"] += 1
            if jump_shot:
                marks["jump"] += 1
            if follow_through:
                marks["follow"] += 1
            if trajectory:
                marks["trajectory"] += 1
            
            total_frames += 1
            
            # Draw Landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Draw Summary
            draw_summary(frame, feet_aligned, feet_distance, shoulder_width, elbow_aligned, elbow_angle, jump_shot, follow_through, trajectory)
        
        # Write Frame to Output Video
        out.write(frame)
    
    # Release Resources
    cap.release()
    out.release()
    
    # Calculate Scores
    marks_out_of_5 = {
        "feet": 1 if marks["feet"] > 0 else 0,
        "elbow": 1 if marks["elbow"] > 0 else 0,
        "jump": 1 if marks["jump"] > 0 else 0,
        "follow": 1 if marks["follow"] > 0 else 0,
        "trajectory": 1 if marks["trajectory"] > 0 else 0,
    }
    
    total_marks = sum(marks_out_of_5.values())
    
    # Determine Areas to Improve
    areas_to_improve = []
    if marks_out_of_5["feet"] == 0:
        areas_to_improve.append("Feet Alignment")
    if marks_out_of_5["elbow"] == 0:
        areas_to_improve.append("Elbow Alignment")
    if marks_out_of_5["jump"] == 0:
        areas_to_improve.append("Jump Shot")
    if marks_out_of_5["follow"] == 0:
        areas_to_improve.append("Follow Through")
    if marks_out_of_5["trajectory"] == 0:
        areas_to_improve.append("Ball Trajectory")
    
    # Save Analysis Results to Video Record
    video_record.analyzed_filename = output_filename
    video_record.overall_marks = total_marks
    video_record.areas_to_improve = ", ".join(areas_to_improve)
    video_record.feet = marks_out_of_5["feet"]
    video_record.elbow = marks_out_of_5["elbow"]
    video_record.jump = marks_out_of_5["jump"]
    video_record.follow = marks_out_of_5["follow"]
    video_record.trajectory = marks_out_of_5["trajectory"]
    db.session.commit()
    
    # Optionally, append to CSV
    csv_path = os.path.join(app.root_path, '..', 'common_analysis_results.csv')
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
    else:
        existing_df = pd.DataFrame(columns=["Serial Number", "Video File Name", "Total Frames", 
                                           "Feet (Total)", "Elbow (Total)", "Jump (Total)", 
                                           "Follow (Total)", "Trajectory (Total)", 
                                           "Overall Marks (Out of 5)", "Areas to Improve"])
    
    current_serial = existing_df.shape[0] + 1
    summary_data = {
        "Serial Number": current_serial,
        "Video File Name": input_video_name,
        "Total Frames": total_frames,
        "Feet (Total)": marks["feet"],
        "Elbow (Total)": marks["elbow"],
        "Jump (Total)": marks["jump"],
        "Follow (Total)": marks["follow"],
        "Trajectory (Total)": marks["trajectory"],
        "Overall Marks (Out of 5)": total_marks,
        "Areas to Improve": ", ".join(areas_to_improve)
    }
    
    # Create a DataFrame from the summary_data dictionary
    summary_df = pd.DataFrame([summary_data])

    # Concatenate the existing DataFrame with the new summary DataFrame
    new_df = pd.concat([existing_df, summary_df], ignore_index=True)

    new_df.to_csv(csv_path, index=False)

# Function to calculate feet alignment and distance
def calculate_feet_alignment_and_angle(left_foot, right_foot, left_shoulder, right_shoulder):
    """
    Calculates feet alignment and the distance between feet.
    
    Returns:
        feet_aligned (bool): True if feet are aligned within a certain threshold.
        feet_distance (float): Distance between left and right foot.
        shoulder_width (float): Distance between left and right shoulder.
    """
    # Calculate normalized positions
    feet_distance = np.sqrt(
        (left_foot.x - right_foot.x) ** 2 +
        (left_foot.y - right_foot.y) ** 2
    )
    
    shoulder_width = np.sqrt(
        (left_shoulder.x - right_shoulder.x) ** 2 +
        (left_shoulder.y - right_shoulder.y) ** 2
    )
    
    alignment_threshold = 0.05  # Adjust based on normalized distances
    
    feet_aligned = feet_distance < (alignment_threshold * shoulder_width)
    
    return feet_aligned, feet_distance, shoulder_width

# Function to calculate elbow alignment and angle
def calculate_elbow_alignment_and_angle(left_shoulder, left_elbow, left_wrist):
    """
    Calculates elbow alignment and measures the elbow angle.
    
    Returns:
        elbow_aligned (bool): True if elbow is aligned within a certain threshold.
        elbow_angle (float): Angle at the elbow joint in degrees.
    """
    a = (left_shoulder.x, left_shoulder.y)
    b = (left_elbow.x, left_elbow.y)
    c = (left_wrist.x, left_wrist.y)
    
    angle = calculate_angle(a, b, c)
    
    ideal_min_angle = 45
    ideal_max_angle = 90
    
    elbow_aligned = ideal_min_angle <= angle <= ideal_max_angle
    
    return elbow_aligned, angle

# Function to analyze jump shot mechanics
def analyze_jump_shot(left_ankle, right_ankle, left_knee, right_knee, left_hip, right_hip):
    """
    Analyzes the jump mechanics of the player.
    
    Returns:
        jump_shot (bool): True if jump mechanics are proper.
    """
    # Calculate knee angles
    a_left = (left_hip.x, left_hip.y)
    b_left = (left_knee.x, left_knee.y)
    c_left = (left_ankle.x, left_ankle.y)
    
    a_right = (right_hip.x, right_hip.y)
    b_right = (right_knee.x, right_knee.y)
    c_right = (right_ankle.x, right_ankle.y)
    
    angle_left = calculate_angle(a_left, b_left, c_left)
    angle_right = calculate_angle(a_right, b_right, c_right)
    
    ideal_knee_min = 70
    ideal_knee_max = 110
    
    jump_shot = (ideal_knee_min <= angle_left <= ideal_knee_max) and \
                (ideal_knee_min <= angle_right <= ideal_knee_max)
    
    return jump_shot

# Function to check follow through
def check_follow_through(left_wrist, right_wrist, left_elbow, right_elbow, left_shoulder, right_shoulder):
    """
    Checks the follow-through mechanics of the player.
    
    Returns:
        follow_through (bool): True if follow-through is proper.
    """
    # Calculate wrist to elbow angles
    a_left = (left_wrist.x, left_wrist.y)
    b_left = (left_elbow.x, left_elbow.y)
    c_left = (left_shoulder.x, left_shoulder.y)
    
    a_right = (right_wrist.x, right_wrist.y)
    b_right = (right_elbow.x, right_elbow.y)
    c_right = (right_shoulder.x, right_shoulder.y)
    
    angle_left = calculate_angle(a_left, b_left, c_left)
    angle_right = calculate_angle(a_right, b_right, c_right)
    
    ideal_follow_min = 30
    ideal_follow_max = 60
    
    follow_through_left = ideal_follow_min <= angle_left <= ideal_follow_max
    follow_through_right = ideal_follow_min <= angle_right <= ideal_follow_max
    
    follow_through = follow_through_left and follow_through_right
    
    return follow_through

# Function to evaluate ball trajectory
def evaluate_ball_trajectory(frame):
    """
    Evaluates the ball trajectory in the current frame.
    
    Returns:
        trajectory_good (bool): True if ball trajectory is consistent.
    """
    # Simple color detection (assuming orange ball)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_color = np.array([5, 100, 100])
    upper_color = np.array([15, 255, 255])
    
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        
        if radius > 5:
            if y < frame.shape[0] / 2:  # Example condition
                return True
    
    return False

# Function to draw summary on the frame
def draw_summary(frame, feet_aligned, feet_distance, shoulder_width, elbow_aligned, elbow_angle, jump_shot, follow_through, trajectory):
    """
    Draws summary text on the frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # White color
    thickness = 2
    y0, dy = 30, 30
    
    summary_texts = [
        f"Feet Aligned: {'Yes' if feet_aligned else 'No'}",
        f"Feet Distance: {feet_distance:.2f}",
        f"Elbow Aligned: {'Yes' if elbow_aligned else 'No'} ({elbow_angle:.1f}°)",
        f"Jump Shot: {'Yes' if jump_shot else 'No'}",
        f"Follow Through: {'Yes' if follow_through else 'No'}",
        f"Ball Trajectory: {'Good' if trajectory else 'Bad'}"
    ]
    
    for i, text in enumerate(summary_texts):
        y = y0 + i * dy
        cv2.putText(frame, text, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)
