import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def load_expected_results():
    df = pd.read_csv(os.path.join(DATA_DIR, "count.csv"))
    expected_results = {}
    for line in df.itertuples():
        video = line.video
        count = line.count 
        expected_results[video] = count
    return expected_results


def preprocess_video(video_name, results):
    video_path = os.path.join(DATA_DIR, video_name + ".mp4")

    cap = cv2.VideoCapture(video_path)
    num_bones = 0
    is_circle_active = False
    num_skipped_frames = 0
    
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        crop_frame = frame[int(0.5*height):int(height), int(0.25*width):int(0.75*width)]

        hsv = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2HSV)
        
        lower_mask = np.array([10, 0, 180])
        upper_mask = np.array([30, 255, 255])
        
        mask = cv2.inRange(hsv, lower_mask, upper_mask)
        
        mask = cv2.medianBlur(mask, 5)
        circles = cv2.HoughCircles(
            mask, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=50,
            param1=50, 
            param2=15, 
            minRadius=20, 
            maxRadius=100
        )
        if circles is not None:
            num_skipped_frames = 0
            if not is_circle_active:
                num_bones += 1
                is_circle_active = True
        else:
            num_skipped_frames += 1
            if num_skipped_frames >= 5:
                is_circle_active = False

    cap.release()
    results[video_name] = num_bones

if __name__ == "__main__":
    expected_results = load_expected_results()
    data_directory = os.path.dirname(os.path.abspath(__file__))
    results = {}
    for video in expected_results.keys():
        preprocess_video(video, results)

    mae = np.mean([np.abs(results[video] - expected_results[video]) for video in expected_results.keys()])
    print(mae)