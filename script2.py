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
    # print(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    sum_of_bones = 0
    bones = []
    is_circle_active = False
    num_frames_without_circle = 0
    frame_num = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        crop_frame = frame[int(0.5*height):int(height), int(0.25*width):int(0.75*width)]

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = gray[int(0.5*height):int(height), int(0.25*width):int(0.75*width)]
        # gray = cv2.medianBlur(gray, 5)

        # # plt.imshow(gray, cmap='gray')
        # edges = cv2.Canny(gray, 60, 160)
        hsv = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2HSV)
        
        # Define range for golden/yellow color of the circle
        # Adjust these values based on the exact color
        lower_gold = np.array([10, 0, 180])
        upper_gold = np.array([30, 255, 255])
        
        # Create mask for golden color
        mask = cv2.inRange(hsv, lower_gold, upper_gold)
        
        # Apply some morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur to reduce noise
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        circles = cv2.HoughCircles(
            mask, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=50,
            param1=50, 
            param2=15, 
            minRadius=30, 
            maxRadius=100
        )
        # circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        #                            param1=160, param2=20, minRadius=20, maxRadius=100)
        display = crop_frame.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            num_frames_without_circle = 0
            if not is_circle_active:
                sum_of_bones += 1
                bones.append((circles[0, 0][2]))
                is_circle_active = True
                event_text = f"New circle detected! "
                color = (0, 255, 255)
                cv2.circle(display, (circles[0, 0][0], circles[0, 0][1]), circles[0, 0][2], color, 4)
                cv2.circle(display, (circles[0, 0][0], circles[0, 0][1]), 2, (0, 0, 255), 3)
                
            else:
                event_text = f"active"
                
                
        else:
            num_frames_without_circle += 1
            if num_frames_without_circle >= 3:
                is_circle_active = False
        cv2.putText(display, f"Count: {sum_of_bones}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, event_text if circles is not None else "No circle", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Mask", mask)
        cv2.imshow("Frame", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # print(bones)
    cap.release()
    cv2.destroyAllWindows()
    results[video_name] = sum_of_bones

if __name__ == "__main__":
    start_time = cv2.getTickCount()
    expected_results = load_expected_results()
    data_directory = os.path.dirname(os.path.abspath(__file__))
    results = {}
    for video in expected_results.keys():
        preprocess_video(video, results)

    mae = np.mean([np.abs(results[video] - expected_results[video]) for video in expected_results.keys()])
    print(f"Mean Absolute Error (MAE): {mae}")
    # preprocess_video("video1", results)
    print("Results:", results)
    cv2.destroyAllWindows()
    end_time = cv2.getTickCount()
    time = (end_time - start_time) / cv2.getTickFrequency()
    print(f"Total processing time: {time} seconds")