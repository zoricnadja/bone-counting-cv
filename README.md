# Bone Counting

This project detects and counts picked-up bones in gameplay videos using classical computer vision techniques.
It was developed as part of the Soft Computing course at the Faculty of Technical Sciences.
Each time a bone is collected, a circular visual effect appears around the player’s body. The algorithm detects this circle and counts each appearance as one bone pickup.

---

## Project Description

- **Input:** Gameplay videos (`.mp4`)
- **Output:** Number of picked-up bones per video
- **Ground truth:** Provided in `count.csv`
- **Evaluation metric:** Mean Absolute Error (MAE)

The same algorithm and parameters are applied to all videos.

---

## Method Overview

The solution is based on HSV color segmentation and circle detection.

### Processing pipeline:
1. Load video using OpenCV
2. Crop the lower half, central region of each frame
3. Convert cropped frame to HSV color space
4. Threshold the image to isolate the gold pickup effect
5. Apply median blur to reduce noise
6. Detect circles using the Hough Circle Transform

---

## Technologies Used

- Python 3
- OpenCV (cv2)
- NumPy
- Pandas
- Matplotlib

---

## How to Run

1. Place all videos and `count.csv` inside the `data/` directory.
2. Run the script:

```bash
python main.py


