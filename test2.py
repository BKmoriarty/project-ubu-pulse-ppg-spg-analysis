import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the video file
video_path = "storage/2024-12-20 14_56_55 tee 2000 300 (200, 200) rate-10/video-0000.avi"
cap = cv2.VideoCapture(video_path)

# Parameters for processing
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize storage for analysis
motion_magnitudes = np.zeros((frame_height, frame_width))
intensity_variations = np.zeros((frame_height, frame_width))

# Read the first frame
ret, prev_frame = cap.read()
if ret:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
else:
    print("Failed to read the video")
    cap.release()

# Optical flow and intensity variation calculations
while ret:
    ret, curr_frame = cap.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute optical flow for motion
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_magnitudes += magnitude  # Accumulate motion magnitudes
    
    # Compute intensity variation
    intensity_variations += np.abs(curr_gray - prev_gray)  # Accumulate intensity changes
    
    prev_gray = curr_gray

cap.release()

# Normalize results
motion_magnitudes /= frame_count
intensity_variations /= frame_count
motion_magnitudes_normalized = cv2.normalize(motion_magnitudes, None, 0, 255, cv2.NORM_MINMAX)
intensity_variations_normalized = cv2.normalize(intensity_variations, None, 0, 255, cv2.NORM_MINMAX)

# Segment regions
motion_threshold = 30  # Threshold for motion
low_motion_mask = motion_magnitudes_normalized < motion_threshold
light_absorption_mask = intensity_variations_normalized > motion_threshold

# Visualize results
plt.figure(figsize=(18, 10))

# Motion heatmap
plt.subplot(1, 3, 1)
plt.title("Motion Heatmap (BCG)")
plt.imshow(motion_magnitudes_normalized, cmap='hot')
plt.colorbar(label='Motion Magnitude')
plt.xlabel("X-axis (pixels)")
plt.ylabel("Y-axis (pixels)")

# Intensity variation heatmap
plt.subplot(1, 3, 2)
plt.title("Light Absorption Heatmap (iPPG)")
plt.imshow(intensity_variations_normalized, cmap='Blues')
plt.colorbar(label='Intensity Variation')
plt.xlabel("X-axis (pixels)")
plt.ylabel("Y-axis (pixels)")

# Combined segmentation
plt.subplot(1, 3, 3)
combined_segmentation = np.zeros_like(motion_magnitudes_normalized)
combined_segmentation[low_motion_mask] = 1  # iPPG areas
combined_segmentation[light_absorption_mask] = 2  # BCG areas
plt.title("Combined Segmentation")
plt.imshow(combined_segmentation, cmap='cool')
plt.colorbar(ticks=[0, 1, 2], label='Region Type (1: iPPG, 2: BCG)')
plt.xlabel("X-axis (pixels)")
plt.ylabel("Y-axis (pixels)")

plt.tight_layout()
plt.show()
