import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('video-aj.avi')

# Get the video dimensions
height, width, _ = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_COUNT)

# Create a mask to track optical flow
prev_gray = None
mask = np.zeros_like(cap.read()[1])
mask[..., 1] = 255

# Keep track of the ROI coordinates
ipPG_roi = None
bcg_roi = None

# Loop through the video frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Visualize the optical flow
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask[..., 0] = ang * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.bitwise_and(frame, mask)

        # Identify ROIs based on motion
        high_motion_roi = np.where(mag > 10, 1, 0)  # BCG ROI
        low_motion_roi = np.where(mag <= 10, 1, 0)  # iPPG ROI

        # Find the contours of the ROIs
        contours_high, _ = cv2.findContours(high_motion_roi.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_low, _ = cv2.findContours(low_motion_roi.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Get the bounding boxes of the ROIs
        if contours_high:
            x, y, w, h = cv2.boundingRect(contours_high[0])
            bcg_roi = (x, y, x+w, y+h)
            cv2.rectangle(gray, (bcg_roi[0], bcg_roi[1]), (bcg_roi[2], bcg_roi[3]), (0, 0, 255), 2)

        if contours_low:
            x, y, w, h = cv2.boundingRect(contours_low[0])
            ipPG_roi = (x, y, x+w, y+h)
            cv2.rectangle(gray, (ipPG_roi[0], ipPG_roi[1]), (ipPG_roi[2], ipPG_roi[3]), (0, 255, 0), 2)
        
        # Display the results
        cv2.imshow('Optical Flow', gray)

    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("iPPG ROI:", ipPG_roi)
print("BCG ROI:", bcg_roi)

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()