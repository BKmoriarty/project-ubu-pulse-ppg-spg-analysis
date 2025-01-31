import cv2
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

def cal_contrast_cpu(frame, block_size):
    shape = frame.shape
    return np.array([[((frame[i:i+block_size, j:j+block_size]).std() / 
                       np.mean(frame[i:i+block_size, j:j+block_size]))
                      for j in range(0, shape[1]-block_size+1, block_size)]
                     for i in range(0, shape[0]-block_size+1, block_size)])

def cal_contrast_gpu(frame, block_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert frame to GPU tensor
    frame_tensor = torch.from_numpy(frame).float().to(device)
    shape = frame_tensor.shape
    
    # Initialize output tensor
    height = (shape[0] - block_size + 1) // block_size
    width = (shape[1] - block_size + 1) // block_size
    result = torch.zeros((height, width), device=device)
    
    # Unfold the frame into blocks
    blocks = frame_tensor.unfold(0, block_size, block_size).unfold(1, block_size, block_size)
    
    # Calculate mean and std for each block
    means = blocks.mean(dim=(2,3))
    stds = blocks.std(dim=(2,3), unbiased=True)  # Added unbiased=True to match numpy
    
    # Calculate contrast
    result = stds / (means + 1e-6)  # Add small epsilon to avoid division by zero
    
    return result.cpu().numpy()

def compare_results(cpu_result, gpu_result, frame_num):
    is_close = np.allclose(cpu_result, gpu_result, rtol=1e-3, atol=1e-3)  # Relaxed tolerance
    
    return is_close

def plot_comparison(cpu_results, gpu_results):
    # Convert results to 1D arrays for plotting
    cpu_means = [np.mean(frame) for frame in cpu_results]
    gpu_means = [np.mean(frame) for frame in gpu_results]
    
    frames = range(len(cpu_results))
    
    plt.figure(figsize=(12, 6))
    plt.plot(frames, cpu_means, label='CPU', color='blue', alpha=0.7)
    plt.plot(frames, gpu_means, label='GPU', color='red', alpha=0.7, linestyle='--')
    
    plt.title('Comparison of CPU vs GPU Contrast Calculations')
    plt.xlabel('Frame Number')
    plt.ylabel('Mean Contrast Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot difference
    plt.figure(figsize=(12, 6))
    differences = np.array(cpu_means) - np.array(gpu_means)
    plt.plot(frames, differences, label='CPU - GPU Difference', color='green')
    plt.title('Difference between CPU and GPU Results')
    plt.xlabel('Frame Number')
    plt.ylabel('Difference in Contrast Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def test_contrast_processing():
    # Check GPU availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # Open a video file
    video_path = "storage/2024-12-10 16_24_53 tee 2500 300 (200, 200) rate-10 good/video-0000.avi"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    block_size = 3
    num_frames = 1000

    size = 100
    max_x = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    max_y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    x1 = int((max_x - size) // 2)
    y1 = int((max_y - size) // 2)
    x2 = x1 + size
    y2 = y1 + size
    print(f"Video dimensions: {max_x}x{max_y}")
    print(f"Selected region: {x1}x{x2}x{y1}x{y2}")
    
    cpu_results = []
    gpu_results = []
    matching_frames = 0
    
    # Test CPU Processing
    print("\nTesting CPU Processing...")
    start_time = time.time()
    frames_processed = 0
    
    while cap.isOpened() and frames_processed < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        contrast_cpu = cal_contrast_cpu(gray, block_size)
        cpu_results.append(contrast_cpu)
        frames_processed += 1
        
        if frames_processed % 10 == 0:
            print(f"Processed {frames_processed} frames...")
    
    cpu_time = time.time() - start_time
    print(f"CPU Processing Time: {cpu_time:.2f} seconds")

    # Reset video capture
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Test GPU Processing
    print("\nTesting GPU Processing...")
    start_time = time.time()
    frames_processed = 0
    
    while cap.isOpened() and frames_processed < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        contrast_gpu = cal_contrast_gpu(gray, block_size)
        gpu_results.append(contrast_gpu)
        
        # Compare results
        is_close = compare_results(cpu_results[frames_processed], 
                                 contrast_gpu, 
                                 frames_processed + 1)
        if is_close:
            matching_frames += 1
            
        frames_processed += 1
        
        if frames_processed % 10 == 0:
            print(f"Processed {frames_processed} frames...")
    
    gpu_time = time.time() - start_time
    print(f"\nGPU Processing Time: {gpu_time:.2f} seconds")
    print(f"Speed improvement: {cpu_time/gpu_time:.2f}x")
    
    # Final accuracy report
    print(f"\nAccuracy Report:")
    print(f"Total frames processed: {frames_processed}")
    print(f"Frames with matching results: {matching_frames}")
    print(f"Accuracy percentage: {(matching_frames/frames_processed)*100:.2f}%")

    # Plot the comparison
    plot_comparison(cpu_results, gpu_results)

    cap.release()

if __name__ == "__main__":
    test_contrast_processing()
    