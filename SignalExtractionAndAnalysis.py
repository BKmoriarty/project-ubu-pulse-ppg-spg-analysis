import cv2
from matplotlib.widgets import MultiCursor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, find_peaks
from time import process_time


class ImageCropper:
    def __init__(self, image):
        self.cropping = False
        self.x_start, self.y_start, self.x_end, self.y_end = 0, 0, 0, 0
        self.image = image
        self.oriImage = self.image.copy()

    def mouse_crop(self, event, x, y, flags, param):
        # grab references to the global variables
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x_start, self.y_start, self.x_end, self.y_end = x, y, x, y
            self.cropping = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cropping:
                self.x_end, self.y_end = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.x_end, self.y_end = x, y
            self.cropping = False
            refPoint = [(self.x_start, self.y_start), (self.x_end, self.y_end)]
            if len(refPoint) == 2:
                roi = self.oriImage[refPoint[0][1]:refPoint[1]
                                    [1], refPoint[0][0]:refPoint[1][0]]
                print('Size', self.x_end - self.x_start,
                      self.y_end - self.y_start)
                cv2.imshow("Cropped", roi)

    def start_cropping(self):
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.mouse_crop)
        while True:
            i = self.image.copy()
            if not self.cropping:
                cv2.imshow("image", self.image)
            elif self.cropping:
                cv2.rectangle(i, (self.x_start, self.y_start),
                              (self.x_end, self.y_end), (255, 0, 0), 2)
                cv2.imshow("image", i)
            # if enter is pressed, break from the loop
            # and pass values x_start, y_start, x_end, y_end to the main function
            if cv2.waitKey(1) & 0xFF == 13:
                break
        cv2.destroyAllWindows()
        return self.x_start, self.y_start, self.x_end, self.y_end


class VideoSaving:

    def init(self, video_path,  frame=[]):
        self.video_path = video_path

        self.frame = frame

    def add(self, frame):
        self.frame.append(frame)

    def save(self):
        height, width, = self.frame[0].shape
        size = (width, height)
        out = cv2.VideoWriter(
            self.video_path, cv2.VideoWriter_fourcc(*'XVID'), self.fps, size)
        for i in range(len(self.frame)):
            out.write(self.frame[i])
        out.release()

    def save_image(self, frame, name):
        cv2.imwrite(f'{name}.jpg', frame)

    def stop(self):
        cv2.destroyAllWindows()


class Analysis_PPG_SPG:

    def __init__(self, video_path, excel_path, size_ppg, size_spg, exposure_time=2200, fps=30, cache=False, cut_time_delay=0.2):
        self.size_ppg = size_ppg
        self.size_spg = size_spg
        self.video_path = video_path
        self.dir_path = video_path.split('/')[1]
        self.excel_path = excel_path
        self.size_block = 5
        self.exposure_time = exposure_time
        self.fps = fps
        self.cache = cache
        self.cut_time_delay = cut_time_delay

    def __del__(self):
        cv2.destroyAllWindows()
        plt.close('all')

    def detect_finger_center(self, frame):
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and improve contour detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thresholding to create a binary image (finger is assumed to be the brightest part)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If contours are detected, find the largest one (assuming it's the finger)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Calculate the center of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2

            return (center_x, center_y, w, h)
        else:
            return None

    def printProgressBar(self, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                         (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()

    def crop_center(self, frame, crop_width, crop_height):
        height, width, _ = frame.shape
        start_x = width // 2 - crop_width // 2
        start_y = height // 2 - crop_height // 2
        return start_x, start_y, crop_width, crop_height

    def get_center_crop_position(self, image_shape, crop_size):
        height, width = image_shape[:2]

        # Ensure crop size is not larger than the image
        crop_size = min(crop_size, min(height, width))

        # Calculate the center of the image
        center_x, center_y = width // 2, height // 2

        # Calculate the top-left and bottom-right corners of the square
        x1 = center_x - crop_size // 2
        y1 = center_y - crop_size // 2
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        return (x1, y1, x2, y2)

    def cal_contrast(self, frame):
        # Calculate the contrast of the frame
        shape = frame.shape

        return np.array([[((frame[i:i+self.size_block, j:j+self.size_block]).std() / np.mean(frame[i:i+self.size_block, j:j+self.size_block]))
                          for j in range(0, shape[1]-self.size_block+1, self.size_block)]
                         for i in range(0, shape[0]-self.size_block+1, self.size_block)])

    def extract_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)

        _, frame = cap.read()

        x1_ppg, y1_ppg, x2_ppg, y2_ppg = self.get_center_crop_position(
            frame.shape, self.size_ppg)
        x1_spg, y1_spg, x2_spg, y2_spg = self.get_center_crop_position(
            frame.shape, self.size_spg)

        color_ppg = (0, 255, 0)
        color_spg = (255, 0, 0)
        thickness = 2
        # cv2.rectangle(frame, (x1_ppg, y1_ppg),
        #               (x2_ppg, y2_ppg), color_ppg, thickness)
        # cv2.rectangle(frame, (x1_spg, y1_spg),
        #               (x2_spg, y2_spg), color_spg, thickness)
        # cv2.imshow('Image with Centered Square', frame)

        # fig, ax01 = plt.subplots(1, 1)
        # ax01.imshow(frame)
        # ax01.set_title('Raw Image')

        # fig, ax02 = plt.subplots(1, 1)
        # ax02.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        # ax02.set_title('Grayscale Image')

        # fig, ax03 = plt.subplots(1, 1)
        # ax03.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y1:y2, x1:x2])
        # ax03.set_title('Cropped Image')

        # frame_crop = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[
        #     y1_spg:y2_spg, x1_spg:x2_spg]

        # fig, ax04 = plt.subplots(1, 1)
        # mean_frame = np.mean(frame_crop)
        # # show the mean pixel intensity in the ROI
        # ax04.imshow(np.ones((frame_crop.shape[0], frame_crop.shape[1]))
        #             * mean_frame, cmap='gray')
        # ax04.set_title(f'Mean Frame: {mean_frame:.2f}')

        # fig, ax = plt.subplots(1, 1)
        # contrast = self.cal_contrast(frame_crop)
        # ax.imshow(contrast, cmap='hot')
        # ax.set_title('Contrast')

        return cap, [(x1_ppg, y1_ppg, x2_ppg, y2_ppg), (x1_spg, y1_spg, x2_spg, y2_spg)]

    def extract_signal(self, cap, position):

        signal_ppg = []
        mean_contrast_frame = []
        mean_exposure_frame = []
        # exposure_time = 1500  # us

        x1_ppg, y1_ppg, x2_ppg, y2_ppg = position[0]
        x1_spg, y1_spg, x2_spg, y2_spg = position[1]

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 1
        i = 0

        self.printProgressBar(0, length, prefix='Progress:',
                              suffix='Complete', length=50)
        start_process = process_time()

        # video = cv2.VideoWriter(
        #     "roi.avi", cv2.VideoWriter_fourcc(*'XVID'), self.fps, (self.size_ppg, self.size_ppg))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Extract the ROI from the grayscale frame
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_ppg = frame[y1_ppg:y2_ppg, x1_ppg:x2_ppg]
            roi_spg = frame[y1_spg:y2_spg, x1_spg:x2_spg]

            roi_ppg = cv2.cvtColor(roi_ppg, cv2.COLOR_BGR2GRAY)
            roi_spg = cv2.cvtColor(roi_spg, cv2.COLOR_BGR2GRAY)

            # Calculate the mean pixel intensity in the ROI
            signal_intensity = np.mean(roi_ppg)
            signal_ppg.append(signal_intensity)

            contrast = self.cal_contrast(roi_spg)
            mean_contrast_frame.append(np.mean(contrast))  # mean contrast

            # Calculate mean exposure using the formula: 1 / (2 * T * K^2)
            T = self.exposure_time
            K = contrast
            epsilon = 1e-10
            exposure = 1 / (2 * T * (np.square(K) + epsilon))
            mean_exposure = np.mean(exposure)
            mean_exposure_frame.append(mean_exposure)

            # video.write(frame)

            if (i == 0):
                fig, ax01 = plt.subplots(1, 1)
                ax01.imshow(frame[y1_ppg:y2_ppg, x1_ppg:x2_ppg])

                fig, ax02 = plt.subplots(1, 1)
                ax02.imshow(roi_ppg, cmap='hot', )

                fig, ax03 = plt.subplots(1, 1)
                ax03.imshow(signal_intensity * np.ones(
                    (roi_ppg.shape[0], roi_ppg.shape[1])), cmap='hot')

                fig, ax04 = plt.subplots(1, 1)
                ax04.imshow(contrast, cmap='hot', )

                fig, ax05 = plt.subplots(1, 1)
                ax05.imshow(exposure, cmap='hot', )

            end_process = process_time()
            i = i+1
            self.printProgressBar(i + 1, length, prefix='Progress:',
                                  suffix=f'time: {(end_process-start_process):.2f} seconds. Complete', length=50)

        # video.release()
        end_process = process_time()
        i = i+1
        self.printProgressBar(i + 1, length, prefix='Progress',
                              suffix=f'time: {(end_process-start_process):.2f} seconds. Complete', length=50)
        cap.release()

        if not signal_ppg:
            print("Warning: No signal values were extracted from the video.")
        else:
            print("Signal values extracted successfully.")

        return np.array(signal_ppg), np.array(mean_contrast_frame), np.array(mean_exposure_frame)

    def plot_signal(self, signal_ppg):
        plt.figure(figsize=(10, 4))
        plt.plot(signal_ppg, color='blue', label='Extracted Signal')
        plt.title('Extracted Heart Wave Signal')
        plt.xlabel('Frame')
        plt.ylabel('Intensity')
        # plt.legend()
        plt.show()

    def plot_spectrum(self, xf, yf):
        plt.figure(figsize=(10, 4))
        plt.plot(xf, yf, color='blue', label='Spectrum')
        plt.title('Frequency Spectrum of the Extracted Signal')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        # plt.legend()
        plt.show()

    def bandpass_filter(self, signal, lowcut, highcut, fs, order=3):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, signal)
        return y

    # Function to create a bandpass filter

    def lowpass_filter(self, signal, cutoff, fs, order=3):
        nyquist = 0.5 * fs
        normalized_cutoff = cutoff / nyquist
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.99  # Ensure it's below 1
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, signal)
        return y

    def highpass_filter(self, signal, cutoff, fs, order=3):
        nyquist = 0.5 * fs
        normalized_cutoff = cutoff / nyquist
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.99  # Ensure it's below 1
        b, a = butter(order, normalized_cutoff, btype='high', analog=False)
        y = filtfilt(b, a, signal)
        return y

    # Perform FFT on the signal

    def perform_fft(self, signal_ppg, frame_rate):
        N = len(signal_ppg)
        yf = fft(signal_ppg)
        xf = fftfreq(N, 1 / frame_rate)
        xf = xf[:N//2]
        yf = np.abs(yf[:N//2])
        return xf, yf

    def define_fft(self, data,):
        fft_data = np.fft.fft(data)
        fft_data = np.abs(fft_data)
        fft_data = fft_data / len(data)
        fft_data = fft_data[0:len(data)//2]
        return fft_data

    # Plot the signal (time domain)

    def plot_signal(self, signal_ppg, title="Signal"):
        plt.figure(figsize=(10, 4))
        plt.plot(signal_ppg, color='blue', label=title)
        plt.title(title)
        plt.xlabel('Frame')
        plt.ylabel('Intensity')
        # plt.legend()
        plt.show()

    # Plot the frequency spectrum (frequency domain)

    def plot_spectrum(self, xf, yf, title="Frequency Spectrum"):
        plt.figure(figsize=(10, 4))
        plt.plot(xf, yf, color='blue', label=title)
        plt.title(title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        # plt.legend()
        plt.show()

    def load_excel(self, file_name_excel):
       # Read the Excel file
        df = pd.read_excel(file_name_excel)
        # Convert the date column to datetime format if it's not already
        df['Time'] = pd.to_datetime(df['Data'], format='%H:%M:%S.%f')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

        # Define your start and end date
        # start_date = pd.to_datetime('17:41:48.318345', format='%H:%M:%S.%f')
        # end_date = pd.to_datetime('17:42:08.318345', format='%H:%M:%S.%f')

        end_time = df['Time'].max()
        start_time = end_time - pd.Timedelta(seconds=20)

        start_date = start_time
        end_date = end_time

        # Filter the data based on the date range
        filtered_df = df[(df['Time'] >= start_date) & (df['Time'] <= end_date)]

        time_excel = (filtered_df['Time'] - filtered_df['Time'].min()
                      ).dt.total_seconds()
        amplitude_excel = filtered_df['Value'].values

        return [time_excel, amplitude_excel]

    def filter_signal(self, signal, fps, highcut, lowcut):
        # filter the signal
        filtered_ppg = self.lowpass_filter(signal, lowcut, fps)
        filtered_ppg = self.highpass_filter(filtered_ppg, highcut, fps)
        filtered_signal_fft = self.perform_fft(filtered_ppg, fps)

        [b, a] = butter(4, 0.5, btype='highpass', fs=fps)
        filtered_ppg = filtfilt(b, a, filtered_ppg)

        return [filtered_ppg, filtered_signal_fft]

    def calculate_snr(self, data, sample_rate, signal_freq_range, noise_freq_ranges):
        # Perform FFT
        n = len(data)
        yf = fft(data)
        xf = fftfreq(n, 1/sample_rate)

        # Calculate power
        power = np.abs(yf)**2 / n

        # Find signal power
        signal_power = np.sum(
            power[(xf >= signal_freq_range[0]) & (xf <= signal_freq_range[1])])

        # Find noise power (sum of both noise ranges)
        noise_power = np.sum(
            power[(xf >= noise_freq_ranges[0][0]) & (xf <= noise_freq_ranges[0][1])])
        noise_power += np.sum(power[(xf >= noise_freq_ranges[1][0])
                                    & (xf <= noise_freq_ranges[1][1])])

        # Calculate SNR
        snr = 10 * np.log10(signal_power / noise_power)

        return snr, xf, power

    def snr_cal(self, fre_ppg, psd_ppg, lowCut_snr, highCut_snr):
        power_ppg = 0
        power_noise_ppg = 0
        power_ppg_count = 0
        power_noise_ppg_count = 0
        power_ppg_mean = 0
        power_noise_ppg_mean = 0

        for something in range(len(fre_ppg)):
            value_fre_ppg = fre_ppg[something]
            if (value_fre_ppg <= lowCut_snr) or (value_fre_ppg >= highCut_snr):
                power_noise_ppg = power_noise_ppg + psd_ppg[something]
                power_noise_ppg_count = power_noise_ppg_count+1
            else:
                power_ppg = power_ppg + psd_ppg[something]
                power_ppg_count = power_ppg_count+1

        power_ppg_mean = power_ppg/power_ppg_count
        power_noise_ppg_mean = power_noise_ppg/power_noise_ppg_count

        SNR_ppg = 10*np.log(power_ppg_mean/power_noise_ppg_mean)  # unit is dB

        return SNR_ppg

    def find_peak_freq(self, data, time):

        distance = self.fps * (50/88)

        peaks, _ = find_peaks(data, height=None,
                              threshold=None, distance=distance)

        # Calculate heart rate (if peaks represent heartbeats)
        time_diff = np.diff(time[peaks])
        heart_rate = 60 / np.mean(time_diff)  # unit is bpm

        return peaks, heart_rate

    def find_one_peaks(self, data):
        i_peaks, _ = find_peaks(data)

        i_max_peak = i_peaks[np.argmax(data[i_peaks])]

        return i_max_peak

    def find_peak_freq_excel(self, data, time):
        distance = self.fps * (60/88)
        # Find peaks
        # Adjust these parameters as needed for your specific data
        peaks, _ = find_peaks(data, height=None,
                              threshold=None, distance=distance)

        # Calculate heart rate (if peaks represent heartbeats)
        time_diff = np.diff(time[peaks])
        heart_rate = 60 / np.mean(time_diff)

        return peaks, heart_rate

    def main(self):
        if (self.cache):
            try:
                signal_ppg = np.load(
                    f'storage/{self.dir_path}/signal_ppg.npy')
                mean_exposure_frame = np.load(
                    f'storage/{self.dir_path}/mean_exposure_frame.npy')
                mean_contrast_frame = np.load(
                    f'storage/{self.dir_path}/mean_contrast_frame.npy')
            except:
                print("Error: Cache files not found. Extracting signal from video.")
                cap, position = self.extract_video_frames(self.video_path)
                signal_ppg, mean_contrast_frame, mean_exposure_frame = self.extract_signal(
                    cap, position)
                np.save(f'storage/{self.dir_path}/signal_ppg.npy', signal_ppg)
                np.save(
                    f'storage/{self.dir_path}/mean_exposure_frame.npy', mean_exposure_frame)
                np.save(
                    f'storage/{self.dir_path}/mean_contrast_frame.npy', mean_contrast_frame)
        else:
            cap, position = self.extract_video_frames(self.video_path)
            signal_ppg, mean_contrast_frame, mean_exposure_frame = self.extract_signal(
                cap, position
            )
            np.save(f'storage/{self.dir_path}/signal_ppg.npy', signal_ppg)
            np.save(
                f'storage/{self.dir_path}/mean_exposure_frame.npy', mean_exposure_frame)
            np.save(
                f'storage/{self.dir_path}/mean_contrast_frame.npy', mean_contrast_frame)

        # =========================== PPG ===========================
        signal_freq_range_ppg = (0.83, 4)  # Hz (typical range for heart rate)
        noise_freq_range_ppg = [
            (0, signal_freq_range_ppg[0]), (signal_freq_range_ppg[1], 15)]

        fps = self.fps
        signal_values_fft = self.perform_fft(signal_ppg, fps)

        fs = signal_ppg.size / 20
        time_ppg = np.arange(signal_ppg.size) / fs

        filtered_ppg, filtered_signal_fft1 = self.filter_signal(
            signal_ppg, fps,  signal_freq_range_ppg[0], signal_freq_range_ppg[1])

        # phase shift the signal 180 degree
        filtered_ppg_reverse = filtered_ppg  # * -1

        # 1/ln
        # filtered_ppg = np.log(filtered_ppg)

        peaks_filtered_signal, heart_rate_filtered_signal = self.find_peak_freq(
            filtered_ppg, time_ppg)

        max_filtered_ppg = self.find_one_peaks(filtered_signal_fft1[1])
        heart_rate_filtered_ppg = filtered_signal_fft1[0][max_filtered_ppg]*60
        print(
            f"Heart rate of the iPPG FFT: {heart_rate_filtered_ppg:.2f} bpm at {filtered_signal_fft1[0][max_filtered_ppg]} Hz")

        fig, (ax4, ax5, ax6) = plt.subplots(3, 1, figsize=(11, 5))

        ax4.plot(time_ppg, signal_ppg, color='b', label='iPPG Raw Signal')
        # ax4.legend()
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Amplitude')

        ax5.plot(
            signal_values_fft[0], signal_values_fft[1], color='b', label='iPPG Raw fft')
        ax5.plot(
            filtered_signal_fft1[0], filtered_signal_fft1[1], color='r', label='iPPG Filtered fft')
        ax5.axvspan(
            signal_freq_range_ppg[0], signal_freq_range_ppg[1], color='green', alpha=0.3)
        ax5.set_ylim([0, np.max(filtered_signal_fft1[1]) * 2])
        ax5.plot(filtered_signal_fft1[0][max_filtered_ppg], filtered_signal_fft1[1][max_filtered_ppg],
                 color='r', label="Peaks iPPG Filtered FFT", marker='o', linestyle='')
        # ax5.legend()
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Amplitude')

        ax6.plot(time_ppg, filtered_ppg, color='r',
                 label='iPPG Filtered Signal')
        ax6.plot(time_ppg[peaks_filtered_signal], filtered_ppg[peaks_filtered_signal],
                 color='r', label="Peaks iPPG", marker='o', linestyle='')
        # ax6.legend()
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Amplitude')

        fig.tight_layout()
        fig.savefig(f'storage/{self.dir_path}/iPPG.png')

        # =========================== SPG ===================================
        # Hz (typical range for heart rate)
        signal_freq_range_spg = (0.83, 4.5)
        noise_freq_range_spg = [
            (0, signal_freq_range_spg[0]), (signal_freq_range_spg[1], 15)]

        fps = self.fps
        time_spg = np.arange(mean_exposure_frame.size) / fps

        mean_exposure_frame_fft = self.perform_fft(mean_exposure_frame, fps)

        filtered_spg, filtered_spg_fft = self.filter_signal(
            mean_exposure_frame, fps,  signal_freq_range_spg[0], signal_freq_range_spg[1])

        peaks_filtered_spg, heart_rate_filtered_spg = self.find_peak_freq(
            filtered_spg, time_spg)

        max_filtered_spg = self.find_one_peaks(filtered_spg_fft[1])
        heart_rate_spg = filtered_spg_fft[0][max_filtered_spg]*60
        print(
            f"Heart rate of the SPG FFT: {heart_rate_spg:.2f} bpm at {filtered_spg_fft[0][max_filtered_spg]} Hz")

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 5))

        ax1.plot(time_spg, mean_exposure_frame,
                 color='b', label='SPG Raw Signal')
        # ax1.legend()
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')

        ax2.plot(
            mean_exposure_frame_fft[0], mean_exposure_frame_fft[1], color='b', label='SPG Raw fft')
        ax2.plot(
            filtered_spg_fft[0], filtered_spg_fft[1], color='r', label='SPG Filtered fft')
        ax2.set_ylim([0, np.max(filtered_spg_fft[1])*2])
        ax2.axvspan(
            signal_freq_range_spg[0], signal_freq_range_spg[1], color='green', alpha=0.3)
        ax2.plot(filtered_spg_fft[0][max_filtered_spg], filtered_spg_fft[1][max_filtered_spg],
                 color='r', label="Peaks SPG Filtered FFT", marker='o', linestyle='')
        # ax2.legend()
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')

        ax3.plot(time_spg, filtered_spg, color='r',
                 label='SPG Filtered Signal')
        ax3.plot(time_spg[peaks_filtered_spg], filtered_spg[peaks_filtered_spg],
                 color='r', label="Peaks SPG", marker='o', linestyle='')
        # ax3.legend()
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')

        fig.tight_layout()
        fig.savefig(f'storage/{self.dir_path}/SPG.png')

        # =========================== import excel ===========================
        excel_data = self.load_excel(self.excel_path)

        # apply filter
        signal_freq_range_excel = (0.83, 4)
        noise_freq_range_excel = [
            (0, signal_freq_range_excel[0]), (signal_freq_range_excel[1], 15)]
        fps_excel = 125  # sampling rate means the number of samples per second

        filtered_excel_fft = self.perform_fft(filtered_ppg, fps)

        [b, a] = butter(4, 0.5, btype='highpass', fs=fps_excel)
        filtered_excel = filtfilt(b, a, excel_data[1])

        # [filtered_excel, filtered_excel_fft]= self.filter_signal(
        #     excel_data[1], fps_excel, signal_freq_range_excel[0], signal_freq_range_excel[1])

        peaks_filtered_excel, heart_rate_filtered_excel = self.find_peak_freq_excel(
            filtered_excel, excel_data[0])

        max_filtered_excel = self.find_one_peaks(
            filtered_excel_fft[1])
        heart_rate_excel = filtered_excel_fft[0][max_filtered_excel]*60
        print(
            f"Heart rate of the cPPG FFT: {heart_rate_excel:.2f} bpm at {filtered_excel_fft[0][max_filtered_excel]} Hz")

        fig, (ax10, ax11, ax12) = plt.subplots(3, 1, figsize=(11, 5))

        ax10.plot(excel_data[0], excel_data[1],
                  color='b', label='cPPG Signal')
        # ax10.legend()
        ax10.set_xlabel('Time (s)')
        ax10.set_ylabel('Amplitude')

        ax11.plot(
            filtered_excel_fft[0], filtered_excel_fft[1], color='r', label='cPPG Filtered fft')
        ax11.set_ylim([0, np.max(filtered_excel_fft[1])*2])
        ax11.axvspan(
            signal_freq_range_excel[0], signal_freq_range_excel[1], color='green', alpha=0.3)
        ax11.plot(filtered_excel_fft[0][max_filtered_excel], filtered_excel_fft[1][max_filtered_excel],
                  color='r', label="Peaks cPPG Filtered FFT", marker='o', linestyle='')
        # ax11.legend()
        ax11.set_xlabel('Time (s)')
        ax11.set_ylabel('Amplitude')

        ax12.plot(excel_data[0], filtered_excel/np.max(filtered_excel), color='r',
                  label='cPPG Filtered Signal')
        ax12.plot(excel_data[0][peaks_filtered_excel], (filtered_excel/np.max(filtered_excel))[peaks_filtered_excel],
                  color='r', label="Peaks cPPG", marker='o', linestyle='')
        # ax12.legend()
        ax12.set_xlabel('Time (s)')
        ax12.set_ylabel('Amplitude')

        fig.tight_layout()
        fig.savefig(f'storage/{self.dir_path}/cPPG.png')

        # =========================== SNR ===========================

        # mean
        snr_signal_values, frequencies_signal_values, power_signal_values = self.calculate_snr(
            signal_ppg, fps, signal_freq_range_ppg, noise_freq_range_ppg)

        # ppg
        snr_filtered_ppg, frequencies_filtered_ppg, power_filtered_ppg = self.calculate_snr(
            filtered_ppg, fps, signal_freq_range_ppg, noise_freq_range_ppg)

        print(
            f"SNR of the iPPG values: {snr_signal_values:.2f} -> {snr_filtered_ppg:.2f} dB")

        # spg
        snr_spg, frequencies_spg, power_spg = self.calculate_snr(
            mean_exposure_frame, fps, signal_freq_range_spg, noise_freq_range_spg)

        snr_filtered_spg, frequencies_filtered_spg, power_filtered_spg = self.calculate_snr(
            filtered_spg, fps, signal_freq_range_spg, noise_freq_range_spg)

        print(
            f"SNR of the SPG values: {snr_spg:.2f} -> {snr_filtered_spg:.2f} dB")

        # excel
        snr_excel, frequencies_excel, power_excel = self.calculate_snr(
            excel_data[1], fps_excel, signal_freq_range_excel, noise_freq_range_excel)

        snr_filtered_excel, frequencies_excel, power_excel = self.calculate_snr(
            filtered_excel, fps_excel, signal_freq_range_excel, noise_freq_range_excel)

        print(
            f"SNR of the cPPG values: {snr_excel:.2f} -> {snr_filtered_excel:.2f} dB")

        # =========================== Time Delay ===========================
        time_plot = np.arange(mean_exposure_frame.size) / fps

        # remove the first peak
        peaks_filtered_signal = peaks_filtered_signal[1:]
        peaks_filtered_spg = peaks_filtered_spg[1:]

        # Convert peak indices to time points
        # not include the first peak
        ppg_peak_times = time_plot[peaks_filtered_signal]
        spg_peak_times = time_plot[peaks_filtered_spg]

        # print("\n".join(map(str, ppg_peak_times)))
        # print("\n".join(map(str, spg_peak_times)))

        delay_threshold = self.cut_time_delay  # seconds

        # Calculate time differences between closest corresponding peaks
        peak_delays = []
        valid_spg_peaks = []
        valid_ppg_peaks = []
        i, j = 0, 0
        while i < len(spg_peak_times) and j < len(ppg_peak_times):
            # Find the time difference for corresponding peaks
            delay = ppg_peak_times[j] - spg_peak_times[i]
            # if (delay < 0.5 and delay > -0.5):
            #     peak_delays.append(delay)

            # Only keep delays within the threshold
            if abs(delay) <= delay_threshold:
                # peak_delays.append(np.abs(delay))
                peak_delays.append(delay)
                # peak_delays.append(delay)
                valid_spg_peaks.append(peaks_filtered_spg[i])
                valid_ppg_peaks.append(peaks_filtered_signal[j])

                # Move to the next pair of peaks
                i += 1
                j += 1
            else:
                # If delay is too large, move the pointer that is behind
                if spg_peak_times[i] < ppg_peak_times[j]:
                    i += 1
                else:
                    j += 1

        diff = filtered_ppg/np.max(filtered_ppg) - \
            filtered_spg/np.max(filtered_spg)
        avg_diff = np.mean(diff)

        # Calculate the average time delay between SPG and PPG peaks
        avg_time_delay = np.mean(peak_delays) if peak_delays else np.nan
        print(np.max(peak_delays))

        print(
            f"Average time delay between SPG and PPG peaks: {avg_time_delay:.6f} seconds")
        # avg_time_delay, spg_peak_times, ppg_peak_times, peak_delays

        fig, (ax13, ax14, ax15) = plt.subplots(3, 1, figsize=(16, 9))  #

        ax13.plot(time_plot, filtered_ppg/np.max(filtered_ppg),
                  color='b', label='iPPG Signal')
        ax13.plot(time_plot[peaks_filtered_signal], (filtered_ppg/np.max(filtered_ppg))[peaks_filtered_signal],
                  color='b', label='Peaks iPPG', marker='o', linestyle='')
        ax13.plot(time_plot, filtered_spg/np.max(filtered_spg),
                  color='g', label="SPG Signal")
        ax13.plot(time_plot[peaks_filtered_spg], (filtered_spg/np.max(filtered_spg))[peaks_filtered_spg],
                  color='g', label="Peaks SPG", marker='o', linestyle='')
        # Plot vertical lines representing time delay between each corresponding peak
        colors = ['r', 'y', 'c', 'm']
        for i in range(min(len(valid_spg_peaks), len(valid_ppg_peaks))):
            color = colors[i % len(colors)]  # Cycle through colors
            # Plot vertical line connecting the peaks
            ax13.plot([time_plot[valid_spg_peaks[i]], time_plot[valid_ppg_peaks[i]]],
                      [(filtered_spg/np.max(filtered_spg))[valid_spg_peaks[i]],
                       (filtered_ppg/np.max(filtered_ppg))[valid_ppg_peaks[i]]],
                      f'{color}--', alpha=0.7)
            # Plot markers at the peaks
            ax13.plot(time_plot[valid_spg_peaks[i]], (filtered_spg/np.max(filtered_spg))[valid_spg_peaks[i]],
                      marker='o', color=color)
            ax13.plot(time_plot[valid_ppg_peaks[i]], (filtered_ppg/np.max(filtered_ppg))[valid_ppg_peaks[i]],
                      marker='o', color=color)
            # Plot lines on the y-axis 0 to 1 for each pair of peaks
            ax13.plot([time_plot[valid_spg_peaks[i]], time_plot[valid_ppg_peaks[i]]],
                      [0, 0], f'{color}-', alpha=0.5)
            ax13.plot([time_plot[valid_spg_peaks[i]], time_plot[valid_spg_peaks[i]]],
                      [0, 1], f'{color}-', alpha=0.5)
            ax13.plot([time_plot[valid_ppg_peaks[i]], time_plot[valid_ppg_peaks[i]]],
                      [0, 1], f'{color}-', alpha=0.5)
            ax13.plot([time_plot[valid_spg_peaks[i]], time_plot[valid_ppg_peaks[i]]],
                      [1, 1], f'{color}-', alpha=0.5)

        ax13.grid()
        ax13.legend()
        ax13.set_title('Time Delay Between PPG and SPG Peaks')
        ax13.set_xlabel('Time (s)')
        ax13.set_ylabel('Normalized Amplitude')

        # Plot time delay between valid peak pairs
        colors = ['r', 'y', 'c', 'm']
        for idx, delay in enumerate(peak_delays):
            color = colors[idx % len(colors)]
            ax14.plot(idx, delay, "o-", color=color)
        # ax14.axhline(y=avg_time_delay, color='gray', linestyle='--',
        #              label=f'Average Delay: {avg_time_delay:.4f} s')
        ax14.set_ylim([-0.2, 0.2])
        ax14.set_xlabel('Peak Pair Index')
        ax14.set_ylabel('Time Delay (s)')
        ax14.set_title('Time Delays Between Corresponding Peaks')
        ax14.grid()
        ax14.legend()

        # Plot difference between SPG and PPG signals
        ax15.plot(time_plot, diff, color='b',
                  label=f'Avg Diff: {avg_diff:.4f}')
        ax15.axhline(y=avg_diff, color='gray', linestyle='--')

        ax15.legend()
        ax15.grid()
        ax15.set_title('Difference Between SPG and PPG Signals')
        ax15.set_xlabel('Time (s)')
        ax15.set_ylabel('Amplitude Difference')

        # save figure to file
        fig.tight_layout()
        fig.savefig(f'storage/{self.dir_path}/time_delay.png')

        # =========================== Plotting ===========================

        fig, (ax7, ax8, ax9) = plt.subplots(3, 1, figsize=(16, 9))  #
        ax7.plot(time_plot, filtered_ppg/np.max(filtered_ppg),
                 color='b', label='iPPG Signal')
        ax7.plot(time_plot[peaks_filtered_signal], (filtered_ppg/np.max(filtered_ppg))[peaks_filtered_signal],
                 color='b', label='Peaks iPPG', marker='o', linestyle='')
        ax7.plot(excel_data[0], filtered_excel/np.max(filtered_excel),
                 color='r', label='cPPG Signal', linestyle='--')
        ax7.plot(time_plot, filtered_spg/np.max(filtered_spg),
                 color='g', label="SPG Signal")
        ax7.plot(time_plot[peaks_filtered_spg], (filtered_spg/np.max(filtered_spg))[peaks_filtered_spg],
                 color='g', label="Peaks SPG", marker='o', linestyle='')
        ax7.grid()
        ax7.legend()
        ax7.set_title('Integrate Signal')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Amplitude')

        offset_ppg = np.mean((filtered_ppg/np.max(filtered_ppg))
                             [peaks_filtered_signal])

        offset_excel = np.mean((filtered_excel/np.max(filtered_excel))
                               [peaks_filtered_excel])

        offset_ppg_excel = 1 + np.abs(offset_ppg - offset_excel)

        ax8.plot(time_plot, (filtered_ppg/np.max(filtered_ppg)) * offset_ppg_excel,
                 color='b', label='iPPG Signal')
        ax8.plot(time_plot[peaks_filtered_signal], (filtered_ppg/np.max(filtered_ppg))[peaks_filtered_signal] * offset_ppg_excel,
                 color='b', label='Peaks iPPG', marker='o', linestyle='')
        ax8.plot(excel_data[0], filtered_excel/np.max(filtered_excel),
                 color='r', label='cPPG Signal', linestyle='--')
        ax8.plot(excel_data[0][peaks_filtered_excel], (filtered_excel/np.max(filtered_excel))[peaks_filtered_excel],
                 color='r', label='Peaks cPPG', marker='o', linestyle='')

        ax8.grid()
        ax8.legend()
        ax8.set_title('Integrate iPPG Signal')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Amplitude')

        ax9.plot(excel_data[0], filtered_excel/np.max(filtered_excel),
                 color='r', label='cPPG Signal', linestyle='--')
        ax9.plot(excel_data[0][peaks_filtered_excel], (filtered_excel/np.max(filtered_excel))[peaks_filtered_excel],
                 color='r', label='Peaks cPPG', marker='o', linestyle='')
        ax9.plot(time_plot, filtered_spg/np.max(filtered_spg),
                 color='g', label="SPG Signal")
        ax9.plot(time_plot[peaks_filtered_spg], (filtered_spg/np.max(filtered_spg))[peaks_filtered_spg],
                 color='g', label="Peaks SPG", marker='o', linestyle='')
        ax9.grid()
        ax9.legend()
        ax9.set_title('Integrate SPG Signal')
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel('Amplitude')

        # Add MultiCursor to sync movement across subplots
        multi = MultiCursor(fig.canvas, (ax7, ax8, ax9), color='y', lw=1,
                            horizOn=True, vertOn=True)

        # save figure to file
        fig.tight_layout()
        fig.savefig(f'storage/{self.dir_path}/integrate_signal.png')

        plt.show()

        return filtered_ppg, filtered_spg, filtered_excel


if __name__ == "__main__":

    exposure_time = 6000  # us
    size_ppg = 200  # W x H 200
    size_spg = 150  # W x H 100
    fps = 88  # Hz 88
    cache = False
    cut_time_delay = 0.2

    folder = "2024-11-14 15_26_51 tee 6000 88 200"
    video_path = f"storage/{folder}/video.avi"
    serial_path = f"storage/{folder}/serial.xlsx"

    analysis = Analysis_PPG_SPG(
        video_path, serial_path, size_ppg, size_spg, exposure_time, fps, cache, cut_time_delay)
    ppg, spg, excel = analysis.main()

    # for i in range(6):

    #     for j in range(12):

    #         analysis_ppg_spg = Analysis_PPG_SPG(
    #             f'D:/college/4/term1/Project/videos/{i}/video.avi', f'D:/college/4/term1/Project/videos/{i}/serial.xlsx', 50*(j+1))
    #         a, b, c = analysis_ppg_spg.main()
    #         snr.append(a)
    #         snr_filtered_excel.append(b)
    #         error.append(c)

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(snr, label='PPG')
    # ax.plot(snr_filtered_excel, label='Excel')
    # ax.plot(error, label='Error')
    # ax.axvspan(0, 12, color='red', alpha=0.3)
    # ax.axvspan(12, 24, color='blue', alpha=0.3)
    # ax.axvspan(24, 36, color='green', alpha=0.3)
    # ax.axvspan(36, 48, color='yellow', alpha=0.3)
    # ax.axvspan(48, 60, color='purple', alpha=0.3)
    # ax.axvspan(60, 72, color='orange', alpha=0.3)
    # ax.legend()
    # ax.grid()
    # plt.show()
