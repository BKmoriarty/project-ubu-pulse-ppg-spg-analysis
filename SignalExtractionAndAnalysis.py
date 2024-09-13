import cv2
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


class Analysis_PPG_SPG:

    def __init__(self, video_path, excel_path, size, exposure_time=2200):
        self.size = size
        self.video_path = video_path
        self.excel_path = excel_path
        self.size_block = 3
        self.exposure_time = exposure_time

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

        x1, y1, x2, y2 = self.get_center_crop_position(
            frame.shape, self.size)

        color = (0, 255, 0)  # Green color in BGR
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.imshow('Image with Centered Square', frame)

        # fig, ax01 = plt.subplots(1, 1)
        # ax01.imshow(frame)
        # ax01.set_title('Raw Image')

        # fig, ax02 = plt.subplots(1, 1)
        # ax02.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        # ax02.set_title('Grayscale Image')

        # fig, ax03 = plt.subplots(1, 1)
        # ax03.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y1:y2, x1:x2])
        # ax03.set_title('Cropped Image')

        # frame_crop = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y1:y2, x1:x2]

        # fig, ax04 = plt.subplots(1, 1)
        # mean_frame = np.mean(frame_crop)
        # # show the mean pixel intensity in the ROI
        # ax04.imshow(np.ones((frame_crop.shape[0], frame_crop.shape[1]))
        #             * mean_frame, cmap='gray')
        # ax04.set_title(f'Mean Frame: {mean_frame:.2f}')

        # fig, ax05 = plt.subplots(1, 1)
        # # crop 3x3 blocks and calculate the contrast
        # contrast = self.cal_contrast(frame_crop)
        # ax05.imshow(contrast)
        # ax05.set_title('Contrast')

        return cap, [x1, y1, x2, y2]

    def extract_signal(self, cap, position):

        signal_ppg = []
        mean_contrast_frame = []
        mean_exposure_frame = []
        # exposure_time = 1500  # us

        x1, y1, x2, y2 = position

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 1
        i = 0

        self.printProgressBar(0, length, prefix='Progress:',
                              suffix='Complete', length=50)
        start_process = process_time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Extract the ROI from the grayscale frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray_frame[y1:y2, x1:x2]

            # Calculate the mean pixel intensity in the ROI
            signal_intensity = np.mean(roi)
            signal_ppg.append(signal_intensity)

            contrast = self.cal_contrast(roi)
            mean_contrast_frame.append(np.mean(contrast))  # mean contrast

            # Calculate mean exposure using the formula: 1 / (2 * T * K^2)
            T = self.exposure_time
            K = contrast
            epsilon = 1e-10
            mean_exposure = np.mean(1 / (2 * T * (np.square(K) + epsilon)))
            mean_exposure_frame.append(mean_exposure)

            end_process = process_time()
            i = i+1
            self.printProgressBar(i + 1, length, prefix='Progress:',
                                  suffix=f'time: {(end_process-start_process):.2f} seconds. Complete', length=50)

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
        plt.legend()
        plt.show()

    def plot_spectrum(self, xf, yf):
        plt.figure(figsize=(10, 4))
        plt.plot(xf, yf, color='blue', label='Spectrum')
        plt.title('Frequency Spectrum of the Extracted Signal')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()
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
        plt.legend()
        plt.show()

    # Plot the frequency spectrum (frequency domain)

    def plot_spectrum(self, xf, yf, title="Frequency Spectrum"):
        plt.figure(figsize=(10, 4))
        plt.plot(xf, yf, color='blue', label=title)
        plt.title(title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()
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

        SNR_ppg = 10*np.log(power_ppg_mean/power_noise_ppg_mean)

        return SNR_ppg

    def find_peak_freq(self, data, time):

        peaks, _ = find_peaks(data, height=None,
                              threshold=None, distance=15)

        # Calculate heart rate (if peaks represent heartbeats)
        time_diff = np.diff(time[peaks])
        heart_rate = 60 / np.mean(time_diff)  # unit is bpm

        return peaks, heart_rate

    def find_one_peaks(self, data):
        i_peaks, _ = find_peaks(data)

        i_max_peak = i_peaks[np.argmax(data[i_peaks])]

        return i_max_peak

    def find_peak_freq_excel(self, data, time):
        # Find peaks
        # Adjust these parameters as needed for your specific data
        peaks, _ = find_peaks(data, height=None,
                              threshold=None, distance=60)

        # Calculate heart rate (if peaks represent heartbeats)
        time_diff = np.diff(time[peaks])
        heart_rate = 60 / np.mean(time_diff)

        return peaks, heart_rate

    def main(self):
        # print("1. exit data")
        # print("2. new data")
        # choose = input("choose your data:")
        choose = '2'

        cap, position = self.extract_video_frames(self.video_path)

        if (choose == '1'):
            signal_ppg = np.load('temp\signal_ppg.npy')
            mean_exposure_frame = np.load('temp\mean_exposure_frame.npy')
            mean_contrast_frame = np.load('temp\mean_contrast_frame.npy')
        else:
            signal_ppg, mean_contrast_frame, mean_exposure_frame = self.extract_signal(
                cap, position
            )
            np.save('temp\signal_ppg.npy', signal_ppg)
            np.save('temp\mean_exposure_frame.npy', mean_exposure_frame)
            np.save('temp\mean_contrast_frame.npy', mean_contrast_frame)

        # =========================== PPG ===========================
        signal_freq_range_ppg = (0.83, 4)  # Hz (typical range for heart rate)
        noise_freq_range_ppg = [
            (0, signal_freq_range_ppg[0]), (signal_freq_range_ppg[1], 15)]

        fps = 30
        signal_values_fft = self.perform_fft(signal_ppg, fps)

        fs = signal_ppg.size / 20
        time_ppg = np.arange(signal_ppg.size) / fs

        filtered_ppg, filtered_signal_fft1 = self.filter_signal(
            signal_ppg, fps,  signal_freq_range_ppg[0], signal_freq_range_ppg[1])

        # phase shift the signal 180 degree
        filtered_ppg_reverse = filtered_ppg  # * -1

        peaks_filtered_signal, heart_rate_filtered_signal = self.find_peak_freq(
            filtered_ppg_reverse, time_ppg)

        max_filtered_ppg = self.find_one_peaks(filtered_signal_fft1[1])
        print(
            f"Heart rate of the ppg FFT: {(filtered_signal_fft1[0][max_filtered_ppg]*60):.2f} bpm")

        fig, (ax4, ax5, ax6) = plt.subplots(3, 1)

        ax4.plot(time_ppg, signal_ppg, color='b', label='PPG Raw Signal')
        ax4.legend()
        ax5.plot(time_ppg, filtered_ppg, color='b',
                 label='PPG Filtered Signal')
        ax5.plot(time_ppg[peaks_filtered_signal], filtered_ppg[peaks_filtered_signal],
                 color='g', label="Peaks PPG", marker='o', linestyle='')
        ax5.legend()
        ax6.plot(
            signal_values_fft[0], signal_values_fft[1], color='b', label='PPG Raw fft')
        ax6.plot(
            filtered_signal_fft1[0], filtered_signal_fft1[1], color='r', label='PPG Filtered fft')
        ax6.axvspan(
            signal_freq_range_ppg[0], signal_freq_range_ppg[1], color='green', alpha=0.3)
        ax6.set_ylim([0, np.max(filtered_signal_fft1[1]) * 2])
        ax6.plot(filtered_signal_fft1[0][max_filtered_ppg], filtered_signal_fft1[1][max_filtered_ppg],
                 color='r', label="Peaks PPG Filtered FFT", marker='o', linestyle='')
        ax6.legend()

        # =========================== SPG ===================================
        # Hz (typical range for heart rate)
        signal_freq_range_spg = (0.83, 4.5)
        noise_freq_range_spg = [
            (0, signal_freq_range_spg[0]), (signal_freq_range_spg[1], 15)]

        fps = 30
        time_spg = np.arange(mean_exposure_frame.size) / fps

        mean_exposure_frame_fft = self.perform_fft(mean_exposure_frame, fps)

        filtered_spg, filtered_spg_fft = self.filter_signal(
            mean_exposure_frame, fps,  signal_freq_range_spg[0], signal_freq_range_spg[1])

        peaks_filtered_spg, heart_rate_filtered_spg = self.find_peak_freq(
            filtered_spg, time_spg)

        max_filtered_spg = self.find_one_peaks(filtered_spg_fft[1])
        print(
            f"Heart rate of the spg FFT: {(filtered_spg_fft[0][max_filtered_spg]*60):.2f}  bpm")

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

        ax1.plot(time_spg, mean_exposure_frame,
                 color='b', label='SPG Raw Signal')
        ax1.legend()
        ax2.plot(time_spg, filtered_spg, color='b',
                 label='SPG Filtered Signal')
        ax2.plot(time_spg[peaks_filtered_spg], filtered_spg[peaks_filtered_spg],
                 color='g', label="Peaks SPG", marker='o', linestyle='')
        ax2.legend()
        ax3.plot(
            mean_exposure_frame_fft[0], mean_exposure_frame_fft[1], color='b', label='SPG Raw fft')
        ax3.plot(
            filtered_spg_fft[0], filtered_spg_fft[1], color='r', label='SPG Filtered fft')
        ax3.set_ylim([0, np.max(filtered_spg_fft[1])*2])
        ax3.axvspan(
            signal_freq_range_spg[0], signal_freq_range_spg[1], color='green', alpha=0.3)
        ax3.plot(filtered_spg_fft[0][max_filtered_spg], filtered_spg_fft[1][max_filtered_spg],
                 color='r', label="Peaks SPG Filtered FFT", marker='o', linestyle='')
        ax3.legend()

        # =========================== import excel ===========================
        excel_data = self.load_excel(self.excel_path)

        # apply filter
        signal_freq_range_excel = (0.83, 4)
        noise_freq_range_excel = [
            (0, signal_freq_range_excel[0]), (signal_freq_range_excel[1], 15)]
        fps_excel = 125  # sampling rate means the number of samples per second
        [filtered_excel, filtered_excel_fft] = self.filter_signal(
            excel_data[1], fps_excel, signal_freq_range_excel[0], signal_freq_range_excel[1])

        max_filtered_excel = self.find_one_peaks(
            filtered_excel_fft[1])
        print(
            f"Heart rate of the excel FFT: {(filtered_excel_fft[0][max_filtered_excel]*60):.2f} bpm")

        # fps_excel = 30
        # # time = excel_data[0]
        # time = np.arange(excel_data[0].size) / fps_excel
        # signal_freq_range_excel = (0.83, 3)
        # noise_freq_range_excel = [
        #     (0, signal_freq_range_excel[0]), (signal_freq_range_excel[1], 15)]

        # # filtered_excel = self.lowpass_filter(
        # #     excel_data[1], signal_freq_range_excel[1], fps)
        # # filtered_excel_fft = self.perform_fft(filtered_excel, fps)

        # filtered_excel, filtered_excel_fft = self.filter_signal(
        #     excel_data[1], fps_excel,  signal_freq_range_excel[0], signal_freq_range_excel[1])

        # peaks_excel, heart_rate_excel = self.find_peak_freq_excel(
        #     filtered_excel, excel_data[0])

        # max_filtered_excel = self.find_one_peaks(filtered_excel_fft[1])
        # print(
        #     f"max_filtered_excel:{filtered_excel_fft[0][max_filtered_excel]}")
        # print(
        #     f"Heart rate of the excel: {heart_rate_excel:.2f} -> FFT: {(filtered_excel_fft[0][max_filtered_excel]*60):.2f} bpm")

        fig, (ax10, ax11, ax12) = plt.subplots(3, 1)

        ax10.plot(excel_data[0], excel_data[1],
                  color='b', label='Excel Raw Signal')
        ax10.legend()
        ax11.plot(excel_data[0], filtered_excel/np.max(filtered_excel), color='b',
                  label='Excel Filtered Signal')
        ax11.legend()
        ax12.plot(
            filtered_excel_fft[0], filtered_excel_fft[1], color='r', label='Excel Filtered fft')
        ax12.set_ylim([0, np.max(filtered_excel_fft[1])*2])
        ax12.axvspan(
            signal_freq_range_excel[0], signal_freq_range_excel[1], color='green', alpha=0.3)
        ax12.plot(filtered_excel_fft[0][max_filtered_excel], filtered_excel_fft[1][max_filtered_excel],
                  color='r', label="Peaks Excel Filtered FFT", marker='o', linestyle='')
        ax12.legend()

        # =========================== SNR ===========================

        # mean
        snr_signal_values, frequencies_signal_values, power_signal_values = self.calculate_snr(
            signal_ppg, fps, signal_freq_range_ppg, noise_freq_range_ppg)

        # ppg
        snr_filtered_ppg, frequencies_filtered_ppg, power_filtered_ppg = self.calculate_snr(
            filtered_ppg, fps, signal_freq_range_ppg, noise_freq_range_ppg)

        print(
            f"SNR of the ppg values: {snr_signal_values:.2f} -> {snr_filtered_ppg:.2f} dB")

        # spg
        snr_spg, frequencies_spg, power_spg = self.calculate_snr(
            mean_exposure_frame, fps, signal_freq_range_spg, noise_freq_range_spg)

        snr_filtered_spg, frequencies_filtered_spg, power_filtered_spg = self.calculate_snr(
            filtered_spg, fps, signal_freq_range_spg, noise_freq_range_spg)

        print(
            f"SNR of the spg values: {snr_spg:.2f} -> {snr_filtered_spg:.2f} dB")

        # excel
        snr_excel, frequencies_excel, power_excel = self.calculate_snr(
            excel_data[1], fps_excel, signal_freq_range_excel, noise_freq_range_excel)

        snr_filtered_excel, frequencies_excel, power_excel = self.calculate_snr(
            filtered_excel, fps_excel, signal_freq_range_excel, noise_freq_range_excel)

        print(
            f"SNR of the excel values: {snr_excel:.2f} -> {snr_filtered_excel:.2f} dB")

        # =========================== Plotting ===========================
        time_plot = np.arange(mean_exposure_frame.size) / fps

        fig, (ax7, ax8, ax9) = plt.subplots(3, 1,)  # figsize=(15, 3)
        ax7.plot(time_plot, filtered_ppg/np.max(filtered_ppg),
                 color='b', label='PPG Signal')
        ax7.plot(time_plot[peaks_filtered_signal], (filtered_ppg/np.max(filtered_ppg))[peaks_filtered_signal],
                 color='b', label='Peaks PPG', marker='o', linestyle='')
        ax7.plot(excel_data[0], filtered_excel/np.max(filtered_excel),
                 color='r', label='Real Signal', linestyle='--')
        ax7.plot(time_plot, filtered_spg/np.max(filtered_spg),
                 color='g', label="SPG Signal")
        ax7.plot(time_plot[peaks_filtered_spg], (filtered_spg/np.max(filtered_spg))[peaks_filtered_spg],
                 color='g', label="Peaks SPG", marker='o', linestyle='')
        ax7.grid()
        ax7.legend()
        ax7.set_title('Integrate Signal')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Amplitude')

        ax8.plot(time_plot, filtered_ppg_reverse/np.max(filtered_ppg_reverse),
                 color='b', label='PPG Signal')
        ax8.plot(time_plot[peaks_filtered_signal], (filtered_ppg_reverse/np.max(filtered_ppg_reverse))[peaks_filtered_signal],
                 color='b', label='Peaks PPG', marker='o', linestyle='')
        ax8.plot(excel_data[0], filtered_excel/np.max(filtered_excel),
                 color='r', label='Real Signal', linestyle='--')
        ax8.grid()
        ax8.legend()
        ax8.set_title('Integrate PPG Signal')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Amplitude')

        ax9.plot(excel_data[0], filtered_excel/np.max(filtered_excel),
                 color='r', label='Real Signal', linestyle='--')
        ax9.plot(time_plot, filtered_spg/np.max(filtered_spg),
                 color='g', label="SPG Signal")
        ax9.plot(time_plot[peaks_filtered_spg], (filtered_spg/np.max(filtered_spg))[peaks_filtered_spg],
                 color='g', label="Peaks SPG", marker='o', linestyle='')
        ax9.grid()
        ax9.legend()
        ax9.set_title('Integrate SPG Signal')
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel('Amplitude')

        plt.tight_layout()
        plt.show()

        return [filtered_ppg, snr_filtered_excel, abs(heart_rate_filtered_signal - heart_rate_filtered_spg)]


if __name__ == "__main__":
    snr, snr_filtered_excel, error = [], [], []

    analysis_ppg_spg = Analysis_PPG_SPG(
        f'D:/college/4/term1/Project/videos/v11_et2200/video.avi', f'D:/college/4/term1/Project/videos/v11_et2200/serial.xlsx', 150, 2000)
    # "./video.avi", "./serial.xlsx", 150)
    a, b, c = analysis_ppg_spg.main()
    # analysis_ppg_spg.extract_video_frames(
    #     f'D:/college/4/term1/Project/videos/v1/video.avi')

    plt.show()

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


# Plot the power spectrum
# plt.figure(figsize=(12, 6))
# plt.plot(frequencies[:len(frequencies)//2], power[:len(frequencies)//2])
# plt.title('Power Spectrum')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power')
# plt.axvspan(signal_freq_range[0], signal_freq_range[1],
#             color='green', alpha=0.3, label=f'Signal Range SNR: {snr_signal_values:.2f} dB')
# plt.axvspan(noise_freq_range[0][0], noise_freq_range[0]
#             [1], color='red', alpha=0.3, label='Noise Range 0-0.83 Hz')
# plt.axvspan(noise_freq_range[1][0], noise_freq_range[1]
#             [1], color='red', alpha=0.3, label='Noise Range 4-15 Hz')
# plt.legend()
# plt.grid(True)
# plt.show()


# =========================== Plotting ===========================
# fig, (ax1, ax3, ax5) = plt.subplots(3, 1, figsize=(15, 10))
# fig, (ax2, ax4, ax6) = plt.subplots(3, 1)


# ax1.plot(time, signal_ppg, color='blue', label='Raw Signal')
# ax1.set_title('Raw Signal')
# ax1.set_xlabel('Frame')
# ax1.set_ylabel('Intensity')
# ax1.grid()

# ax2.plot(signal_values_fft[0], signal_values_fft[1],
#          color='blue', label='Spectrum')
# ax2.set_title('Frequency Spectrum of the Raw Signal')
# ax2.set_xlabel('Frequency (Hz)')
# ax2.set_ylabel('Amplitude')
# ax2.grid()

# ax3.plot(time, filtered_ppg, color='blue', label='Filtered Signal')
# ax3.set_title('Filtered Signal')
# ax3.set_xlabel('Frame')
# ax3.set_ylabel('Intensity')
# ax3.grid()

# ax4.plot(filtered_signal_fft[0], filtered_signal_fft[1],
#          color='blue', label='Filtered Signal')
# ax4.set_title('Frequency Spectrum of the Filtered Signal')
# ax4.set_xlabel('Frequency (Hz)')
# ax4.set_ylabel('Amplitude')
# ax4.grid()

# ax5.plot(excel_data[0], excel_data[1], color='blue', label='Extracted Signal')
# ax5.set_title('Extracted Heart Wave Signal')
# ax5.set_xlabel('Frame')
# ax5.set_ylabel('Intensity')
# ax5.grid()

# ax6.plot(filtered_excel_fft[0], filtered_excel_fft[1],
#          color='blue', label='Filtered Signal')
# ax6.set_title('Frequency Spectrum of the Filtered Signal')
# ax6.set_xlabel('Frequency (Hz)')
# ax6.set_ylabel('Amplitude')
# ax6.grid()
