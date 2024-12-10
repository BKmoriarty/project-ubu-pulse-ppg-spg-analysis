import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

class PPGQualityAssessment:
    def __init__(self, sampling_rate=30.0):
        self.fs = sampling_rate
        # Define frequency bands of interest
        self.freq_hr_min = 0.5  # 30 BPM
        self.freq_hr_max = 5.0  # 300 BPM
        
        # Define quality thresholds
        self.snr_threshold = 2.0  # Minimum acceptable SNR
        self.peak_prominence_threshold = 0.1
        self.baseline_variation_threshold = 0.2

    def calculate_snr(self, ppg_signal):
        """
        Calculate Signal-to-Noise Ratio using frequency domain analysis
        """
        # Compute FFT
        n = len(ppg_signal)
        fft_data = fft(ppg_signal)
        freq = fftfreq(n, 1/self.fs)
        
        # Get positive frequencies only
        pos_freq_idx = freq >= 0
        freq = freq[pos_freq_idx]
        fft_data = np.abs(fft_data[pos_freq_idx])
        
        # Define signal and noise regions
        signal_mask = (freq >= self.freq_hr_min) & (freq <= self.freq_hr_max)
        noise_mask = ~signal_mask & (freq > 0)  # Exclude DC component
        
        # Calculate power in signal and noise regions
        signal_power = np.sum(np.square(fft_data[signal_mask]))
        noise_power = np.sum(np.square(fft_data[noise_mask]))
        
        # Calculate SNR
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        return snr

    def assess_baseline_stability(self, ppg_signal):
        """
        Assess baseline stability using moving average
        """
        # Make window size adaptive to signal length
        window = min(int(self.fs), len(ppg_signal) - 1)
        # Ensure window is odd (required for Savitzky-Golay filter)
        window = window - 1 if window % 2 == 0 else window
        # Ensure window is at least 3 (minimum required for quadratic fitting)
        window = max(3, window)
        
        # Ensure window is not larger than the signal length
        if window > len(ppg_signal):
            return False, 0
        
        baseline = signal.savgol_filter(ppg_signal, window, 2)
        variation = np.std(baseline) / np.mean(ppg_signal)
        return variation < self.baseline_variation_threshold, variation

    def detect_heart_rate(self, ppg_signal):
        """
        Detect heart rate and assess its reliability
        """
        # Apply bandpass filter
        nyquist = self.fs / 2
        b, a = signal.butter(2, [self.freq_hr_min/nyquist, self.freq_hr_max/nyquist], 'band')
        filtered_signal = signal.filtfilt(b, a, ppg_signal)
        
        # Find peaks
        peaks, properties = signal.find_peaks(filtered_signal, 
                                            distance=int(self.fs/self.freq_hr_max),
                                            prominence=self.peak_prominence_threshold)
        
        if len(peaks) < 2:
            return 0, False
            
        # Calculate heart rate
        peak_intervals = np.diff(peaks) / self.fs  # Convert to seconds
        hr = 60 / np.mean(peak_intervals)  # Convert to BPM
        
        # Check reliability
        hr_reliable = (hr >= 30) and (hr <= 300)  # Normal human heart rate range
        return hr, hr_reliable

    def calculate_signal_quality_index(self, ppg_signal):
        """
        Calculate overall signal quality index
        """
        # Calculate individual metrics
        snr = self.calculate_snr(ppg_signal)
        baseline_stable, baseline_var = self.assess_baseline_stability(ppg_signal)
        hr, hr_reliable = self.detect_heart_rate(ppg_signal)
        
        # Normalize SNR score (0 to 1)
        snr_score = min(max(snr / 10, 0), 1)  # Assuming SNR of 10 dB is excellent
        
        # Combine metrics into single quality score
        quality_score = (snr_score * 0.4 +  # 40% weight to SNR
                        (1 - baseline_var) * 0.3 +  # 30% weight to baseline stability
                        float(hr_reliable) * 0.3)  # 30% weight to HR reliability
        
        return quality_score

    def is_signal_acceptable(self, ppg_signal):
        """
        Determine if signal quality is acceptable based on all metrics
        """
        quality_score = self.calculate_signal_quality_index(ppg_signal)
        snr = self.calculate_snr(ppg_signal)
        baseline_stable, _ = self.assess_baseline_stability(ppg_signal)
        _, hr_reliable = self.detect_heart_rate(ppg_signal)
        
        # Define acceptance criteria
        is_acceptable = (
            quality_score >= 0.6 and  # Overall quality score threshold
            snr >= self.snr_threshold and
            baseline_stable and
            hr_reliable
        )
        
        return is_acceptable, quality_score

    def cal_spectrum(self, ppg_signal):

        # Plot PPG signal
        time = np.arange(len(ppg_signal)) / self.fs
        # ax1.plot(time, ppg_signal, 'b-', label='PPG Signal')
        # ax1.set_xlim(0, len(ppg_signal) / self.fs)
        # ax1.set_ylim(np.min(ppg_signal), np.max(ppg_signal))
        # ax1.set_xlabel('Time (s)')
        # ax1.set_ylabel('Amplitude')
        # ax1.set_title('PPG Signal')
        # ax1.legend()

        # Plot frequency spectrum
        fft_data = fft(ppg_signal)
        freq = fftfreq(len(ppg_signal), 1/self.fs)
        pos_freq_idx = freq >= 0
        
        # ax2.plot(freq[pos_freq_idx], np.abs(fft_data[pos_freq_idx]), 'r-', label='Frequency Spectrum')
        # ax2.set_xlim(0, self.fs / 2)
        # ax2.set_ylim(0, np.max(np.abs(fft_data)))
        # ax2.set_xlabel('Frequency (Hz)')
        # ax2.set_ylabel('Magnitude')
        # ax2.set_title('Frequency Spectrum')
        # ax2.axvline(self.freq_hr_min, color='g', linestyle='--', label='HR Band')
        # ax2.axvline(self.freq_hr_max, color='g', linestyle='--')
        # ax2.set_ylim(0, 600)
        # ax2.legend()
        
        return time, freq[pos_freq_idx], np.abs(fft_data[pos_freq_idx])


    def analyze_and_plot(self, ppg_signal):
        """
        Analyze signal quality and create diagnostic plots
        """
        # Perform analysis
        quality_score = self.calculate_signal_quality_index(ppg_signal)
        snr = self.calculate_snr(ppg_signal)
        hr, hr_reliable = self.detect_heart_rate(ppg_signal)
        is_acceptable, _ = self.is_signal_acceptable(ppg_signal)
        
        # # Create plots
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # # Time domain plot
        # time = np.arange(len(ppg_signal)) / self.fs
        # ax1.plot(time, ppg_signal, 'b-', label='PPG Signal')
        # ax1.set_xlabel('Time (s)')
        # ax1.set_ylabel('Amplitude')
        # ax1.set_title(f'PPG Signal (Quality Score: {quality_score:.2f}, SNR: {snr:.2f} dB)')
        
        # # Frequency domain plot
        # n = len(ppg_signal)
        # fft_data = fft(ppg_signal)
        # freq = fftfreq(n, 1/self.fs)
        # pos_freq_idx = freq >= 0
        
        # ax2.plot(freq[pos_freq_idx], np.abs(fft_data[pos_freq_idx]), 'r-')
        # ax2.set_xlabel('Frequency (Hz)')
        # ax2.set_ylabel('Amplitude')
        # ax2.set_title(f'Frequency Spectrum (HR: {hr:.1f} BPM)')
        
        # # Highlight HR frequency band
        # ax2.axvspan(self.freq_hr_min, self.freq_hr_max, color='g', alpha=0.2)
        
        # plt.tight_layout()
        return True, is_acceptable
