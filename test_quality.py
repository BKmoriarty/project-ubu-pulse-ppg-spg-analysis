import numpy as np
from SignalQualityAssessment import PPGQualityAssessment
from matplotlib import pyplot as plt
import mplcursors
import os

# Example usage
if __name__ == "__main__":
    
    # loop load npy from storage
    # get all folder save to list
    folder_list = [f for f in os.listdir('storage') if os.path.isdir(os.path.join('storage', f))]
    
    signal_acceptable = []
    snr = []
    quality = []
    valid_folders = []

    fig , (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    for folder in folder_list:
        if not os.path.exists(f'storage/{folder}/signal_ppg.npy'):
            print(f'No signal in {folder}')
            continue
        
        signal_ppg = np.load(f'storage/{folder}/signal_ppg.npy')
        
        try:
            sampling_rate = float(folder.split(' ')[4])
        except (ValueError, IndexError):
            print(f'Invalid sampling rate in folder name: {folder}')
            continue
        
        if len(signal_ppg) < int(sampling_rate):
            print(f'Signal length is shorter than sampling rate in {folder}')
            continue
        
        quality_assessor = PPGQualityAssessment(sampling_rate=sampling_rate)
        
        # Analyze signal quality
        _ , is_acceptable = quality_assessor.analyze_and_plot(signal_ppg)
        signal_acceptable.append(is_acceptable)
        snr_value = quality_assessor.calculate_snr(signal_ppg)
        quality_score = quality_assessor.calculate_signal_quality_index(signal_ppg)
        snr.append(snr_value)
        quality.append(quality_score)
        valid_folders.append(folder)
        print(f"Signal Acceptable: {is_acceptable} Quality: {quality_score} SNR: {snr_value} in {folder}")
        
        # reset plot before next plot
        ax1.clear()
        ax2.clear()
        
        time , freq, spectrum = quality_assessor.cal_spectrum(signal_ppg)
        
        ax1.plot(time, signal_ppg, 'b-', label='PPG Signal')
        ax1.set_xlim(0, len(signal_ppg) / sampling_rate)
        ax1.set_ylim(np.min(signal_ppg), np.max(signal_ppg))
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('PPG Signal')
        ax1.legend()
        
        ax2.plot(freq, spectrum, 'r-', label='Frequency Spectrum')
        ax2.set_xlim(0, sampling_rate / 2)
        ax2.set_ylim(0, np.max(spectrum))
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Frequency Spectrum')
        ax2.axvline(quality_assessor.freq_hr_min, color='g', linestyle='--', label='HR Band')
        ax2.axvline(quality_assessor.freq_hr_max, color='g', linestyle='--')
        ax2.set_ylim(0, 600)
        ax2.legend()
        
        plt.tight_layout()
        
    
    # plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(valid_folders, signal_acceptable, 'o', label='Signal Acceptable')
    ax.plot(valid_folders, snr, 'o', label='SNR')
    ax.plot(valid_folders, quality/np.max(quality), 'o', label='Quality')
    ax.set_xlabel('Folder')
    ax.set_ylabel('Value')
    ax.set_title('Signal Acceptable, SNR, and Quality')
    ax.legend()
    plt.tight_layout()
    crs = mplcursors.cursor(ax,hover=True)
    crs.connect("add", lambda sel: sel.annotation.set_text(
        'Point {},{}'.format(sel.target[0], sel.target[1])))
    plt.show()
    
    
    # # Real data
    # noisy_signal = np.load('storage/2024-11-14 15_26_51 tee 6000 88 200/signal_ppg.npy')
    # # noisy_signal = np.load('storage/2024-11-26 16_49_26 tee 6000.0 88 (200, 200)/signal_ppg.npy')
    # # noisy_signal = np.load('storage/2024-12-06 17_21_02 nam 3000 350 (200, 200)/signal_ppg.npy')
    # sampling_rate = 88
    
    # # Initialize quality assessment
    # quality_assessor = PPGQualityAssessment(sampling_rate=sampling_rate)
    
    # # Analyze signal quality
    # fig, is_acceptable = quality_assessor.analyze_and_plot(noisy_signal)
    
    # # plot spectrum
    # quality_assessor.cal_spectrum(noisy_signal)
    
    # # Print results
    # print(f"Signal Acceptable: {is_acceptable}")
    # print(f"SNR: {quality_assessor.calculate_snr(noisy_signal):.2f} dB")
    # print(f"Quality Score: {quality_assessor.calculate_signal_quality_index(noisy_signal):.2f}")
    
    # # Show plots
    # plt.show()
