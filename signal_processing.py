import numpy as np
import pandas as pd
from scipy import signal
import neurokit2 as nk

class SignalProcessing:

    def bandpass_filter(self, input_signal, lp_freq, hp_freq, sampling_rate=200, order=4):
        nyquist = 0.5 * sampling_rate
        low = lp_freq / nyquist
        high = hp_freq / nyquist
        
        b, a = signal.butter(order, [low, high], 'bandpass', analog=False)
        y = signal.filtfilt(b, a, input_signal, axis=0)
        
        return y
    
    # Fast Fourier Transform of the signal
    def fft(signal1, sampling_rate=200):
        fft_ = np.fft.fft(signal1) / len(signal1)
        fft_ = fft_[range(int(len(signal1)/2))]
        tim_p = len(signal1) / sampling_rate
        freq = np.arange(int(len(signal1) / 2)) / tim_p

        return freq, abs(fft_)

    
    def rectify(self, signal):
        return abs(signal)
    
    # Amplitude envelope
    def amplitude_envelope(signal, frame_size, hop_length):
        amplitude_envelope = []

        for i in range(0, len(signal), hop_length):
            c_frame_env = max(signal[i:i+frame_size])

            amplitude_envelope.append(c_frame_env)
        
        return np.array(amplitude_envelope)
    
    # RMS 
    def rms_envelope(signal, frame_length, hop_length): 
        rms = []
        for i in range(0, len(signal), hop_length):
            rms_ = np.sqrt(np.sum(signal[i:i+frame_length]**2) / frame_length)
            rms.append(rms_)
        return np.array(rms)