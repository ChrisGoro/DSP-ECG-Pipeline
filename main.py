import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks


class ECG_Pipeline: 

    def __init__(self, patient_number, run_batch = False):

        self.patient_number = patient_number
        self.raw_ecg = None
        self.clean_ecg = None
        self.peaks = None
        self.bpm = 0
        self.sdnn = 0
        self.pvc_count = 0
        self.run_batch = run_batch


    def load_data(self): 

        """
        Retrieves 4000 samples of ECG data from the MIT-BIH Arrhythmia Database.
        Extracts the primary lead (column 0) for analysis.
        """

        print(f"Initializing data extraction for Patient {self.patient_number}...")
        record = wfdb.rdrecord(f'{self.patient_number}', pn_dir='mitdb', sampto=4000)
        data = pd.DataFrame(record.p_signal, columns=record.sig_name)
        self.raw_ecg = data.iloc[:, 0] 
        

    def apply_bandpass(self): 

        """
        Applies a 2nd-order Butterworth bandpass filter (0.5Hz - 40.0Hz) using zero-phase 
        filtering (filtfilt) to remove baseline wander and high-frequency noise.
        Slices the edges to prevent startup transients.
        """
        
        sample_rate = 360 
        nyquist = 0.5 * sample_rate
        low = 0.5 / nyquist
        high = 40.0 / nyquist

        b, a = butter(2, [low, high], btype='band')
        clean_ecg = filtfilt(b, a, self.raw_ecg)

        self.clean_ecg = clean_ecg[200:3800]
        self.raw_ecg = self.raw_ecg.values[200:3800]
        

    def find_peaks(self): 

        """
        Autonomously corrects phase inversion using 1st/99th percentiles.
        Calculates a dynamic amplitude threshold (60% of the 99.5th percentile) 
        to detect R-peaks while ignoring T-waves via a 120-sample refractory period.
        """

        bottom_floor = abs(np.percentile(self.clean_ecg, 1))
        top_ceiling = np.percentile(self.clean_ecg, 99)
        
        if bottom_floor > top_ceiling:
            self.clean_ecg = self.clean_ecg * -1
            
        max_voltage = np.percentile(self.clean_ecg, 99.5)
        dynamic_threshold = max_voltage * 0.60
        self.peaks, _ = find_peaks(self.clean_ecg, height=dynamic_threshold, distance=120)

        
    def calculate_bpm(self): 

        """
        Calculates the average Beats Per Minute (BPM) based on the mean 
        sample distance between detected R-peaks.
        """

        intervals = np.diff(self.peaks)
        avg_interval_samples = np.mean(intervals)
    
        # Convert that sample distance into seconds
        avg_interval_sec = avg_interval_samples / 360.0
    
        # Convert seconds per beat into Beats Per Minute
        self.bpm = 60.0 / avg_interval_sec
    
        print(f"Calculated Heart Rate: {round(self.bpm, 1)} BPM")
    

    def calculate_hrv(self):

        """
        Calculates the Standard Deviation of Normal-to-Normal intervals (SDNN).
        Utilizes a dynamic median-based time filter (±20% tolerance) to identify 
        and exclude arrhythmic intervals (e.g., PVCs, dropped beats) from the calculation.
        """
        
        # Grab intervals and convert to milliseconds
        intervals_samples = np.diff(self.peaks)
        intervals_ms = (intervals_samples / 360.0) * 1000.0
        
        # Find the median
        median_interval = np.median(intervals_ms)
        
        lower_bound = median_interval * 0.80  # 20% faster (Catches the premature beat)
        upper_bound = median_interval * 1.20  # 20% slower (Catches the compensatory pause)
        
        # Keep only the intervals strictly inside the bounds
        normal_intervals = intervals_ms[(intervals_ms >= lower_bound) & (intervals_ms <= upper_bound)]
        
        # Calculate SDNN on the clean normal beats
        self.sdnn = np.std(normal_intervals)
        
        # Count how many anomalies we caught for the output
        self.pvc_count = len(intervals_ms) - len(normal_intervals)
        
        print(f"Calculated HRV (SDNN): {round(self.sdnn, 2)} ms")
        if self.pvc_count > 0:
            print(f"   -> [ALERT] PVC Filter active: Removed {self.pvc_count} abnormal intervals.")


    def plot_results(self, return_fig=False): 

        """
        Generates a comparative matplotlib visualization of the raw noisy signal 
        overlaid with the filtered signal and detected R-peak markers.
        """
        
        fig = plt.figure(figsize=(12, 6))
        
        # Plot the raw, messy data in the background (faded out)
        plt.plot(self.raw_ecg, label="Raw Noisy Data", color="gray", alpha=0.5)
        
        # Plot the clean, zero-phase filtered data on top
        plt.plot(self.clean_ecg, label="Filtered Data", color="#1f77b4", linewidth=2)
        
        # Plant the red X's on the exact peak coordinates
        plt.plot(self.peaks, self.clean_ecg[self.peaks], "rx", markersize=8, label="Detected Beats")
        
        plt.title(f"DSP Pipeline Results - Patient {self.patient_number}")
        plt.xlabel("Samples (Time)")
        plt.ylabel("Voltage (mV)")
        plt.legend()
        plt.grid(True)
        
        if self.run_batch == True:
            # If running the loop, save it to the folder
            plt.savefig(f"graphs/patient_{self.patient_number}_report.png", bbox_inches='tight')
            plt.close()
        elif return_fig == True:
            return fig 
        else:
            # If running standalone, pop it open on the screen
            plt.show()
        

class Advanced_ECG_Pipeline(ECG_Pipeline): # Subclass for experimental, from-scratch DSP algorithms

    def __init__(self, patient_number, run_batch=False):
        super().__init__(patient_number, run_batch)
        

    # --- CONVOLUTION (FIR FILTER) ---
    def custom_fir_filter(self, window_size=15):

        """
        Applies a from-scratch Finite Impulse Response (FIR) moving-average filter 
        using mathematical convolution to smooth the signal and demonstrate phase delay.
        """

        kernel = np.ones(window_size) / window_size 
        filtered_signal = np.zeros(len(self.raw_ecg))
        padded_signal = np.pad(self.raw_ecg, (window_size//2, window_size//2), mode='edge')
        
        for i in range(len(self.raw_ecg)):
            window_data = padded_signal[i : i + window_size]
            filtered_signal[i] = np.sum(window_data * kernel)
            
        self.clean_ecg = filtered_signal
        

    def custom_fft(self, x):

        """
        A recursive Radix-2 Cooley-Tukey Fast Fourier Transform (FFT) algorithm 
        written from scratch. Converts time-domain data to the frequency domain.
        Note: Input array length (N) must be a power of 2.
        """

        N = len(x)
        if N <= 1:
            return x
        even = self.custom_fft(x[0::2])
        odd =  self.custom_fft(x[1::2])
        T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
        return [even[k] + T[k] for k in range(N // 2)] + \
               [even[k] - T[k] for k in range(N // 2)]


    def analyze_frequency(self, inject_noise=False):

        """
        Truncates the signal to a power of 2 (N=2048) and executes the custom FFT 
        to extract frequency magnitudes, isolating powerline interference (e.g., 60Hz).
        Allows for synthetic noise injection for algorithmic stress-testing.
        """

        if inject_noise == True:
            print("-> [TESTING] Injecting synthetic 60Hz noise...")
            time_array = np.arange(len(self.raw_ecg)) / 360.0 
            fake_60hz_hum = 0.5 * np.sin(2 * np.pi * 60 * time_array)
            self.raw_ecg = self.raw_ecg + fake_60hz_hum

        N = 2048 # Must be power of 2
        sample_data = self.raw_ecg[:N]

        # Convert to standard numpy array
        sample_data = np.array(sample_data)
        fft_result = self.custom_fft(sample_data)
        
        self.frequencies = np.array([k * 360.0 / N for k in range(N//2)])
        self.magnitudes = np.abs(fft_result)[:N//2]


    def plot_results(self, return_fig=False):

        """
        Overrides the parent class plotter to generate a two-pane visualization:
        1. Time Domain: Custom FIR convolution smoothing.
        2. Frequency Domain: FFT magnitude output.
        """

        # Create a 2-story graph
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Only plot the first 1000 samples
        ax1.plot(self.raw_ecg[:1000], label='Raw Noisy Data', color='silver', alpha=0.8)
        ax1.plot(self.clean_ecg[:1000], label='Custom FIR Filtered', color='#1f77b4', linewidth=2)
        ax1.set_title(f'Time Domain: Custom Convolution FIR Filter (Patient {self.patient_number})', fontsize=14)
        ax1.set_xlabel('Samples (Time)')
        ax1.set_ylabel('Voltage (mV)')
        ax1.legend(loc='upper right')
        ax1.grid(True)

        # Check if we ran the FFT before trying to plot it
        if hasattr(self, 'frequencies'):
            ax2.plot(self.frequencies, self.magnitudes, color='darkred', linewidth=1.5)
            ax2.set_title('Frequency Domain: Custom Fast Fourier Transform (FFT)', fontsize=14)
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Signal Magnitude (Power)')
            ax2.grid(True)
            
            # Zoom the camera in to the 0-100Hz range
            ax2.set_xlim(0, 100) 
            
        else:
            ax2.text(0.5, 0.5, 'FFT not run. Call analyzer.analyze_frequency()', 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)

        plt.tight_layout()
        
        if return_fig == True:
            return fig
        else:
            plt.show()


if __name__ == "__main__":

    # ==========================================
    #              MASTER CONTROL
    # ==========================================

    # Select 100-series or 200-series patient
    patient_number = 208      

    # True: Use custom FIR/FFT | False: Use MVP pipeline      
    test_advanced = False    

    # True: Force 60Hz powerline hum into the signal
    inject_60hz_noise = False 

    # True: Run full directory | False: Run single patient 
    run_batch = False    

    # Choose which database to run: 100, 200, or "ALL"
    batch_target = 200

    # ==========================================

    if run_batch == True:

        # The official valid list of 100-series patients
        patients_100 = [100, 101, 102, 103, 104, 105, 106, 107, 108, 
                        109, 111, 112, 113, 114, 115, 116, 117, 118, 
                        119, 121, 122, 123, 124]

        # The official valid list of 200-series patients
        patients_200 = [200, 201, 202, 203, 205, 207, 208, 209, 210, 
                        212, 213, 214, 215, 217, 219, 220, 221, 222, 
                        223, 228, 230, 231, 232, 233, 234]

        print("Initializing Batch Processing...")

        if batch_target == 100:
            target_list = patients_100
        elif batch_target == 200:
            target_list = patients_200
        else:
            target_list = patients_100 + patients_200 

        for p_id in target_list:
            
            try:

                # Initialize pipeline for batch processing
                analyzer = ECG_Pipeline(p_id, run_batch = True)
                
                # Execute DSP stages
                analyzer.load_data()
                analyzer.apply_bandpass()
                analyzer.find_peaks()
                analyzer.calculate_bpm()
                analyzer.calculate_hrv()
                
                # Export visualizations
                analyzer.plot_results()
                
                print(f"-> Patient {p_id} processed successfully.")
                
            except Exception as e:
                # If anything breaks (like a missing column or math error),
                # it prints the error but doesn't stop the loop
                print(f"FAILED on Patient {p_id}. Error: {e}\n")

        print("Batch Processing Complete.")


    elif test_advanced == True:

        analyzer = Advanced_ECG_Pipeline(patient_number, run_batch=False)

        analyzer.load_data()

        # Swap out the scipy bandpass for the custom FIR filter
        analyzer.custom_fir_filter(window_size=15) 
        analyzer.find_peaks()
        analyzer.calculate_bpm()
        analyzer.calculate_hrv()

        # Run the frequency analyzer
        analyzer.analyze_frequency(inject_noise=inject_60hz_noise) 

        analyzer.plot_results()


    else:

        my_analyzer = ECG_Pipeline(patient_number, run_batch = False)

        # Run the algorithm steps in order
        my_analyzer.load_data()
        my_analyzer.apply_bandpass()
        my_analyzer.find_peaks()
        my_analyzer.calculate_bpm()
        my_analyzer.calculate_hrv()
        my_analyzer.plot_results()
