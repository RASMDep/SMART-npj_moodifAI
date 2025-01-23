import numpy as np
import pandas as pd
from datetime import timedelta
import neurokit2 as nk
from sensors.vivalink_patch.ecg.preprocessing.ecg_preprocessing import fix_baseline_wander
from datetime import timedelta

# define sensor specific values and predefined thresholds
sampling_rate = 128
gradthreshweight = 2 # used by neurokit peak detector


class HRVCalculator:
    def __init__(self, ecg_data):
        self.ecg_data = ecg_data
        if not ecg_data.empty:
            time_unit = 's' if self.ecg_data.iloc[0]['value.time'] < 1e12 else 'ms'
            if  time_unit == 's':
               self.ecg_data["value.time"] =  self.ecg_data["value.time"]*1000
        
        self.hrv = pd.DataFrame()
        self.status_legend = {
            0: 'Success',
            1: 'Window Length Mismatch',
            2: 'Not Enough Peaks in Window',
            3: 'Peak Detection Failed',
            4: 'Unknown Error',
            5: 'No Data',
        }

    def calculate_hrv(self, win_len, overlap):
        """
        Calculate HRV metrics for segmented ECG data.

        Args:
            win_len (int): Length of each window in minutes (1 to 30).
            overlap (float): Overlap between windows (0, 0.25, 0.5, 0.75).

        Returns:
            pandas.DataFrame: DataFrame containing HRV metrics for each window.
        """
        # Validate input parameters
        if overlap not in [0, 0.25, 0.5, 0.75]:
            raise ValueError("Invalid overlap value. Choose from 0.25, 0.5, or 0.75.")

        if not (1 <= win_len <= 30):
            raise ValueError("Invalid win_len value. Choose a value between 1 and 30 minutes.")
        
        window_length_seconds = win_len * 60
        overlap_seconds = int(window_length_seconds * overlap)
        start_time = pd.to_datetime(self.ecg_data.iloc[0]['value.time'], unit='ms', utc = True)
        # Set minutes and seconds to 0 while keeping the hour intact
        start_time = start_time.replace(minute=0, second=0)
        end_time = pd.to_datetime(self.ecg_data.iloc[-1]['value.time'], unit='ms', utc = True)
        # compute actual length in seconds
        ecg_length_seconds =  int((end_time-start_time).total_seconds()) 

        min_beats = int((window_length_seconds / 2) + 1)
        max_beats = int((window_length_seconds / 0.375) - 1)

        for win in range(0, ecg_length_seconds, window_length_seconds - overlap_seconds):
            hrv_win = self.process_window(win, window_length_seconds, start_time, min_beats, max_beats)
            self.hrv = pd.concat((self.hrv, hrv_win))

        self.hrv = self.hrv.reset_index()


        # Reorder columns to make "t_start" the first column after the index
        desired_order = ['t_start_utc', 'value.time'] + [col for col in self.hrv.columns if col not in ['t_start_utc','value.time']]
        self.hrv = self.hrv[desired_order]

        return self.hrv

    def process_window(self, win, window_length_seconds, start_time, min_beats, max_beats):
        """
        Process a single window of ECG data.

        Args:
            win (int): Current window index.
            window_length_seconds (int): Length of each window in seconds.
            start_time (datetime): Start time of the ECG data.
            min_beats (int): Minimum number of beats for valid processing.
            max_beats (int): Maximum number of beats for valid processing.

        Returns:
            pandas.DataFrame: DataFrame containing HRV metrics for the processed window.
        """
        try:
            t_start = start_time + timedelta(seconds=win)
            t_end = t_start + timedelta(seconds=window_length_seconds-(1/sampling_rate)) 

            ecg_win = self.ecg_data[
                (self.ecg_data['value.time'] >= t_start.timestamp() * 1000) &
                (self.ecg_data['value.time'] <= t_end.timestamp() * 1000)
            ]
            ecg_win = ecg_win.reset_index()

            if ecg_win.shape[0] == 0:
                hrv_win = pd.DataFrame({"t_start_utc": t_start, "value.time": np.nan}, index=[0])
                hrv_win["sqi_avg"] = np.nan
                hrv_win["sqi"] = np.nan
                hrv_win['ProcessingStatus'] = 5
                return hrv_win

            if ecg_win.shape[0] < sampling_rate * window_length_seconds -1 or ecg_win.shape[0] > sampling_rate * window_length_seconds + 1 :
                hrv_win = pd.DataFrame({"t_start_utc": t_start, "value.time": ecg_win.iloc[0]["value.time"]})
                hrv_win["sqi_avg"] = np.nan
                hrv_win["sqi"] = np.nan
                hrv_win['ProcessingStatus'] = 1
            else:
                filtered_ecg = fix_baseline_wander(ecg_win["value.ecg"], fs=sampling_rate)
                try:
                    pks = nk.ecg_findpeaks(filtered_ecg, sampling_rate=sampling_rate, gradthreshweight=gradthreshweight)["ECG_R_Peaks"]
                except:
                    hrv_win = pd.DataFrame({"t_start_utc": t_start, "value.time": ecg_win.iloc[0]["value.time"]}, index=[0])
                    hrv_win["sqi_avg"] = np.nan
                    hrv_win["sqi"] = np.nan
                    hrv_win["pks"] = np.nan
                    hrv_win['ProcessingStatus'] = 3

                    return hrv_win


                sqi_avg = np.mean(nk.ecg_quality(filtered_ecg, rpeaks=pks, sampling_rate=sampling_rate, method='averageQRS', approach=None))
                sqi = nk.ecg_quality(filtered_ecg, rpeaks=pks, sampling_rate=sampling_rate, method='zhao2018', approach=None)

                if min_beats < pks.shape[0] < max_beats:
                    hrv_win = nk.hrv(pks, sampling_rate=sampling_rate, show=False)
                    hrv_win["t_start_utc"] = t_start
                    hrv_win["value.time"] = ecg_win.iloc[0]["value.time"]
                    hrv_win["sqi_avg"] = sqi_avg
                    hrv_win["sqi"] = sqi
                    hrv_win["pks"] = [pks]
                    hrv_win['ProcessingStatus'] = 0
                else:
                    hrv_win = pd.DataFrame({"t_start_utc": t_start,  "value.time": ecg_win.iloc[0]["value.time"],"sqi_avg": sqi_avg, "sqi": sqi, "ProcessingStatus": 2}, index=[0])

            return hrv_win

        except Exception as e:
            hrv_win = pd.DataFrame({"t_start_utc": t_start, "value.time": ecg_win.iloc[0]["value.time"]}, index=[0])
            hrv_win["sqi_avg"] = np.nan
            hrv_win["sqi"] = np.nan
            hrv_win["pks"] = np.nan
            hrv_win['ProcessingStatus'] = 4

            return hrv_win







