import sensormotion as sm
import numpy as np
from datetime import datetime, timedelta
from math import floor
import pandas as pd
from sensors.vivalink_patch.accelerometer.feature_extraction.acc_step_counter import *
from datetime import timedelta

def calculate_activity_counts(vm,win_len=60):
    """
    Compute activity counts for ACC data.
    This function takes ACC data with timestamps, and returns the amount of physical activity (counts).

    Args:
        win_len (int): Length of each window in seconds.

    Returns:
        pandas.DataFrame: DataFrame containing activity counts.

    Note:
        The function uses the sensormotion library.
    """
    def divide_list(l,n):
        for i in range(0,len(l),n):
            yield l[i:i+n]

    fs = 25
    size = fs*win_len
    vm_win = list(divide_list(vm,size))
    act_counts = [np.sum(np.abs(win)) for win in vm_win]
    
    return act_counts


class ActigraphyCalculator:

    def __init__(self,acc_data):

        self.acc_data = acc_data.drop_duplicates()
        if not acc_data.empty:
            time_unit = 's' if self.acc_data.iloc[0]['value.time'] < 1e12 else 'ms'
            if  time_unit == 's':
               self.acc_data["value.time"] =  self.acc_data["value.time"]*1000
        
        self.time_unit = time_unit
        t = np.array(acc_data['value.time'])
        self.sampling_rate = int(1000/(t[1]-t[0]))

        self.acc = pd.DataFrame()


    def calculate_actigraphy(self, win_len):
        
        window_length_seconds = win_len * 60
        start_time = pd.to_datetime(self.acc_data.iloc[0]['value.time'], unit='ms', utc = True)
        end_time = pd.to_datetime(self.acc_data.iloc[-1]['value.time'], unit='ms', utc = True)
        # Set minutes and seconds to 0 while keeping the hour intact
        start_time = start_time.replace(minute=0, second=0)
        # compute actual length in seconds
        acc_length_seconds =  int((end_time-start_time).total_seconds()) 

        for win in range(0, acc_length_seconds, window_length_seconds):
            acc_win = self.process_window(win, window_length_seconds, start_time)
            self.acc = pd.concat((self.acc, acc_win))

        self.acc = self.acc.reset_index()

        # Reorder columns to make "t_start" the first column after the index
        desired_order = ['t_start_utc', 'value.time'] + [col for col in self.acc.columns if col not in ['t_start_utc','value.time']]
        self.acc = self.acc[desired_order]

        return self.acc
    
    def process_window(self,win,window_length_seconds,start_time):

        t_start = start_time + timedelta(seconds=win)
        t_end = t_start + timedelta(seconds=window_length_seconds)

        try:

            acc_win = self.acc_data[
                (self.acc_data['value.time'] >= t_start.timestamp() * 1000) &
                (self.acc_data['value.time'] < t_end.timestamp() * 1000)
            ]
            acc_win = acc_win.reset_index()

            t = np.array(acc_win['value.time'])
            x = np.array(acc_win['value.x'])/2048
            y = np.array(acc_win['value.y'])/2048
            z = np.array(acc_win['value.z'])/2048


            ### step count and walking time
            # preprocess data fragment
            t_bout_interp, vm_bout, x, y, z = preprocess_bout(t/1000, x, y, z, fs=25)

            ### TO DO:  activity count 
            act_counts = calculate_activity_counts(vm_bout)

            # find walking and estimate cadence
            cadence_bout = find_walking(vm_bout, fs=25)
            cadence_bout = cadence_bout[np.where(cadence_bout > 0)]

            step_count = int(np.sum(cadence_bout))
            walkingtime = len(cadence_bout[np.where(cadence_bout > 0)])

            #cadence is in step per second
            #walkingtime is in seconds

            act_win = pd.DataFrame({"t_start_utc": t_start,  "value.time": acc_win.iloc[0]["value.time"],
                                    "activity_counts": np.mean(act_counts), "cadence": np.mean(cadence_bout), "step_count": step_count, "run_walk_time": walkingtime}, index=[0])
        except:
            act_win = pd.DataFrame({"t_start_utc": t_start,  "value.time": np.nan,
                                    "activity_counts": np.nan, "cadence": np.nan, "step_count": np.nan, "run_walk_time": np.nan}, index=[0])

        return act_win
    


