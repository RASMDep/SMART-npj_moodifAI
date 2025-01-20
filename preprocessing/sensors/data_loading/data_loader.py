import os
import pandas as pd
from datetime import datetime, timedelta
import glob
from os.path import join as pjoin
import numpy as np

def load_sensor_data(subject_dir, topic, start_time, end_time=None):
    """
    Load data for a specified range of hours or a single hour.

    Parameters:
    - subject_dir (str): Path to the subject's directory.
    - topic (str): e.g., vivalnk_vv330_ecg for ecg data, works for phone sensors data as well
    - start_time (str): Start time in the format 'YYYYMMDD_HH00'.
    - end_time (str, optional): End time in the format 'YYYYMMDD_HH00'. If not specified, loads data for a single hour.

    Returns:
    - data (pd.DataFrame): Concatenated sensor data for the specified range of hours or a single hour.
    """

    data = []

    current_time = datetime.strptime(start_time, '%Y%m%d_%H%M')
    

    if end_time is None:
        end_time = current_time + timedelta(hours=1)
    else:
        end_time = datetime.strptime(end_time, '%Y%m%d_%H%M')

    while current_time < end_time:
        file_path = os.path.join(subject_dir, topic , current_time.strftime('%Y%m%d'), current_time.strftime('%Y%m%d_%H00.csv.gz'))
        
        if os.path.exists(file_path):
            data_hour = pd.read_csv(file_path, compression='gzip')
            data.append(data_hour)

        current_time += timedelta(hours=1)

    if data:
        data = pd.concat(data, ignore_index=True)
        data.sort_values('value.time')
        return data
    else:
        return None

def load_sensor_data_day(subject_dir,topic):
    """
    Load all data for a subject and a day.

    Parameters:
    - subject_dir (str): Path to the subject's directory.
    - topic (str): e.g., vivalnk_vv330_ecg for ecg data, works for phone sensors data as well

    Returns:
    - data (pd.DataFrame): Concatenated sensor data for the specified subject, for the specified date.
    """
    
    data = []
    for f in glob.glob(pjoin(subject_dir, topic,'*.csv.gz')):
        try:
            data.append(pd.read_csv(f, compression= 'gzip'))
        except:
            print('exception')
            continue

    if data:
        data = pd.concat(data, ignore_index=True)
        data.sort_values('value.time')
        return data
    else:
        return None


def load_sensor_data_all(subject_dir, topic):
    """
    Load all data for a subject.

    Parameters:
    - subject_dir (str): Path to the subject's directory.
    - topic (str): e.g., vivalnk_vv330_ecg for ecg data, works for phone sensors data as well

    Returns:
    - data (pd.DataFrame): Concatenated sensor data for the specified subject.
    """

    data = []
    for d in glob.glob(pjoin(subject_dir,topic, '20*')):
        for f in glob.glob(pjoin(d, '*.csv.gz')):
            try:
                data.append(pd.read_csv(f, compression= 'gzip'))
            except:
                print('exception')
                continue
    
    if data:
        data = pd.concat(data, ignore_index=True)
        data.sort_values('value.time')
        return data
    else:
        return None
