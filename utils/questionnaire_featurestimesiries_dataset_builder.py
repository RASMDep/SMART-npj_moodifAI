import os
import pickle
import pandas as pd
import tqdm
import numpy as np
ids_p = ['SMART_201', 'SMART_001', 'SMART_003', 'SMART_004', 'SMART_006', 'SMART_008', 'SMART_009', 'SMART_007', 'SMART_010', 'SMART_012', 'SMART_015', 'SMART_016', 'SMART_018', 'SMART_019']

def get_three_level_value(value):
    if value <= 2:
        return 0
    elif value <= 4:
        return 1
    else:
        return 2
    
def nan_percentage(arr):
        
    # Count the number of NaN values in the array
    nan_count = np.isnan(arr).sum()
    # Calculate the total number of elements in the array
    total_elements = arr.size
    # Calculate the percentage of NaN values
    percentage_nan = (nan_count / total_elements) * 100

    return percentage_nan


# define sensor specific values and predefined thresholds
sampling_rate = 128
gradthreshweight = 2 # used by neurokit peak detector
window_size_samples = sampling_rate * 360
pks_mindelay =0.3

# Define data directories
#data_dir = "/cluster/work/smslab/ambizione/SMART_derived_features/"

data_dir = "/run/user/1000/gvfs/smb-share:server=hest.nas.ethz.ch,share=green_groups_sms_public/Projects_Current/Ambizione/01_Confidential_Data/read_only/SMART_derived_features/"

# Specify the filename for your pickle file
out_dir = data_dir#'/cluster/work/smslab/ambizione/SMART_ML_ECG_Dataset'


# Load and merge questionnaire files
questionnaire_files = [
    "morning_questionnaire_allparticipants.csv",
    "afternoon_questionnaire_allparticipants.csv",
    "evening_questionnaire_allparticipants.csv"
]


questionnaire_answers = pd.concat([pd.read_csv(os.path.join(data_dir, "questionnaires", file)) for file in questionnaire_files])

# Initialize an empty list to store multiple entries
dataset = {}


for pid in questionnaire_answers.key_smart_id.unique():

    if pid in ids_p:
        depression = 1
    else:
        depression = 0 
    try:

        iter = 0

        # load the hrv and acc features files
        hrv = pd.read_csv(os.path.join(data_dir, pid, "hrvmetrics", pid+"_hrvmetrics_winlen5_overlap_0.csv" ))
        hrv['t_start_utc'] = pd.to_datetime(hrv['t_start_utc'], utc=True)  
        hrv['t_start_utc'] = hrv['t_start_utc'].dt.floor('S')
        hrv = hrv.set_index('t_start_utc')
        resampled_hrv = hrv.resample('5T').asfreq()
    

        acc = pd.read_csv(os.path.join(data_dir, pid, "accmetrics", pid+"_accmetrics_winlen_5.csv" ))
        acc['t_start_utc'] = pd.to_datetime(acc['t_start_utc'], utc=True)  
        acc['t_start_utc'] = acc['t_start_utc'].dt.floor('S')
        acc = acc.set_index('t_start_utc')
        resampled_acc = acc.resample('5T').asfreq()
        # get questionnaire dataframe for specific participant
        quest = questionnaire_answers[questionnaire_answers.key_smart_id==pid]

        for _, row in tqdm.tqdm(quest.iterrows()):
            pid = row["key_smart_id"]
            quest_time_utc = row["value.time"]

            start_time =  pd.to_datetime(quest_time_utc, unit='s', utc=True) 
            start_time_24_hours_before = start_time - pd.Timedelta(hours=24)
            # Filter the DataFrame
            filtered_hrv = resampled_hrv[(resampled_hrv.index >= start_time_24_hours_before) & (resampled_hrv.index  < pd.to_datetime(start_time, unit='s'))]
            filtered_hrv = filtered_hrv.reset_index()

            if nan_percentage(filtered_hrv["HRV_MeanNN"])>30:
                continue
            
            filtered_acc = resampled_acc[(resampled_acc.index >= start_time_24_hours_before) & (resampled_acc.index  < pd.to_datetime(start_time, unit='s'))]
            filtered_acc = filtered_acc.reset_index()

            if filtered_hrv is not None and filtered_acc is not None:
                entry = {
                    'Participant_ID': pid,
                    'Date': quest_time_utc,
                    'HR_Data': 60000/filtered_hrv["HRV_MeanNN"],
                    'RMSSD_Data': filtered_hrv["HRV_RMSSD"],
                    'SampEn_Data': filtered_hrv["HRV_SampEn"],
                    'activity_counts': filtered_acc["activity_counts"],
                    'cadence': filtered_acc["cadence"],
                    'step_count': filtered_acc["step_count"],
                    'run_walk_time': filtered_acc["run_walk_time"],
                    'quest_type': row["value.name"],
                    'imq1': row["imq_1.value"],
                    'imq2': 6-row["imq_2.value"],
                    'imq3': row["imq_3.value"],
                    'imq4': 6-row["imq_4.value"],
                    'imq5': row["imq_5.value"],
                    'imq6': 6-row["imq_6.value"],
                    'kss': row["kss.value"],
                    'arousal_level': get_three_level_value((row["imq_1.value"] + (6 - row["imq_4.value"])) / 2),
                    'valence_level': get_three_level_value(((6-row["imq_2.value"]) + row["imq_5.value"]) / 2),
                    'depression': depression
                }

                dataset[pid,iter] = entry  # Add the 'entry' dictionary to the list
                iter +=1

        # Save the dataset using pickle
        with open(os.path.join(out_dir, "HRV_ACC_timeseries_24hours_clean.pkl"), 'wb') as file:
            pickle.dump(dataset, file)



    except Exception as e:
        print(f"An error occurred: {e}")
        continue

