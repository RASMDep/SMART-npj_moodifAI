import os
import pickle
import pandas as pd
import tqdm
import numpy as np
import pytz


# Define data directories
data_dir = "/home/gdapoian/Ambizione/01_Confidential_Data/SMART_derived_features/"

# Specify the filename for your pickle file
out_dir = data_dir

# Specify % of missing data 

perc_missing = 25

#####################################################
#####################################################
#####################################################

# Constants
ids_p = ['SMART_201', 'SMART_001', 'SMART_003', 'SMART_004', 'SMART_006', 'SMART_008', 'SMART_009', 'SMART_007', 'SMART_010', 
         'SMART_012', 'SMART_015', 'SMART_016', 'SMART_018', 'SMART_019']

def get_three_level_value(value):
    if value <= 2:
        return 0
    elif value <= 4:
        return 1
    else:
        return 2

def nan_percentage(arr):
    nan_count = np.isnan(arr).sum()
    total_elements = arr.size
    return (nan_count / total_elements) * 100

def load_and_resample(data_dir, pid, subfolder, filename, freq, start_time, end_time):
    df = pd.read_csv(os.path.join(data_dir, pid, subfolder, filename))
    df['t_start_utc'] = pd.to_datetime(df['t_start_utc'], utc=True)
    df['t_start_utc'] = df['t_start_utc'].dt.floor(freq)
    df = df.set_index('t_start_utc')  
    resampled_df = df.resample(freq).asfreq()
    return resampled_df[(resampled_df.index >= start_time) & (resampled_df.index < end_time)].reset_index()

def process_participant(data_dir, questionnaire_answers, ids_p, out_dir, perc_missing):
    dataset = {}
    depression_mapping = {pid: 1 if pid in ids_p else 0 for pid in questionnaire_answers['key_smart_id'].unique()}
    
    for pid in tqdm.tqdm(questionnaire_answers['key_smart_id'].unique()):
        if pid in ids_p:
            depression = 1
        else:
            depression = 0

        try:
            iter = 0
            # Load and merge questionnaire files
            quest = questionnaire_answers[questionnaire_answers.key_smart_id == pid]

            for _, row in quest.iterrows():
                quest_time_utc = row['value.time']
                start_time = pd.to_datetime(quest_time_utc, unit='s', utc=True)
                start_time_hours_before = (start_time - pd.Timedelta(hours=24)).replace(tzinfo=pytz.UTC)
                end_time = pd.to_datetime(quest_time_utc, unit='s').replace(tzinfo=pytz.UTC)

                filtered_hrv = load_and_resample(data_dir, pid, 'hrvmetrics', f"{pid}_hrvmetrics_winlen5_overlap_0.csv", '5T', start_time_hours_before, end_time)
                if nan_percentage(filtered_hrv['HRV_MeanNN']) > perc_missing:
                    continue

                filtered_acc = load_and_resample(data_dir, pid, 'accmetrics', f"{pid}_accmetrics_winlen_5.csv", '5T', start_time_hours_before, end_time)
                #filtered_gps = load_and_resample(data_dir, pid, 'gpsmetrics', f"{pid}_gpsmetrics_winlen_60.csv", '60T', start_time_24_hours_before, end_time)

                entry = {
                    'Participant_ID': pid,
                    'Date': quest_time_utc,
                    'HR_Data': 60000 / filtered_hrv['HRV_MeanNN'],
                    'RMSSD_Data': filtered_hrv['HRV_RMSSD'],
                    'SampEn_Data': filtered_hrv['HRV_SampEn'],
                    'activity_counts': filtered_acc['activity_counts'],
                    'cadence': filtered_acc['cadence'],
                    'step_count': filtered_acc['step_count'],
                    'run_walk_time': filtered_acc['run_walk_time'],
                    #'time_home': filtered_gps['time_home'],
                    #"gyration": filtered_gps['gyration'],
                    #"max_loc_home": filtered_gps['max_loc_home'],
                    #"rand_entropy": filtered_gps['rand_entropy'],
                    #"real_entropy": filtered_gps['real_entropy'],
                    #"max_dist": filtered_gps['max_dist'],
                    #"nr_visits": filtered_gps['nr_visits'],
                    #"rand_entropy": filtered_gps['rand_entropy'],
                    'quest_type': row['value.name'],
                    'imq1': row['imq_1.value'],
                    'imq2': 6 - row['imq_2.value'],
                    'imq3': row['imq_3.value'],
                    'imq4': 6 - row['imq_4.value'],
                    'imq5': row['imq_5.value'],
                    'imq6': 6 - row['imq_6.value'],
                    'kss': row['kss.value'],
                    'arousal_class': get_three_level_value((row['imq_1.value'] + (6 - row['imq_4.value'])) / 2),
                    'valence_class': get_three_level_value(((6 - row['imq_2.value']) + row['imq_5.value']) / 2),
                    'arousal_level': (row['imq_1.value'] + (6 - row['imq_4.value'])) / 2,
                    'valence_level': ((6 - row['imq_2.value']) + row['imq_5.value']) / 2,
                    'mood1': row['mood_1.value'],
                    'mood2': row['mood_2.value'],
                    'depression': depression
                }

                dataset[(pid, iter)] = entry
                iter += 1

            # Save the dataset using pickle
            with open(os.path.join(out_dir, "HRV_ACC_timeseries_24hour_clean_" + str(perc_missing) +"percent.pkl"), 'wb') as file:
                pickle.dump(dataset, file)

        except Exception as e:
           print(pid)
           print(f"An error occurred: {e}")
           continue


# Load and merge questionnaire files
questionnaire_files = [
    "morning_questionnaire_allparticipants.csv",
    "afternoon_questionnaire_allparticipants.csv",
    "evening_questionnaire_allparticipants.csv"
]

questionnaire_answers = pd.concat([pd.read_csv(os.path.join(data_dir, "questionnaires", file)) for file in questionnaire_files])

# Process participants
process_participant(data_dir, questionnaire_answers, ids_p, out_dir, perc_missing)
