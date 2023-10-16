import os
import pickle
import pandas as pd
import tqdm


def get_three_level_value(value):
    if value <= 2.5:
        return 0
    elif value <= 4.5:
        return 1
    else:
        return 2
    

# define sensor specific values and predefined thresholds
sampling_rate = 128
gradthreshweight = 2 # used by neurokit peak detector
window_size_samples = sampling_rate * 360
pks_mindelay =0.3

# Define data directories
#data_dir = "/cluster/work/smslab/ambizione/SMART_derived_features/"

#data_dir = "/run/user/1000/gvfs/smb-share:server=hest.nas.ethz.ch,share=green_groups_sms_public/Projects_Current/Ambizione/01_Confidential_Data/read_only/SMART_derived_features/"
data_dir = "/Volumes/green_groups_sms_public/Projects_Current/Ambizione/01_Confidential_Data/read_only/SMART_derived_features/"

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
dataset = pd.DataFrame()

for pid in questionnaire_answers.key_smart_id.unique():
    try:

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

        # load the gps features files
        gps = pd.read_csv(os.path.join(data_dir,pid,"gpsmetrics", pid+"_gpsmetrics_winlen_60.csv" ))
        gps['t_start_utc'] = pd.to_datetime(gps['t_start_utc'], utc=True)  
        gps['t_start_utc'] = gps['t_start_utc'].dt.floor('H')
        gps = gps.set_index('t_start_utc')
        resampled_gps = gps.resample('60T').asfreq()

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

            filtered_acc = resampled_acc[(resampled_acc.index >= start_time_24_hours_before) & (resampled_acc.index  < pd.to_datetime(start_time, unit='s'))]
            filtered_acc = filtered_acc.reset_index()

            filtered_gps = resampled_gps[(resampled_gps.index >= start_time_24_hours_before) & (resampled_gps.index  < pd.to_datetime(start_time, unit='s'))]
            filtered_gps = filtered_gps.reset_index()

            if filtered_hrv is not None and filtered_acc is not None:
                entry = {
                    'Participant_ID': pid,
                    'Date': quest_time_utc,
                    'HR_Data': 60000/filtered_hrv["HRV_MeanNN"],
                    'RMSSD_Data': filtered_hrv["HRV_RMSSD"],
                    'SampEn_Data': filtered_hrv["HRV_SampEn"],
                    'activity_counts ': filtered_acc["activity_counts"],
                    'cadence': filtered_acc["cadence"],
                    'step_count': filtered_acc["step_count"],
                    'run_walk_time': filtered_acc["run_walk_time"],
                    'time_home': filtered_gps["time_home"],
                    "gyration": filtered_gps["gyration"],
                    "max_loc_home": filtered_gps["max_loc_home"],
                    "rand_entropy": filtered_gps["rand_entropy"],
                    "real_entropy": filtered_gps["real_entropy"],
                    "max_dist": filtered_gps["max_dist"],
                    "nr_visits": filtered_gps["nr_visits"],
                    "rand_entropy": filtered_gps["rand_entropy"],
                    'quest_type': row["value.name"],
                    'imq1': row["imq_1.value"],
                    'imq2': 6-row["imq_2.value"],
                    'imq3': row["imq_3.value"],
                    'imq4': 6-row["imq_4.value"],
                    'imq5': row["imq_5.value"],
                    'imq6': 6-row["imq_6.value"],
                    'kss': row["kss.value"],
                    'arousal_level': get_three_level_value((row["imq_1.value"] + (6 - row["imq_4.value"])) / 2),
                    'valence_level': get_three_level_value((6-row["imq_2.value"]) + row["imq_5.value"]) / 2,
                }

                dataset = pd.concat((dataset,pd.DataFrame(entry)),ignore_index=True)

    except Exception as e:
        print(f"An error occurred: {e}")
        continue


    # Save the dataset using pickle
    with open(os.path.join(out_dir, "HRV_ACC_GPS_timeseries_24hours.pkl"), 'wb') as file:
        pickle.dump(dataset, file)
