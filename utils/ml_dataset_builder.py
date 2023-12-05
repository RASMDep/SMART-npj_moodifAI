import os
import pickle
import pandas as pd
import tqdm
import numpy as np
import pytz


class DataProcessor:
    def __init__(self, data_dir, out_dir, perc_missing, ids_p):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.perc_missing = perc_missing
        self.ids_p = ids_p

    @staticmethod
    def get_three_level_value(value):
        if value <= 2.5:
            return 0
        elif value <= 3.5:
            return 1
        else:
            return 2
    
    @staticmethod
    def get_three_kss_value(value):
        if value <= 3:
            return 0
        elif value <= 6:
            return 1
        else:
            return 2

    @staticmethod
    def nan_percentage(arr):
        nan_count = np.isnan(arr).sum()
        total_elements = arr.size
        if total_elements==0:
            return 100
        else:
            return (nan_count / total_elements) * 100

    @staticmethod
    def load_and_resample(data_dir, pid, subfolder, filename, freq, start_time, end_time):
        
        df = pd.read_csv(os.path.join(data_dir, pid, subfolder, filename))
        df['t_start_utc'] = pd.to_datetime(df['t_start_utc'], utc=True,format='ISO8601')
        df['t_start_utc'] = df['t_start_utc'].dt.floor(freq)
        df = df.set_index('t_start_utc')
        resampled_df = df.resample(freq).asfreq()
        return resampled_df[(resampled_df.index >= start_time) & (resampled_df.index < end_time)].reset_index()
        

    def process_participant(self, questionnaire_answers, include_hrv = False,include_acc=False, include_gps=False,include_rr=False):
        dataset = {}
        data_used = []
        if include_hrv:
            data_used.append("HRV")
        if include_acc:
            data_used.append('ACC')
        if include_gps:
            data_used.append('GPS')
        if include_rr:
            data_used.append('RR')

        for pid in tqdm.tqdm(questionnaire_answers['key_smart_id'].unique()):
            if pid in self.ids_p:
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

                    filtered_hrv = self.load_and_resample(
                        self.data_dir, pid, 'hrvmetrics', f"{pid}_hrvmetrics_winlen5_overlap_0.csv", '5T',
                        start_time_hours_before, end_time
                    )

                    if self.nan_percentage(filtered_hrv['HRV_MeanNN']) > self.perc_missing:
                        continue
                
                    # Conditionally load and resample acc and gps
                    if include_acc:
                        filtered_acc = self.load_and_resample(
                            self.data_dir, pid, 'accmetrics', f"{pid}_accmetrics_winlen_5.csv", '5T',
                            start_time_hours_before, end_time
                        )

                    if include_gps:
                        filtered_gps = self.load_and_resample(
                            self.data_dir, pid, 'gpsmetrics', f"{pid}_gpsmetrics_winlen_60.csv", '60T',
                            start_time_hours_before, end_time
                        )
                    
                    if include_rr:
                        filtered_rr = self.load_and_resample(
                            self.data_dir, pid, 'rrmetrics', f"{pid}_rrmetrics_winlen_5.csv", '5T',
                            start_time_hours_before, end_time
                        )

                    entry = {
                        'Participant_ID': pid,
                        'Date': quest_time_utc,
                        'Minute': filtered_hrv.t_start_utc.dt.hour * 60 + filtered_hrv.t_start_utc.dt.minute,
                        'HR_Data': 60000 / filtered_hrv['HRV_MeanNN'],
                        'RMSSD_Data': filtered_hrv['HRV_RMSSD'],
                        'SampEn_Data': filtered_hrv['HRV_SampEn'],
                        'quest_type': row['value.name'],
                        'imq1': row['imq_1.value'],
                        'imq2': 6 - row['imq_2.value'],
                        'imq3': row['imq_3.value'],
                        'imq4': 6 - row['imq_4.value'],
                        'imq5': row['imq_5.value'],
                        'imq6': 6 - row['imq_6.value'],
                        'kss': row['kss.value'],
                        'kss_class': self.get_three_kss_value(row['kss.value']),
                        'arousal_class': self.get_three_level_value((row['imq_1.value'] + (6 - row['imq_4.value'])) / 2),
                        'valence_class': self.get_three_level_value(((6 - row['imq_2.value']) + row['imq_5.value']) / 2),
                        'arousal_level': (row['imq_1.value'] + (6 - row['imq_4.value'])) / 2,
                        'valence_level': ((6 - row['imq_2.value']) + row['imq_5.value']) / 2,
                        'mood1': row['mood_1.value'],
                        'mood2': row['mood_2.value'],
                        'depression': depression
                    }


                    # Add acc and gps data if conditions are met
                    if include_acc:
                        entry['activity_counts'] = filtered_acc['activity_counts']
                        entry['cadence'] = filtered_acc['cadence']
                        entry['step_count'] = filtered_acc['step_count']
                        entry['run_walk_time'] = filtered_acc['run_walk_time']

                    if include_gps:
                        entry['time_home'] = filtered_gps['time_home']
                        entry['gyration'] = filtered_gps['gyration']
                        entry['max_loc_home'] = filtered_gps['max_loc_home']
                        entry['rand_entropy'] = filtered_gps['rand_entropy']
                        entry['real_entropy'] = filtered_gps['real_entropy']
                        entry['max_dist'] = filtered_gps['max_dist']
                        entry['nr_visits'] = filtered_gps['nr_visits']
                        entry['rand_entropy'] = filtered_gps['rand_entropy']
                    
                    if include_rr:
                        entry['resp_rate'] = filtered_rr['resp_rate']


                    dataset[(pid, iter)] = entry
                    iter += 1

                # Save the dataset using pickle
                filename = f"{'_'.join(data_used)}_timeseries_24hour_clean_{self.perc_missing}percent_tightclasses_291123.pkl"

                with open(os.path.join(self.out_dir, filename), 'wb') as file:
                    pickle.dump(dataset, file)

            except Exception as e:
                print(pid)
                print(f"An error occurred: {e}")
                continue


def main(include_pilot_data=True, include_hrv =False, include_acc=False, include_gps=False, include_rr=False):
    # Define data directories
    data_dir = "/Volumes/green_groups_sms_public/Projects_Current/Ambizione/01_Confidential_Data/read_only/SMART_derived_features"
    out_dir = "/Users/crisgallego/Documents/phd/00_Data/02_SMART_STUDY"
    perc_missing = 25
    ids_p = ['SMART_201', 'SMART_003', 'SMART_004','SMART_006', 'SMART_007','SMART_008', 'SMART_009','SMART_010', 
            'SMART_012','SMART_015', 'SMART_016','SMART_018','SMART_024',
            'SMART_019','SMART_020','SMART_027']

    # Specify questionnaire files
    questionnaire_files = [
        "morning_questionnaire_allparticipants.csv",
        "afternoon_questionnaire_allparticipants.csv",
        "evening_questionnaire_allparticipants.csv",
    ]

    if include_pilot_data:
        questionnaire_files += [
            "morning_questionnaire_allparticipants_pilot.csv",
            "afternoon_questionnaire_allparticipants_pilot.csv",
            "evening_questionnaire_allparticipants_pilot.csv",
        ]

    # Load and merge questionnaire files
    questionnaire_answers = pd.concat([pd.read_csv(os.path.join(data_dir, "questionnaires", file)) for file in questionnaire_files])

    # Create an instance of the DataProcessor class
    data_processor = DataProcessor(data_dir, out_dir, perc_missing, ids_p)

    # Process participants
    data_processor.process_participant(questionnaire_answers,include_acc=include_acc, include_gps=include_gps,include_rr=include_rr)


# Example usage
main(include_pilot_data=False,include_hrv=True,include_acc=True, include_gps=False,include_rr=True)  # Set to True if you want to include pilot data