import pickle
import pandas as pd
import os 
from datetime import datetime 
from datetime import timedelta
import gzip 
import numpy as np

quest_dir = "/run/user/1000/gvfs/smb-share:server=hest.nas.ethz.ch,share=green_groups_sms_public/Projects_Current/Ambizione/01_Confidential_Data/read_only/SMART_derived_features/questionnaires"
ecg_dir = "/run/user/1000/gvfs/smb-share:server=hest.nas.ethz.ch,share=green_groups_sms_public/Projects_Current/Ambizione/01_Confidential_Data/read_only/SMART_STUDY/"

# Define the function to load and merge ECG data
def load_and_merge_ecg_data(pid, quest_time_utc):
    # Convert quest_time_utc to a datetime object
    quest_time = datetime.utcfromtimestamp(quest_time_utc)

    # Create an empty DataFrame to store the merged data
    merged_df = pd.DataFrame()

    # Loop through the last 24 hours before quest_time
    for i in range(24):
        # Calculate the date and hour to load
        current_time = quest_time - timedelta(hours=i)
        date_str = current_time.strftime("%Y%m%d")
        hour_str = current_time.strftime("%H00")
        
        # Build the file path
        file_path = os.path.join(ecg_dir,pid, "vivalnk_vv330_ecg", date_str,date_str+"_"+hour_str+".csv.gz")

        # Check if the file exists
        if os.path.exists(file_path):
            # Load the CSV data from the gzip file
            with gzip.open(file_path, 'rt') as file:
                ecg_df = pd.read_csv(file)
            
            # Append the loaded data to the merged DataFrame
            merged_df = pd.concat([merged_df, ecg_df], ignore_index=True)
            merged_df = merged_df.sort_values(by="value.time").reset_index()

        # Filter the DataFrame to include only rows within the specified time range
        filtered_df = merged_df[(merged_df['value.time'] > quest_time_utc-(3600*24)*1000) & (merged_df['value.time'] <= quest_time_utc*1000)]

        # Define the length of the vector
        vector_length = 24 * 128 * 3600

        # Create an empty numpy array filled with NaN values
        result_vector = np.full(vector_length, np.nan)

        # Iterate through the rows of the DataFrame and populate the vector
        for index, row in filtered_df.iterrows():
            # Calculate the index in the vector corresponding to the timestamp
            timestamp = row['value.time']
            index_in_vector = int((timestamp - filtered_df['value.time'].min())/1000 * 128)

            # Update the value in the result_vector with "value.ecg"
            result_vector[index_in_vector] = row['value.ecg']


    
    return merged_df




#load and merge questionnairs files
questionnaire_answers = pd.concat((pd.read_csv(os.path.join(quest_dir,"morning_questionnaire_allparticipants.csv")),
                                   pd.read_csv(os.path.join(quest_dir,"afternoon_questionnaire_allparticipants.csv")),
                                   pd.read_csv(os.path.join(quest_dir,"evening_questionnaire_allparticipants.csv"))
                                   ))


for _,row in  questionnaire_answers.iterrows():

    pid =  row["key_smart_id"]    
    quest_time_utc = row["value.time"]

    merged_ecg_data = load_and_merge_ecg_data(pid, quest_time_utc)

    # Create a Python dictionary to store your data
    dataset = {
        'Participant_ID': participant_ids,
        'Date': dates,
        'ECG_Data': ecg_data,
        'ACC_Data': acc_data,
        'Outcome_Label_1': outcome_label_1,
        'Outcome_Label_2': outcome_label_2,
        'Outcome_Label_3': outcome_label_3,
        'Outcome_Label_4': outcome_label_4
    }

    # Specify the filename for your pickle file
    pickle_filename = 'your_dataset.pkl'

    # Save the dataset using pickle
    with open(pickle_filename, 'wb') as file:
        pickle.dump(dataset, file)