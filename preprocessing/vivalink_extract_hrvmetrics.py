import os
import pandas as pd
import argparse
from sensors.data_loading.data_loader import load_sensor_data
from sensors.vivalink_patch.ecg.feature_extraction.ecg_hrv_feature_extraction import HRVCalculator
from tqdm import tqdm  # Import tqdm for progress bars

# Define command-line arguments
parser = argparse.ArgumentParser(description="HRV Calculation Script")
parser.add_argument("--win_len", type=int, default=5, help="Specify the window length in minutes for HRV calculation")
parser.add_argument("--overlap", type=int, default=0, help="Specify the percentage overlap for HRV calculation")
parser.add_argument("--data_path", help="Specify the data path")
parser.add_argument("--output_dir",  help="Specify the output directory")
parser.add_argument("--participant_id", help="Specify the participant ID")
args = parser.parse_args()


topic = "vivalnk_vv330_ecg"

def process_ecg_data(ecg_data, win_len, overlap):
    hrv_calculator = HRVCalculator(ecg_data=ecg_data)
    hrvmetrics = hrv_calculator.calculate_hrv(win_len=win_len, overlap=overlap)
    return hrvmetrics

# Load ECG data for a specific participant
subject_dir = os.path.join(args.data_path, args.participant_id)
out_file_name = args.participant_id + "_hrvmetrics_winlen" + str(args.win_len) + "_overlap_" + str(args.overlap) + ".csv"


hrv_participant = []  # Initialize an empty list for the current day's HRV metrics

# Work on a daily basis
for day in tqdm(os.listdir(os.path.join(subject_dir, topic)), desc="Processing Days"):
    day_dir = os.path.join(subject_dir, topic, day)
    if not os.path.isdir(day_dir):
        continue
    # Filter files to only consider those that start with '2' and end with '.csv.gz'
    file_names = [filename for filename in os.listdir(day_dir) if filename.startswith('2') and filename.endswith('.csv.gz')]
    file_names.sort()
    if file_names:
        # Work on an hourly basis
        for file_name in tqdm(file_names, desc="Processing Hours", leave=False):
            hrv_metrics_hour = pd.DataFrame()
            start_time = file_name.split('.')[0]

            try:
                # Load ECG data for the current hour
                ecg_data = load_sensor_data(subject_dir, topic, start_time)

                # Process and calculate HRV metrics
                hrv_metrics_hour = process_ecg_data(ecg_data, args.win_len, args.overlap)

                # Append the HRV metrics to the current day's list
                hrv_participant.append(hrv_metrics_hour)
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

        # Create a single DataFrame from the accumulated HRV metrics for the day
        if hrv_participant:
            df_hrv_metrics_day = pd.concat(hrv_participant, ignore_index=True)
        else:
            df_hrv_metrics_day = pd.DataFrame()

        # Save the DataFrame as a CSV file (overwrite if it already exists)
        output_csv_path = os.path.join(args.output_dir, args.participant_id, "hrvmetrics", out_file_name)
        # Create the output directory if it doesn't exist
        if not os.path.exists(os.path.join(args.output_dir, args.participant_id, "hrvmetrics")):
            os.makedirs(os.path.join(args.output_dir, args.participant_id, "hrvmetrics"))

        df_hrv_metrics_day = df_hrv_metrics_day.sort_values(by='t_start_utc')
        df_hrv_metrics_day.to_csv(output_csv_path, index=False)
    
