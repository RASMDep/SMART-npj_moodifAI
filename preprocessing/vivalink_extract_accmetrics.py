import os
import pandas as pd
import argparse
from sensors.data_loading.data_loader import load_sensor_data
from sensors.vivalink_patch.accelerometer.feature_extraction.acc_feature_extraction import ActigraphyCalculator
from tqdm import tqdm  # Import tqdm for progress bars

# Define command-line arguments
parser = argparse.ArgumentParser(description="ACC Calculation Script")
parser.add_argument("--win_len", type=int, default=5, help="Specify the window length in minutes for ACC calculation")
parser.add_argument("--data_path", default="/cluster/work/smslab/ambizione/SMART_STUDY/", help="Specify the data path")
parser.add_argument("--output_dir", default="/cluster/work/smslab/ambizione/SMART_derived_features/", help="Specify the output directory")
parser.add_argument("--participant_id", default="SMART_019", help="Specify the participant ID")
args = parser.parse_args()


topic = "vivalnk_vv330_acceleration"

def process_acc_data(acc_data, win_len):
    acc_calculator = ActigraphyCalculator(acc_data)
    accmetrics = acc_calculator.calculate_actigraphy(win_len=win_len)
    return accmetrics

# Load ACC data for a specific participant
subject_dir = os.path.join(args.data_path, args.participant_id)
out_file_name = args.participant_id + "_accmetrics_winlen_" + str(args.win_len) + ".csv"


acc_participant = []  # Initialize an empty list for the current day's ACC metrics

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
            acc_metrics_hour = pd.DataFrame()
            start_time = file_name.split('.')[0]

            try:
                # Load ACC data for the current hour
                acc_data = load_sensor_data(subject_dir, topic, start_time)

                # Process and calculate ACC metrics
                acc_metrics_hour = process_acc_data(acc_data, args.win_len)

                # Append the ACC metrics to the current day's list
                acc_participant.append(acc_metrics_hour)
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

        # Create a single DataFrame from the accumulated ACC metrics for the day
        if acc_participant:
            df_acc_metrics_day = pd.concat(acc_participant, ignore_index=True)
        else:
            df_acc_metrics_day = pd.DataFrame()

        # Save the DataFrame as a CSV file (overwrite if it already exists)
        output_csv_path = os.path.join(args.output_dir, args.participant_id, "accmetrics", out_file_name)
        # Create the output directory if it doesn't exist
        if not os.path.exists(os.path.join(args.output_dir, args.participant_id, "accmetrics")):
            os.makedirs(os.path.join(args.output_dir, args.participant_id, "accmetrics"))
        df_acc_metrics_day = df_acc_metrics_day.sort_values(by='t_start_utc')  # Sort by 't_start_utc'    
        df_acc_metrics_day.to_csv(output_csv_path, index=False)
