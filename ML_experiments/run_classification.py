from dev.developing_suite import DevelopingSuite
import sklearn.metrics as skm
import pickle
import os
import pandas as pd
import argparse

# Directories and files
data_dir = "/home/cgallego/valence_arousal_paper"
save_out_dir = "/home/cgallego/valence_arousal_paper/results_all"
data_file = "HRV_ACC_RR_timeseries_24hour_clean_25percent_classes_041124.pkl"
save_dir = './results'

# Targets and feature combinations
targets = ["valence_class", "arousal_class", "kss_class"]
combinations = ['hr', 'rmssd', 'en', 'act', 'rr', 'hr+rmssd', 'hr+en', 'hr+act', 'hr+rr',
                'hr+rmssd+en', 'hr+act+rr', 'hr+rmssd+en+act', 'hr+rmssd+en+rr', 'hr+rmssd+en+act+rr']

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run experiments for classification.")
parser.add_argument("--mode", type=str, default="all_participants", choices=["all_participants", "loso"],
                    help="Mode of evaluation: all participants or leave-one-subject-out.")
args = parser.parse_args()

# Load dataset
df = pd.DataFrame(pd.read_pickle(os.path.join(data_dir, data_file))).transpose()
subjects = df['Participant_ID'].sort_values().unique()

# Main loop
for target in targets:
    for comb in combinations:
        nchan = str(len(comb.split('+')))
        common_args = [
            "--target=" + target,
            "--mode=train",
            "--n-channels=" + nchan,
            "--data=patch_data",
            "--model=Conv1dNet",
            "--num-class=3",
            "--n-kfold=10",
            "--model-filename=baseline",
            "--save-dir=" + save_dir,
            "--data-path=" + data_dir,
            "--data-file=" + data_file,
            "--n-covariates=2",
            "--save-model=best",
            "--epochs=100",
            "--scheduler=exp",
            "--batch-size=16",
            "--logstep-train=4",
            "--optimizer=adam",
            "--lr=0.001",
        ]

        out_data = {}

        if args.mode == "loso":
            # Leave-One-Subject-Out (LOSO) loop
            for test_subject in subjects:
                print(f"Evaluating LOSO for subject: {test_subject}")
                str_args = common_args + ["--test-subject=" + test_subject]
                parsed_args = parser.parse_args(str_args)
                parsed_args.combs = comb

                developing_suite = DevelopingSuite(parsed_args)
                developing_suite.train_and_eval()

                pid, date, outputs_all, targets_all, y_true, y_scores = developing_suite.eval_model_stats()
                entry = {
                    'pid': pid,
                    'date': date,
                    'outputs': outputs_all.cpu(),
                    'targets': targets_all.cpu(),
                    'y_true': y_true.cpu(),
                    'y_scores': y_scores.cpu()
                }
                out_data[test_subject] = entry

        else:
            # All participants cross-validation loop
            for fold in range(10):
                print(f"Evaluating fold: {fold}")
                str_args = common_args + ["--fold-test=" + str(fold)]
                parsed_args = parser.parse_args(str_args)
                parsed_args.combs = comb

                developing_suite = DevelopingSuite(parsed_args)
                developing_suite.train_and_eval()

                pid, date, outputs_all, targets_all, y_true, y_scores = developing_suite.eval_model_stats()
                entry = {
                    'pid': pid,
                    'date': date,
                    'outputs': outputs_all.cpu(),
                    'targets': targets_all.cpu(),
                    'y_true': y_true.cpu(),
                    'y_scores': y_scores.cpu()
                }
                out_data[fold] = entry

        # Save results
        mode_suffix = "loso" if args.mode == "loso" else "all_participants"
        file_path = os.path.join(save_out_dir, f'outputs_{target}_{comb}_{mode_suffix}.pkl')
        with open(file_path, 'wb') as file:
            pickle.dump(out_data, file)

        print(skm.classification_report(targets_all.cpu(), outputs_all.cpu()))
