from dev.developing_suite import DevelopingSuite
from dev.developing_suite import *
import sklearn.metrics  as skm
#from torchmetrics.classification import MulticlassCohenKappa


data_dir="/Users/giulia/Desktop/SMART_derived_features"
save_dir="/Users/giulia/Desktop/SMART_derived_features"
data_file= "HRV_timeseries_24hour_clean_25percent_withpilot.pkl"
model_dir = './runs'
save_dir = './results'
target = "valence_class"


str_args = [
        "--target="+target,
        "--mode=train",
        "--n-channels=1",
        "--data=passive_data",
        "--model=Conv1dNet",
        "--num-class=3",
        "--n-kfold=10",
        "--test-subject=none",
        "--per-change=0",
        "--loss=BCE",
        "--model-filename=baseline",
        "--save-dir="+save_dir,
        "--data-path="+data_dir,
        "--data-file="+data_file,
        "--gamma-neg=5",
        "--gamma-pos=1",
        "--n-covariates=0",
        "--save-model=best",
        "--mode=train",
        "--epochs=30",
        "--scheduler=exp",  # feature_branch: no
        "--batch-size=10",  # feature_branch: 16
        "--logstep-train=4",
        "--optimizer=adam",
        "--lr=0.001",
        "--experiment-folder=models/out/",
    ]
args = parser.parse_args(str_args)

developingSuite=DevelopingSuite(args)
developingSuite.train_and_eval()

noise = 0
outputs_all,targets_all= developingSuite.eval_model_stats()

print(skm.classification_report(targets_all.cpu(),outputs_all.cpu()))
print(skm.confusion_matrix(outputs_all.cpu(),targets_all.cpu()))

