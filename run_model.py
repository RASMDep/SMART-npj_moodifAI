from dev.developing_suite import DevelopingSuite
from dev.developing_suite import *
import sklearn.metrics  as skm
from torchmetrics.classification import MulticlassCohenKappa


data_dir="/home/gdapoian/Ambizione/01_Confidential_Data/MoodDetection"
save_dir="/home/gdapoian/Ambizione/01_Confidential_Data/MoodDetection"
model_dir = './runs'
save_dir = './results'

str_args = [
        "--mode=train",
        "--n-channels=2",
        "--data=passive_data",
        "--model=Conv1dNet",
        "--num-class=1",
        "--n-kfold=5",
        "--test-subject=SMART_019",
        "--per-change=0",
        "--loss=BCE",
        "--model-filename=baseline",
        "--save-dir="+save_dir,
        "--data-path="+data_dir,
        "--gamma-neg=5",
        "--gamma-pos=1",
        "--n-covariates=1",
        "--save-model=best",
        "--mode=train",
        "--epochs=100",
        "--scheduler=exp",  # feature_branch: no
        "--batch-size=8",  # feature_branch: 16
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

