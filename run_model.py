from dev.developing_suite import DevelopingSuite
from dev.developing_suite import *
import sklearn.metrics  as skm
from torchmetrics.classification import MulticlassCohenKappa


data_dir="/home/gdapoian/CardioSleep"
save_dir="/home/gdapoian/CardioSleep"
model_dir = './runs'
save_dir = './results'

str_args = [
        "--mode=train",
        "--less-features",
        "--n-channels=6",
        "--data=ecg_data",
        "--model=Conv1dNet",
        "--num-class=3",
        "--n-kfold=5",
        "--test-subject=S008",
        "--per-change=0",
        "--loss=ASL",
        "--model-filename=baseline",
        "--save-dir="+save_dir,
        "--data-path="+data_dir,
        "--gamma-neg=5",
        "--gamma-pos=1",
        "--n-covariates=1",
        "--save-model=best",
        "--mode=train",
        "--num-processes=-1",
        "--epochs=30",
        "--scheduler=exp",  # feature_branch: no
        "--batch-size=10",  # feature_branch: 16
        "--logstep-train=10",
        "--optimizer=adam",
        "--lr=0.001",
        "--experiment-folder=models/out/",
        "--recompute-covariates",
        "--covariates-save-path=covariates/"
    ]
args = parser.parse_args(str_args)

developingSuite=DevelopingSuite(args)
developingSuite.train_and_eval()

noise = 0
print("results for noise:" + str(noise))
outputs_all_arousal1,outputs_all_valence1,_,targets_all_arousal1,targets_all_valence1,_, outputs_all_arousal2,outputs_all_valence2,targets_all_arousal2,targets_all_valence2 = developingSuite.eval_model_stats()


print("arosual")
arosual1 = torch.argmax(torch.sigmoid(targets_all_arousal1), dim=1 ).float()
print(skm.classification_report(arosual1.cpu(),outputs_all_arousal1.cpu()))
print(skm.confusion_matrix(arosual1.cpu(),outputs_all_arousal1.cpu()))

arosual2 = torch.argmax(torch.sigmoid(targets_all_arousal2), dim=1 ).float()
print(skm.classification_report(arosual2.cpu(),outputs_all_arousal2.cpu()))
print(skm.confusion_matrix(arosual2.cpu(),outputs_all_arousal2.cpu()))

#print("valence")
#valence1 = torch.argmax(torch.sigmoid(targets_all_valence1), dim=1 ).float()
#print(skm.classification_report(valence1.cpu(),outputs_all_valence1.cpu()))
#print(skm.confusion_matrix(valence1.cpu(),outputs_all_valence1.cpu()))

#valence2 = torch.argmax(torch.sigmoid(targets_all_valence2), dim=1 ).float()
#print(skm.classification_report(valence2.cpu(),outputs_all_valence2.cpu()))
#print(skm.confusion_matrix(valence2.cpu(),outputs_all_valence2.cpu()))

#print("day part")
#time = torch.argmax(torch.sigmoid(targets_all_daypart), dim=1 ).float()
#print(skm.classification_report(time.cpu(),outputs_all_daypart.cpu()))
#print(skm.confusion_matrix(time.cpu(),outputs_all_daypart.cpu()))

