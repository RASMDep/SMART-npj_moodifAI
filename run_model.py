from dev.developing_suite import DevelopingSuite
from dev.developing_suite import *
import sklearn.metrics  as skm

data_dir="/run/user/1000/gvfs/smb-share:server=hest.nas.ethz.ch,share=green_groups_sms_public/Projects_Current/Ambizione/10_Studies/2_SMART/Data/valence-arousal-paper"
save_dir="/run/user/1000/gvfs/smb-share:server=hest.nas.ethz.ch,share=green_groups_sms_public/Projects_Current/Ambizione/10_Studies/2_SMART/Data/valence-arousal-paper"
data_file= "HRV_ACC_timeseries_24hour_clean_25percent_smart.pkl"
model_dir = './runs'
save_dir = './results'
target = "valence_class"


str_args = [
        "--target="+target,
        "--mode=train",
        "--n-channels=2",
        "--data=patch_data",
        "--model=Conv1dNet",
        "--num-class=3",
        "--n-kfold=10",
        "--fold-test=2",
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
        "--batch-size=32",  # feature_branch: 16
        "--logstep-train=4",
        "--optimizer=adam",
        "--lr=0.001",
        "--experiment-folder=models/out/",
    ]
args = parser.parse_args(str_args)

developingSuite=DevelopingSuite(args)
developingSuite.train_and_eval()

noise = 0
outputs_all,targets_all,y_true,y_scores = developingSuite.eval_model_stats()

print(skm.classification_report(targets_all.cpu(),outputs_all.cpu()))
print(skm.confusion_matrix(outputs_all.cpu(),targets_all.cpu()))

y_true = y_true.cpu()
y_scores = y_scores.cpu()


import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, average_precision_score, precision_recall_curve

fig, ax = plt.subplots(figsize=(6, 6))
colors = ["aqua", "darkorange", "cornflowerblue"]
target_names = ["low valence", "neutral", "high valence"]
for class_id, color in zip(range(3), colors):
    RocCurveDisplay.from_predictions(
        y_true[:, class_id],
        y_scores[:, class_id],
        name=f"ROC curve for {target_names[class_id]}",
        color=color,
        ax=ax,
    )
    if class_id == 2:
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance Level')


plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
plt.legend()
plt.show()


fig2, ax2 = plt.subplots(figsize=(6, 6))
colors = ["aqua", "darkorange", "cornflowerblue"]
target_names = ["low valence", "neutral", "high valence"]

for class_id, color in zip(range(3), colors):
    precision, recall, _ = precision_recall_curve(y_true[:, class_id], y_scores[:, class_id])
    average_precision = average_precision_score(y_true[:, class_id], y_scores[:, class_id])

    ax2.plot(recall, precision, color=color, lw=2,
            label=f"PR curve for {target_names[class_id]} (AUC = {average_precision:.2f})")
    chance_precision = sum(y_true[:, class_id]) / len(y_true[:, class_id])
    ax2.plot([0, 1], [chance_precision, chance_precision], color='navy', lw=2, linestyle='--', label='Chance Level')


ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve')
ax2.legend(loc="lower left")
plt.show()

# Calculate AUROC for each class
auroc_scores = []
for i in range(y_true.shape[1]):
    try:
        auroc = skm.roc_auc_score(y_true[:, i], y_scores[:, i])
        auroc_scores.append(auroc)
    except:
        auroc_scores.append(np.nan)

# Calculate AUPRC for each class
auprc_scores = []
for i in range(y_true.shape[1]):
    try:
        auprc = skm.average_precision_score(y_true[:, i], y_scores[:, i])
        auprc_scores.append(auprc)
    except:
        auprc_scores.append(np.nan)

# Print or use the results as needed
print("AUROC Scores:", auroc_scores)
print("AUPRC Scores:", auprc_scores)

