from dev.developing_suite import DevelopingSuite
from dev.developing_suite import *
import sklearn.metrics  as skm

data_dir="/Users/crisgallego/Documents/phd/00_Data/02_SMART_STUDY"
save_dir="/Users/crisgallego/Documents/phd/00_Data/02_SMART_STUDY"
data_file= "ACC_RR_timeseries_24hour_clean_25percent_tightclasses_291123.pkl"
model_dir = './runs'
save_dir = './results'
target = "valence_class"


# Move the common arguments outside the loop
common_args = [
    "--target=" + target,
    "--mode=train",
    "--n-channels=3",
    "--data=patch_data",
    "--model=Conv1dNet",
    "--num-class=3",
    "--n-kfold=10",
    "--test-subject=none",
    "--model-filename=baseline",
    "--save-dir=" + save_dir,
    "--data-path=" + data_dir,
    "--data-file=" + data_file,
    "--n-covariates=0",
    "--save-model=best",
    "--mode=train",
    "--epochs=30",
    "--scheduler=exp",
    "--batch-size=16",
    "--logstep-train=4",
    "--optimizer=adam",
    "--lr=0.001",
]


# Lists to store results
auroc_scores_list = []
auprc_scores_list = []
cm_normalized_list = []

# Cross-validation loop
for fold in range(0, 10):
    # Modify the fold argument for each iteration
    str_args = common_args + ["--fold-test=" + str(fold)]
    args = parser.parse_args(str_args)


    developingSuite = DevelopingSuite(args)
    developingSuite.train_and_eval()

    noise = 0
    outputs_all, targets_all, y_true, y_scores = developingSuite.eval_model_stats()

    print(skm.classification_report(targets_all.cpu(), outputs_all.cpu()))
    cm = skm.confusion_matrix(outputs_all.cpu(), targets_all.cpu())
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)

    cm_normalized_list.append(cm_normalized)
    y_true = y_true.cpu()
    y_scores = y_scores.cpu()

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

    # Append scores to the lists
    auroc_scores_list.append(auroc_scores)
    auprc_scores_list.append(auprc_scores)

# Convert the lists to NumPy arrays for easier handling
auroc_scores_array = np.array(auroc_scores_list)
auprc_scores_array = np.array(auprc_scores_list)

# Calculate standard deviation across folds
std_auroc = np.nanstd(auroc_scores_array, axis=0)
std_auprc = np.nanstd(auprc_scores_array, axis=0)

# Plotting ROC curves with shaded area for standard deviation
fig, ax = plt.subplots(figsize=(6, 6))
colors = ["aqua", "darkorange", "cornflowerblue"]
target_names = ["low valence", "neutral", "high valence"]

for class_id, color in zip(range(3), colors):
    display = skm.RocCurveDisplay.from_predictions(
        y_true[:, class_id],
        y_scores[:, class_id],
        name=f"ROC curve for {target_names[class_id]}",
        color=color,
        ax=ax,
    )
    if class_id == 2:
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance Level')

    # Plot shaded area for standard deviation
    ax.fill_between(display.fpr, display.tpr - std_auroc[class_id], display.tpr + std_auroc[class_id],
                    color=color, alpha=0.2)

plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
plt.legend()
plt.show()

# Plotting PR curves with shaded area for standard deviation
fig2, ax2 = plt.subplots(figsize=(6, 6))
colors = ["aqua", "darkorange", "cornflowerblue"]

for class_id, color in zip(range(3), colors):
    precision, recall, _ = skm.precision_recall_curve(y_true[:, class_id], y_scores[:, class_id])
    average_precision = skm.average_precision_score(y_true[:, class_id], y_scores[:, class_id])

    ax2.plot(recall, precision, color=color, lw=2,
             label=f"PR curve for {target_names[class_id]} (AUC = {average_precision:.2f})")
    chance_precision = sum(y_true[:, class_id]) / len(y_true[:, class_id])
    ax2.plot([0, 1], [chance_precision, chance_precision], color='navy', lw=2, linestyle='--', label='Chance Level')

    # Plot shaded area for standard deviation
    ax2.fill_between(recall, precision - std_auprc[class_id], precision + std_auprc[class_id],
                     color=color, alpha=0.2)

ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve')
ax2.legend(loc="lower left")
plt.show()
