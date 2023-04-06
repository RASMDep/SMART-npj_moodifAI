import os
from os.path import join as pjoin
from fnmatch import fnmatch
import pandas as pd
from datetime import datetime as dt
import datetime
import numpy as np
from visbrain.io.rw_hypno import read_hypno
import warnings
warnings.filterwarnings("ignore")
import tqdm

root = "/run/user/1000/gvfs/smb-share:server=hest.nas.ethz.ch,share=green_groups_sms_public/Projects_Current/Ambizione/01_Confidential_Data/read_only/ETH_RASMDEP_PILOT"
save_to = "/run/user/1000/gvfs/smb-share:server=hest.nas.ethz.ch,share=green_groups_sms_public/Projects_Current/Ambizione/01_Confidential_Data/MoodDetection"
topic_ecg = "vivalnk_vv330_ecg"
topic_scoring = "sleeploop_scoring"
topic_acc =  "vivalnk_vv330_acceleration"
subjects = [s for s in os.listdir(root) if not s.startswith('.')]

def load_ecg_night_data(start_date, before, after, subject):

    ecg_df = pd.DataFrame()

    for h in range(-before,after+1):
        # load all csv files from ecg starting from start_date and for the next 10 hours
        new_date = start_date + datetime.timedelta(hours=h)
        date = dt.strftime(new_date, "%Y%m%d")
        ecg_fname =  dt.strftime(new_date, "%Y%m%d_%H%M")+'.csv.gz'
        ecg_df = pd.concat( (ecg_df, pd.read_csv(os.path.join(root, subject, topic_ecg,  date, ecg_fname), compression="gzip") ))

    ecg_df['value.time'] = pd.to_datetime(ecg_df['value.time'],unit='s')
    ecg_timetable = ecg_df.pivot_table(values="value.ecg", index="value.time")
    ecg_timetable

    return ecg_timetable



def load_acc_night_data(start_date, before, after, subject):

    acc_df = pd.DataFrame()

    for h in range(-before,after+1):
        try:
            # load all csv files from ecg starting from start_date and for the next hours
            new_date = start_date + datetime.timedelta(hours=h)
            date = dt.strftime(new_date, "%Y%m%d")
            acc_fname =  dt.strftime(new_date, "%Y%m%d_%H%M")+'.csv.gz'
            acc_df = pd.concat( (acc_df, pd.read_csv(os.path.join(root, subject, topic_acc,  date, acc_fname), compression="gzip") ))
        except: 
            pass

    acc_df['value.time'] = pd.to_datetime(acc_df['value.time'],unit='s')
    acc_timetable = acc_df.pivot_table(values=["value.x","value.y","value.z"], index="value.time")
    acc_timetable

    return acc_timetable




#load the "label" file
quest = pd.read_csv("datahandling/questionnaire_allsubjects.csv")
quest.loc[quest["value.name"]=="MORNING", "value.name"] = 0 
quest.loc[quest["value.name"]=="kss_mood", "value.name"] = 1 
quest.loc[quest["value.name"]=="EVENING", "value.name"] = 2 

# creat dataframe with all nights with corresponding sleep scoring avilable
scoring_files = pd.DataFrame()
ecg_acc_and_imq = pd.DataFrame()

for subject in tqdm.tqdm(quest["Subject"].unique()): 
    print("Processing data for subject: " + subject)
    quest_sub = quest[quest["Subject"]== subject]
    for _, quest_answer in tqdm.tqdm(quest_sub.iterrows()):
        try:
            fdate = dt.utcfromtimestamp(quest_answer["value.time"])#
            qtime= dt.utcfromtimestamp(quest_answer["value.time"]).time()
            
            before = 6 # hours
            after = 1
            ecg_timetable = load_ecg_night_data(fdate.replace(minute=0, second=0), before,after, subject)
            acc_timetable = load_acc_night_data(fdate.replace(minute=0, second=0), before,after, subject)

            ecg_data = ecg_timetable.between_time((fdate-datetime.timedelta(hours=5)).time(), (fdate+datetime.timedelta(minutes=15)).time())
            acc_data = acc_timetable.between_time((fdate-datetime.timedelta(hours=5)).time(), (fdate+datetime.timedelta(minutes=15)).time())
            if ecg_data["value.ecg"].sum()==0:
                pass
            else:
                ecg_acc_and_imq = pd.concat((ecg_acc_and_imq, pd.DataFrame({"subject": subject, "date":dt.strftime(fdate,"%Y%m%d_%H%M%S"), "daypart": quest_answer["value.name"],
                                                            "ecg":[np.asarray(ecg_data["value.ecg"] )], 
                                                            "acc_x":[np.asarray(acc_data["value.x"] )],"acc_y":[np.asarray(acc_data["value.y"] )],"acc_z":[np.asarray(acc_data["value.z"] )],
                                                            "label": [[quest_answer["imq_1"],quest_answer["imq_2"],quest_answer["imq_4"],quest_answer["imq_5"]]]})))        

            
            print(fdate)
        except Exception as e:
            pass
            #print(e)

ecg_acc_and_imq = ecg_acc_and_imq.reset_index()
ecg_acc_and_imq.to_csv(pjoin(save_to,"longecg_longacc_and_imq.csv"))

empty_idx = []
for _,e in ecg_acc_and_imq.iterrows():
    if e["ecg"].sum()==0:
        empty_idx.append(e["index"])
ecg_acc_and_imq = ecg_acc_and_imq.drop(ecg_acc_and_imq.index[empty_idx])
ecg_acc_and_imq = ecg_acc_and_imq.reset_index(drop=True)


import pickle
data_dict = {"x": ecg_acc_and_imq["ecg"], "acc_x": ecg_acc_and_imq["acc_x"], "acc_y": ecg_acc_and_imq["acc_y"], "acc_z": ecg_acc_and_imq["acc_z"],
            "uid": ecg_acc_and_imq["subject"], "night":ecg_acc_and_imq["date"], "y":ecg_acc_and_imq["label"], "daypart": ecg_acc_and_imq["daypart"]}
pickle.dump(data_dict, open(pjoin(save_to,"longecg_longacc_and_imq.pickle"), "wb"))