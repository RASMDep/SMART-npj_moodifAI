<h1 align="center">
<img src="logo/moodifAI.jpg" align="left" width="200px"/>
<b> MoodifAI: Unveiling Psychophysiological States Through AI and Wearable Data  </b>
<br clear="left"/>
</h1><br>


## Overview

This repository contains the codebase for **MoodifAI**, a framework designed to explore and analyze mood patterns using advanced AI techniques. The project focuses on preprocessing wearable patch data, generating time series features, and training deep learning models for psychophysiological state assessment.  

The work is associated with the paper:  
**"Longitudinal and Objective Psychophysiological State Assessment in Real-World Settings: A Data-Driven Analysis of Wearable Patch Data"**.  

## Repository Contents  

### 1. `preprocessing/`  
Scripts and utilities for cleaning and preparing raw wearable patch data  

To preprocess raw accelerometer data and extract derived features, use the following command:  

```bash
python preprocessing/vivalink_extract_accmetrics.py \
  --participant_id=SMART_999 \
  --data_path="data_examples/raw_data" \
  --output_dir="data_examples/derived_features"

likewise to extract HRV metrics

```bash
python preprocessing/vivalink_extract_hrvmetrics.py \
  --participant_id=SMART_999 \
  --data_path="data_examples/raw_data" \
  --output_dir="data_examples/derived_features"


### 2. `ML_experiments/`  
Deep learning experiments for psychophysiological state assessment:  
- **`ML_experiments/utils/ml_dataset_builder.py`**: Processes the time-series features generated during preprocessing, formats the data, and saves it as a dataset for use in experiments.
- **`run_classification.py`**: End-to-end script for training and evaluating deep learning models.  

### 3. `figures/`  
Scripts to reproduce the figures used in the paper: 

### 4. data_examples/

This folder contains example data to help understand the input and output formats:

    raw_data/: Includes samples of raw data from wearable devices before preprocessing.
    derived_features/: Contains examples of time-series features generated during preprocessing.

## 💓 Wearable ECG patch data - Vivalink - used in the paper
  [Vivalink](https://www.vivalink.com/) is a wearable ECG patch with the following features:

  -  **Single lead ECG sensors** (128 Hz)

    Raw ECG data are stored in a `CSV` format compressed using `gzip`. 
    Each user/day/hour has a separte file.
    The file includes the following columns:

    | Column Name   | Description                                       | Example                |
    |---------------|---------------------------------------------------|------------------------|
    | `value.time`  | Timestamp of the ECG data, represented as epoch time (milliseconds since 1970).| `1.672110e+12`          |
    | `value.ecg`   | ECG signal value recorded at a specific timestamp. This is a continuous signal, representing the heart's electrical activity. | `0.456`                |
    | `key.projectId` | Identifier for the project or study this data belongs to. | `SMART`                |
    | `key.userId`  | Identifier for the user or participant. This is typically an anonymized ID. | `SMART_999`            |

    
  -  **Accelerometer sensor** (3 axis, 25 Hz - custom rate)

    Raw ACC data are stored in a `CSV` format compressed using `gzip`. 
    Each user/day/hour has a separte file.
    The file includes the following columns:

    | Column Name   | Description                                       | Example                |
    |---------------|---------------------------------------------------|------------------------|
    | `value.time`  | Timestamp of the ECG data, represented as epoch time (milliseconds since 1970).| `1.672110e+12`          |
    | `value.x`   | accelerometer readings along the X-axis  | `-461`                |
    | `value.y`   | accelerometer readings along the Y-axis |  `1921`  |
    | `value.z`   | accelerometer readings along the Z-axis  | `547`  |
    | `key.projectId` | Identifier for the project or study this data belongs to. | `SMART`                |
    | `key.userId`  | Identifier for the user or participant. This is typically an anonymized ID. | `SMART_999`            |



  - Derived  **Respiration Rate**
