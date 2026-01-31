
# Baseline models for MultiAct

## 1. Slowfast Backbone

- Relevant code is located in the `backbone/` directory.
- This implementation is a **condensed and adapted clone** of the official EPIC-Sounds repository:  
  https://github.com/epic-kitchens/epic-sounds-annotations/tree/main/src

### 1.1 Prerequisites & Data Preparation
Before training or inference, you must extract the audio and convert it to the required format:

* **Audio Extraction:** Run `slowfast/audio_extraction/extract_audio.py` to extract raw audio from video sources.
* **HDF5 Conversion:** Run `slowfast/audio_extraction/wav_to_hdf5.py` to package extracted `.wav` files into `.hdf5` files for efficient loading.



### 1.2  Fine-tuning SlowFast
To fine-tune Slow-Fast on EPIC-Sounds, run:

```(python)
python tools/run_net.py \
--cfg configs/EPIC-Sounds/slowfast/SLOWFASTAUDIO_8x8_R50.yaml \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/outpur_dir \
EPICSOUNDS.AUDIO_DATA_FILE /path/to/EPIC_audio.hdf5 \
EPICSOUNDS.ANNOTATIONS_DIR /path/to/annotations \
TRAIN.CHECKPOINT_FILE_PATH /path/to/SLOWFAST_VGG.pyth
```

### 1.3 For Detection Tasks

To extract features for action detection, ensure `SAVE_DETECTION` is set to `True`. This will generate `.npz` files containing the embeddings.

```(python)
python tools/run_net.py \
--cfg configs/EPIC-Sounds/slowfast/SLOWFASTAUDIO_8x8_R50.yaml \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/outpur_dir \
EPICSOUNDS.AUDIO_DATA_FILE /path/to/EPIC_audio.hdf5 \
EPICSOUNDS.ANNOTATIONS_DIR /path/to/annotations \
SAVE_DETECTION True \
DETECTION_FEATS_PATH /path/to/SLOWFAST_det_feats.npz
```

### 1.4 For Classification Tasks

To extract features for action detection, ensure `SAVE_CLASSIFICATION` is set to `True`. This will generate `.pkl` files containing the embeddings.

```(python)
python tools/run_net.py \
--cfg configs/EPIC-Sounds/slowfast/SLOWFASTAUDIO_8x8_R50.yaml \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/outpur_dir \
EPICSOUNDS.AUDIO_DATA_FILE /path/to/EPIC_audio.hdf5 \
EPICSOUNDS.ANNOTATIONS_DIR /path/to/annotations \
SAVE_CLASSIFICATION True \
CLS_FEATS_PATH /path/to/SLOWFAST_cls_feats.pkl
```

## 2. Activity / Sub-activity / Event Classification

All relevant scripts and modules are located in the `classification/` directory.

### 2.1 Event Classification
For **Event Classification**, training and testing can be performed directly using the **SlowFast Backbone** codebase.

### 2.2 Sub-activity Classification
- **Training:** Please refer to `train_subactivity_net.py` for training procedures.
- **Testing:** Please refer to `test_subactivity_net.py` for evaluation and testing.

### 2.3 Activity Classification
- **Training:** Please refer to `train_activity_net.py` for training procedures.
- **Testing:** Please refer to `test_activity_net.py` for evaluation and testing.

## 3. Sub-activity / Event Detection

- **Codebase Path:** All relevant source code is located in the `detection/` directory.
- **Implementation Note:** This module is a condensed and adapted version of the official repository: https://github.com/epic-kitchens/C10-epic-sounds-detection.

### 3.1 Training Data Preparation
During the training phase for both **Sub-activity** and **Event Detection**, training data is generated using a sliding window approach over features extracted from the **SlowFast backbone**. The sliding window parameters are configured as follows:
* `CLIP_SECS`: 1.999 s
* `STRIDE_SECS`: 0.2 s   

### 3.2 Training & Testing Workflow
The implementation for Sub-activity and Event detection follows a similar architectural framework. While the core model training logic remains largely consistent, they differ primarily in terms of **input data** and specific **training configurations**.

## 4. Sub-activity Sequence Prediction

- **Codebase Path:** All relevant scripts and modules are located in the `seq_pred/` directory.

### 4.1 Training Data Preparation
The training data generation process for Sequential Prediction follows the same protocol as **Sub-activity Detection**. Features are extracted from the **SlowFast backbone** using a sliding window approach to ensure temporal consistency across the dataset.

## 5. Activity Captioning

- **Codebase Path:** All relevant scripts and modules are located in the `captioning/` directory.