
# Baseline models for MultiAct

### 1. Slowfast Backbone Model

- Relevant code is located in the `backbone/` directory.
- This implementation is a **condensed and adapted clone** of the official EPIC-Sounds repository:  
  https://github.com/epic-kitchens/epic-sounds-annotations/tree/main/src

#### 1.1 Prerequisites & Data Preparation
Before training or inference, you must extract the audio and convert it to the required format:

* **Audio Extraction:** Run `slowfast/audio_extraction/extract_audio.py` to extract raw audio from video sources.
* **HDF5 Conversion:** Run `slowfast/audio_extraction/wav_to_hdf5.py` to package extracted `.wav` files into `.hdf5` files for efficient loading.



#### 1.2  Fine-tuning SlowFast
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

#### 1.3 For Detection Tasks

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

#### 1.4 For Classification Tasks

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
