import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from misc.require_lib import *



def timestamp_to_seconds(ts):
        parts = ts.strip().split(':')
        parts = [float(p) if '.' in p else int(p) for p in parts]

        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = 0
            m, s = parts
        # h, m, s = ts.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

ALL_SUBACTIVITY_CLASSES = [str(i) for i in range(12)]
label_map = {label_str: i + 1 for i, label_str in enumerate(ALL_SUBACTIVITY_CLASSES)}

class TrainSeqPrediction(Dataset):
    def __init__(self, train_json_file, feat_file_dir, feat_index_dir):
        super(TrainSeqPrediction, self).__init__()

        self.dataset = []

        with open(train_json_file, "r") as f:
            train_json_data = json.load(f)
        
        for act_id, act_data in train_json_data.items():
            self.subact_item = {}
            act_video_id = act_data["video_id"]
            act_st_time = act_data["start_time"]
            act_et_time = act_data["end_time"]
            # sub activity sequence label
            sub_cls_ids = [sub["sub_class_id"] for sub in act_data.get("subactivities", [])]
            # sub_label_int_ids = [int(label) for label in sub_cls_ids]
            sub_label_int_ids = [label_map.get(str(label)) for label in sub_cls_ids]
            # feat_index
            feat_index_file = os.path.join(feat_index_dir, f"{act_video_id}.csv")
            feat_index_df = pd.read_csv(feat_index_file)
            feat_index_df["start_sec"] = feat_index_df["start_timestamp"].apply(timestamp_to_seconds)
            feat_index_df["end_sec"] = feat_index_df["stop_timestamp"].apply(timestamp_to_seconds)

            matching_rows = feat_index_df[
                (feat_index_df["start_sec"] < act_et_time) & 
                (feat_index_df["end_sec"] > act_st_time)
            ]
            start_idx = matching_rows.index.min()
            end_idx = matching_rows.index.max()
            # feat
            feat_file = os.path.join(feat_file_dir, f"{act_video_id}_feats.npz")
            if not os.path.exists(feat_file):
                raise FileNotFoundError(f"No feature file found for video_id: {act_video_id}")
            feat_data = np.load(feat_file) # 'feats': 6459, 2304  

            selected_feat = feat_data['feats'][start_idx: end_idx + 1]

            self.subact_item['slowfast_feat'] = selected_feat
            self.subact_item['label'] = sub_label_int_ids
            self.dataset.append(self.subact_item)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]



class TrainSeqPredictionFixNum(Dataset):
    def __init__(self, MAX_SUBACTIVITIES, train_json_file, feat_file_dir, feat_index_dir):
        super(TrainSeqPredictionFixNum, self).__init__()

        self.dataset = []

        keep_short_sequences=True

        with open(train_json_file, "r") as f:
            train_json_data = json.load(f)
        
        for act_id, act_data in train_json_data.items():
            
            sub_activities = act_data.get("subactivities", [])
            num_sub_acts = len(sub_activities)

            sub_sequences_to_process = []
            if num_sub_acts >= MAX_SUBACTIVITIES:

                for i in range(num_sub_acts - MAX_SUBACTIVITIES + 1):
                    sub_sequences_to_process.append(sub_activities[i : i + MAX_SUBACTIVITIES])
            elif keep_short_sequences:

                sub_sequences_to_process.append(sub_activities)


            for target_sub_acts in sub_sequences_to_process:
                if not target_sub_acts:
                    continue

                sub_act_item = {}
                act_video_id = act_data["video_id"]

                clip_start_time = target_sub_acts[0]["start_time"]
                clip_end_time = target_sub_acts[-1]["end_time"]

                # 获取子序列的标签
                sub_cls_ids = [sub["sub_class_id"] for sub in target_sub_acts]

                sub_label_int_ids = [label_map.get(str(label)) for label in sub_cls_ids]

                try:
                    feat_index_file = os.path.join(feat_index_dir, f"{act_video_id}.csv")
                    feat_index_df = pd.read_csv(feat_index_file)
                    

                    clip_st_sec = clip_start_time
                    clip_et_sec = clip_end_time
            
                    feat_index_df["start_sec"] = feat_index_df["start_timestamp"].apply(timestamp_to_seconds)
                    feat_index_df["end_sec"] = feat_index_df["stop_timestamp"].apply(timestamp_to_seconds)
                    
                    matching_rows = feat_index_df[
                        (feat_index_df["end_sec"] > clip_st_sec) &
                        (feat_index_df["start_sec"] < clip_et_sec)
                    ]
                    
                    if matching_rows.empty:
                        continue

                    start_idx = matching_rows.index.min()
                    end_idx = matching_rows.index.max()

                    feat_file = os.path.join(feat_file_dir, f"{act_video_id}_feats.npz")
                    if not os.path.exists(feat_file):
                        raise FileNotFoundError(f"No feature file found for video_id: {act_video_id}")

                    feat_data = np.load(feat_file)
                    selected_feat = feat_data['feats'][start_idx : end_idx + 1]

                    sub_act_item['slowfast_feat'] = selected_feat
                    sub_act_item['label'] = sub_label_int_ids
                    self.dataset.append(sub_act_item)

                except Exception as e:
                    print(f"Error processing a sub-sequence from {act_id} in video {act_video_id}: {e}")
                    continue

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]





