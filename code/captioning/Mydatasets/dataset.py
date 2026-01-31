

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from misc.require_lib import *

from transformers import BartTokenizer

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


ACTIVITY_TEMPLATES_EN = {
    "Prep": [
        "During the preparation phase, {sub_activities_str}.",
    ],
    "Cook": [
        "During the cooking phase, {sub_activities_str}.",
    ],
    "Cleanup": [
        "During the cleanup phase, {sub_activities_str}.",
    ],
}

SUB_ACTIVITY_PHRASES_EN = {
    'HandleObject': 'handles an object',
    'OpenCloseStorage': 'opens or closes a storage unit',
    'WashScrub': 'washes or scrubs dishes or surfaces',
    'CutChop': 'cuts or chops ingredients',
    'PreheatIgnite': 'preheats or ignites an appliance',
    'SeasonFood': 'seasons food',
    'WipeTidy': 'wipes or tidies up',
    'ApplianceRunning': 'operates an appliance',
    'PanCook': 'cooks in a pan',
    'MixKnead': 'mixes or kneads ingredients',
    'PlateFood': 'plates the food',
    'UnwrapPackage': 'unwraps a package',
    'default': 'performs an unknown action'
}
TRANSITION_WORDS_EN = ['then', 'next', 'after that', 'and then']

def generate_sequential_sub_caption(sub_activity_list):

    if not sub_activity_list:
        return "nothing happens"
    
    num_sub_activities = len(sub_activity_list)

    get_sub_activity_phrase = lambda sub_activity: SUB_ACTIVITY_PHRASES_EN.get(sub_activity, SUB_ACTIVITY_PHRASES_EN['default'])
    
    # Case 1: Only one activity
    if num_sub_activities == 1:
        return f"the user {get_sub_activity_phrase(sub_activity_list[0])}"

    # Case 2: Multiple activities
    parts = []
    # First activity
    parts.append(f"the user first {get_sub_activity_phrase(sub_activity_list[0])}")

    # Middle activities
    for i in range(1, num_sub_activities - 1):
        transition = TRANSITION_WORDS_EN[(i - 1) % len(TRANSITION_WORDS_EN)]
        parts.append(f"{transition} {get_sub_activity_phrase(sub_activity_list[i])}")
    
    # Last activity
    parts.append(f"and finally {get_sub_activity_phrase(sub_activity_list[-1])}")

    return ", ".join(parts)

class TrainCaption(Dataset):
    def __init__(self, train_json_file, feat_file_dir, feat_index_dir, tokenizer, task='caption', max_feature_len=9300, max_caption_len=512):
        super(TrainCaption, self).__init__()
        
        self.feat_file_dir = feat_file_dir
        self.feat_index_dir = feat_index_dir
        self.tokenizer = tokenizer
        self.max_feature_len = max_feature_len
        self.max_caption_len = max_caption_len
        self.task = task

        with open(train_json_file, "r") as f:
            self.metadata = json.load(f)
        
        self.act_ids = list(self.metadata.keys())

    def __len__(self):
        return len(self.act_ids)

    def __getitem__(self, index):

        act_id = self.act_ids[index]
        act_data = self.metadata[act_id]

        act_class_label = act_data["act_class"]
        act_sub_seq_label = act_data["sub_seq"]
        act_sub_seq_caption = generate_sequential_sub_caption(act_sub_seq_label)
        act_caption = ACTIVITY_TEMPLATES_EN[act_class_label][0]
        prefix_text = act_caption.format(sub_activities_str=act_sub_seq_caption)

        act_video_id = act_data["video_id"]
        act_st_time = act_data["start_time"]
        act_et_time = act_data["end_time"]

        feat_index_file = os.path.join(self.feat_index_dir, f"{act_video_id}.csv")
        feat_index_df = pd.read_csv(feat_index_file)

        feat_index_df["start_sec"] = feat_index_df["start_timestamp"].apply(timestamp_to_seconds) 
        feat_index_df["end_sec"] = feat_index_df["stop_timestamp"].apply(timestamp_to_seconds)

        matching_rows = feat_index_df[
            (feat_index_df["start_sec"] < act_et_time) & 
            (feat_index_df["end_sec"] > act_st_time)
        ]
        start_idx = matching_rows.index.min()
        end_idx = matching_rows.index.max()

        feat_file = os.path.join(self.feat_file_dir, f"{act_video_id}_feats.npz")
        feat_data = np.load(feat_file)
        features = feat_data['feats'][start_idx : end_idx + 1] # Shape: (Time, 2304)


        if features.shape[0] > self.max_feature_len:
            features = features[:self.max_feature_len, :]

        feature_len = features.shape[0]
        padding_len = self.max_feature_len - feature_len

        encoder_attention_mask = torch.ones(feature_len)
        if padding_len > 0:
            padding_array = np.zeros((padding_len, features.shape[1]), dtype=features.dtype)
            features = np.vstack([features, padding_array])
            encoder_attention_mask = torch.cat([encoder_attention_mask, torch.zeros(padding_len)])

        features = torch.from_numpy(features).float()


        if self.task == 'caption':
            caption_text = act_data['caption_en']
        else: 
            caption_text = act_data['summary_en']
        
        
        prefix_tokens = self.tokenizer(prefix_text, add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0)
        caption_tokens = self.tokenizer(caption_text, add_special_tokens=True, return_tensors='pt')['input_ids'].squeeze(0)
        prefix_labels = torch.full_like(prefix_tokens, -100)
        bos_token = caption_tokens[0:1]
        rest_of_caption = caption_tokens[1:]
        labels = torch.cat([bos_token, prefix_labels, rest_of_caption])
        
        current_len = len(labels)
        if current_len > self.max_caption_len:

            labels = labels[:self.max_caption_len]

            labels[-1] = self.tokenizer.eos_token_id
        elif current_len < self.max_caption_len:

            padding_len = self.max_caption_len - current_len

            padding = torch.full((padding_len,), -100)
            labels = torch.cat([labels, padding])
        
        return {
            "inputs_embeds": features, 
            "attention_mask": encoder_attention_mask.long(), 
            "labels": labels.long(), 
        }
        

class TrainAudioTextCaption(Dataset):
    def __init__(self, train_json_file, feat_file_dir, feat_index_dir, tokenizer, task='caption', max_feature_len=9300, max_text_input_len=512, max_caption_len=512):
        super(TrainAudioTextCaption, self).__init__()
        
        self.feat_file_dir = feat_file_dir
        self.feat_index_dir = feat_index_dir
        self.tokenizer = tokenizer
        
        self.max_feature_len = max_feature_len
        self.max_caption_len = max_caption_len
        self.max_text_input_len = max_text_input_len
        self.task = task

        with open(train_json_file, "r") as f:
            self.metadata = json.load(f)
        
        self.act_ids = list(self.metadata.keys())

    def __len__(self):
        return len(self.act_ids)

    def __getitem__(self, index):

        act_id = self.act_ids[index]
        act_data = self.metadata[act_id]
        

        act_class_label = act_data["act_class"]
        act_sub_seq_label = act_data["sub_seq"]
        act_sub_seq_caption = generate_sequential_sub_caption(act_sub_seq_label)
        act_caption = ACTIVITY_TEMPLATES_EN[act_class_label][0]
        prefix_text = act_caption.format(sub_activities_str=act_sub_seq_caption)
        

        text_input_tokenized = self.tokenizer(
            prefix_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_input_len, 
            return_tensors='pt'
        )

        input_ids = text_input_tokenized['input_ids'].squeeze(0)
        text_attention_mask = text_input_tokenized['attention_mask'].squeeze(0)

        act_video_id = act_data["video_id"]
        act_st_time = act_data["start_time"]
        act_et_time = act_data["end_time"]

        feat_index_file = os.path.join(self.feat_index_dir, f"{act_video_id}.csv")
        feat_index_df = pd.read_csv(feat_index_file)

        feat_index_df["start_sec"] = feat_index_df["start_timestamp"].apply(timestamp_to_seconds) 
        feat_index_df["end_sec"] = feat_index_df["stop_timestamp"].apply(timestamp_to_seconds)

        matching_rows = feat_index_df[
            (feat_index_df["start_sec"] < act_et_time) & 
            (feat_index_df["end_sec"] > act_st_time)
        ]
        start_idx = matching_rows.index.min()
        end_idx = matching_rows.index.max()

        feat_file = os.path.join(self.feat_file_dir, f"{act_video_id}_feats.npz")
        feat_data = np.load(feat_file)
        features = feat_data['feats'][start_idx : end_idx + 1] # Shape: (Time, 2304)


        if features.shape[0] > self.max_feature_len:
            features = features[:self.max_feature_len, :]

        feature_len = features.shape[0]
        padding_len = self.max_feature_len - feature_len

        audio_attention_mask = torch.ones(feature_len)
        if padding_len > 0:
            padding_array = np.zeros((padding_len, features.shape[1]), dtype=features.dtype)
            features = np.vstack([features, padding_array])
            audio_attention_mask = torch.cat([audio_attention_mask, torch.zeros(padding_len)])
        features = torch.from_numpy(features).float()


        if self.task == 'caption':
            caption_text = act_data['caption_en']
        else: # 'summary'
            caption_text = act_data['summary_en']
        
        labels_tokenized = self.tokenizer(
            caption_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_caption_len, 
            return_tensors='pt'
        )
        
        labels = labels_tokenized['input_ids'].squeeze(0)

        labels[labels == self.tokenizer.pad_token_id] = -100
        

        return {

            "input_ids": input_ids.long(),
            "attention_mask": text_attention_mask.long(),
            

            "audio_inputs_embeds": features,
            "audio_attention_mask": audio_attention_mask.long(),
            

            "labels": labels.long(),
        }
        


class EvalCaption(Dataset):
    def __init__(self, train_json_file, feat_file_dir, feat_index_dir, tokenizer, task='caption'):
        super(EvalCaption, self).__init__()
        
        self.feat_file_dir = feat_file_dir
        self.feat_index_dir = feat_index_dir
        self.tokenizer = tokenizer
        self.task = task

        with open(train_json_file, "r") as f:
            self.metadata = json.load(f)
        
        self.act_ids = list(self.metadata.keys())

    def __len__(self):
        return len(self.act_ids)

    def __getitem__(self, index):

        act_id = self.act_ids[index]
        act_data = self.metadata[act_id]

        act_video_id = act_data["video_id"]
        act_st_time = act_data["start_time"]
        act_et_time = act_data["end_time"]
        
        prefix_text = act_data["prefix_text_summary"]

        feat_index_file = os.path.join(self.feat_index_dir, f"{act_video_id}.csv")
        feat_index_df = pd.read_csv(feat_index_file)

        feat_index_df["start_sec"] = feat_index_df["start_timestamp"].apply(timestamp_to_seconds) 
        feat_index_df["end_sec"] = feat_index_df["stop_timestamp"].apply(timestamp_to_seconds)

        matching_rows = feat_index_df[
            (feat_index_df["start_sec"] < act_et_time) & 
            (feat_index_df["end_sec"] > act_st_time)
        ]
        start_idx = matching_rows.index.min()
        end_idx = matching_rows.index.max()

        feat_file = os.path.join(self.feat_file_dir, f"{act_video_id}_feats.npz")
        feat_data = np.load(feat_file)
        features = feat_data['feats'][start_idx : end_idx + 1] # Shape: (Time, 2304)

        features_tensor = torch.from_numpy(features).float()
        

        if self.task == 'caption':
            caption_text = act_data['caption_en']
        else: 
            caption_text = act_data['summary_en']

        
        return {
            "inputs_embeds": features_tensor,
            "prefix_text": prefix_text,
            "reference_text": caption_text
        }
        
class EvalAudioTextCaption(Dataset):
    def __init__(self, train_json_file, feat_file_dir, feat_index_dir, tokenizer, task='caption', max_text_input_len=512):
        super(EvalAudioTextCaption, self).__init__()
        
        self.feat_file_dir = feat_file_dir
        self.feat_index_dir = feat_index_dir
        self.tokenizer = tokenizer
        self.task = task
        self.max_text_input_len = max_text_input_len

        with open(train_json_file, "r") as f:
            self.metadata = json.load(f)
        
        self.act_ids = list(self.metadata.keys())

    def __len__(self):
        return len(self.act_ids)

    def __getitem__(self, index):

        act_id = self.act_ids[index]
        act_data = self.metadata[act_id]

        act_video_id = act_data["video_id"]
        act_st_time = act_data["start_time"]
        act_et_time = act_data["end_time"]
        
        # prefix_text = act_data["prefix_text_summary"]
        prefix_text = act_data["prefix_text_caption"]
        
        text_input_tokenized = self.tokenizer(
            prefix_text,
            truncation=True,
            max_length=self.max_text_input_len,
            return_tensors='pt'
        )
        input_ids = text_input_tokenized['input_ids'].squeeze(0)

        feat_index_file = os.path.join(self.feat_index_dir, f"{act_video_id}.csv")
        feat_index_df = pd.read_csv(feat_index_file)

        feat_index_df["start_sec"] = feat_index_df["start_timestamp"].apply(timestamp_to_seconds) 
        feat_index_df["end_sec"] = feat_index_df["stop_timestamp"].apply(timestamp_to_seconds)

        matching_rows = feat_index_df[
            (feat_index_df["start_sec"] < act_et_time) & 
            (feat_index_df["end_sec"] > act_st_time)
        ]
        start_idx = matching_rows.index.min()
        end_idx = matching_rows.index.max()

        feat_file = os.path.join(self.feat_file_dir, f"{act_video_id}_feats.npz")
        feat_data = np.load(feat_file)
        features = feat_data['feats'][start_idx : end_idx + 1] # Shape: (Time, 2304)

        features_tensor = torch.from_numpy(features).float()
        

        if self.task == 'caption':
            caption_text = act_data['caption_en']
        else:
            caption_text = act_data['summary_en']

        
        return {
            "input_ids": input_ids.long(), 
            "audio_inputs_embeds": features_tensor,
            "reference_text": caption_text 
        }

if __name__ == '__main__':
    
    # Captioning Task

    train_json_file = '/mnt/fast/nobackup/scratch4weeks/pz00220/dataset/LFAAC/annotation_captioning/train/activity_label.json'
    slowfast_feat_train_index_dir = '/mnt/fast/nobackup/scratch4weeks/pz00220/dataset/LFAAC/data/slowfast_feats/annotation_files'  # + video_id.csv
    slowfast_feat_train_dir = '/mnt/fast/nobackup/scratch4weeks/pz00220/dataset/LFAAC/data/slowfast_feats/feats' # + video_id_feats.npz

    dataset = TrainCaption(train_json_file=train_json_file, feat_file_dir=slowfast_feat_train_dir, feat_index_dir=slowfast_feat_train_index_dir)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=0,
                            drop_last=True)

    for idx in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            print(idx)

    # # Sequence Prediction Task
    # total_json_label_train_file = '/mnt/fast/nobackup/scratch4weeks/pz00220/dataset/LFAAC/annotation_captioning/train/activity.json'
    # slowfast_feat_train_index_dir = '/mnt/fast/nobackup/scratch4weeks/pz00220/dataset/LFAAC/data/slowfast_feats/annotation_files'  # + video_id.csv
    # slowfast_feat_train_dir = '/mnt/fast/nobackup/scratch4weeks/pz00220/dataset/LFAAC/data/slowfast_feats/feats' # + video_id_feats.npz

    # dataset = TrainSeqPrediction(train_json_file=total_json_label_train_file, feat_file_dir=slowfast_feat_train_dir, feat_index_dir=slowfast_feat_train_index_dir)
    # dataloader = DataLoader(dataset=dataset,
    #                         batch_size=1,
    #                         shuffle=True,
    #                         num_workers=0,
    #                         drop_last=False)

    # for idx in enumerate(tqdm(dataloader)):
    #     with torch.no_grad():
    #         print(idx)

    # # Classification Task
    # train_event_embed_path = '/Users/maisyzhang/Desktop/DataAnns/P07_113/P07_113_slowfast_emb.pkl'
    # train_subactivity_embed_path = '/Users/maisyzhang/Desktop/DataAnns/P07_113/P07_113_subactivity_emb.pkl'
    
    # train_activity_label_path = '/Users/maisyzhang/Desktop/DataAnns/P07_113/P07_113_activity.csv'
    # train_subactivity_label_path = '/Users/maisyzhang/Desktop/DataAnns/P07_113/P07_113_subactivity.csv'

    # # dataset = TrainActivityDataset_wo_Event(train_subactivity_embed_path=train_subactivity_embed_path,
    # #                                         train_activity_label_path=train_activity_label_path)

    # dataset = TrainActivityDataset_w_Event(train_event_embed_path=train_event_embed_path, 
    #                                        train_subactivity_embed_path=train_subactivity_embed_path, 
    #                                        train_subactivity_label_path=train_subactivity_label_path, 
    #                                        train_activity_label_path=train_activity_label_path)

    # dataloader = DataLoader(dataset=dataset,
    #                         batch_size=1,
    #                         shuffle=True,
    #                         num_workers=0,
    #                         drop_last=False)

    # for idx in enumerate(tqdm(dataloader)):
    #     with torch.no_grad():
    #         print(idx)









