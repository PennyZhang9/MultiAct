import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from misc.require_lib import *

class TrainSubActivityDataset(Dataset):
    def __init__(self, train_event_embed_path, train_subactivity_label_path):
        super(TrainSubActivityDataset, self).__init__()
        
        self.train_event_embed = pickle.load(open(train_event_embed_path, 'rb'))
        self.subactivity_df = pd.read_csv(train_subactivity_label_path)

        self.dataset = []
        for _, row in self.subactivity_df.iterrows():
            self.subactivity_item = {}
            annotation_event_id = row['annotation_event_id']
            parts = annotation_event_id.split('_')
            base_key = '_'.join(parts[:2]) # P07_113
            start_event = int(parts[2])
            end_event = int(parts[3]) 

            # combine multiple event embedding
            slow_embedding_list = []
            fast_embedding_list = []
            for i in range(start_event, end_event + 1):
                key = f"{base_key}_{i}"
                slow_embedding = self.train_event_embed[key]['slow'] # [1, 2048, 25, 4]
                fast_embedding = self.train_event_embed[key]['fast'] # [1, 256, 100, 4]
                slow_embedding_list.append(slow_embedding)
                fast_embedding_list.append(fast_embedding)

            cat_slow_embedding = np.concatenate(slow_embedding_list, axis=0)
            cat_fast_embedding = np.concatenate(fast_embedding_list, axis=0)

            self.subactivity_item['event_embed'] = [cat_slow_embedding, cat_fast_embedding]
            self.subactivity_item['class_id'] = int(row['class_id'])
            self.subactivity_item['metadata'] = row['class']

            self.dataset.append(self.subactivity_item) 
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):

        item = self.dataset[index]
        slow_seq, fast_seq = item['event_embed']
        label = item['class_id']

        slow_seq = torch.tensor(slow_seq, dtype=torch.float32)
        fast_seq = torch.tensor(fast_seq, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return slow_seq, fast_seq, label

class TestSubActivityDataset(Dataset):
    def __init__(self, test_subactivity_path, test_subactivity_label_path):
        super(TestSubActivityDataset, self).__init__()

        self.test_event_embed = pickle.load(open(test_subactivity_path, 'rb'))
        self.subactivity_df = pd.read_csv(test_subactivity_label_path)

        self.dataset = []
        for _, row in self.subactivity_df.iterrows():
            self.subactivity_item = {}
            annotation_event_id = row['annotation_event_id']
            parts = annotation_event_id.split('_')
            base_key = '_'.join(parts[:2]) # P07_113
            start_event = int(parts[2])
            end_event = int(parts[3]) 

            # combine multiple event embedding
            slow_embedding_list = []
            fast_embedding_list = []
            for i in range(start_event, end_event + 1):
                key = f"{base_key}_{i}"
                slow_embedding = self.test_event_embed[key]['slow'] # [1, 2048, 25, 4]
                fast_embedding = self.test_event_embed[key]['fast'] # [1, 256, 100, 4]
                slow_embedding_list.append(slow_embedding)
                fast_embedding_list.append(fast_embedding)

            cat_slow_embedding = np.concatenate(slow_embedding_list, axis=0)
            cat_fast_embedding = np.concatenate(fast_embedding_list, axis=0)

            self.subactivity_item['event_embed'] = [cat_slow_embedding, cat_fast_embedding]
            self.subactivity_item['class_id'] = int(row['class_id'])
            self.subactivity_item['metadata'] = row['annotation_id']

            self.dataset.append(self.subactivity_item) 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        slow_seq, fast_seq = item['event_embed']
        label = item['class_id']
        metadata = item['metadata']

        slow_seq = torch.tensor(slow_seq, dtype=torch.float32)
        fast_seq = torch.tensor(fast_seq, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        length = torch.tensor(slow_seq.shape[0], dtype=torch.long)

        return slow_seq, fast_seq, label, metadata, length

class TrainActivityDataset_wo_Event(Dataset):
    def __init__(self, train_subactivity_embed_path, train_activity_label_path):
        super(TrainActivityDataset_wo_Event, self).__init__()
        
        self.train_subactivity_embed = pickle.load(open(train_subactivity_embed_path, 'rb'))

        self.activity_df = pd.read_csv(train_activity_label_path)

        self.dataset = []
        for _, row in self.activity_df.iterrows():   # 按照 activity 进行遍历 sub activity 
            self.activity_item = {}

            # extract sub activity embedding
            annotation_subactivity_id = row['annotation_subactivity_id']
            subactivity_parts = annotation_subactivity_id.split('_')
            base_key = '_'.join(subactivity_parts[:2]) # P07_113
            start_subactivity = int(subactivity_parts[2])
            end_subactivity = int(subactivity_parts[3])

            subactivity_embedding_list = []
            for j in range(start_subactivity, end_subactivity + 1):
                key = f"{base_key}_{j}"
                subactivity_embedding = self.train_subactivity_embed[key]['sub_embed']
                subactivity_embedding = np.expand_dims(subactivity_embedding, axis=0)
                subactivity_embedding_list.append(subactivity_embedding)
            cat_subactivity_embedding = np.concatenate(subactivity_embedding_list, axis=0)

            # construct activity
            self.activity_item['subactivity_embed'] = cat_subactivity_embedding
            self.activity_item['class_id'] = int(row['class_id'])
            self.activity_item['metadata'] = row['annotation_id']

            self.dataset.append(self.activity_item) 
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):

        item = self.dataset[index]
        subactivity_seq = item['subactivity_embed'] 
        label = item['class_id'] - 1

        subactivity_seq = torch.tensor(subactivity_seq, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return subactivity_seq, label
    

class TestActivityDataset_only(Dataset):
    def __init__(self, test_subactivity_embed_path, test_activity_label_path):
        super(TestActivityDataset_only, self).__init__()
        
        self.test_subactivity_embed = pickle.load(open(test_subactivity_embed_path, 'rb'))

        self.activity_df = pd.read_csv(test_activity_label_path)

        self.dataset = []
        for _, row in self.activity_df.iterrows():   # 按照 activity 进行遍历 sub activity 
            self.activity_item = {}

            # extract sub activity embedding
            annotation_subactivity_id = row['annotation_subactivity_id']
            subactivity_parts = annotation_subactivity_id.split('_')
            base_key = '_'.join(subactivity_parts[:2]) # P07_113
            start_subactivity = int(subactivity_parts[2])
            end_subactivity = int(subactivity_parts[3])

            subactivity_embedding_list = []
            for j in range(start_subactivity, end_subactivity + 1):
                key = f"{base_key}_{j}"
                subactivity_embedding = self.test_subactivity_embed[key]['sub_embed']
                subactivity_embedding = np.expand_dims(subactivity_embedding, axis=0)
                subactivity_embedding_list.append(subactivity_embedding)
            cat_subactivity_embedding = np.concatenate(subactivity_embedding_list, axis=0)

            # construct activity
            self.activity_item['subactivity_embed'] = cat_subactivity_embedding
            self.activity_item['class_id'] = int(row['class_id'])
            self.activity_item['metadata'] = row['annotation_id']

            self.dataset.append(self.activity_item) 
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):

        item = self.dataset[index]
        subactivity_seq = item['subactivity_embed'] 
        label = item['class_id'] - 1
        

        subactivity_seq = torch.tensor(subactivity_seq, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        length = torch.tensor(subactivity_seq.shape[0], dtype=torch.long)

        return subactivity_seq, label, length


class TestActivityDataset_joint(Dataset):
    def __init__(self, test_event_embed_path, test_subactivity_label_path, test_activity_label_path):
        super(TestActivityDataset_joint, self).__init__()

        self.test_event_embed = pickle.load(open(test_event_embed_path, 'rb'))

        self.subactivity_df = pd.read_csv(test_subactivity_label_path)
        self.activity_df = pd.read_csv(test_activity_label_path)

        self.dataset = []

        for _, row in self.activity_df.iterrows():   # 按照 activity 进行遍历 sub activity 

            activity_label = int(row['class_id'])

            annotation_id = row['annotation_id']
            parts = annotation_id.split('_')
            base_key = '_'.join(parts[:2]) 

            # subactivty
            annotation_subactivity_id = row['annotation_subactivity_id']
            subactivity_parts = annotation_subactivity_id.split('_')
            sub_start = int(subactivity_parts[2])
            sub_end = int(subactivity_parts[3])

            subactivities = []

            for j in range(sub_start, sub_end + 1):
                sub_key = f"{base_key}_{j}"
                sub_row = self.subactivity_df[self.subactivity_df['annotation_id'].str.contains(sub_key)]

                sub_activity_label = int(sub_row.iloc[0]['class_id'])

                annotation_event_id = sub_row.iloc[0]['annotation_event_id']
                parts = annotation_event_id.split('_')

                start_event = int(parts[2])
                end_event = int(parts[3]) 

                slow_embedding_list = []
                fast_embedding_list = []
                for i in range(start_event, end_event + 1):
                    key = f"{base_key}_{i}"
                    slow_embedding = self.test_event_embed[key]['slow'] # [1, 2048, 25, 4]
                    fast_embedding = self.test_event_embed[key]['fast'] # [1, 256, 100, 4]
                    slow_embedding_list.append(slow_embedding)
                    fast_embedding_list.append(fast_embedding)

                events_slow = np.concatenate(slow_embedding_list, axis=0) # [Event_num, 2048, 25, 4]
                events_fast = np.concatenate(fast_embedding_list, axis=0) # [Event_num, 256, 100, 4]

                subactivity_item = {
                    'subactivity_label': sub_activity_label,
                    'events_slow': events_slow,
                    'events_fast': events_fast 
                }

                subactivities.append(subactivity_item)
        
            activity_item = {
                'activity_label': activity_label,
                'sub_activities': subactivities
            }

            self.dataset.append(activity_item)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):

        item = self.dataset[index]
        
        activity_label =  int(item['activity_label']) - 1

        sub_activities = []
        for sub_act in item['sub_activities']:
            sub_activity_label = int(sub_act['subactivity_label'])

            events_slow = torch.as_tensor(sub_act['events_slow'], dtype=torch.float32)
            events_fast = torch.as_tensor(sub_act['events_fast'], dtype=torch.float32)

            sub_activities.append({
                'subactivity_label': sub_activity_label,
                'events_slow': events_slow,
                'events_fast': events_fast
            })

        return {
            'activity_label': activity_label,
            'sub_activities': sub_activities
        }



class TrainActivityDataset_w_Event(Dataset):
    def __init__(self, train_event_embed_path, train_subactivity_embed_path, train_subactivity_label_path, train_activity_label_path):
        super(TrainActivityDataset_w_Event, self).__init__()
        
        self.train_event_embed = pickle.load(open(train_event_embed_path, 'rb'))
        # self.train_subactivity_embed = pickle.load(open(train_subactivity_embed_path, 'rb'))

        self.subactivity_df = pd.read_csv(train_subactivity_label_path)
        self.activity_df = pd.read_csv(train_activity_label_path)

        self.dataset = []

        for _, row in self.activity_df.iterrows():   # 按照 activity 进行遍历 sub activity 

            activity_label = int(row['class_id'])

            annotation_id = row['annotation_id']
            parts = annotation_id.split('_')
            base_key = '_'.join(parts[:2]) 

            # subactivty
            annotation_subactivity_id = row['annotation_subactivity_id']
            subactivity_parts = annotation_subactivity_id.split('_')
            sub_start = int(subactivity_parts[2])
            sub_end = int(subactivity_parts[3])

            subactivities = []

            for j in range(sub_start, sub_end + 1):
                sub_key = f"{base_key}_{j}"
                sub_row = self.subactivity_df[self.subactivity_df['annotation_id'].str.contains(sub_key)]

                sub_activity_label = int(sub_row.iloc[0]['class_id'])

                annotation_event_id = sub_row.iloc[0]['annotation_event_id']
                parts = annotation_event_id.split('_')

                start_event = int(parts[2])
                end_event = int(parts[3]) 

                slow_embedding_list = []
                fast_embedding_list = []
                for i in range(start_event, end_event + 1):
                    key = f"{base_key}_{i}"
                    slow_embedding = self.train_event_embed[key]['slow'] # [1, 2048, 25, 4]
                    fast_embedding = self.train_event_embed[key]['fast'] # [1, 256, 100, 4]
                    slow_embedding_list.append(slow_embedding)
                    fast_embedding_list.append(fast_embedding)

                events_slow = np.concatenate(slow_embedding_list, axis=0) # [Event_num, 2048, 25, 4]
                events_fast = np.concatenate(fast_embedding_list, axis=0) # [Event_num, 256, 100, 4]

                subactivity_item = {
                    'subactivity_label': sub_activity_label,
                    'events_slow': events_slow,
                    'events_fast': events_fast 
                }

                subactivities.append(subactivity_item)
        
            activity_item = {
                'activity_label': activity_label,
                'sub_activities': subactivities
            }

            self.dataset.append(activity_item)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):

        item = self.dataset[index]
        
        activity_label =  int(item['activity_label']) - 1

        sub_activities = []
        for sub_act in item['sub_activities']:
            sub_activity_label = int(sub_act['subactivity_label'])

            events_slow = torch.as_tensor(sub_act['events_slow'], dtype=torch.float32)
            events_fast = torch.as_tensor(sub_act['events_fast'], dtype=torch.float32)

            sub_activities.append({
                'subactivity_label': sub_activity_label,
                'events_slow': events_slow,
                'events_fast': events_fast
            })

        return {
            'activity_label': activity_label,
            'sub_activities': sub_activities
        }

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

