from trainers.base_trainer import *
from losses.loss import CE
from misc.require_lib import *
from datasets.dataset import TrainActivityDataset_wo_Event, TrainActivityDataset_w_Event
from models.activity_model import ActivityNet, HierarchicalActivityNet

def activity_collate_fn(batch):

    subactivity_seqs = [item[0] for item in batch]
    labels = torch.LongTensor([item[1] for item in batch])
    

    lengths = [len(seq) for seq in subactivity_seqs]
    max_len = max(lengths)

    _, f_s = subactivity_seqs[0].shape

    padded_subactivity = torch.zeros(len(batch), max_len, f_s)
    mask = torch.zeros(len(batch), max_len)


    for i, sub in enumerate(subactivity_seqs):
        end = lengths[i]
        padded_subactivity[i, :end, ...] = sub
        mask[i, :end] = 1.0
        
    return padded_subactivity, mask, labels, torch.LongTensor(lengths) 

def hierarchical_collate_fn(batch):

    activity_labels = []

    all_sub_activity_labels = []
    all_event_slow_features = []
    all_event_fast_features = []


    sub_activities_lengths = []
    event_lengths_per_sub = []

    for sample in batch:

        activity_labels.append(sample['activity_label'])

        sub_activities_lengths.append(len(sample['sub_activities']))

        for sub_act in sample['sub_activities']:
            all_sub_activity_labels.append(sub_act['subactivity_label'])

            event_count = sub_act['events_slow'].shape[0]
            event_lengths_per_sub.append(event_count)
            
            all_event_slow_features.append(sub_act['events_slow'])
            all_event_fast_features.append(sub_act['events_fast'])

    batched_event_slow = torch.cat(all_event_slow_features, dim=0) # [3+15, 2048, 25, 4]
    batched_event_fast = torch.cat(all_event_fast_features, dim=0) # [3+15, 256, 100, 4]

    batch_sub_labels = torch.tensor(all_sub_activity_labels, dtype=torch.long)

    return {
        'activity_labels': torch.tensor(activity_labels, dtype=torch.long),
        'sub_labels': batch_sub_labels,

        'event_features': {
            'slow': batched_event_slow,
            'fast': batched_event_fast
        },

        'structure':{
            'sub_lengths': torch.tensor(sub_activities_lengths, dtype=torch.long),
            'event_lengths': torch.tensor(event_lengths_per_sub, dtype=torch.long)
        }
    }


class Trainer(baseTrainer):
    def __init__(self, params, model_dir, train_event_embed_path, train_subactivity_embed_path, train_subactivity_label_path, train_activity_label_path):
        self.params = params
        self.model_dir = model_dir
        self.train_event_embed_path = train_event_embed_path
        self.train_subactivity_embed_path = train_subactivity_embed_path
        self.train_subactivity_label_path = train_subactivity_label_path
        self.train_activity_label_path = train_activity_label_path

        self.device = torch.device(self.params.train_type)

        print('[LOG] - Loading Train Dataset.')
        # 不加入event特征输入
        # self.train_dataset = TrainActivityDataset_wo_Event(train_subactivity_embed_path=self.train_subactivity_embed_path,
        #                                                    train_activity_label_path=self.train_activity_label_path)

        # self.train_dataloader = DataLoader(self.train_dataset,
        #                                    batch_size=self.params.train_batch_size,
        #                                    shuffle=True,
        #                                    num_workers=self.params.num_workers,
        #                                    pin_memory=True,
        #                                    drop_last=True,
        #                                    collate_fn=activity_collate_fn)
        
        # 加入 event 特征输入
        self.train_dataset = TrainActivityDataset_w_Event(train_event_embed_path, 
                                                          train_subactivity_embed_path, 
                                                          train_subactivity_label_path, 
                                                          train_activity_label_path)

        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.params.train_batch_size,
                                           shuffle=True,
                                           num_workers=self.params.num_workers,
                                           pin_memory=True,
                                           drop_last=True,
                                           collate_fn=hierarchical_collate_fn)  

        # self.network = ActivityNet().to(self.device)
        self.network = HierarchicalActivityNet().to(self.device)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.params.learning_rate)
        self.loss = CE().to(self.device)
        self.model = os.path.join(model_dir, "nnet")
        self.model_log = os.path.join(model_dir, "log")
        self.checkpoint_dir = os.path.join(model_dir, "checkpoint")
    

    def hierarchical_train(self, epoch):
        
        sum_loss, act_correct_preds, sub_correct_preds, sum_act_samples, sum_sub_samples = 0, 0, 0, 0, 0
        progress_bar = tqdm(enumerate(self.train_dataloader))

        for batch_idx, batch in progress_bar:
            sum_act_samples += len(batch['activity_labels'])
            sum_sub_samples += len(batch['sub_labels'])

            outputs = self.network(batch)
            activity_logits = outputs['main_logits']
            sub_logits = outputs['sub_logits']

            activity_labels = batch['activity_labels'].to(self.device)
            sub_labels = batch['sub_labels'].to(self.device)

            loss_activity = self.loss(activity_logits, activity_labels)
            loss_sub = self.loss(sub_logits, sub_labels)

            loss = loss_activity + loss_sub

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            sum_loss += loss.item()

            act_pred = torch.argmax(activity_logits, dim=1)

            act_correct = (act_pred == batch['activity_labels'].to(act_pred.device)).sum().item()
            act_correct_preds += act_correct

            sub_pred = torch.argmax(sub_logits, dim=1)
            sub_correct = (sub_pred == batch['sub_labels'].to(sub_pred.device)).sum().item()
            sub_correct_preds += sub_correct

            progress_bar.set_description('[LOG] - Train Epoch: {:3d} [{:4d}/{:4d} ({:.3f}%)] Loss: {:.4f} Act-Acc: {:.2f}% Sub-Acc: {:.2f}%'.format(
            epoch, batch_idx + 1, len(self.train_dataloader),
            100. * (batch_idx + 1) / len(self.train_dataloader),
            sum_loss / sum_act_samples,
            100. * act_correct_preds / sum_act_samples,
            100. * sub_correct_preds / sum_sub_samples,
            ))

        self.save(epoch=epoch, model=self.network, optimizer=self.optimizer, checkpoint_dir=self.checkpoint_dir)

        return

    def train(self, epoch):
   
        sum_loss, sum_samples, correct_preds = 0, 0, 0
        progress_bar = tqdm(enumerate(self.train_dataloader))

        for batch_idx, (padded_subactivity, mask, label, lengths) in progress_bar:
            sum_samples += len(label)

            padded_subactivity = padded_subactivity.to(self.device)
            mask = mask.to(self.device)
            label = label.to(self.device)
            lengths = lengths.to(self.device)

            _, logits = self.network(padded_subactivity, mask, lengths)
            loss = self.loss(logits, label)

            sum_loss += loss.item() * len(label)

            pred = torch.argmax(logits, dim=1)
            correct = (pred == label).sum().item()
            correct_preds += correct

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            progress_bar.set_description('[LOG] - Train Epoch: {:3d} [{:4d}/{:4d} ({:.3f}%)] Loss: {:.4f} Acc: {:.2f}%'.format(
            epoch, batch_idx + 1, len(self.train_dataloader),
            100. * (batch_idx + 1) / len(self.train_dataloader),
            sum_loss / sum_samples,
            100. * correct_preds / sum_samples
        ))
            
        with open(os.path.join(self.model_log, "epoch_loss_log"), 'a') as epoch_f:
            epoch_f.write(
                "Time:{}, Epoch:{}, Loss:{:.4f}, Acc:{:.2f}%\n".format(
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                epoch,
                sum_loss / sum_samples,
                100. * correct_preds / sum_samples
            )
        )
        self.save(epoch=epoch, model=self.network, optimizer=self.optimizer, checkpoint_dir=self.checkpoint_dir)


