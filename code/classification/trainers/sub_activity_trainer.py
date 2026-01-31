from trainers.base_trainer import *
from losses.loss import CE
from misc.require_lib import *
from datasets.dataset import TrainSubActivityDataset
from models.sub_activity_model import SubactivityNet

def subactivity_collate_fn(batch):

    slow_seqs = [item[0] for item in batch]
    fast_seqs = [item[1] for item in batch]
    labels = torch.LongTensor([item[2] for item in batch])

    lengths = [len(seq) for seq in slow_seqs]
    max_len = max(lengths)

    _, c_s, t_s, h_s = slow_seqs[0].shape
    _, c_f, t_f, h_f = fast_seqs[0].shape

    padded_slow = torch.zeros(len(batch), max_len, c_s, t_s, h_s)
    padded_fast = torch.zeros(len(batch), max_len, c_f, t_f, h_f)
    mask = torch.zeros(len(batch), max_len) 


    for i, (sl, fl) in enumerate(zip(slow_seqs, fast_seqs)):
        end = lengths[i]
        padded_slow[i, :end, ...] = sl
        padded_fast[i, :end, ...] = fl
        mask[i, :end] = 1.0 
        
    return padded_slow, padded_fast, mask, labels, torch.LongTensor(lengths)

class Trainer(baseTrainer):
    def __init__(self, params, model_dir, train_event_embed_path, train_subactivity_label_path):
        self.params = params
        self.model_dir = model_dir
        self.train_event_embed_path = train_event_embed_path
        self.train_subactivity_label_path = train_subactivity_label_path

        self.device = torch.device(self.params.train_type)

        print('[LOG] - Loading Train Dataset.')

        self.train_dataset = TrainSubActivityDataset(train_event_embed_path=self.train_event_embed_path,
                                                     train_subactivity_label_path=self.train_subactivity_label_path)

        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.params.train_batch_size,
                                           shuffle=True,
                                           num_workers=self.params.num_workers,
                                           pin_memory=True,
                                           drop_last=True,
                                           collate_fn=subactivity_collate_fn) 
        
        self.network = SubactivityNet().to(self.device)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.params.learning_rate)
        self.loss = CE().to(self.device)
        self.model = os.path.join(model_dir, "nnet")
        self.model_log = os.path.join(model_dir, "log")
        self.checkpoint_dir = os.path.join(model_dir, "checkpoint")


    def train(self, epoch):
   
        sum_loss, sum_samples, correct_preds = 0, 0, 0
        progress_bar = tqdm(enumerate(self.train_dataloader))

        for batch_idx, (padded_slow, padded_fast, mask, label, lengths) in progress_bar:
            padded_slow = padded_slow.to(self.device)
            padded_fast = padded_fast.to(self.device)
            mask = mask.to(self.device)
            lengths = lengths.to(self.device) 
            label = label.to(self.device)     
            sum_samples += len(label)
            _, logits = self.network(padded_slow, padded_fast, mask, lengths)
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




