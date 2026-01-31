from trainers.base_trainer import *
from misc.require_lib import *
from Mydatasets.dataset import TrainCaption, EvalCaption, TrainAudioTextCaption, EvalAudioTextCaption
from transformers import BartTokenizer, AdamW, get_scheduler
from models.caption_model_v2 import create_audio_text_bart_model
from torch.nn.utils.rnn import pad_sequence
import logging
from functools import partial

from typing import List, Dict
import shutil


class Trainer(baseTrainer):
    def __init__(self, params, model_dir, train_total_json, train_slowfast_feat_index_dir, train_slowfast_feat_dir):
        self.params = params
        self.model_dir = model_dir
        self.device = torch.device(self.params.train_type)
        
        print('[LOG] - Loading Tokenizer and Model.')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.network = create_audio_text_bart_model(feature_dim=2304)  
        self.network.to(self.device)

        print('[LOG] - Loading Train Dataset.')
        self.train_dataset = TrainAudioTextCaption(train_json_file=train_total_json,
                                                   feat_file_dir=train_slowfast_feat_dir,
                                                   feat_index_dir=train_slowfast_feat_index_dir,
                                                   tokenizer=self.tokenizer,
                                                   task='caption')

        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.params.train_batch_size,
                                           shuffle=True,
                                           num_workers=self.params.num_workers,
                                           pin_memory=True,
                                           drop_last=True)


        print('[LOG] - Setting up Optimizer and Scheduler.')
        self.optimizer = AdamW(self.network.parameters(), lr=self.params.learning_rate)

        num_training_steps = params.num_epochs * len(self.train_dataloader)

        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps  # 
        )

        self.model = os.path.join(model_dir, "nnet")
        self.model_log = os.path.join(model_dir, "log")
        self.log_file = os.path.join(self.model_log, "training_log.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file), 
                logging.StreamHandler()           
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.checkpoint_dir = os.path.join(model_dir, "checkpoint")

    def train_epoch(self, epoch):
        self.network.train()
        total_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.params.num_epochs}")  #

        for batch in progress_bar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.network(**batch)
            loss = outputs.loss
            loss.backward()
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss / (progress_bar.n + 1):.4f}'})

        epoch_checkpoint_dir = os.path.join(self.checkpoint_dir, f"epoch_{epoch+1}")
        self.network.save_pretrained(epoch_checkpoint_dir)
        self.tokenizer.save_pretrained(epoch_checkpoint_dir)





