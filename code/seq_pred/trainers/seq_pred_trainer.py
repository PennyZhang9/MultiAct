from trainers.base_trainer import *
from misc.require_lib import *
from Mydatasets.dataset import TrainSeqPrediction, TrainSeqPredictionFixNum
from models.seq_pred_model import CTCBaseline, SlowFastConformerCTC
from torch.optim.lr_scheduler import ReduceLROnPlateau
import jiwer
# from ctcdecode import CTCBeamDecoder
from collections import defaultdict

def seq_pred_collate_fn(batch):

    batch.sort(key=lambda x: x['slowfast_feat'].shape[0], reverse=True)

    # convert np.ndarray to torch.Tensor
    features = [torch.tensor(item['slowfast_feat'], dtype=torch.float32) for item in batch]
    labels = [torch.tensor(item['label'], dtype=torch.long) for item in batch]

    feature_lengths = torch.LongTensor([f.shape[0] for f in features])
    label_lengths = torch.LongTensor([l.shape[0] for l in labels])

    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True)
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0) 
        
    return padded_features, feature_lengths, padded_labels, label_lengths

class Trainer(baseTrainer):
    def __init__(self, params, model_dir, train_total_json, val_total_json, train_slowfast_feat_index_dir, train_slowfast_feat_dir):
        self.params = params
        self.model_dir = model_dir
        self.device = torch.device(self.params.train_type)
        BLANK_ID = 0

        print('[LOG] - Loading Train Dataset.')

        self.train_dataset = TrainSeqPredictionFixNum(MAX_SUBACTIVITIES=self.params.max_subactivity,
                                                      train_json_file=train_total_json,
                                                      feat_file_dir=train_slowfast_feat_dir,
                                                      feat_index_dir=train_slowfast_feat_index_dir)
        # self.train_dataset = TrainSeqPrediction(train_json_file=train_total_json,
        #                                         feat_file_dir=train_slowfast_feat_dir,
        #                                         feat_index_dir=train_slowfast_feat_index_dir)

        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.params.train_batch_size,
                                           shuffle=True,
                                           num_workers=self.params.num_workers,
                                           pin_memory=True,
                                           drop_last=True,
                                           collate_fn=seq_pred_collate_fn)  # 这里可能需要修改
        
        
        self.val_dataset = TrainSeqPrediction(train_json_file=val_total_json,
                                                feat_file_dir=train_slowfast_feat_dir,
                                                feat_index_dir=train_slowfast_feat_index_dir)

        self.val_dataloader = DataLoader(self.val_dataset,
                                           batch_size=self.params.test_batch_size,
                                           num_workers=self.params.num_workers,
                                           pin_memory=True,
                                           collate_fn=seq_pred_collate_fn)
        
        self.network = SlowFastConformerCTC(slowfast_dim=2304, num_classes=self.params.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.params.learning_rate)


        self.loss = nn.CTCLoss(blank=BLANK_ID, reduction='mean', zero_infinity=True).to(self.device)
        self.model = os.path.join(model_dir, "nnet")
        self.model_log = os.path.join(model_dir, "log")
        self.checkpoint_dir = os.path.join(model_dir, "checkpoint")

    def greedy_decoder(self, log_probs, blank_id=0):
        best_paths = torch.argmax(log_probs.transpose(0, 1), dim=2)
        decoded_seqs = []
        for path in best_paths:
            collapsed_path = [p for i, p in enumerate(path) if i == 0 or p != path[i-1]]
            decoded_seq = [p.item() for p in collapsed_path if p != blank_id]
            decoded_seqs.append(decoded_seq)
        return decoded_seqs
    
    def beam_search_decoder(self, log_probs, beam_width=5, blank_id=0):
        """
        Args:
            log_probs (Tensor): (T, B, C) — time-major log-probabilities
            beam_width (int): number of beams to keep
            blank_id (int): index of the blank label

        Returns:
            List[List[int]]: list of decoded label indices per batch sample
        """

        T, B, C = log_probs.size()
        log_probs = log_probs.cpu().detach().numpy()

        results = []

        for b in range(B):
            beam = [([], 0.0)]  # List of tuples: (sequence, log_prob)

            for t in range(T):
                new_beam = defaultdict(lambda: -float('inf'))

                for seq, score in beam:
                    for c in range(C):
                        new_seq = seq + [c]
                        new_score = score + log_probs[t][b][c]

                        # Merge identical sequences with max prob
                        key = tuple(new_seq)
                        new_beam[key] = max(new_beam[key], new_score)

                # Keep top-k beams
                beam = sorted(new_beam.items(), key=lambda x: x[1], reverse=True)[:beam_width]
                beam = [(list(seq), score) for seq, score in beam]

            # Take the best sequence
            best_seq, _ = beam[0]

            # Collapse repeats and remove blanks
            final = []
            prev = None
            for token in best_seq:
                if token != blank_id and token != prev:
                    final.append(token)
                prev = token

            results.append(final)

        return results

    def calculate_aer(self, gt_seqs, pred_seqs):

        gt_seqs_str = [[str(token) for token in seq] for seq in gt_seqs]
        pred_seqs_str = [[str(token) for token in seq] for seq in pred_seqs]

        identity = lambda x: x
        output = jiwer.process_words(
            gt_seqs_str,
            pred_seqs_str,
            reference_transform=identity,
            hypothesis_transform=identity
        )

        s = output.substitutions
        d = output.deletions
        i = output.insertions
        h = output.hits
        n = s + d + h 

        if n == 0:
            return {
                'aer': 0.0,
                'sub_rate': 0.0,
                'del_rate': 0.0,
                'ins_rate': 0.0,
                's_count': 0,
                'd_count': 0,
                'i_count': 0,
            }
        total_aer = (s + d + i) / n
        detailed_results = {
            'aer': total_aer, 
            'sub_rate': s / n,
            'del_rate': d / n,
            'ins_rate': i / n,
            's_count': s,
            'd_count': d,
            'i_count': i
        }
        
        return detailed_results
    
    def aggregate_sequences(self, all_preds):

        if not all_preds: return []
        concatenated = [item for sublist in all_preds for item in sublist]
        if not concatenated: return []
        final_sequence = [concatenated[0]]
        for item in concatenated[1:]:
            if item != final_sequence[-1]:
                final_sequence.append(item)
        return final_sequence

    def train(self, epoch):
        
        self.network.train()

        sum_loss, sum_samples = 0, 0
        progress_bar = tqdm(self.train_dataloader, desc=f"Train Epoch {epoch+1}")

        for batch_idx, (features, feature_lengths, targets, target_lengths) in enumerate(progress_bar):

            features = features.to(self.device)
            feature_lengths = feature_lengths.to(self.device)
            target_lengths = target_lengths.to(self.device)
            targets = targets.to(self.device)

            num_samples_in_batch = features.size(0)
            sum_samples += num_samples_in_batch

            log_probs, output_lengths = self.network(features, feature_lengths)

            loss = self.loss(log_probs, targets, output_lengths, target_lengths)

            if torch.isinf(loss) or torch.isnan(loss):
                print(f"Warning: Skipping batch {batch_idx} due to inf/nan loss")
                continue
                
            sum_loss += loss.item() * num_samples_in_batch
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            current_avg_loss = sum_loss / sum_samples
            progress_bar.set_postfix(loss=f'{current_avg_loss:.4f}')
        
        epoch_avg_loss = sum_loss / sum_samples
        print(f"[LOG] - Train Epoch: {epoch+1:3d} | Average Loss: {epoch_avg_loss:.4f}")


        log_file_path = os.path.join(self.model_log, "epoch_train_metrics.log") 

        
        print("Calculating AER on the validation set...")
        # chunk 测试
        with torch.no_grad():
            chunk_sizes_to_test = [700]
            for chunk_size in chunk_sizes_to_test:
                print(f"\n{'='*20} Testing VAL with CHUNK_SIZE = {chunk_size} {'='*20}")
                stride = chunk_size // 2
                all_gt_seqs = []
                all_final_pred_seqs = []
                sample_idx = 0
                for batch_data in tqdm(self.val_dataloader, desc=f" chunk={chunk_size} "):
                    features, feature_lengths, targets, target_lengths = batch_data
                    features        = features.to(self.device)
                    feature_lengths = feature_lengths.to(self.device)
                    targets         = targets.to(self.device)
                    target_lengths  = target_lengths.to(self.device)

                    valid_T = int(feature_lengths[0].item())
                    full_features = features[0, :valid_T]

                    # --- 分块推理 ---
                    chunk_predictions = []
                    total_frames = full_features.shape[0]

                    for start_frame in range(0, total_frames, stride):
                        end_frame = start_frame + chunk_size
                        chunk_feat_2d = full_features[start_frame:end_frame, :]   # [t, F...]
                        if chunk_feat_2d.shape[0] == 0:
                            continue

                        # 增 batch 维 -> [1, t, F...]
                        chunk_feat = chunk_feat_2d.unsqueeze(0).to(self.device)
                        chunk_len  = torch.tensor([chunk_feat.shape[1]], device=self.device)

                        # 前向 + 解码
                        log_probs, _ = self.network(chunk_feat, chunk_len)
                        pred_seqs = self.greedy_decoder(log_probs.detach().cpu(), blank_id=0)

                        if pred_seqs and len(pred_seqs[0]) > 0:
                            chunk_predictions.append(pred_seqs[0])

                    # 聚合各 chunk 的预测得到最终序列
                    final_pred_seq = self.aggregate_sequences(chunk_predictions) if len(chunk_predictions) > 0 else []
                    all_final_pred_seqs.append(final_pred_seq)

                    # 取 GT（按 target_lengths 截断）
                    gt_len = int(target_lengths[0].item())
                    if targets.dim() == 2:   # [1, L]
                        gt_seq = targets[0, :gt_len].detach().cpu().tolist()
                    else:                    # [L]
                        gt_seq = targets[:gt_len].detach().cpu().tolist()
                    all_gt_seqs.append(gt_seq)
                    
                    gt_str = " ".join(map(str, gt_seq)) if len(gt_seq) > 0 else "<empty>"
                    pred_str = " ".join(map(str, final_pred_seq)) if len(final_pred_seq) > 0 else "<empty>"
                    with open(log_file_path, "a", encoding="utf-8") as f:
                        f.write(f"[sample {sample_idx:06d}]\n")
                        f.write(f"GT: {gt_str}\n")
                        f.write(f"Pred: {pred_str}\n\n")
                    sample_idx += 1

                val_metrics = self.calculate_aer(all_gt_seqs, all_final_pred_seqs)

                val_aer = val_metrics['aer']
                val_sub_rate = val_metrics['sub_rate']
                val_del_rate = val_metrics['del_rate']
                val_ins_rate = val_metrics['ins_rate']

                print(f"[LOG] - Validation Set (Epoch: {epoch+1:3d}):\n"
                      f"[chunk={chunk_size}] "
                      f"AER={val_metrics['aer']:.4f} | "
                      f"sub={val_metrics['sub_rate']:.4f} | "
                      f"del={val_metrics['del_rate']:.4f} | "
                      f"ins={val_metrics['ins_rate']:.4f}")
                
                log_message = (
                        "Time:{time}, Epoch:{epoch}, Loss:{loss:.4f}, "
                        "Chunk:{chunk_size}, "
                        "VAL AER:{val_aer:.4f} (S:{val_s_rate:.4f}, D:{val_d_rate:.4f}, I:{val_i_rate:.4f})\n"
                    )
                
                with open(log_file_path, 'a') as epoch_f:
                    epoch_f.write(log_message.format(
                                time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                epoch=epoch,  # 这里假设你外部有 epoch 变量
                                loss=epoch_avg_loss,  # 这里假设你外部有这个变量
                                chunk_size=chunk_size,
                                val_aer=val_aer,
                                val_s_rate=val_sub_rate,
                                val_d_rate=val_del_rate,
                                val_i_rate=val_ins_rate))


        # # 非chunk测试
        # val_all_gt_seqs = []
        # val_all_pred_seqs = []
        # with torch.no_grad():
        #     # 再次遍历训练集，但这次不进行训练，只为解码
        #     for batch_data in tqdm(self.val_dataloader, desc="  Evaluating on train set"):
        #         features, feature_lengths, targets, target_lengths = batch_data
        #         features = features.to(self.device)
        #         feature_lengths = feature_lengths.to(self.device)
        #         target_lengths = target_lengths.to(self.device)
        #         targets = targets.to(self.device)
                
        #         log_probs_val, output_lengths_val = self.network(features, feature_lengths)
                
        #         pred_seqs = self.greedy_decoder(log_probs_val.cpu(), blank_id=0)
        #         val_all_pred_seqs.extend(pred_seqs)
        #         # decoded_val = self.beam_search_decoder(
        #         #     log_probs_val,       # shape (T, B, C)
        #         #     beam_width=5,
        #         #     blank_id=0
        #         # )
        #         # all_pred_seqs.extend(decoded_val)
                

        #         for i in range(targets.size(0)):
        #             length = target_lengths[i]
        #             val_all_gt_seqs.append(targets[i][:length].cpu().tolist())

        self.save(epoch=epoch, model=self.network, optimizer=self.optimizer, checkpoint_dir=self.checkpoint_dir)





