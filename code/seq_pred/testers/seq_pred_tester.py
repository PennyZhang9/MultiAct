from trainers.base_trainer import *
from misc.require_lib import *
from datasets.dataset import TrainSeqPrediction
from models.seq_pred_model import CTCBaseline, SlowFastConformerCTC
import jiwer
from collections import defaultdict

def seq_pred_collate_fn(batch):
    
    batch.sort(key=lambda x: x['slowfast_feat'].shape[0], reverse=True)

    # convert np.ndarray to torch.Tensor
    features = [torch.tensor(item['slowfast_feat'], dtype=torch.float32) for item in batch]
    labels = [torch.tensor(item['label'], dtype=torch.long) for item in batch]

    feature_lengths = torch.LongTensor([f.shape[0] for f in features])
    label_lengths = torch.LongTensor([l.shape[0] for l in labels])

    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True)
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0) # 使用0作为padding值
        
    return padded_features, feature_lengths, padded_labels, label_lengths

class Tester(baseTrainer):
    def __init__(self, params, model_dir, test_total_json, test_slowfast_feat_index_dir, test_slowfast_feat_dir):
        self.params = params
        self.model_dir = model_dir
        self.model_log = os.path.join(model_dir, "eval_log")
        os.makedirs(self.model_log, exist_ok=True)
        self.device = torch.device(self.params.train_type)
        # BLANK_ID = 0

        print('[LOG] - Loading Test Dataset.')

        self.test_dataset = TrainSeqPrediction(train_json_file=test_total_json,
                                                feat_file_dir=test_slowfast_feat_dir,
                                                feat_index_dir=test_slowfast_feat_index_dir)

        self.test_dataloader = DataLoader(self.test_dataset,
                                           batch_size=self.params.test_batch_size,
                                           num_workers=self.params.num_workers,
                                           pin_memory=True,
                                           collate_fn=seq_pred_collate_fn)
        
        # self.network = CTCBaseline(input_dim=2304, hidden_dim=256, num_classes=self.params.num_classes).to(self.device)
        self.network = SlowFastConformerCTC(slowfast_dim=2304, num_classes=self.params.num_classes).to(self.device)

    def greedy_decoder(self, log_probs, blank_id=0):
        best_paths = torch.argmax(log_probs.transpose(0, 1), dim=2)
        decoded_seqs = []
        for path in best_paths:
            collapsed_path = [p for i, p in enumerate(path) if i == 0 or p != path[i-1]]
            decoded_seq = [p.item() for p in collapsed_path if p != blank_id]
            decoded_seqs.append(decoded_seq)
        return decoded_seqs

    def calculate_aer(self, gt_seqs, pred_seqs):
        total_distance = 0
        total_length = 0
        for gt, pred in zip(gt_seqs, pred_seqs):
            total_distance += editdistance.eval(gt, pred)
            total_length += len(gt)
        return total_distance / total_length if total_length > 0 else float('inf')

    def aggregate_sequences(self, all_preds):

        if not all_preds: return []
        concatenated = [item for sublist in all_preds for item in sublist]
        if not concatenated: return []
        final_sequence = [concatenated[0]]
        for item in concatenated[1:]:
            if item != final_sequence[-1]:
                final_sequence.append(item)
        return final_sequence


    def test_sliding_window(self, resume_model):
        
        model_dir, model_name = os.path.split(resume_model)
        self.load(self.network, None, model_dir, model_name)
        self.network.eval()

        with torch.no_grad():
            chunk_sizes_to_test = [500, 600, 700, 800, 900]
            for chunk_size in chunk_sizes_to_test:
                print(f"\n{'='*20} Testing with CHUNK_SIZE = {chunk_size} {'='*20}")
                stride = chunk_size // 2
                all_gt_seqs = []
                all_final_pred_seqs = []
                for batch_data in tqdm(self.test_dataloader, desc=f" chunk={chunk_size} "):
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

                eval_metrics = self.calculate_aer(all_gt_seqs, all_final_pred_seqs)
                print(f"[chunk={chunk_size}] "
              f"AER={eval_metrics['aer']:.4f} | "
              f"sub={eval_metrics['sub_rate']:.4f} | "
              f"del={eval_metrics['del_rate']:.4f} | "
              f"ins={eval_metrics['ins_rate']:.4f}")
        pass


    def test(self, resume_model):
        
        model_dir, model_name = os.path.split(resume_model)
        self.load(self.network, None, model_dir, model_name)
        self.network.eval()

        all_gt_seqs = []
        all_pred_seqs = []

        progress_bar = tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader))

        with torch.no_grad():
            for batch_idx, batch_data in progress_bar:

                features, feature_lengths, targets, target_lengths = batch_data
                features = features.to(self.device)
                targets = targets.to(self.device)

                log_probs, _ = self.network(features, feature_lengths)

                pred_seqs = self.greedy_decoder(log_probs.cpu(), blank_id=0) # 解码在CPU上进行
                all_pred_seqs.extend(pred_seqs)
                
                for i in range(targets.size(0)):
                    length = target_lengths[i]
                    all_gt_seqs.append(targets[i][:length].cpu().tolist())


        test_aer = self.calculate_aer(all_gt_seqs, all_pred_seqs)
        print(f"\n--- Test Results ---")
        print(f"Model: {resume_model}")
        print(f"Total Test Samples: {len(all_gt_seqs)}")
        print(f"Activity Error Rate (AER): {test_aer:.4f} ({test_aer:.2%})")
        print(f"--------------------\n")


        with open(os.path.join(self.model_log, "test_result_log"), 'a') as log_f: 
            log_f.write(f"Test results for model: {resume_model}\n")
            log_f.write(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
            log_f.write(f"Final Test AER: {test_aer:.4f}\n\n")
            log_f.write("--- Per-sample predictions ---\n")
            for i, (gt, pred) in enumerate(zip(all_gt_seqs, all_pred_seqs)):
                # gt_str = [self.inverse_label_map.get(idx, '?') for idx in gt]
                # pred_str = [self.inverse_label_map.get(idx, '?') for idx in pred]
                log_f.write(f"Sample {i+1}:\n")
                log_f.write(f"  GT:   {gt}\n")
                log_f.write(f"  Pred: {pred}\n")
                log_f.write(f"  AER:  {editdistance.eval(gt, pred) / len(gt) if len(gt) > 0 else float('inf')}\n\n")
            


        # self.save(epoch=epoch, model=self.network, optimizer=self.optimizer, checkpoint_dir=self.checkpoint_dir)




