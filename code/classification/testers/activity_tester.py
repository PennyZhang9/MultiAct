
from trainers.base_trainer import *
from misc.require_lib import *
import slowfast.utils.metrics as metrics
from models.activity_model import ActivityNet, HierarchicalActivityNet
from Mydatasets.dataset import TestActivityDataset_only, TestActivityDataset_joint


def hierarchical_collate_fn(batch):

    activity_labels = []

    all_sub_activity_labels = []
    all_event_slow_features = []
    all_event_fast_features = []

    # 每个
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


class Tester(baseTrainer):
    def __init__(self, params, model_dir, test_event_embed, test_subactivity_embed, test_activity_label_path, test_subactivity_label_path, tag_dir):
        self.model_dir = model_dir
        self.model_log = os.path.join(model_dir, "{}_log".format(tag_dir))
        os.makedirs(self.model_log, exist_ok=True)

        self.params = params
        self.device = torch.device(self.params.train_type)

        print('[LOG] - Loading Evaluation Dataset.')
        # self.test_dataset_only_activity = TestActivityDataset_only(test_subactivity_embed_path=test_subactivity_embed,
        #                                                            test_activity_label_path=test_activity_label_path)
        # self.test_dataloader = DataLoader(self.test_dataset_only_activity,
        #                                  batch_size=self.params.test_batch_size,
        #                                  pin_memory=True,
        #                                  num_workers=self.params.num_workers)
        
        self.test_dataset_joint = TestActivityDataset_joint(test_event_embed_path=test_event_embed, 
                                                            test_subactivity_label_path=test_subactivity_label_path, 
                                                            test_activity_label_path=test_activity_label_path)
        self.test_dataloader = DataLoader(self.test_dataset_joint,
                                         batch_size=self.params.test_batch_size,
                                         num_workers=1,
                                         collate_fn=hierarchical_collate_fn)

        
        # self.network = ActivityNet().to(self.device)
        self.network = HierarchicalActivityNet().to(self.device)
    
    def test_only(self, resume_model):

        model_dir, model_name = os.path.split(resume_model)
        self.load(self.network, None, model_dir, model_name)
        self.network.eval()

        all_preds = []
        all_labels = []

        progress_bar = tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader))

        for batch_idx, (sub_seq, label, length) in progress_bar:

            sub_seq = sub_seq.to(self.device)
            length = length.to(self.device)

            T = sub_seq.shape[1]
            mask = torch.ones(1, T, dtype=torch.bool)
            mask = mask.to(self.device)

            with torch.no_grad():
                _, logits = self.network(sub_seq, mask, length)
            
            all_preds.append(logits.cpu())
            all_labels.append(label.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        topk_correct = metrics.topks_correct(all_preds, all_labels, ks=[1])
        top1_acc = (topk_correct[0] / all_preds.size(0)) * 100.0

        stats = metrics.get_stats(all_preds.numpy(), all_labels.numpy())
        mAP = stats["mAP"]
        mAUC = stats["mAUC"]
        mPCA = stats["mPCA"]

        with open(os.path.join(self.model_log, "test_result_log"), 'a') as log_f:
            log_f.write(
            "Time: {}, Model: {}, Top1: {:.2f}%, mAP: {:.4f}, mAUC: {:.4f}, mPCA: {:.4f}\n".format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            os.path.basename(resume_model),
            top1_acc,
            mAP,
            mAUC,
            mPCA))

        pass

    def test_joint(self, resume_model):
        
        model_dir, model_name = os.path.split(resume_model)
        self.load(self.network, None, model_dir, model_name)
        self.network.eval()

        all_subactivity_preds = []
        all_activity_preds = []

        all_subactivity_labels = []
        all_activity_labels = []
        
        all_pred_id = []

        progress_bar = tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader))

        for batch_idx, batch in progress_bar:

            with torch.no_grad():
                outputs = self.network(batch)
                activity_logits = outputs['main_logits']
                sub_logits = outputs['sub_logits']

                pred_class_ids = torch.argmax(activity_logits, dim=1)
                all_pred_id.append(pred_class_ids.cpu())
                # print(f"Sample {batch_idx} predicted class ID: {batch_idx.item()}")

                activity_labels = batch['activity_labels'].to(self.device)
                sub_labels = batch['sub_labels'].to(self.device)

                all_activity_preds.append(activity_logits.cpu())
                all_activity_labels.append(activity_labels.cpu())

                all_subactivity_preds.append(sub_logits.cpu())
                all_subactivity_labels.append(sub_labels.cpu())
        
        all_activity_preds = torch.cat(all_activity_preds)
        all_activity_labels = torch.cat(all_activity_labels)

        all_subactivity_preds = torch.cat(all_subactivity_preds)
        all_subactivity_labels = torch.cat(all_subactivity_labels)
        
        # ======== Activity 分类指标 ========
        topk_activity = metrics.topks_correct(all_activity_preds, all_activity_labels, ks=[1])
        activity_top1_acc = topk_activity[0] / all_activity_preds.size(0) * 100.0

        stats_activity = metrics.get_stats(all_activity_preds.numpy(), all_activity_labels.numpy())
        activity_mAP = stats_activity["mAP"]
        activity_mAUC = stats_activity["mAUC"]
        activity_mPCA = stats_activity["mPCA"]

        # ======== Sub-Activity 分类指标 ========
        topk_sub = metrics.topks_correct(all_subactivity_preds, all_subactivity_labels, ks=[1, 5])
        subactivity_top1_acc = topk_sub[0] / all_subactivity_preds.size(0) * 100.0
        subactivity_top5_acc = topk_sub[1] / all_subactivity_preds.size(0) * 100.0

        stats_sub = metrics.get_stats(all_subactivity_preds.numpy(), all_subactivity_labels.numpy())
        subactivity_mAP = stats_sub["mAP"]
        subactivity_mAUC = stats_sub["mAUC"]
        subactivity_mPCA = stats_sub["mPCA"]

        # ======== 写入日志 ========
        with open(os.path.join(self.model_log, "test_result_log"), 'a') as log_f:
            log_f.write(
            "Time: {}, Model: {}\n".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), os.path.basename(resume_model))
        )
            log_f.write(
            "Activity   => Top1: {:.2f}%, mAP: {:.4f}, mAUC: {:.4f}, mPCA: {:.4f}\n".format(
                activity_top1_acc, activity_mAP, activity_mAUC, activity_mPCA
            )
        )
            log_f.write(
            "SubActivity => Top1: {:.2f}%, Top5: {:.2f}% mAP: {:.4f}, mAUC: {:.4f}, mPCA: {:.4f}\n".format(
                subactivity_top1_acc, subactivity_top5_acc, subactivity_mAP, subactivity_mAUC, subactivity_mPCA
            )
        )
            log_f.write("\nPredicted Activity Class IDs:\n")
            for pred_ids in all_pred_id:
                ids_str = " ".join(map(str, pred_ids.tolist()))
                log_f.write(f"{ids_str}\n")

        pass
