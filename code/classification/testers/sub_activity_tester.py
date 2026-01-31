
from trainers.base_trainer import *
from misc.require_lib import *
from models.sub_activity_model import SubactivityNet
from datasets.dataset import TestSubActivityDataset
import slowfast.utils.metrics as metrics

class Tester(baseTrainer):
    def __init__(self, params, model_dir, test_subactivity_path, test_subactivity_label_path, tag_dir):
        self.model_dir = model_dir
        self.model_log = os.path.join(model_dir, "{}_log".format(tag_dir))
        self.embed_dir = os.path.join(model_dir, "{}_subactivity_emb".format(tag_dir))
        os.makedirs(self.model_log, exist_ok=True)
        os.makedirs(self.embed_dir, exist_ok=True)
        self.test_subactivity_path = test_subactivity_path
        # self.test_save_embed_dir = test_save_embed_dir
        self.test_subactivity_label_path = test_subactivity_label_path

        self.params = params
        self.device = torch.device(self.params.train_type)

        print('[LOG] - Loading Evaluation Dataset.')
        self.test_dataset = TestSubActivityDataset(test_subactivity_path=self.test_subactivity_path, 
                                                   test_subactivity_label_path=self.test_subactivity_label_path)
        self.test_dataloader = DataLoader(self.test_dataset,
                                         batch_size=self.params.test_batch_size,
                                         pin_memory=True,
                                         num_workers=self.params.num_workers)

        self.network = SubactivityNet().to(self.device)

    def test(self, resume_model, is_save_embed=False):
        model_dir, model_name = os.path.split(resume_model)
        self.load(self.network, None, model_dir, model_name)
        self.network.eval()

        all_preds = []
        all_labels = []

        all_records = {}

        progress_bar = tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader))

        for batch_idx, (slow_input, fast_input, label, metadata, length) in progress_bar:

            slow_input = slow_input.to(self.device)
            fast_input = fast_input.to(self.device)
            length = length.to(self.device)

            T = slow_input.shape[1]
            mask = torch.ones(1, T, dtype=torch.bool)
            mask = mask.to(self.device)
    
            with torch.no_grad():
                embedding, logits = self.network(slow_input, fast_input, mask, length)
            
            all_preds.append(logits.cpu())
            all_labels.append(label.cpu())

            ann_id = metadata[0]
            
            record = {
            'sub_embed': embedding[0].detach().cpu().numpy(),   # shape (1, 512)
            'meta': ann_id              
            }

            all_records[ann_id] = record
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        topk_correct = metrics.topks_correct(all_preds, all_labels, ks=[1, 5])
        top1_acc = (topk_correct[0] / all_preds.size(0)) * 100.0
        top5_acc = (topk_correct[1] / all_preds.size(0)) * 100.0

        stats = metrics.get_stats(all_preds.numpy(), all_labels.numpy())
        mAP = stats["mAP"]
        mAUC = stats["mAUC"]
        mPCA = stats["mPCA"]

        with open(os.path.join(self.model_log, "test_result_log"), 'a') as log_f:
            log_f.write(
            "Time: {}, Model: {}, Top1: {:.2f}%, Top5: {:.2f}%, mAP: {:.4f}, mAUC: {:.4f}, mPCA: {:.4f}\n".format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            os.path.basename(resume_model),
            top1_acc,
            top5_acc,
            mAP,
            mAUC,
            mPCA))

        if is_save_embed:
            save_path = os.path.join(self.embed_dir, "{}_subactivity_emb.pkl".format(model_name.split('.')[0]))
            with open(save_path, 'wb') as f:
                pickle.dump(all_records, f)
            print(f'Subactivuty Embeddings saved.')

        

