from misc.require_lib import *
from trainers.caption_trainer import Trainer
from misc.utils import save_codes_and_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='nnet_conf/caption.json')
parser.add_argument("--model-dir", type=str, default='exp')
parser.add_argument("--train-total-json", type=str, default='train/activity_label.json')
parser.add_argument("--train-slowfast-feat-index-dir", type=str, default='data/slowfast_feats/annotation_files')
parser.add_argument("--train-slowfast-feat-dir", type=str, default='data/slowfast_feats/feats')


if __name__ == '__main__':
    args = parser.parse_args() 

    params = save_codes_and_config(config=args.config, model_dir=args.model_dir)

    model_dir = os.path.join(args.model_dir, "nnet")
    os.environ['CUDA_VISIBLE_DEVICES'] = params.train_gpu_id
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    trainer = Trainer(params=params,
                      model_dir=args.model_dir,
                      train_total_json=args.train_total_json,
                      train_slowfast_feat_index_dir=args.train_slowfast_feat_index_dir,
                      train_slowfast_feat_dir=args.train_slowfast_feat_dir)

    start_epoch = 0
    for epoch in range(start_epoch, params.num_epochs):
        trainer.train_epoch(epoch)