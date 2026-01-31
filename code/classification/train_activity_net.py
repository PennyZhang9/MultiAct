from misc.require_lib import *
from trainers.activity_trainer import Trainer
from misc.utils import save_codes_and_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--model-dir", type=str, default=None)
parser.add_argument("--train-event-embed", type=str, default=None)
parser.add_argument("--train-subactivity-embed", type=str, default=None)
parser.add_argument("--train-subactivity-label", type=str, default=None)
parser.add_argument("--train-activity-label", type=str, default=None)

if __name__ == '__main__':
    args = parser.parse_args()

    params = save_codes_and_config(config=args.config, model_dir=args.model_dir)

    model_dir = os.path.join(args.model_dir, "nnet")
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    trainer = Trainer(params=params,
                      model_dir=args.model_dir,
                      train_event_embed_path=args.train_event_embed,
                      train_subactivity_embed_path=args.train_subactivity_embed,
                      train_subactivity_label_path=args.train_subactivity_label,
                      train_activity_label_path=args.train_activity_label)

    start_epoch = 0
    for epoch in range(start_epoch, params.num_epochs):
        # trainer.train(epoch)
        trainer.hierarchical_train(epoch)