from misc.require_lib import *
from trainers.sub_activity_trainer import Trainer
from misc.utils import save_codes_and_config

parser = argparse.ArgumentParser()
parser.add_argument("--config",                  type=str, help="The config file path.",        default=None)
parser.add_argument("--model-dir",               type=str, help="The models directory.",        default=None)
parser.add_argument("--train-event-embed",       type=str, help="Training event embedding.",    default=None)
parser.add_argument("--train-subactivity-label", type=str, help="Training sub activity label.", default=None)

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
                      train_event_embed_path=args.train_event_embed,
                      train_subactivity_label_path=args.train_subactivity_label)

    start_epoch = 0
    for epoch in range(start_epoch, params.num_epochs):
        trainer.train(epoch)