from misc.require_lib import *
from misc.utils import Params
from testers.activity_tester import Tester


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--model-dir", type=str, default=None)
parser.add_argument("--test-event-embed", type=str, default=None)
parser.add_argument("--test-subactivity-embed", type=str, default=None)
parser.add_argument("--test-activity-label-path", type=str, default=None)
parser.add_argument("--test-subactivity-label-path", type=str, default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    params = Params(args.config)

    test_model_list = glob.glob(args.model_dir + '/checkpoint/*.pickle')
    tag_dir = 'val' # val, eval
    tester = Tester(params, args.model_dir, args.test_event_embed, args.test_subactivity_embed, args.test_activity_label_path, args.test_subactivity_label_path, tag_dir)

    for resume_model in test_model_list:
        # tester.test_only(resume_model=resume_model)
        tester.test_joint(resume_model=resume_model)


