from misc.require_lib import *
from misc.utils import Params
from testers.sub_activity_tester import Tester

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--model-dir", type=str, default=None)
parser.add_argument("--test_subactivity_path", type=str, default=None)
parser.add_argument("--test_subactivity_label_path", type=str, default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    params = Params(args.config)

    test_model_list = glob.glob(args.model_dir + '/checkpoint/*.pickle')
    tag_dir = 'evaluation' # val, train
    tester = Tester(params, args.model_dir, args.test_subactivity_path, args.test_subactivity_label_path, tag_dir)

    for resume_model in test_model_list:
        tester.test(resume_model=resume_model, is_save_embed=True)


