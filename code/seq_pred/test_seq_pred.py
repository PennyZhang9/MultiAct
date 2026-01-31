from misc.require_lib import *
from misc.utils import Params
from testers.seq_pred_tester import Tester

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='nnet_conf/seq_pred.json')
parser.add_argument("--model-dir", type=str, default='exp')
parser.add_argument("--test-total-json", type=str, default='evaluation/activity.json')
parser.add_argument("--test-slowfast-feat-index-dir", type=str, default='data/slowfast_feats/annotation_files')
parser.add_argument("--test-slowfast-feat-dir", type=str, default='data/slowfast_feats/feats')


if __name__ == '__main__':
    args = parser.parse_args()  # 外部传入  
    params = Params(args.config)

    test_model_list = glob.glob(args.model_dir + '/checkpoint/*.pickle')
    test_model_list = ['{}/checkpoint/{}.pickle'.format(args.model_dir, '223')]

    tester = Tester(params=params,
                    model_dir=args.model_dir,
                    test_total_json=args.test_total_json,
                    test_slowfast_feat_index_dir=args.test_slowfast_feat_index_dir,
                    test_slowfast_feat_dir=args.test_slowfast_feat_dir)

    for resume_model in test_model_list:
        tester.test(resume_model=resume_model)
