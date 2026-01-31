
import re, os
import json
import shutil
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def _java_available() -> bool:
    return shutil.which("java") is not None

if _java_available():
    try:
        from pycocoevalcap.meteor.meteor import Meteor
    except Exception:
        print("Warning: METEOR library could not be imported.")
        Meteor = None
else:
    print("Warning: Java is not found. METEOR score will not be calculated.")
    Meteor = None

def calculate_all_metrics_pycocoevalcap(predictions, references):

    gts = {
        key: [{"caption": s.strip()} for s in val_list]
        for key, val_list in references.items()
    }
    res = {
        key: [{"caption": val_str.strip()}]
        for key, val_str in predictions.items()
    }
    
    try:
        tokenizer = PTBTokenizer()
        gts_tokenized = tokenizer.tokenize(gts)
        res_tokenized = tokenizer.tokenize(res)
    except Exception as e:
        print(f"ERROR: PTBTokenizer failed. Is Java installed? Error: {e}")
        return {}

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    if Meteor:
        scorers.append((Meteor(), "METEOR"))
    
    eval_metrics = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts_tokenized, res_tokenized)
        if isinstance(method, list):
            for i, m in enumerate(method):
                eval_metrics[m] = score[i]
        else:
            eval_metrics[method] = score
            
    return eval_metrics


def parse_and_evaluate_log(log_file_path):

    predictions = []
    references = []


    ref_pattern = re.compile(r"REFERENCE\s*:\s*(.*)")
    pred_pattern = re.compile(r"PREDICTION:\s*(.*)")

    print(f"Reading log file: {log_file_path}")
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                ref_match = ref_pattern.search(line)
                if ref_match:
                    references.append(ref_match.group(1).strip())
                    continue
                
                pred_match = pred_pattern.search(line)
                if pred_match:
                    predictions.append(pred_match.group(1).strip())
                    
    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_file_path}'")
        return


    if len(predictions) != len(references):
        print(f"Warning: Found {len(references)} references but {len(predictions)} predictions. Evaluation might be skewed.")

        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
        
    if not predictions:
        print("No prediction/reference pairs found in the log file.")
        return

    print(f"Found {len(predictions)} prediction/reference pairs to evaluate.")


    preds_dict = {i: p for i, p in enumerate(predictions)}
    refs_dict = {i: [r] for i, r in enumerate(references)}
    

    metrics = calculate_all_metrics_pycocoevalcap(preds_dict, refs_dict)


    if metrics:
        print("\n" + "="*50)
        print(f"      PERFORMANCE REPORT FOR {os.path.basename(log_file_path)}")
        print("="*50)
        metric_order = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE_L", "CIDEr"]
        for name in metric_order:
            if name in metrics:
                score = metrics[name]
                print(f"  - {name:<10}: {score * 100:.2f}")
        print("="*50)
    else:
        print("\nCould not compute evaluation metrics from the log file.")

