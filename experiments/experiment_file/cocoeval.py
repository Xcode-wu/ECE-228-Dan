import os
import glob
import sys
import pickle

import numpy as np

PROJECT_ROOT = os.path.abspath("../../")

sys.path.append(PROJECT_ROOT)

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

EVAL_METRICS = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]

class EvaluationScorer(object):
    def __init__(self):
        print('Initializing COCO-EVAL scorer')
            
    def score(self, ground_truths, predictions, ids, result_file):
        self.eval_results = {}
        self.img_eval = {}
        gt = {}
        pred = {}
        for ID in ids:
            gt[ID] = ground_truths[ID]
            pred[ID] = predictions[ID]

        tokenizer = PTBTokenizer()
        gt = tokenizer.tokenize(gt)
        pred = tokenizer.tokenize(pred)

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        eval_results = {}
        self.final_scores = []
        for scorer, method in scorers:
            print('Computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gt, pred)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.set_eval_results(sc, m)
                    self.set_img_eval_scores(scs, ids, m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.set_eval_results(score, method)
                self.set_img_eval_scores(scores, ids, method)
                print("%s: %0.3f"%(method, score))
        
        print()
        for metric in EVAL_METRICS:
            self.final_scores.append(self.eval_results[metric])
        self.final_scores = np.array(self.final_scores)

        return self.eval_results
    
    def set_eval_results(self, score, method):
        self.eval_results[method] = score

    def set_img_eval_scores(self, scores, img_ids, method):
        for img_id, score in zip(img_ids, scores):
            if not img_id in self.img_eval:
                self.img_eval[img_id] = {}
                self.img_eval[img_id]["image_id"] = img_id
            self.img_eval[img_id][method] = score

def read_predictions(prediction_file):
    ground_truths = {}
    predictions = {}
    with open(prediction_file, 'r') as f:
        lines = f.read().split('\n')

    for i in range(0, len(lines) - 4, 4):
        id_line = lines[i+1]
        gt_line = lines[i+2]
        pd_line = lines[i+3]
            
        curr_gt = {}
        curr_gt['image_id'] = id_line
        curr_gt['cap_id'] = 0
        curr_gt['caption'] = gt_line
        ground_truths[id_line] = [curr_gt]
            
        curr_pd = {}
        curr_pd['image_id'] = id_line
        curr_pd['caption'] = pd_line
        predictions[id_line] = [curr_pd]
    
    return ground_truths, predictions

def evaluate_iit_v2c():
    prediction_files = sorted(glob.glob(os.path.join(PROJECT_ROOT, 'checkpoints', 'prediction', '*.txt')))
    
    scorer = EvaluationScorer()
    max_scores = np.zeros((len(EVAL_METRICS), ), dtype=np.float32)
    max_file = None
    for prediction_file in prediction_files:
        ground_truths, predictions = read_predictions(prediction_file)
        ids = list(ground_truths.keys())
        scorer.score(ground_truths, predictions, ids, prediction_file)
        if np.sum(scorer.final_scores) > np.sum(max_scores):
            max_scores = scorer.final_scores
            max_file = prediction_file

    print('Maximum Score with file', max_file)
    for i in range(len(max_scores)):
        print('%s: %0.3f' % (EVAL_METRICS[i], max_scores[i]))

if __name__ == '__main__':
    evaluate_iit_v2c()
