import os
import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
        
class Metric:
    def __init__(self, metric_list, tokenizer):
        self.metric_list = metric_list
        self.tokenizer = tokenizer
        self.post_process = lambda x: ' '.join(
            [i for i in x.replace('\n', '').replace('### Reason:', ' ').split()]
        )
        
    def add_batch(self, preds, labels, ans_only=False):
        self.preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        self.labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        if ans_only:
            self.preds = [self.preds[0].split('### Answer:')[-1]]
            self.labels = [self.labels[0].split('### Answer:')[-1]]
        self.preds = [self.post_process(item) for item in self.preds]
        self.labels = [self.post_process(item) for item in self.labels]
        
        for m in self.metric_list:
            m.add_batch(predictions=self.preds, references=self.labels)
            
        return self.preds, self.labels
    
    def compute(self):
        res = {}
        for m in self.metric_list:
            key = m.__class__.__name__.lower()
            if key=='bertscore':
                f1_scores = m.compute(model_type='bert-base-multilingual-cased', lang='ml')['f1']
                res[key] = sum(f1_scores)/len(f1_scores)
                continue
            res[key] = m.compute()
        return res