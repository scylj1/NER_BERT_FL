'''
evaluate
'''

import numpy as np
import pandas as pd

bert = pd.read_table('val_result/bert.txt', header=None, names=['token','label', 'bio_only', 'upos', 'prediction']).dropna()
weight = pd.read_table('val_result/bert_weight.txt', header=None, names=['token','label', 'bio_only', 'upos', 'prediction']).dropna()
focal = pd.read_table('val_result/bert_focal.txt', header=None, names=['token','label', 'bio_only', 'upos', 'prediction']).dropna()


# Evaluation
# The code is copy from 'Task 2' notebook
def wnut_evaluate(txt):
  #entity evaluation: we evaluate by whole named entities
  npred = 0; ngold = 0; tp = 0
  nrows = len(txt)
  for i in txt.index:
    if txt['prediction'][i]=='B' and txt['bio_only'][i]=='B':
      npred += 1
      ngold += 1
      for predfindbo in range((i+1),nrows):
        if txt['prediction'][predfindbo]=='O' or txt['prediction'][predfindbo]=='B':
          break  # find index of first O (end of entity) or B (new entity)
      for goldfindbo in range((i+1),nrows):
        if txt['bio_only'][goldfindbo]=='O' or txt['bio_only'][goldfindbo]=='B':
          break  # find index of first O (end of entity) or B (new entity)
      if predfindbo==goldfindbo:  # only count a true positive if the whole entity phrase matches
        tp += 1
    elif txt['prediction'][i]=='B':
      npred += 1
    elif txt['bio_only'][i]=='B':
      ngold += 1
  
  fp = npred - tp  # n false predictions
  fn = ngold - tp  # n missing gold entities
  prec = tp / (tp+fp)
  rec = tp / (tp+fn)
  f1 = (2*(prec*rec)) / (prec+rec)
  print('Sum of TP and FP = %i' % (tp+fp))
  print('Sum of TP and FN = %i' % (tp+fn))
  print('True positives = %i, False positives = %i, False negatives = %i' % (tp, fp, fn))
  print('Precision = %.3f, Recall = %.3f, F1 = %.3f' % (prec, rec, f1))

print('New evaluation:')
wnut_evaluate(bert)
wnut_evaluate(weight)
wnut_evaluate(focal)

