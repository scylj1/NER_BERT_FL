'''
Data statistics of train, development and test sets
'''
import pandas as pd
import numpy as np

wnuttrain = 'https://storage.googleapis.com/wnut-2017_ner-shared-task/wnut17train_clean_tagged.txt'
train = pd.read_table(wnuttrain, header=None, names=['token', 'label', 'bio_only', 'upos'])
# NB: don't drop the empty lines between texts yet, they are needed for sequence splits (they show up as NaN in the data frame)
train.head(n=30)

# the dev set
wnutdev = 'https://storage.googleapis.com/wnut-2017_ner-shared-task/wnut17dev_clean_tagged.txt'
dev = pd.read_table(wnutdev, header=None, names=['token', 'label', 'bio_only', 'upos'])

# the test set
wnuttest = 'https://storage.googleapis.com/wnut-2017_ner-shared-task/wnut17test_annotated_clean_tagged.txt'
test = pd.read_table(wnuttest, header=None, names=['token','label', 'bio_only', 'upos'])

print(train.describe())
print(train['bio_only'].value_counts())

print(dev.describe())
print(dev['bio_only'].value_counts())


print(test.describe())
print(test['bio_only'].value_counts())

def tokens2sequences(txt_in,istest=False):
  '''
  Takes panda dataframe as input, copies, and adds a sequence index based on full-stops.
  Outputs a dataframe with sequences of tokens, named entity labels, and token indices as lists.
  '''
  txt = txt_in.copy()
  txt['sequence_num'] = 0
  seqcount = 0
  for i in txt.index:  # in each row...
    txt.loc[i,'sequence_num'] = seqcount  # set the sequence number
    if pd.isnull(txt.loc[i,'token']):  # increment sequence counter at empty lines
      seqcount += 1
  # now drop the empty lines, group by sequence number and output df of sequence lists
  txt = txt.dropna()
  if istest:  # looking ahead: the test set doesn't have labels
    txt_seqs = txt.groupby(['sequence_num'],as_index=False)[['token']].agg(lambda x: list(x))
  else:  # the dev and training sets do have labels
    txt_seqs = txt.groupby(['sequence_num'],as_index=False)[['token', 'label']].agg(lambda x: list(x))
  return txt_seqs

print("This cell takes a little while to run: be patient :)")
train_seqs = tokens2sequences(train)
dev_seqs = tokens2sequences(dev)
test_seqs = tokens2sequences(test)

print(train_seqs.describe())
print(dev_seqs.describe())
print(test_seqs.describe())
