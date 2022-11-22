# -*- coding: utf-8 -*-
"""task3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Kx0_B6Ngn8-_h0le4OtC0ct_J85AzZ4q
"""
import torch
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
    txt_seqs = txt.groupby(['sequence_num'],as_index=False)[['token', 'bio_only']].agg(lambda x: list(x))
  return txt_seqs

print("This cell takes a little while to run: be patient :)")
train_seqs = tokens2sequences(train)
train_seqs.head()

dev_seqs = tokens2sequences(dev)

test_seqs = tokens2sequences(test)
#print(test_seqs)

def read_wnut(file):
    
    token_docs = []
    tag_docs = []

    for num in range (len(file['sequence_num'])):
        
        token_docs.append(file['token'][num])
        tag_docs.append(file['bio_only'][num])

    return token_docs, tag_docs

train_texts, train_tags = read_wnut(train_seqs)
print(train_texts[0][10:17], train_tags[0][10:17], sep='\n')
val_texts, val_tags = read_wnut(dev_seqs)
test_texts, test_tags = read_wnut(test_seqs)
print(test_texts[0][10:17], test_tags[0][10:17], sep='\n')
unique_tags = set(tag for doc in train_tags for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

from transformers import DistilBertTokenizerFast, RobertaTokenizerFast,BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
#tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large',add_prefix_space=True)
#tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
test_encodings = tokenizer(test_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

import numpy as np

def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)
test_labels = encode_tags(test_tags, test_encodings)

import torch

class WNUTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_encodings.pop("offset_mapping") # we don't want to pass this to the model
val_encodings.pop("offset_mapping")
test_encodings.pop("offset_mapping")

train_dataset = WNUTDataset(train_encodings, train_labels)
val_dataset = WNUTDataset(val_encodings, val_labels)
test_dataset = WNUTDataset(test_encodings, test_labels)

# length 长度
train_data_size = len(train_dataset)
val_data_size = len(val_dataset)

import torch.nn as nn
from torch import ones, log, sum, rand_like, cuda
from transformers import BertPreTrainedModel, BertModel,BertForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config, pos_weight = None, focal_loss = False, gamma=2, linear_dropout_prob = 0.5):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.dropout = nn.Dropout(linear_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.pos_weight = self.init_pos_weight(pos_weight)
        self.loss_fct = nn.CrossEntropyLoss()
        self.loss_fct_focal = self.init_focal_loss(focal_loss, gamma, self.pos_weight)
        
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
            position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)

        # Apply classifier to encoder representation
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)

        if labels is not None:
            if self.loss_fct_focal is not None:
                #print(logits.view(-1, self.num_labels))
                #print(labels.view(-1))
                loss = self.loss_fct_focal(logits.view(-1, self.num_labels), labels.view(-1))
                #print(loss)
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, 
                                    hidden_states=outputs.hidden_states,
                                    attentions=outputs.attentions)
    
    def init_pos_weight(self, pos_weight):
        if(pos_weight == None or pos_weight.shape[0] != self.num_labels):
            return None
        return pos_weight.cuda() if cuda.is_available() else pos_weight

    def init_focal_loss(self, focal_loss, gamma, pos_weight):
        if focal_loss:
            return FocalLoss(gamma,  pos_weight)
        else: 
            focal_loss = None
        
    
class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        log_pt = torch.log_softmax(input, dim=1)
        #print(log_pt)
        pt = torch.exp(log_pt)
        #print((1 - pt) ** self.gamma)        
        #print(pt)
        log_pt = (1 - pt) ** self.gamma * log_pt       
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        #print(loss)
           
        return loss

from transformers import AutoTokenizer
my_model_name = "bert-base-cased"

from transformers import AutoConfig
#会覆盖掉原来的默认值配置
my_config = AutoConfig.from_pretrained(my_model_name,num_labels=len(unique_tags),id2label=id2tag, label2id=tag2id)


from transformers import DistilBertForTokenClassification, RobertaForTokenClassification, BertForTokenClassification, DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer)
#model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=len(unique_tags))
#model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(unique_tags))
#model = RobertaForTokenClassification.from_pretrained('roberta-large', num_labels=len(unique_tags))
device = torch.device('cuda')

model = (BertForMultiLabelClassification.from_pretrained(my_model_name, config=my_config, focal_loss=False, pos_weight=torch.tensor([0.36, 6.21, 12.44])))

import numpy as np
def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []
    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            # Ignore label IDs = -100
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(id2tag[label_ids[batch_idx][seq_idx]])
                example_preds.append(id2tag[preds[batch_idx][seq_idx]])
        labels_list.append(example_labels)
        preds_list.append(example_preds)
    return preds_list, labels_list

from seqeval.metrics import f1_score, accuracy_score, precision_score, recall_score
def compute_metrics(eval_pred):
    y_pred, y_true = align_predictions(eval_pred.predictions,eval_pred.label_ids)
    #print(y_pred[0:10])
    #print(y_true[0:10])
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": precision_score(y_true, y_pred), "recall": recall_score(y_true, y_pred), "f1": f1_score(y_true, y_pred)}

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch import nn
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        #print(inputs)
        labels = inputs.get("labels")
        #print(labels)
        # forward pass
        #model = model.to(torch.device('cuda'))
        #model.cuda()
        outputs = model(**inputs)
        #print(outputs)
        logits = outputs.get("logits")
        #print(logits)
        #print(logits.view(-1, self.model.config.num_labels))
        #print(labels.view(-1))
        softmax_func=nn.Softmax(dim=1)
        soft_output=softmax_func(logits.view(-1, self.model.config.num_labels))
        print('soft_output:\n',soft_output)
        logpt = torch.gather(soft_output, dim=1, index=labels.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        # compute custom loss (suppose one has 3 labels with different weights)
        #loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.36, 6.92, 11.28]).to(device))
        loss_fct = nn.CrossEntropyLoss()
        #loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.36, 6.21, 12.44]).to(device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=2,              # total number of training epochs
    learning_rate=2e-5,
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    #warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    seed=3407,
    
)

#trainer = CustomTrainer(  
trainer =  Trainer(
    model=model,                        # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    tokenizer=tokenizer,
    #data_collator=data_collator,
)

trainer.train()

trainer.evaluate()

#model.save_pretrained("./model/%s-%sepoch" % ('distilbert', 20))

res = trainer.predict(test_dataset) #.metrics["test_f1"]
y_pred, y_true = align_predictions(res.predictions,res.label_ids)

#print(y_pred)
bio_preds = []
for y in y_pred:
    for p in y:
        bio_preds.append(p)

test = test.dropna()
test['prediction'] = bio_preds
print(test.describe())
print(test['prediction'].value_counts())
print(res.metrics["test_f1"])
test.to_csv('test2.txt', sep='\t', index=False)
