'''
data preprocessing, model training
'''
from transformers import DistilBertTokenizerFast, BertTokenizerFast
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch import ones, log, sum, rand_like, cuda
from transformers import BertPreTrainedModel, BertModel, BertForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from seqeval.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import DistilBertForTokenClassification, RobertaForTokenClassification, BertForTokenClassification, DataCollatorForTokenClassification
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch import nn

#------------------------------------------------------------------------------------------------------------------
# preprocessing
wnuttrain = 'https://storage.googleapis.com/wnut-2017_ner-shared-task/wnut17train_clean_tagged.txt'
train = pd.read_table(wnuttrain, header=None, names=[
                      'token', 'label', 'bio_only', 'upos'])
# the dev set
wnutdev = 'https://storage.googleapis.com/wnut-2017_ner-shared-task/wnut17dev_clean_tagged.txt'
dev = pd.read_table(wnutdev, header=None, names=[
                    'token', 'label', 'bio_only', 'upos'])
# the test set
wnuttest = 'https://storage.googleapis.com/wnut-2017_ner-shared-task/wnut17test_annotated_clean_tagged.txt'
test = pd.read_table(wnuttest, header=None, names=[
                     'token', 'label', 'bio_only', 'upos'])

result = pd.read_table('val_result/results.txt', header=None, names=[
                       'model',   'seed',    'epoch',    'acc', 'pre', 'rec', 'f1'])

hyperp = pd.read_table('val_result/gamma.txt', header=None, names=[
                       'model',   'gamma',    'epoch',    'acc', 'pre', 'rec', 'f1'])

# evaluate on only B and I
#is_inside = dev['bio_only']!='O'
#dev = dev[is_inside]

def tokens2sequences(txt_in, istest=False):
    '''
    Takes panda dataframe as input, copies, and adds a sequence index based on full-stops.
    Outputs a dataframe with sequences of tokens, named entity labels, and token indices as lists.
    '''
    txt = txt_in.copy()
    txt['sequence_num'] = 0
    seqcount = 0
    for i in txt.index:  # in each row...
        txt.loc[i, 'sequence_num'] = seqcount  # set the sequence number
        if pd.isnull(txt.loc[i, 'token']):  # increment sequence counter at empty lines
            seqcount += 1
    # now drop the empty lines, group by sequence number and output df of sequence lists
    txt = txt.dropna()
    if istest:  # looking ahead: the test set doesn't have labels
        txt_seqs = txt.groupby(['sequence_num'], as_index=False)[
            ['token']].agg(lambda x: list(x))
    else:  # the dev and training sets do have labels
        txt_seqs = txt.groupby(['sequence_num'], as_index=False)[
            ['token', 'bio_only']].agg(lambda x: list(x))
    return txt_seqs

print("Change tokens to sequences")
train_seqs = tokens2sequences(train)
dev_seqs = tokens2sequences(dev)
test_seqs = tokens2sequences(test)
# print(test_seqs)

def read_wnut(file):
    token_docs = []
    tag_docs = []
    for num in range(len(file['sequence_num'])):
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

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
#tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

train_encodings = tokenizer(train_texts, is_split_into_words=True,
                            return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True,
                           return_offsets_mapping=True, padding=True, truncation=True)
test_encodings = tokenizer(test_texts, is_split_into_words=True,
                             return_offsets_mapping=True, padding=True, truncation=True)

def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:, 0] == 0) & (
            arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)
test_labels = encode_tags(test_tags, test_encodings)

class WNUTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# we don't want to pass this to the model
train_encodings.pop("offset_mapping")
val_encodings.pop("offset_mapping")
test_encodings.pop("offset_mapping")

train_dataset = WNUTDataset(train_encodings, train_labels)
val_dataset = WNUTDataset(val_encodings, val_labels)
test_dataset = WNUTDataset(test_encodings, test_labels)

# length
train_data_size = len(train_dataset)
val_data_size = len(val_dataset)
#------------------------------------------------------------------------------------------------------------
# model defination
class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config, pos_weight=None, focal_loss=False, gamma=1.2, linear_dropout_prob=0.5):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.dropout = nn.Dropout(linear_dropout_prob)
        self.classifier = nn.Linear(
            config.hidden_size, self.config.num_labels)
        self.pos_weight = self.init_pos_weight(pos_weight)
        self.loss_fct = nn.CrossEntropyLoss()
        self.loss_fct_focal = self.init_focal_loss(
            focal_loss, gamma, self.pos_weight)

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
                # print(labels.view(-1))
                loss = self.loss_fct_focal(
                    logits.view(-1, self.num_labels), labels.view(-1))
                # print(loss)
            else:
                loss = self.loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits,
                                        hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)

    def init_pos_weight(self, pos_weight):
        if (pos_weight == None or pos_weight.shape[0] != self.num_labels):
            return None
        return pos_weight.cuda() if cuda.is_available() else pos_weight

    def init_focal_loss(self, focal_loss, gamma, pos_weight):
        if focal_loss:
            return FocalLoss(gamma, pos_weight)
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
        # print(log_pt)
        pt = torch.exp(log_pt)
        #print((1 - pt) ** self.gamma)
        # print(pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(
            log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        # print(loss)
        return loss


#----------------------------------------------------------------------------------------------------------
# model training
epoch = 10
myseeds = [3407]  # , 42, 522, 227, 2]
mygamma=2

for myseed in myseeds:
    num_epoch = 1   
    my_model_name = "bert-base-cased"
    my_config = AutoConfig.from_pretrained(my_model_name, num_labels=len(
        unique_tags), id2label=id2tag, label2id=tag2id)

    data_collator = DataCollatorForTokenClassification(tokenizer)
    #model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=len(unique_tags))
    #model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(unique_tags))
    device = torch.device('cuda')

    model = (BertForMultiLabelClassification.from_pretrained(my_model_name, config=my_config, focal_loss=False, pos_weight=torch.tensor([0.143, 0.952, 1.905])))  # pos_weight=torch.tensor([0.143, 0.952, 1.905])

    def align_predictions(predictions, label_ids):
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        labels_list, preds_list = [], []
        for batch_idx in range(batch_size):
            example_labels, example_preds = [], []
            for seq_idx in range(seq_len):
                # Ignore label IDs = -100
                if label_ids[batch_idx, seq_idx] != -100:
                    example_labels.append(
                        id2tag[label_ids[batch_idx][seq_idx]])
                    example_preds.append(id2tag[preds[batch_idx][seq_idx]])
            labels_list.append(example_labels)
            preds_list.append(example_preds)
        return preds_list, labels_list

    def compute_metrics(eval_pred):
        #print(eval_pred)
        y_pred, y_true = align_predictions(
            eval_pred.predictions, eval_pred.label_ids)
        # print(y_pred[0:10])
        # print(y_true[0:10])
        global num_epoch
        alist = ['bert_weight_onlyBI', myseed, num_epoch, accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)]
        #alist = ['focal3407', mygamma, num_epoch, accuracy_score(y_true, y_pred), precision_score( y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)]
        result.loc[len(result)] = alist
        #hyperp.loc[len(hyperp)] = alist
        num_epoch = num_epoch+1
        #result.append({'f1':f1_score(y_true, y_pred)}, ignore_index=True)
        return {"accuracy": accuracy_score(y_true, y_pred), "precision": precision_score(y_true, y_pred), 
                "recall": recall_score(y_true, y_pred), "f1": f1_score(y_true, y_pred)}
    
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=epoch,              # total number of training epochs
        learning_rate=2e-5,
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        # warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        evaluation_strategy="epoch",
        disable_tqdm=False,
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        seed=myseed,

    )

    trainer = Trainer(
        model=model,                        # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        tokenizer=tokenizer,
        # data_collator=data_collator,
    )
    
    trainer.train()
    
#--------------------------------------------------------------------------------------------------------
# save results

#result.to_csv('./val_result/results.txt', sep='\t', index=False)
#trainer.evaluate()
#model.save_pretrained("./model/%s-%sepoch" % ('bert1.5', 10))
#hyperp.to_csv('./val_result/gamma.txt', sep='\t', index=False)

res = trainer.predict(test_dataset) 
y_pred, y_true = align_predictions(res.predictions, res.label_ids)
# print(y_pred)
bio_preds = []
for y in y_pred:
    for p in y:
        bio_preds.append(p)
test = test.dropna()
test['prediction'] = bio_preds
print(test.describe())
print(test['prediction'].value_counts())
print(res.metrics["test_f1"])
#test.to_csv('./test_result/test_result13.txt', sep='\t', index=False)
result.to_csv('./val_result/results.txt', sep='\t', index=False)

'''
val = dev.dropna()
val['prediction'] = bio_preds
print(val.describe())
print(val['prediction'].value_counts())
print(res.metrics["test_f1"])
val.to_csv('bert_focal.txt', sep='\t', index=False)
'''
