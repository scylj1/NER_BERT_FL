import pandas as pd
import numpy as np
result = 'res.txt'
res = pd.read_table(result, header=None, names=['model',	'seed',	'epoch',	'acc',	'pre',	'rec',	'f1']).dropna()
res_an = pd.read_table('res_an.txt', header=None, names=['model',	'epoch',	'acc',	'pre',	'rec',	'f1']).dropna()

models = ['bert', 'bertfocal','bert_weight', 'bertfocal_weight','bert13','bertfocal13']
for model in models:
    m_index = res['model']==model 
    #print(m_index)
    m = res[m_index]
    #print(m)
    for epoch in range (1,11):
        e_index = m['epoch'] == str(epoch)
        e = m[e_index]
        #print(e)
        acc = 0
        pre=0;rec=0;f1=0
        for index,row in e.iterrows():
            #print(row)
            #print(row['acc'])
            acc = acc+float(row['acc'])
            pre = pre+float(row['pre'])
            rec = rec+float(row['rec'])
            f1 = f1+float(row['f1'])

        alist = [model, epoch, acc/5, pre/5, rec/5, f1/5]
        res_an.loc[len(res_an)]=alist

res_an.to_csv('res_an.txt', sep='\t', index=False)