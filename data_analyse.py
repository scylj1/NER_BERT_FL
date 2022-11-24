import pandas as pd
import numpy as np
result = 'res.txt'
res = pd.read_table(result, header=None, names=['model',	'seed',	'epoch',	'acc',	'pre',	'rec',	'f1']).dropna()
res_an = pd.read_table('res_an.txt', header=None, names=['model',	'epoch',	'acc',	'pre',	'rec',	'f1', 'stdacc','stdpre','stdrec','stdf1']).dropna()

def stdfunc(list1):
    std=(sum((i-(sum(list1)/len(list1)))**2 for i in list1)/len(list1))**0.5
    return std

models = ['bert', 'bertfocal','bert_weight', 'bertfocal_weight','bert13','bertfocal13','distilbert']
for model in models:
    m_index = res['model']==model 
    #print(m_index)
    m = res[m_index]
    #print(m)
    for epoch in range (1,11):
        e_index = m['epoch'] == str(epoch)
        e = m[e_index]
        #print(e)
        acc = []; pre=[];rec=[];f1=[]
        for index,row in e.iterrows():
            #print(row)
            #print(row['acc'])
            acc.append(float(row['acc']))
            pre.append(float(row['pre']))
            rec.append(float(row['rec']))
            f1.append(float(row['f1']))

        std=(sum((i-(sum(f1)/len(f1)))**2 for i in f1)/len(f1))**0.5
        alist = [model, epoch, sum(acc)/len(acc), sum(pre)/len(pre), sum(rec)/len(rec), sum(f1)/len(f1), stdfunc(acc), stdfunc(pre), stdfunc(rec), stdfunc(f1)]
        res_an.loc[len(res_an)]=alist

res_an.to_csv('res_an.txt', sep='\t', index=False)
print('finished')