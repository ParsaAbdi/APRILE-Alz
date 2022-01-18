#!/usr/bin/env python
# coding: utf-8

# In[1]:


from aprile.model import *
from torch_geometric.data import Data
from aprile.utils import sparse_id

# load data from file
gdata = AprileQuery.load_from_pkl('gdata_dict.pkl')
# convert data type to PyG Data
gdata = Data.from_dict(gdata)

# add proteins' features to data
gdata.p_feat = sparse_id(gdata.n_prot)

aprile = Aprile(gdata, device='cuda')


from aprile.model import *
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os


# In[2]:


# basedir = 'out/CTD'
# files = os.listdir(basedir)
# print(len(files))


# ### Alzheimer's
# - not included as a side effect
# - drugs whose indication is Alzheimer's: 
#     - not in sider
#     - drugbank?

# In[3]:


df_db = pd.read_csv('drugbank.tsv', sep='\t')


# In[4]:


df_db.description.fillna('', inplace=True)
df_db.categories.fillna('', inplace=True)


# In[5]:


df_db[df_db.description.str.contains('lzheimer')].shape


# In[6]:


from torch_geometric.data import Data

# load data from file
gdata = AprileQuery.load_from_pkl('gdata_dict.pkl')
# convert data type  to PyG Data
gdata = Data.from_dict(gdata)


# In[7]:


### Alzheimer's not a side effect
for k,v in gdata.side_effect_name_to_idx.items():
    if 'lzheimer' in k:
        print(k,v)

ses = [k.lower() for k in gdata.side_effect_name_to_idx.keys()]


# In[8]:


df_indication = pd.read_csv('meddra_all_indications.tsv', sep='\t',
                           names=['CID','ID','method','concept',
                                  'concept_type','concept_id','concept_name'])


# In[9]:


### Alzheimer's drugs in SIDER but these aren't in Pose
df_indication[df_indication.concept_name.str.contains('lzheimer')].shape


# In[10]:


pose_cids = list(gdata.drug_id_to_idx.keys())


# In[11]:


df_indication.head()


# In[12]:


df_indication[df_indication.CID.isin(pose_cids)]


# In[13]:


drug_names = list(gdata.drug_idx_to_name.values())
drug_names = [d.lower() for d in drug_names]


# In[14]:


df_db_alz = df_db[df_db.description.str.contains('lzheimer')]
df_db_alz.shape


# In[15]:


df_db_dementia = df_db[df_db.description.str.lower().str.contains('dement')]
df_db_dementia.shape


# In[16]:


df_drugs_alz = df_db_alz[df_db_alz.name.str.lower().isin(drug_names)]
print('Alzheimer in description:', df_drugs_alz.shape[0])
df_drugs_alz.iloc[1].description


# In[17]:


df_drugs_dementia = df_db_dementia[df_db_dementia.name.str.lower().isin(drug_names)]
print('Dementia in description:', df_drugs_dementia.shape[0])
df_drugs_dementia.iloc[0].description


# Alzheimer's drug in Pose data
# - Donepezil: mainly used for Alzheimer's treatment. Used to increase cortical acetylcholine.
#     - Drugbank: "Donepezil (Aricept), is a centrally acting reversible acetyl cholinesterase inhibitor. Its main therapeutic use is in the treatment of Alzheimer's disease where it is used to increase cortical acetylcholine. Donepezil is postulated to exert its therapeutic effect by enhancing cholinergic function. This is accomplished by increasing the concentration of acetylcholine through reversible inhibition of its hydrolysis by acetylcholinesterase. If this proposed mechanism of action is correct, donepezil's effect may lessen as the disease process advances and fewer cholinergic neurons remain functionally intact. Donepezil has been tested in other cognitive disorders including Lewy body dementia and Vascular dementia, but it is not currently approved for these indications. Donepezil has also been studied in patients with Mild Cognitive Impairment, schizophrenia, attention deficit disorder, post-coronary bypass cognitive impairment, cognitive impairment associated with multiple sclerosis, and Down syndrome."
# - Memantine: under investigation for Alzheimer's but no clinical support.
#     - "Memantine is an amantadine derivative with low to moderate-affinity for NMDA receptors. It is a noncompetitive NMDA receptor antagonist that binds preferentially to NMDA receptor-operated cation channels. It  blocks the effects of excessive levels of glutamate that may lead to neuronal dysfunction. It is under investigation for the treatment of Alzheimer's disease, but there has been no clinical support for the prevention or slowing of disease progression."

# Some dementia drugs in Pose data:
# - Risperidone: antipsychotic drug. High affinity for 5-HT and dopamine D2 receptors
#     - manage schizophrenia, inappropriate behavior in severe dementia and manic episodes associated with bipolar I disorder
# - Donepezil (Aricept): primarily for Alzheimer's. Not approved for but tested for Lewy body dementia, Vascular dementia. Studied in Mild Cognitive Impairment, schizophrenia, attention deficit disorder, post-coronary bypass cognitive impairment, cognitive impairment associated with multiple sclerosis, Down syndrome.
# - Citalopram: Antidepressant. Unlabeled indications include mild dementia-associated agitation in nonpsychotic patients, smoking cessation, ethanol abuse, obsessive-compulsive disorder (OCD) in children, and diabetic neuropathy

# In[18]:


drugs = {}

for k,v in gdata.drug_idx_to_name.items():
    if v.lower() in ['donepezil','memantine']:
        print(k,v)
        drugs[k] = v


# In[19]:


for k,v in gdata.drug_idx_to_name.items():
    if v.lower() in ['risperidone','donepezil','citalopram']:
        print(k,v)
        drugs[k] = v


# In[20]:


drugs


# ## Now, we can investigate the drug IDs above against all drugs and all side effects

# In[21]:


df_summ = pd.read_csv('exp_summary.csv', index_col=0)
df_summ


# In[22]:


# given a drug we interest
# generate result

drugid = 191
aprile = Aprile(gdata, device='cpu')





# query_191 = AprileQuery.load_from_pkl('drug-191.pkl')


# In[24]:


from aprile.utils import sparse_id
gdata.p_feat = sparse_id(gdata.n_prot)
aprile.get_prediction_train()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[26]:


SE_Results = []

for drug_idx in drugs.keys():
    # se_idx = gdata.side_effect_name_to_idx[se_name]    
    query = AprileQuery.load_from_pkl('drug-%s.pkl'%drug_idx)
    # df_ctdi = df_ctd[df_ctd.DiseaseName.str.lower()==se_name.lower()]
    # ctd_genes = df_ctdi.GeneID.unique()
    # run_permut_query(se_idx, query, gdata, ctd_genes, N_RAND=10000)


# - reporting system of side effects
#     - Pose can help design this so it is easier for patients with dementia

# In[ ]:





# In[112]:


plt.hist(query.probability)


# In[120]:


import numpy as np
np.median(query.probability)


# In[123]:


(np.array(query.probability)>0.9).sum()


# In[113]:


plt.hist(query.ppiu_score)

