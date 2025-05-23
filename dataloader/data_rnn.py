# Import Pytorch and Pytorch Lightning
import torch  
import pytorch_lightning as pl  
  
# Data Preprocessing & Tools 
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision import datasets, transforms 
from torch.utils.data import Dataset, random_split, DataLoader 
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np



# ------------
# Google SDK
# ------------

project_id = 'som-nero-nigam-bmi215'

import os 

# If locally:
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'/Users/{os.getlogin()}/.config/gcloud/application_default_credentials.json' 
# If on Nero Nigam:
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'/home/{os.getlogin()}/.config/gcloud/application_default_credentials.json' 

os.environ['GCLOUD_PROJECT'] = project_id
from google.cloud import bigquery
client=bigquery.Client()

# ---------------------
# Data functions & SQL
# ---------------------

# Getting y (i.e binary ground-truth)
def get_y(task):
    # Shape should be (batch, 1)
    if task == 'eol':
        work_dataset_id = 'end_of_life_task'
        cohort = '__eol_cohort'
    if task == 'surgical':
        work_dataset_id = 'surgical_task'
        cohort = '__surg_cohort'
    sql_y = """
    select
    example_id,
    y
    from {project_id}.{schema_name}.{cohort}
    where example_id in (select example_id from {project_id}.{schema_name}.all_features group by 1)
    and example_id <= 100
    order by example_id
    """.format(
        project_id = project_id,
        schema_name = work_dataset_id,
        cohort = cohort
    )
    query_job_y = client.query(sql_y)
    y = torch.from_numpy(query_job_y.to_dataframe().y.values)
    y = torch.unsqueeze(y,1).float()
    return y

# Getting x (i.e patient x visits x concepts)
def get_x(task):
    # Shape should be (batch, seq, feature)
    if task == 'eol':
        work_dataset_id = 'end_of_life_task'
    if task == 'surgical':
        work_dataset_id = 'surgical_task'
    sql_x = """
    select
    *,
    1 as counts,
    DENSE_RANK() OVER (ORDER BY concept_id ASC) AS idx_concept
    from
    {project_id}.{schema_name}.all_features 
    where example_id <= 100
    order by example_id, visit_number
    """.format(
        project_id = project_id,
        schema_name = work_dataset_id
    )
    query_job_x = client.query(sql_x)
    data = query_job_x.to_dataframe() 
    # Concept IDs as separate features to one-hot encode
    classes = [[i] for i in list(np.unique(data.idx_concept))]
    mlb = MultiLabelBinarizer()
    mlb.fit(classes)
        
    # Getting x, turning it into a list of torch tensors 
    # and padding (with -1 so we can easily mask later)
    x = [np.stack([mlb.transform(list([data[(data.example_id == i) 
                                            & (data.visit_number == j)].idx_concept])) 
                    for j in list(np.unique(data[data.example_id==i].visit_number))]) 
            for i in list(np.unique(data.example_id))]
        
    x = [torch.from_numpy(np.squeeze(arr)) for arr in x]
    x = pad_sequence(x, batch_first=True, padding_value=-1)
    return x

# Getting t (i.e patient x temporal marker for visits)
def get_t(task):
    # Shape should be (batch, seq)
    if task == 'eol':
        work_dataset_id = 'end_of_life_task'
    if task == 'surgical':
        work_dataset_id = 'surgical_task'
    sql_t = """
    select
    example_id,
    visit_number,
    number_days_from_end
    from
    {project_id}.{schema_name}.all_features 
    where example_id <= 100
    group by 1,2,3
    order by example_id, visit_number
    """.format(
        project_id = project_id,
        schema_name = work_dataset_id
    )
    query_job_t = client.query(sql_t)
    data = query_job_t.to_dataframe() 
    data_pivot = (data.pivot_table('number_days_from_end', 'example_id', 
                                   'visit_number', aggfunc='sum').fillna(-1))
    
    t = np.array(data_pivot)
    
    return torch.from_numpy(t)

# --------------
# Dataset
# --------------

class get_data(Dataset):  
    def __init__(self, task):
        # Getting x and y
        self.x = get_x(task)
        self.y = get_y(task)
        self.t = get_t(task)
        self.len = len(self.y)

    def __getitem__(self, index):
        sample = self.x[index], self.y[index], self.t[index] 
        return sample

    def __len__(self):
        return self.len


# --------------------
# PL EoL Data Module
# --------------------

class DataModuleEoL(pl.LightningDataModule): 
    def __init__(self, batch_size=16): 
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        full_data = get_data(task='eol') 
        full_len = len(full_data)
        data_split = [int(0.7*full_len), int(0.15*full_len), full_len - int(0.15*full_len) - int(0.7*full_len)] 
        self.train, self.val, self.test = random_split(full_data, data_split)
        self.embed_size = full_data.x.shape[2]
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
    

# --------------------
# PL Surg Data Module
# --------------------

class DataModuleSurg(pl.LightningDataModule): 
    def __init__(self, batch_size=16, split=(0.7, 0.15, 0.15)): 
        super().__init__()
        self.batch_size = batch_size
        #self.split = split

    def setup(self, stage=None):
        full_data = get_data(task='surgical') 
        full_len = len(full_data)
        data_split = [int(0.7*full_len), int(0.15*full_len), 
                      full_len - int(0.15*full_len) - int(0.7*full_len)] 
        self.train, self.val, self.test = random_split(full_data, data_split)
        self.embed_size = full_data.x.shape[2]
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)