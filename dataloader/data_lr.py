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

# ------------------------------------
# Google SDK
# ------------------------------------

project_id = 'som-nero-nigam-bmi215'

import os 

# If locally:
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'/Users/{os.getlogin()}/.config/gcloud/application_default_credentials.json' 
# If on Nero Nigam:
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'/home/{os.getlogin()}/.config/gcloud/application_default_credentials.json' 

os.environ['GCLOUD_PROJECT'] = project_id
from google.cloud import bigquery
client=bigquery.Client()

# ------------------------------------
# ------------------------------------
# Data functions & SQL for y, x and t
# ------------------------------------
# ------------------------------------

# ------------------------------------
# Getting y (i.e binary ground-truth)
# ------------------------------------
def get_y(task):
    '''
    Output is:
        y = binary vector of shape (batch,1)
    For either "End of Life" or for "Likelihood of Surgery"
    '''
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
    and example_id <= 5000 
    order by example_id
    """.format(
        project_id = project_id,
        schema_name = work_dataset_id,
        cohort = cohort
    )
    query_job_y = client.query(sql_y)
    y = torch.from_numpy(query_job_y.to_dataframe().y.values).float()
    y = torch.unsqueeze(y, 1)
    return y

# ------------------------------------
# Getting x (i.e patient x visit info)
# ------------------------------------
def get_x(task):
    '''
    Output is:
        x = list of lists of tensors built as patient list x visit lists x concepts
        num_concepts = number of unique concepts in the dataset
    For either "End of Life" or for "Likelihood of Surgery"
    '''
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
    where example_id <= 5000
    order by example_id, visit_number
    """.format(
        project_id = project_id,
        schema_name = work_dataset_id
    )
    query_job_x = client.query(sql_x)
    raw_data = query_job_x.to_dataframe() 
    
    num_concepts = np.max(raw_data.idx_concept)
    num_patients = len(np.unique(raw_data.example_id))
    
    data = np.array(list(
            [
                (np.unique(raw_data[(raw_data.example_id == i) 
                                            & (raw_data.visit_number == j)].idx_concept))
                for j in list(np.unique(raw_data[raw_data.example_id == i].visit_number))
            ]
            for i in list(np.unique(raw_data.example_id))
    ))
    
    X_counts = np.zeros((num_patients, num_concepts))
    for patient_id, visit_codes in enumerate(data):
        all_visit_codes = np.concatenate(visit_codes)
        unique, counts = np.unique(all_visit_codes, return_counts=True)

        for concept_id, num_occurrences in zip(unique, counts):
            X_counts[patient_id][concept_id - 1] = num_occurrences # need to subtract one from the concept_id !
            
    X_counts_torch = torch.from_numpy(X_counts)
                                   
    return X_counts_torch, num_concepts

# ------------------------------------
# Getting t (i.e patient x visit time)
# ------------------------------------
def get_t(task):
    '''
    Output is:
        t = padded vector of shape (batch, max_visits)
        max_days = max number of days between now and first visit
        max_visits = max number of visits per patient
    For either "End of Life" or for "Likelihood of Surgery"
    '''
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
    where example_id <= 5000
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
    max_days = int(t.max())
    max_visits = t.shape[1]
    
    return torch.from_numpy(t), max_days, max_visits

# ------------------------------------
# Set up full dataset with get_data()
# ------------------------------------

class get_data(Dataset):  
    def __init__(self, task):
        # Getting x, y, t and params
        self.x, self.num_concepts = get_x(task)
        self.y = get_y(task)
        self.t, self.max_days, self.max_visits = get_t(task)
        self.len = len(self.y)

    def __getitem__(self, index):
        sample = self.x[index], self.y[index], self.t[index] 
        return sample

    def __len__(self):
        return self.len
    
    def parameters(self):
        return {'max_days': self.max_days, 'max_visits': self.max_visits, 
                'num_concepts': self.num_concepts}


    

# ----------------------------------------------------------------------------------
# Custom Collate function for Dataloader to support variable size input (i.e. x)
# ----------------------------------------------------------------------------------

def collate(batch):
    x = [item[0] for item in batch]
    y = torch.stack([item[1] for item in batch])
    t = torch.stack([item[2] for item in batch])
    return [x, y, t]
         
# ------------------------------------
# ------------------------------------
# Pytorch Lightning Data Module
# ------------------------------------
# ------------------------------------

class DataModuleTask(pl.LightningDataModule): 
    def __init__(self, batch_size=16, task='eol'): 
        super().__init__()
        self.batch_size = batch_size
        self.task = task
        self.rnn = True

    def setup(self, stage=None, split=(0.7, 0.15)):
        self.full_data = get_data(task=self.task) 
        self.full_len = len(self.full_data)
        if sum(split) > 1: 
            print("split cannot sum to more than 1.")
            return
        self.train_len = int(split[0]*self.full_len)
        self.val_len = int(split[1]*self.full_len)
        self.test_len = self.full_len - self.train_len - self.val_len
        data_split = [self.train_len, self.val_len, self.test_len] 
        self.train, self.val, self.test = random_split(self.full_data, data_split)
        
        params = self.full_data.parameters()
        self.max_days = params['max_days']
        self.max_visits = params['max_visits']
        self.num_concepts = params['num_concepts'] 
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=collate)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=collate)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=collate)