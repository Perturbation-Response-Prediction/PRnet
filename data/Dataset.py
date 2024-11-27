# -*- coding: utf-8 -*-
# @Author: Xiaoning Qi
# @Date:   2022-05-10 09:04:03
# @Last Modified by:   Xiaoning Qi
# @Last Modified time: 2024-03-21 21:48:29
import os
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import random

import torch 
from torch.utils.data import Dataset
from ._utils import Condition_encoder, Drug_SMILES_encode, rank_genes_groups_by_cov, Drug_dose_encoder

class DrugDoseAnnDataset(Dataset):
    '''
    Dataset for loading tensors from AnnData objects.
    ''' 
    def __init__(self,
                 adata,
                 dtype='train',
                 obs_key='cov_drug',
                 comb_num=1
                 ):
        self.dtype = dtype
        self.obs_key = obs_key        
        
        
        self.dense_adata = adata
        print(self.dense_adata)

        if sparse.issparse(adata.X):
            self.dense_adata  = sc.AnnData(X=adata.X.A, obs=adata.obs.copy(deep=True), var=adata.var.copy(deep=True), uns=adata.uns.copy(deep=True))
        
                    
        self.drug_adata = self.dense_adata[self.dense_adata.obs['dose']!=0.0] 
         
     
        self.data = torch.tensor(self.drug_adata.X, dtype=torch.float32)
        self.dense_data = torch.tensor(self.dense_adata.X, dtype=torch.float32)

 
        self.paired_control_index = self.drug_adata.obs['paired_control_index'].tolist()
        self.dense_adata_index = self.dense_adata.obs.index.to_list()


        # Encode condition strings to integer
        self.drug_type_list = self.drug_adata.obs['SMILES'].to_list()
        self.dose_list = self.drug_adata.obs['dose'].to_list()
        self.obs_list = self.drug_adata.obs[obs_key].to_list()
        self.encode_drug_doses = Drug_dose_encoder(self.drug_type_list, self.dose_list, comb_num=comb_num)

        self.encode_drug_doses = torch.tensor(self.encode_drug_doses, dtype=torch.float32)



    def __len__(self):
        return len(self.drug_adata)

    def __getitem__(self, index):
        outputs = dict()
        outputs['x'] = self.data[index, :]

        control_index = self.dense_adata_index.index(self.paired_control_index[index]) 
            
        outputs['control'] = self.dense_data[control_index,:]
        outputs['drug_dose'] = self.encode_drug_doses[index, :]
        outputs['label'] = outputs['drug_dose']

        obs_info = self.obs_list[index]

        outputs['cov_drug'] = obs_info

        
        return {'features':(outputs['control'], outputs['x']), 'label':outputs['label'], 'cov_drug': outputs['cov_drug']}


