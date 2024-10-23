# -*- coding: utf-8 -*-
# @Author: Xiaoning Qi
# @Date:   2022-06-13 09:47:44
# @Last Modified by:   Xiaoning Qi
# @Last Modified time: 2024-10-23 11:27:21
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import sys
print(sys.path)

import argparse 
from datetime import datetime
import scanpy as sc

from trainer.PRnetTrainer import PRnetTrainer

def parse_args():
    parse = argparse.ArgumentParser(description='perturbation-conditioned generative model ')  
    parse.add_argument('--split_key', default='lincs_split', type=str, help='split key of data')  
    args = parse.parse_args()  
    return args



if __name__ == "__main__":
    args_train = parse_args()
    start_time = datetime.now()
    

    config_kwargs = {
        'batch_size' : 512,
        'comb_num' : 1,
        'save_dir' : './checkpoint/',
        'results_dir' : './results/lincs/',
        'n_epochs' : 100,
        'split_key' : args_train.split_key,
        'x_dimension' : 978,
        'hidden_layer_sizes' : [128],
        'z_dimension' : 64,
        'adaptor_layer_sizes' : [128],
        'comb_dimension' : 64, 
        'drug_dimension': 1024,
        'dr_rate' : 0.05,
        'n_epochs' : 100,
        'lr' : 1e-3, 
        'weight_decay' : 1e-8,
        'scheduler_factor' : 0.5,
        'scheduler_patience' : 5,
        'n_genes' : 20,
        'loss' : ['GUSS'], 
        'obs_key' : 'cov_drug_dose_name'
    }  


    print(os.getcwd())

    adata = sc.read('./datasets/Lincs_L1000.h5ad')

    


    Trainer = PRnetTrainer(
                            adata,
                            batch_size=config_kwargs['batch_size'],
                            comb_num=config_kwargs['comb_num'],
                            split_key=config_kwargs['split_key'],
                            model_save_dir=config_kwargs['save_dir'],
                            x_dimension=config_kwargs['x_dimension'],
                            hidden_layer_sizes=config_kwargs['hidden_layer_sizes'],
                            z_dimension=config_kwargs['z_dimension'],
                            adaptor_layer_sizes=config_kwargs['adaptor_layer_sizes'],
                            comb_dimension=config_kwargs['comb_dimension'],
                            drug_dimension=config_kwargs['drug_dimension'],
                            dr_rate=config_kwargs['dr_rate'],
                            n_genes=config_kwargs['n_genes'],
                            loss = config_kwargs['loss'],
                            obs_key = config_kwargs['obs_key']
                                )

    Trainer.test('./checkpoint/lincs_best_epoch_all.pt')


    end_time = datetime.now()

    during_time = (end_time-start_time).seconds/60

    print(f'start time: {start_time} end_time: {end_time} time:{during_time} min')
