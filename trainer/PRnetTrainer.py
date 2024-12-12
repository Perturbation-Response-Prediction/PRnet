# -*- coding: utf-8 -*-
# @Author: Xiaoning Qi
# @Date:   2022-06-23 02:24:40
# @Last Modified by:   Xiaoning Qi
# @Last Modified time: 2024-10-31 15:26:11
import os
from sklearn import metrics

import torch
import torch.nn as nn
from torch.autograd import Variable, grad
from torch.nn import functional as F
from torch.distributions import NegativeBinomial, normal

import math
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy import sparse
from anndata import AnnData
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error


from data.Dataset import  DrugDoseAnnDataset
from models.PRnet import PRnet

from ._utils import train_valid_test



    
def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

class PRnetTrainer:
    """
    This class contains the implementation of the PRnetTrainer Trainer
    Parameters
    ----------
    model: PRnet
    adata: : `~anndata.AnnData`
        Annotated Data Matrix for training PRnet.
    batch_size: integer
        size of each batch to be fed to network.
    comb_num: int
        Number of combined compounds.
    shuffle: bool
        if `True` shuffles the training dataset.
    split_key: string
        Attributes of data split.
    model_save_dir: string
        Save dir of model. 
    x_dimension: int
        Dimention of x
    hidden_layer_sizes: list
        A list of hidden layer sizes
    z_dimension: int
        Dimention of latent space
    adaptor_layer_sizes: list
        A list of adaptor layer sizes
    comb_dimension: int
        Dimention of perturbation latent space
    drug_dimension: int
        Dimention of rFCGP
    n_genes: int
        Dimention of different expressed gene
    n_epochs: int
        Number of epochs to iterate and optimize network weights.
    train_frac: Float
        Defines the fraction of data that is used for training and data that is used for validation.
    dr_rate: float
        dropout_rate
    loss: list
        Loss of model, subset of 'NB', 'GUSS', 'KL', 'MSE'
    obs_key:
        observation key of data
    """
    def __init__(self, adata, batch_size = 32, comb_num = 2, shuffle = True, split_key='random_split', model_save_dir = './checkpoint/',results_save_dir = './results/', x_dimension = 5000, hidden_layer_sizes = [128], z_dimension = 64, adaptor_layer_sizes = [128], comb_dimension = 64, drug_dimension = 1031, n_genes=20,  dr_rate = 0.05, loss = ['GUSS'], obs_key = 'cov_drug_name', **kwargs): # maybe add more parameters
        
        assert set(loss).issubset(['NB', 'GUSS', 'KL', 'MSE']), "loss should be subset of ['NB', 'GUSS', 'KL', 'MSE']"

        self.x_dim = x_dimension
        self.split_key = split_key
        self.z_dimension = z_dimension
        self.comb_dimension = comb_dimension

        self.model = PRnet(adata, x_dimension=self.x_dim, hidden_layer_sizes=hidden_layer_sizes, z_dimension=z_dimension, adaptor_layer_sizes=adaptor_layer_sizes, comb_dimension=comb_dimension, comb_num=comb_num, drug_dimension=drug_dimension,dr_rate=dr_rate)

        self.model_save_dir = model_save_dir
        self.results_save_dir = results_save_dir
        self.loss = loss
        self.modelPGM = self.model.get_PGM()


        self.seed = kwargs.get("seed", 2024)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            if(torch.cuda.device_count() > 1):
                self.modelPGM = nn.DataParallel(self.modelPGM, device_ids=[i for i in range(torch.cuda.device_count())])         
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelPGM = self.modelPGM.to(self.device)

        self.modelPGM.apply(self.weight_init)
        print(self.modelPGM)


        self.adata = adata
        #self.adata_deg_list = adata.uns['rank_genes_groups_cov']
        self.de_n_genes = n_genes
        self.adata_var_names = adata.var_names
        self.train_data, self.valid_data, self.test_data = train_valid_test(self.adata, split_key = split_key)

        
        if self.train_data is not None:
            self.train_dataset = DrugDoseAnnDataset(self.train_data, dtype='train', obs_key=obs_key, comb_num=comb_num)     
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        if self.valid_data is not None:
            self.valid_dataset = DrugDoseAnnDataset(self.valid_data, dtype='valid', obs_key=obs_key, comb_num=comb_num)
            self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=True)
        if self.test_data is not None:
            self.test_dataset = DrugDoseAnnDataset(self.test_data, dtype='test', obs_key=obs_key, comb_num=comb_num)
            self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)

        if set(['NB']).issubset(loss):
            self.criterion = NBLoss()
        if set(['GUSS']).issubset(loss):
            self.criterion = nn.GaussianNLLLoss()
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

        self.shuffle = shuffle
        self.batch_size = batch_size

        # Optimization attributes

        self.epoch = -1  # epoch = self.epoch + 1 in compute metrics
        self.best_state_dictPGM = None


        self.PGM_losses = []
        self.r2_score_mean = []
        self.r2_score_var = []
        self.mse_score = []
        self.r2_score_mean_de = []
        self.r2_score_var_de = []
        self.mse_score_de = []
        self.best_mse = np.inf
        self.patient = 0



    def train(self, n_epochs = 100, lr = 0.001, weight_decay= 1e-8, scheduler_factor=0.5,scheduler_patience=10,**extras_kwargs):
        self.n_epochs = n_epochs
        self.params = filter(lambda p: p.requires_grad, self.model.parameters())
        paramsPGM = filter(lambda p: p.requires_grad, self.modelPGM.parameters())

        self.optimPGM = torch.optim.Adam(
            paramsPGM, lr=lr, weight_decay= weight_decay) # consider changing the param. like weight_decay, eps, etc.
        #self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(self.optimPGM, step_size=10)
        self.scheduler_autoencoder = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimPGM, 'min',factor=scheduler_factor,verbose=1,min_lr=1e-8,patience=scheduler_patience)


        for self.epoch in range(self.n_epochs):
            loop = tqdm(enumerate(self.train_dataloader), total =len(self.train_dataloader))
            for i, data in loop:
                self.modelPGM.zero_grad()
                (control, target) = data['features']
                encode_label = data['label']

                control = control.to(self.device, dtype=torch.float32)
                if set(['NB']).issubset(self.loss):
                    control = torch.log1p(control)
                target = target.to(self.device, dtype=torch.float32)
                

                encode_label = encode_label.to(self.device, dtype=torch.float32)
                b_size = control.size(0)
                

                noise = self.make_noise(b_size, 10)
                
                gene_reconstructions = self.modelPGM(control, encode_label, noise)
                dim = gene_reconstructions.size(1) // 2
                gene_means = gene_reconstructions[:, :dim]
                gene_vars = gene_reconstructions[:, dim:]
                gene_vars = F.softplus(gene_vars)

      
                if set(['GUSS']).issubset(self.loss):
                    reconstruction_loss = self.criterion(input=gene_means, target=target, var=gene_vars)

                    dist = normal.Normal(
                        torch.clamp(
                            torch.Tensor(gene_means),
                            min=1e-3,
                            max=1e3,
                        ),
                        torch.clamp(
                            torch.Tensor(gene_vars.sqrt()),
                            min=1e-3,
                            max=1e3,
                        )           
                    )
                if set(['NB']).issubset(self.loss):
                    reconstruction_loss =  self.criterion(gene_means, target, gene_vars)

                    counts, logits = self._convert_mean_disp_to_counts_logits(
                        torch.clamp(
                            torch.Tensor(gene_means),
                            min=1e-3,
                            max=1e3,
                        ),
                        torch.clamp(
                            torch.Tensor(gene_vars),
                            min=1e-3,
                            max=1e3,
                        )
                    )
                    
                    dist = NegativeBinomial(
                        total_count=counts,
                        logits=logits
                    )


                    
                nb_sample = dist.sample()   

                if set(['MSE']).issubset(self.loss):
                    mse_loss = self.mse_loss(nb_sample, target)
                    reconstruction_loss += mse_loss * 10
                if set(['KL']).issubset(self.loss):
                    kl_loss = self.kl_loss(nb_sample, target)
                    reconstruction_loss += kl_loss * 0.01
                reconstruction_loss.backward()

                # Update PGM
                self.optimPGM.step()
                
                #self.scheduler_autoencoder.step()

                # Output training stats               
                # Save Losses for plotting later
                self.PGM_losses.append(reconstruction_loss.item())


                loop.set_description(f'Epoch [{self.epoch}/{self.n_epochs}] [{i}/{len(self.train_dataloader)}]')
                #loop.set_postfix(Loss_NB=nb_loss.item(), Loss_MSE=mse_loss.item())
                loop.set_postfix(Loss=reconstruction_loss.item())
        

            loop_v = tqdm(enumerate(self.valid_dataloader), total =len(self.valid_dataloader))
            self.r2_sum_mean = 0
            self.r2_sum_var = 0
            self.mse_sum = 0
            self.r2_sum_mean_de = 0
            self.r2_sum_var_de = 0
            self.mse_sum_de = 0
            for j, vdata in loop_v:
                (control, target) = vdata['features']
                encode_label = vdata['label']
                data_cov_drug = vdata['cov_drug']

                control = control.to(self.device, dtype=torch.float32)
                if set(['NB']).issubset(self.loss):
                    control = torch.log1p(control)
                target = target.to(self.device, dtype=torch.float32)

                encode_label = encode_label.to(self.device, dtype=torch.float32)
                b_size = control.size(0)
                
                noise = self.make_noise(b_size, 10)
                gene_reconstructions = self.modelPGM(control, encode_label, noise).detach()
                dim = gene_reconstructions.size(1) // 2
                gene_means = gene_reconstructions[:, :dim]
                gene_vars = gene_reconstructions[:, dim:]
                gene_vars = F.softplus(gene_vars)
                
                
                if set(['GUSS']).issubset(self.loss):
                    reconstruction_loss = self.criterion(input=gene_means, target=target, var=gene_vars)

                    dist = normal.Normal(
                        torch.clamp(
                            torch.Tensor(gene_means),
                            min=1e-3,
                            max=1e3,
                        ),
                        torch.clamp(
                            torch.Tensor(gene_vars.sqrt()),
                            min=1e-3,
                            max=1e3,
                        )           
                    )
                if set(['NB']).issubset(self.loss):
                    reconstruction_loss =  self.criterion(gene_means, target, gene_vars)

                    counts, logits = self._convert_mean_disp_to_counts_logits(
                        torch.clamp(
                            torch.Tensor(gene_means),
                            min=1e-3,
                            max=1e3,
                        ),
                        torch.clamp(
                            torch.Tensor(gene_vars),
                            min=1e-3,
                            max=1e3,
                        )
                    )
                    
                    dist = NegativeBinomial(
                        total_count=counts,
                        logits=logits
                    )

                    
                nb_sample = dist.sample().cpu().numpy()
                yp_m = nb_sample.mean(0)
                yp_v = nb_sample.var(0)

                y_true = target.cpu().numpy()
                yt_m = y_true.mean(axis=0)
                yt_v = y_true.var(axis=0)
                
                
                r2_score_mean = r2_score(yt_m, yp_m)
                self.r2_sum_mean += r2_score_mean
                r2_score_var = r2_score(yt_v, yp_v)
                self.r2_sum_var += r2_score_var               

                mse_score =  mean_squared_error(y_true, nb_sample)
                self.mse_sum += mse_score

                loop_v.set_description(f'Epoch [{self.epoch}/{self.n_epochs}] [{j}/{len(self.valid_dataloader)}]')
                loop_v.set_postfix(r2_score_mean=r2_score_mean, r2_score_var=r2_score_var, mse_score=mse_score)


            self.r2_score_mean.append(self.r2_sum_mean/len(self.valid_dataloader))
            self.r2_score_var.append(self.r2_sum_var/len(self.valid_dataloader))
            self.mse_score.append(self.mse_sum/len(self.valid_dataloader))


            print('mean mse of validation datastes:', self.mse_score[-1])
            print('mean r2 of validation datastes:', self.r2_score_mean[-1])
            #print('mean mse DEG of validation datastes:', self.mse_score_de[-1])
            #print('mean r2 DEG of validation datastes:', self.r2_score_mean_de[-1])  
            self.scheduler_autoencoder.step(self.mse_score[-1])                       

         
            if self.mse_score[-1] < self.best_mse:
                self.patient = 0
                print("Saving best state of network...")
                print("Best State was in Epoch", self.epoch)
                self.best_state_dictG = self.modelPGM.module.state_dict()
                torch.save(self.best_state_dictG, self.model_save_dir+self.split_key+'_best_epoch_all.pt')
                self.best_mse = self.mse_score[-1]
            elif self.patient <= 20:
                self.patient += 1   
            else:
                print("The mse of validation datastes has not improve in 20 epochs!")
                break

        
        loss_dict = {'Loss_PGM': self.PGM_losses}
        metrics_dict = {'r2':self.r2_score_mean, 'mse':self.mse_score}
        loss_df = pd.DataFrame(loss_dict)
        metrics_df = pd.DataFrame(metrics_dict)
        loss_df.to_csv(self.model_save_dir+self.split_key+'loss_comb.csv')
        metrics_df.to_csv(self.model_save_dir+self.split_key+'metrics_comb.csv')
       

    def test(self, model_path, return_dict = False):

        self.modelPGM.load_state_dict(torch.load(model_path))
        self.modelPGM.eval()

        x_true_array = np.zeros((0, self.x_dim))
        y_true_array = np.zeros((0, self.x_dim))
        y_pre_array = np.zeros((0, self.x_dim))
        cov_drug_list = []


        #y_true_de_array_ = np.zeros((0, self.de_n_genes))
        #y_pre_de_array_ = np.zeros((0, self.de_n_genes))

        loop_t = tqdm(enumerate(self.test_dataloader), total =len(self.test_dataloader))
        
        for j, vdata in loop_t:
            (control, target) = vdata['features']
            encode_label = vdata['label']
            data_cov_drug = vdata['cov_drug']
            cov_drug_list = cov_drug_list + data_cov_drug

            with open(self.results_save_dir+self.split_key+"_cov_drug_array.csv", 'a') as f:
                for i in data_cov_drug:
                    f.write(i+'\n')

            control = control.to(self.device, dtype=torch.float32)
            if set(['NB']).issubset(self.loss):
                    control = torch.log1p(control)
            target = target.to(self.device, dtype=torch.float32)

            encode_label = encode_label.to(self.device, dtype=torch.float32)
            b_size = control.size(0)
            
            noise = self.make_noise(b_size, 10)
            gene_reconstructions = self.modelPGM(control, encode_label, noise).detach()
            dim = gene_reconstructions.size(1) // 2
            gene_means = gene_reconstructions[:, :dim]
            gene_vars = gene_reconstructions[:, dim:]
            gene_vars = F.softplus(gene_vars)
                
                
            if set(['GUSS']).issubset(self.loss):
                dist = normal.Normal(
                    torch.clamp(
                        torch.Tensor(gene_means),
                        min=1e-3,
                        max=1e3,
                    ),
                    torch.clamp(
                        torch.Tensor(gene_vars.sqrt()),
                        min=1e-3,
                        max=1e3,
                    )           
                )
            if set(['NB']).issubset(self.loss):
                counts, logits = self._convert_mean_disp_to_counts_logits(
                    torch.clamp(
                        torch.Tensor(gene_means),
                        min=1e-3,
                        max=1e3,
                    ),
                    torch.clamp(
                        torch.Tensor(gene_vars),
                        min=1e-3,
                        max=1e3,
                    )
                )
                
                dist = NegativeBinomial(
                    total_count=counts,
                    logits=logits
                )

                
            nb_sample = dist.sample().cpu().numpy()

            y_pre_array = np.concatenate((y_pre_array, nb_sample),axis=0)

            x_true = control.cpu().numpy()
            x_true_array = np.concatenate((x_true_array, x_true),axis=0)

            y_true = target.cpu().numpy()
            y_true_array = np.concatenate((y_true_array, y_true), axis=0)
            with open(self.results_save_dir + self.split_key + "_y_true_array.csv", 'a+') as f:
                np.savetxt(f, y_true, delimiter=",")

            
            with open(self.results_save_dir+self.split_key+"_y_pre_array.csv", 'a+') as f:
                np.savetxt(f, nb_sample, delimiter=",")

            with open(self.results_save_dir+self.split_key+"_x_true_array.csv", 'a+') as f:
                np.savetxt(f, x_true, delimiter=",")
        
        print("Attention! The data were saved with append mode!If you want to change to overwrite mode, you can change the save mode to 'w'")
        print("The Gound Truth has been saved to:", self.results_save_dir+self.split_key+"_x_true_array.csv", 'with append mode!')
        print("The Prediction has been saved to:", self.results_save_dir+self.split_key+"_y_pre_array.csv", 'with append mode!')
        print("The Split Key has been saved to:", self.results_save_dir+self.split_key+"_cov_drug_array.csv", 'with append mode!')

        if return_dict == True:
              
            return x_true_array,y_true_array,y_pre_array,cov_drug_list
 
    @staticmethod
    def _anndataToTensor(adata: AnnData) -> torch.Tensor:
        data_ndarray = adata.X.A
        data_tensor = torch.from_numpy(data_ndarray)
        return data_tensor
    
    def make_noise(self, batch_size, shape, volatile=False):
        tensor = torch.randn(batch_size, shape)
        noise = Variable(tensor, volatile)
        noise = noise.to(self.device, dtype=torch.float32)
        return noise

    def weight_init(self, m):  
        # initialize the weights of the model
        if isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.normal_(1, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
    

    def get_per_latent(self, model_path):
    
        self.modelPGM.load_state_dict(torch.load(model_path))
        self.modelPGM.eval()

        cov_drug_list = []
        
        latent_array = np.zeros((0, self.comb_dimension))

        loop_t = tqdm(enumerate(self.test_dataloader), total =len(self.test_dataloader))
        
        for j, vdata in loop_t:
            (control, target) = vdata['features']
            encode_label = vdata['label']
            data_cov_drug = vdata['cov_drug']
            cov_drug_list = cov_drug_list + data_cov_drug

            control = control.to(self.device, dtype=torch.float32)
            if set(['NB']).issubset(self.loss):
                    control = torch.log1p(control)
            target = target.to(self.device, dtype=torch.float32)

            encode_label = encode_label.to(self.device, dtype=torch.float32)
           
            b_size = control.size(0)
            noise = self.make_noise(b_size, 10)

            latent = self.model.get_per_latent(control, encode_label, noise)

            latent_array = np.concatenate((latent_array, latent),axis=0)

        return latent_array, cov_drug_list


    def get_latent(self, model_path):
        
        self.modelPGM.load_state_dict(torch.load(model_path))
        self.modelPGM.eval()

        cov_drug_list = []
        
        latent_array = np.zeros((0, self.z_dimension))

        loop_t = tqdm(enumerate(self.test_dataloader), total =len(self.test_dataloader))
        
        for j, vdata in loop_t:
            (control, target) = vdata['features']
            encode_label = vdata['label']
            data_cov_drug = vdata['cov_drug']
            cov_drug_list = cov_drug_list + data_cov_drug

            control = control.to(self.device, dtype=torch.float32)
            if set(['NB']).issubset(self.loss):
                    control = torch.log1p(control)
            target = target.to(self.device, dtype=torch.float32)

            encode_label = encode_label.to(self.device, dtype=torch.float32)
           
            b_size = control.size(0)
            noise = self.make_noise(b_size, 10)
            

            latent = self.model.get_latent(control, encode_label, noise)

            latent_array = np.concatenate((latent_array, latent),axis=0)

        return latent_array, cov_drug_list

    @staticmethod
    def pearson_mean(data1, data2):
        sum_pearson_1 = 0
        sum_pearson_2 = 0
        for i in range(data1.shape[0]):
            pearsonr_ = pearsonr(data1[i], data2[i])
            sum_pearson_1 += pearsonr_[0]
            sum_pearson_2 += pearsonr_[1]
        return sum_pearson_1/data1.shape[0], sum_pearson_2/data1.shape[0]
    
    @staticmethod
    def r2_mean(data1, data2):
        sum_r2_1 = 0
        for i in range(data1.shape[0]):
            r2_score_ = r2_score(data1[i], data2[i])
            sum_r2_1 += r2_score_           
        return sum_r2_1/data1.shape[0]

    @staticmethod
    def _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6):
        r"""NB parameterizations conversion. Reference: https://github.com/theislab/chemCPA/tree/main.
    Parameters
    ----------
    mu :
        mean of the NB distribution.
    theta :
        inverse overdispersion.
    eps :
        constant used for numerical log stability. (Default value = 1e-6)
    Returns
    -------
    type
        the number of failures until the experiment is stopped
        and the success probability.
    """
        assert (mu is None) == (theta is None), "If using the mu/theta NB parameterization, both parameters must be specified"
        logits = (mu + eps).log() - (theta + eps).log()
        total_count = theta
        return total_count, logits
    
    @staticmethod
    def _sample_z(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
            Samples from standard Normal distribution with shape [size, z_dim] and
            applies re-parametrization trick. It is actually sampling from latent
            space distributions with N(mu, var) computed by the Encoder.

        Parameters
            ----------
        mean:
        Mean of the latent Gaussian
            log_var:
        Standard deviation of the latent Gaussian
            Returns
            -------
        Returns Torch Tensor containing latent space encoding of 'x'.
        The computed Tensor of samples with shape [size, z_dim].
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps


class NBLoss(torch.nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, yhat, y, eps=1e-8):
        """Negative binomial log-likelihood loss. It assumes targets `y` with n
        rows and d columns, but estimates `yhat` with n rows and 2d columns.
        The columns 0:d of `yhat` contain estimated means, the columns d:2*d of
        `yhat` contain estimated variances. This module assumes that the
        estimated mean and inverse dispersion are positive---for numerical
        stability, it is recommended that the minimum estimated variance is
        greater than a small number (1e-3). Reference: https://github.com/theislab/chemCPA/tree/main.
        Parameters
        ----------
        yhat: Tensor
                Torch Tensor of reeconstructed data.
        y: Tensor
                Torch Tensor of ground truth data.
        eps: Float
                numerical stability constant.
        """
        dim = yhat.size(1) // 2
        # means of the negative binomial (has to be positive support)
        mu = yhat[:, :dim]
        # inverse dispersion parameter (has to be positive support)
        theta = yhat[:, dim:]

        if theta.ndimension() == 1:
            # In this case, we reshape theta for broadcasting
            theta = theta.view(1, theta.size(0))
        t1 = (
            torch.lgamma(theta + eps)
            + torch.lgamma(y + 1.0)
            - torch.lgamma(y + theta + eps)
        )
        t2 = (theta + y) * torch.log(1.0 + (mu / (theta + eps))) + (
            y * (torch.log(theta + eps) - torch.log(mu + eps))
        )
        final = t1 + t2
        final = _nan2inf(final)

        return torch.mean(final)

    @staticmethod
    def _sample_z(mu, log_var):
        
        std = np.exp(0.5 * log_var)
        eps = torch.random.randn(std)
        return mu + std * eps

