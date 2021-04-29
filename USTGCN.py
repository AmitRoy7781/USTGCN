# -*- coding: utf-8 -*-
"""
*Model: USTGCN_Same_Time_Zone*

# Packages
"""

import sys
import os
import torch
import argparse
import pyhocon
import random
import math
import copy
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from datetime import datetime

"""# Data Center"""

class DataCenter(object):

    def __init__(self, config):
        super(DataCenter, self).__init__()
        self.config = config

    def getPositionEmbedding(self,pos):
      input = np.arange(0,pos+1,1)
      a = input * 360
      day = a / 288
      week = a / 2016
      month = a / 8640
      day = np.deg2rad(day)
      week = np.deg2rad(week)
      day = np.sin(day)
      week = np.sin(week)
      combined = day+week
      return combined

  

    def load_data(self,ds,st_day,en_day,hr_sample,day,pred_len):

      content_file = self.config['file_path.' + ds + '_content']

      if ds=="PeMSD8" or ds == "PeMSD4":
        timestamp_data = np.load(content_file)
        timestamp_data = timestamp_data[:,:,2]
      else:
        timestamp_data = []

        with open(content_file) as fp:
          for i, line in enumerate(fp):
            info = line.strip().split(",")
            info = [float(x) for x in info]

            timestamp_data.append(info)

      timestamp_data = np.asarray(timestamp_data)
      timestamp_data = timestamp_data.transpose()
      tot_node = timestamp_data.shape[0]
      tot_ts = timestamp_data.shape[1]
      
      st_day -= 1
      timestamp = 24 * hr_sample
      
      ts_data = []
      label_data = []
      
      for idx in range(st_day,en_day+1-day,1):

        st_point = idx*timestamp
        en_point = (idx+1)*timestamp


        last_hour = False
        for st in range(st_point, en_point):
          a_data = []
          
          if st + 12 + ((day-1)*timestamp) + pred_len == tot_ts:
            break

          for it in range(st,st+12):
            his = timestamp_data[:,(it+pred_len): it+pred_len+((day-1)*timestamp) :timestamp]
            cur = timestamp_data[:,it + (day-1)*timestamp].reshape(tot_node,1)
            a = np.concatenate((his,cur),axis=1).reshape(1,tot_node,day)
            a_data.append(a)
          
          a_data = np.concatenate(a_data,axis=0)
          
          pred_data = []
          for pred in range(pred_len):
            gt = timestamp_data[:,st + 12 + ((day-1)*timestamp) + pred].reshape(tot_node,1)
            pred_data.append(gt)
          
          pred_data = np.concatenate(pred_data,axis =1)

          ts_data.append(a_data)
          label_data.append(pred_data)
              
              
      return ts_data,label_data
    
    def load_adj(self,ds):
      W = self.load_PeMSD(self.config['file_path.'+ ds +'_cites'])
      adj_lists = defaultdict(set)
      for row in range(len(W)):
        adj_lists[row] = set()
        for col in range(len(W)):
          if float(W[row][col]) >0 :
            adj_lists[row].add(col)
            adj_lists[col].add(row)
      
      adj = torch.zeros((len(adj_lists),len(adj_lists)))
      for u in adj_lists:
        for v in adj_lists[u]:
            adj[u][v] = 1
            adj[v][u] = 1
      return adj
        

    def load_PeMSD(self,file_path, sigma2=0.1, epsilon=0.5, scaling=True):

      try:
          W = pd.read_csv(file_path, header=None).values
      except FileNotFoundError:
          print('ERROR: No File Found.')
      
      n = W.shape[0]
      W = W / 10000.
      W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)

      return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask

"""# Utility Functions

"""

def evaluate(test_nodes,raw_features,labels, USTGCN, regression, device,test_loss):


    models = [USTGCN, regression]

    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                params.append(param)

    
    val_nodes = test_nodes
    embs = USTGCN(raw_features,False)
    predicts = regression(embs)
    loss_sup = torch.nn.MSELoss()(predicts, labels)
    loss_sup /= len(val_nodes)
    test_loss += loss_sup.item()

    for param in params:
        param.requires_grad = True

    return predicts,test_loss

def RMSELoss(yhat,y):
    yhat = torch.FloatTensor(yhat)
    y = torch.FloatTensor(y)
    return torch.sqrt(torch.mean((yhat-y)**2)).item()


def mean_absolute_percentage_error(y_true, y_pred):

  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)

  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

"""# Downstream Task"""

class Regression(nn.Module):

    def __init__(self, emb_size, out_size):
        super(Regression, self).__init__()

        self.layer = nn.Sequential(nn.Linear(emb_size, emb_size),
                                nn.ReLU(),
                                nn.Linear(emb_size, out_size),
                                nn.ReLU()
                                )
            

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, embds):
        logists = self.layer(embds)
        return logists

"""# Data Loader"""

class DataLoader:

  def __init__(self, config,ds,pred_len):
    
    super(DataLoader, self).__init__()

    self.ds = ds
    self.dataCenter = DataCenter(config)

    if ds == "PeMSD7":
      train_st = 1
      train_en = 22

      test_st = 23
      test_en = 44 
      
    elif ds == "PeMSD8":
      train_st = 1
      train_en = 50

      test_st = 51
      test_en = 62 
    
    elif ds == "PeMSD4" :
      train_st = 1
      train_en = 47

      test_st = 48
      test_en = 58 
  
    self.train_st = train_st
    self.train_en = train_en
    self.test_st = test_st
    self.test_en = test_en

    self.hr_sample = 12
    self.day = 8
    self.pred_len = pred_len
    
  def load_data(self):
    print("Loading Data...")
    train_data,train_label = self.dataCenter.load_data(self.ds,self.train_st,self.train_en,self.hr_sample,self.day,self.pred_len)
    test_data,test_label = self.dataCenter.load_data(self.ds,self.test_st,self.test_en,self.hr_sample,self.day,self.pred_len)
    adj = self.dataCenter.load_adj(self.ds)
    print("Data Loaded")
    print("Dataset: ", self.ds)
    print("Total Nodes: ",adj.shape[0])
    print("Train timestamps: ",len(train_data))
    print("Test timestamps: ",len(test_data))
    print("Predicting After: ",self.pred_len*5,"minutes")

    return train_data,train_label,test_data,test_label,adj

"""# Traffic Model"""

class TrafficModel:

    def __init__(self, train_data,train_label,test_data,test_label,adj, 
                 config, ds, input_size, out_size,GNN_layers,
                epochs, device,num_timestamps, pred_len,save_flag,PATH,t_debug,b_debug):
      
      super(TrafficModel, self).__init__()
      
      
      self.train_data,self.train_label,self.test_data,self.test_label,self.adj = train_data,train_label,test_data,test_label,adj
      self.all_nodes = [i for i in range(self.adj.shape[0])]

      self.ds = ds
      self.input_size = input_size
      self.out_size = out_size
      self.GNN_layers = GNN_layers
      self.day = input_size 
      self.device = device
      self.epochs = epochs
      self.regression = Regression(input_size * num_timestamps, pred_len)
      self.num_timestamps = num_timestamps
      self.pred_len = pred_len

      self.node_bsz = 512
      self.PATH = PATH
      self.save_flag = save_flag

      self.train_data = torch.FloatTensor(self.train_data).to(device)
      self.test_data = torch.FloatTensor(self.test_data).to(device)
      self.train_label = torch.FloatTensor(self.train_label).to(device)
      self.test_label = torch.FloatTensor(self.test_label).to(device)
      self.all_nodes = torch.LongTensor(self.all_nodes).to(device)
      self.adj = torch.FloatTensor(self.adj).to(device)
      
      self.t_debug = t_debug
      self.b_debug = b_debug

      
    def run_model(self):


      timeStampModel = CombinedGNN(self.input_size,self.out_size,self.adj, 
      self.device,1,self.GNN_layers,self.num_timestamps,self.day)
      timeStampModel.to(self.device)

      regression = self.regression
      regression.to(self.device)

      min_RMSE = float("Inf") 
      min_MAE = float("Inf") 
      min_MAPE = float("Inf")
      best_test = float("Inf")
      

      lr = 0.001
        
      train_loss = torch.tensor(0.).to(self.device)  
      for epoch in range(1,epochs):

        print("Epoch: ",epoch," running...")

        tot_timestamp = len(self.train_data)
        if self.t_debug:
          tot_timestamp = 60
        idx = np.random.permutation(tot_timestamp)

        for data_timestamp in idx:

          tr_data = self.train_data[data_timestamp]
          tr_label = self.train_label[data_timestamp]

          timeStampModel, regression, train_loss = apply_model(self.all_nodes,timeStampModel, 
          regression,self.node_bsz, self.device,tr_data,tr_label,train_loss,lr)

          if self.b_debug:
            break

        train_loss /= len(idx)
        if epoch<= 24 and epoch%8==0:
          lr *= 0.5
        else:
          lr = 0.0001


        print("Train avg loss: ",train_loss)


        pred = []
        label = []
        tot_timestamp = len(self.test_data)

        if self.t_debug:
          tot_timestamp = 60

        idx = np.random.permutation(tot_timestamp)
        test_loss = torch.tensor(0.).to(self.device)
        for data_timestamp in idx:

          #test_label
          raw_features = self.test_data[data_timestamp]
          test_label = self.test_label[data_timestamp]
          
          #evaluate
          temp_predicts,test_loss = evaluate(self.all_nodes,raw_features,test_label,
            timeStampModel, regression, self.device,test_loss)

          label = label + test_label.detach().tolist()
          pred = pred + temp_predicts.detach().tolist()

          if self.b_debug:
            break

        
        test_loss /= len(idx)
        print("Average Test Loss: ",test_loss)

        if test_loss <= best_test:
          best_test = test_loss
          pred_after = self.pred_len * 5
          if self.save_flag:
              torch.save(timeStampModel, self.PATH + "/" + self.ds + "/bestTmodel_" + str(pred_after) +"minutes.pth")
              torch.save(regression, self.PATH + "/" + self.ds + "/bestRegression_" + str(pred_after) +"minutes.pth")

        RMSE = torch.nn.MSELoss()(torch.FloatTensor(pred), torch.FloatTensor(label))
        RMSE = torch.sqrt(RMSE).item()
        MAE = mean_absolute_error(pred,label)
        MAPE = mean_absolute_percentage_error(label,pred)
        
        
        print("Epoch:", epoch)
        print("RMSE: ", RMSE)
        print("MAE: ", MAE)
        print("MAPE: ", MAPE)
        print("===============================================")

        min_RMSE = min(min_RMSE,RMSE)
        min_MAE = min(min_MAE,MAE)
        min_MAPE = min(min_MAPE,MAPE)
        
        print("Min RMSE: ", min_RMSE)
        print("Min MAE: ", min_MAE)
        print("Min MAPE: ", min_MAPE)
        
        print("===============================================")



      return
    
    def run_Trained_Model(self):
      pred_after = self.pred_len * 5
      timeStampModel = torch.load(self.PATH + "/" + self.ds +  "/bestTmodel_" + str(pred_after) +"minutes.pth")
      regression = torch.load(self.PATH + "/" + self.ds +  "/bestRegression_" + str(pred_after) +"minutes.pth")
      pred = []
      label = []
      tot_timestamp = len(self.test_data)
      idx = np.random.permutation(tot_timestamp+1-self.num_timestamps)
      test_loss = torch.tensor(0.).to(self.device)
      for data_timestamp in idx:

          #test_label
          raw_features = self.test_data[data_timestamp]
          test_label = self.test_label[data_timestamp]
          
          #evaluate
          temp_predicts,test_loss = evaluate(self.all_nodes,raw_features,test_label,
            timeStampModel, regression, self.device,test_loss)

          label = label + test_label.detach().tolist()
          pred = pred + temp_predicts.detach().tolist()

      
      test_loss /= len(idx)
      print("Average Test Loss: ",test_loss)

      
      RMSE = torch.nn.MSELoss()(torch.FloatTensor(pred), torch.FloatTensor(label))
      RMSE = torch.sqrt(RMSE).item()
      MAE = mean_absolute_error(pred,label)
      MAPE = mean_absolute_percentage_error(label,pred)
      
      
      print("RMSE: ", RMSE)
      print("MAE: ", MAE)
      print("MAPE: ", MAPE)
      print("===============================================")

"""# Spatio-Temporal GNN"""

class SPTempGNN(nn.Module):
  def __init__(self,D_temporal,A_temporal,num_timestamps,out_size,tot_nodes):
    super(SPTempGNN, self).__init__()
    
    self.tot_nodes = tot_nodes
    self.sp_temp = torch.mm(D_temporal,torch.mm(A_temporal,D_temporal))

    

    self.his_temporal_weight = nn.Parameter(torch.FloatTensor(num_timestamps,out_size))

    self.his_final_weight = nn.Parameter(torch.FloatTensor(2*(out_size),out_size))

  
  def forward(self,his_raw_features):
    his_self = his_raw_features
    his_temporal = self.his_temporal_weight.repeat(self.tot_nodes,1) * his_raw_features
    his_temporal = torch.mm(self.sp_temp,his_temporal)

    his_combined = torch.cat([his_self,his_temporal], dim=1)
    his_raw_features =F.relu(his_combined.mm(self.his_final_weight))


    return his_raw_features

"""# Combined GraphSAGE

"""

class CombinedGNN(nn.Module):
    def __init__(self,input_size,out_size, adj_lists,
                 device,st,GNN_layers,num_timestamps,day):
        super(CombinedGNN, self).__init__()

        self.st = st
        self.num_timestamps = num_timestamps
        self.out_size = out_size
        self.tot_nodes = adj_lists.shape[0]
        self.device = device
        self.adj_lists = adj_lists 
        self.GNN_layers = GNN_layers

        
        self.day = day

                
        self.his_weight = nn.Parameter(torch.FloatTensor(out_size, self.num_timestamps*out_size))
        self.cur_weight = nn.Parameter(torch.FloatTensor(1, self.num_timestamps*1))

        A = self.adj_lists
        dim = self.num_timestamps*self.tot_nodes

        A_temporal = torch.zeros(dim,dim).to(device)
        D_temporal = torch.zeros(dim,dim).to(device)
        identity = torch.eye(self.tot_nodes).to(device)

        for i in range(0, self.num_timestamps):
          for j in range(0, i+1):

            row_st = i*self.tot_nodes
            row_en = row_st + self.tot_nodes
            col_st = j*self.tot_nodes
            col_en = col_st + self.tot_nodes

            if i == j: #adj matrix  
              A_temporal[row_st:row_en,col_st:col_en] = A 
            else: #identity matrix
              A_temporal[row_st:row_en,col_st:col_en] = identity + A
        
        row_sum = torch.sum(A_temporal,0)

        for i in range(dim):
          D_temporal[i,i] = 1/max(torch.sqrt(row_sum[i]),1)      
        
        for i in range(GNN_layers):
          sp_temp = SPTempGNN(D_temporal,A_temporal,num_timestamps,out_size,self.tot_nodes)
          setattr(self, 'sp_temp_layer'+str(i), sp_temp)
          

        dim2 = self.num_timestamps*(out_size)
        self.final_weight = nn.Parameter(torch.FloatTensor(dim2, dim2))

        self.init_params()
        

        

    def init_params(self):
      

      for param in self.parameters():
          if(len(param.shape)>1):
            nn.init.xavier_uniform_(param)


    def forward(self,his_raw_features,isTrain):
      
      dim = self.num_timestamps*self.tot_nodes
      his_raw_features = his_raw_features[:,:,:self.day].view(dim,self.day)

      
      for i in range(self.GNN_layers):
        sp_temp = getattr(self, 'sp_temp_layer'+str(i))
        his_raw_features = sp_temp(his_raw_features)
        
      
      his_list = []

      for timestamp in range(self.num_timestamps):
        st = timestamp * self.tot_nodes
        en = (timestamp+1) * self.tot_nodes
        his_list.append(his_raw_features[st:en,:])

      his_final_embds = torch.cat(his_list,dim=1)


      final_embds = his_final_embds
      final_embds = F.relu(self.final_weight.mm(final_embds.t()).t())



      return final_embds

"""# Applying Model"""

def apply_model(train_nodes, CombinedGNN, regression, 
                node_batch_sz, device,train_data,train_label,avg_loss,lr):


    models = [CombinedGNN, regression]
    params = []
    for model in models:
      for param in model.parameters():
          if param.requires_grad:
              params.append(param)



    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0)

    optimizer.zero_grad()  # set gradients in zero...
    for model in models:
      model.zero_grad()  # set gradients in zero

    node_batches = math.ceil(len(train_nodes) / node_batch_sz)

    loss = torch.tensor(0.).to(device)
    #window slide
    raw_features = train_data
    labels = train_label
    for index in range(node_batches):

      nodes_batch = train_nodes[index * node_batch_sz:(index + 1) * node_batch_sz]
      nodes_batch = nodes_batch.view(nodes_batch.shape[0],1)
      labels_batch = labels[nodes_batch]      
      labels_batch = labels_batch.view(len(labels_batch),pred_len)
      embs_batch = CombinedGNN(raw_features,True)  # Finds embeddings for all the ndoes in nodes_batch

      logists = regression(embs_batch)


      loss_sup = torch.nn.MSELoss()(logists, labels_batch)

      loss_sup /= len(nodes_batch)
      loss += loss_sup



    avg_loss += loss.item()

    loss.backward()
    for model in models:
      nn.utils.clip_grad_norm_(model.parameters(), 5) 
    optimizer.step()

    optimizer.zero_grad()
    for model in models:
      model.zero_grad()

    return CombinedGNN, regression,avg_loss

"""#Training"""

parser = argparse.ArgumentParser(description='pytorch version of USTGCN')
parser.add_argument('-f')  


parser.add_argument('--dataset', type=str, default='PeMSD7')
parser.add_argument('--GNN_layers', type=int, default=3)
parser.add_argument('--num_timestamps', type=int, default=12)
parser.add_argument('--pred_len', type=int, default=3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cuda', action='store_true',help='use CUDA')
parser.add_argument('--trained_model', action='store_true')
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--input_size', type=int, default=8)
args = parser.parse_args()





device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
print('DEVICE:', device)

 
"""# Main Function"""

print('Traffic Forecasting GNN with Historical and Current Model')

#set user given seed to every random generator
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

PATH = os.getcwd() + "/"
config_file = PATH + "experiments.conf"

config = pyhocon.ConfigFactory.parse_file(config_file)
ds = args.dataset
pred_len = args.pred_len
data_loader = DataLoader(config,ds,pred_len)
train_data,train_label,test_data,test_label,adj = data_loader.load_data()

num_timestamps = args.num_timestamps
GNN_layers = args.GNN_layers
input_size = args.input_size
out_size = args.input_size
epochs = args.epochs


save_flag = args.save_model
t_debug = False
b_debug = False
hModel = TrafficModel(train_data,train_label,test_data,test_label,adj,config, ds, input_size, 
                       out_size,GNN_layers,epochs, device,num_timestamps,pred_len,save_flag,
                       PATH,t_debug,b_debug)


if not args.trained_model: #train model and evaluate
  print("Running Trained Model...")
  hModel.run_model()
else:
  print("Running Trained Model...")
  hModel.run_Trained_Model() #run trained model
