# USTGCN: Unified Spatio-Temporal Modeling for Traffic Forecasting using Graph Neural Network

**Authors**
- [Amit Roy](https://amitroy7781.github.io/)
- [Kashob Kumar Roy](https://www.linkedin.com/in/forkkr/) 
- [Amin Ahsan Ali](http://www.cse.iub.edu.bd/faculties/53)
- [M Ashraful Amin](http://www.cse.iub.edu.bd/faculties/25) 
- [A K M Mahbubur Rahman](http://www.cse.iub.edu.bd/faculties/56)

This is a pytorch implementation of our [paper](https://arxiv.org/abs/2104.12518) "Unified Spatio-Temporal Modeling for Traffic Forecasting using Graph Neural Network" which has been accepted by IJCNN 2021.  Check the video presentation of our paper [here](https://youtu.be/95EJAFOsUmY).



# Abstract
Research in deep learning models to forecast traffic intensities has gained great attention in recent years due to their capability to capture the complex spatio-temporal relationships within the traffic data. However, most state-of-the-art approaches have designed spatial-only (e.g. Graph Neural Networks) and temporal-only (e.g. Recurrent Neural Networks) modules to separately extract spatial and temporal features. However, we argue that it is less effective to extract the complex spatio-temporal relationship with such factorized modules. Besides, most existing works predict the traffic intensity of a particular time interval only based on the traffic data of the previous one hour of that day. And thereby ignores the repetitive daily/weekly pattern that may exist in the last hour of data. Therefore, we propose a Unified Spatio-Temporal Graph Convolution Network (USTGCN) for traffic forecasting that performs both spatial and temporal aggregation through direct information propagation across different timestamp nodes with the help of spectral graph convolution on a spatio-temporal graph. Furthermore, it captures historical daily patterns in previous days and current-day patterns in current-day traffic data. Finally, we validate our work's effectiveness through experimental analysis, which shows that our model USTGCN can outperform state-of-the-art performances in three popular benchmark datasets from the Performance Measurement System (PeMS). Moreover, the training time is reduced significantly with our proposed USTGCN model.

# Motivation
![Motivation Figure](motivation_figure.png?raw=true "Title")
Factorized Spatial-only and Temporal-only Aggregation (Left) vs. Unified Spatio-Temporal Aggregation (Right). For a target node in a physical traffic network state-of-the-art approaches capture spatial information from neighbor nodes in each timestamps and aggregate the information for the corresponding node at different timestamps. In contrary, capturing the traffic information for a target node from both spatial and temporal component in a unified manner can learn the inter-relationsip from neighbor nodes at different timestamps more comprehensively.

# USTGCN 
![USTGCN](USTGCN.png?raw=true "Title")
Unified Spatio-Temporal Graph Convolutional Network, USTGCN. The unified spatio-temporal adjacency matrix, **A<sub>ST</sub>** showcases the cross space-time connections among nodes from different timestamps which consists of three types of submatrix: **A** as diagonal submatrix, **Ã** as lower submatrix and **0** as upper submatrix. **A<sub>ST</sub>**, a lower triangular matrix, facilitates traffic feature propagation from neighboring nodes only from the previous timestamps. The input features of different timestamps at convolution layer **l** are stacked into **X<sup>l</sup><sub>self</sub>** which is element-wise multiplied with broadcasted temporal weight parameter **W<sup>l</sup><sub>Temp</sub>** indicating the importance of the feature at the different timestamp. Afterwards, graph convolution is performed followed by weighted combination of self representation, **X<sup>l</sup><sub>self</sub>** and spatio-temporal aggregated vector,  **X<sup>l</sup><sub>ST</sub>**  to compute the representation **X<sup>l+1</sup><sub>self</sub>** that is used as input features at next layer, **l+1** or fed into the regression task.

# Model Architecture
![USTGCN Model](USTGCN_model.png?raw=true "Title")

To learn both daily and current-day traffic pattern, for each node we stack the traffic speeds of the last seven days  (traffic pattern during 09:30 AM - 10:30 AM for the last  week depicted with green color) along with the current-day traffic pattern for the past hour (traffic speed during 9:05 AM - 10:00 AM on current day i.e. Tuesrday depicted with red color) into the corresponding feature vector. We feed the feature matrix stacked for **N** nodes in the traffic network across **T = 12** timestamps to the USTGCN model of **K** convolution layers to compute spatio-temporal embedding. Finally, the regression module predicts future traffic intensities by utilizing the spatio-temporal embeddings.

# Comarison with Baselines
![Baseline Model](baseline_comparison.png?raw=true "Title")

# Envirnoment Set-Up 

Clone the git project:

```
$ git clone https://github.com/AmitRoy7781/USTGCN
```

Create a new conda Environment and install required packages (Commands are for ubuntu 16.04)

```
$ conda create -n TrafficEnv python=3.7
$ conda activate TrafficEnv
$ pip install -r requirements.txt
```

# Basic Usage:

Main Parameters:

```
--dataset           The input traffic dataset(default:PeMSD7)
--GNN_layers        Number of layers in GNN(default:3)
--num_timestamps    Number of timestamps in Historical and current model(default:12)
--pred_len          Traffic Prediction after how many timestamps(default:3)
--epochs            Number of epochs during training(default:200)
--seed              Random seed. (default: 42)
--cuda              Use GPU if declared
--save_model        Save model if declared
--trained_model     Run pretrained model if declaired
```

# Example Usage

**Train Model Using:**
```
$ python3 USTGCN.py --cuda --dataset PeMSD7 --pred_len 3 --save_model
```

<!-- **Run Trained Model:**

Please download the trained USTGCN models from [Google drive]() and place it in `saved_model/PeMSD7` folder

```
$ python3 USTGCN.py --cuda --dataset PeMSD7  --pred_len 3 --trained_model
```

**Run Trained Model:**

Please download the trained SSTGNN models from [Google drive]() and place them in `PeMSD7` folder

```
$ python3 USTGCN.py --cuda --dataset PeMSD7 --pred_len 3 --trained_model
```
!-->
 
## Cite

If you find our paper or repo useful then please cite our paper:

```bibtex
@inproceedings{roy2021unified,
  title={Unified spatio-temporal modeling for traffic forecasting using graph neural network},
  author={Roy, Amit and Roy, Kashob Kumar and Ali, Amin Ahsan and Amin, M Ashraful and Rahman, AKM Mahbubur},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2021},
  organization={IEEE}
}
```

