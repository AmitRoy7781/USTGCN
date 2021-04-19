# USTGCN: Unified Spatio-Temporal Modeling for Traffic Forecasting using Graph Neural Network

**Authors**
- [Amit Roy](https://amitroy7781.github.io/)
- [Kashob Kumar Roy](https://www.linkedin.com/in/forkkr/) 
- [Amin Ahsan Ali](http://www.cse.iub.edu.bd/faculties/53)
- [M Ashraful Amin](http://www.cse.iub.edu.bd/faculties/25) 
- [A K M Mahbubur Rahman](http://www.cse.iub.edu.bd/faculties/56)

This is a pytorch implementation of "Unified Spatio-Temporal Modeling for Traffic Forecasting using Graph Neural Network"
which has been accepted by IJCNN 2021.

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
 -->