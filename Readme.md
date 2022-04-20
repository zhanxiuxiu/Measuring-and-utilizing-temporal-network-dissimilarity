# Measuring and utilizing temporal network dissimilarity

## About
Quantifying the structural and functional differences of temporal networks is a fundamental and challenging problem in the era of big data. This work proposes a temporal dissimilarity measure for temporal network comparison based on the fastest arrival distance distribution and spectral entropy based Jensen-Shannon divergence. 

For more details, please refer to the following paper: 
> Zhan, Xiu-Xiu, Chuang Liu, Zhipeng Wang, Huijuang Wang, Petter Holme, and Zi-Ke Zhang. Measuring and utilizing temporal network dissimilarity. arXiv preprint arXiv:2111.01334 (2021).

Three folders are given here, i.e., data illustrates the datasets that are used in this work, synthetic_data_generation gives the code of generating temporal networks by using activity driven model, TD describes the methods of computing the fastest arrival distance between nodes in a temporal network as well as the computation of temporal dissimilarity between two temporal networks.

### Synthetic temporal network generation:
1.   Generating temporal network with uniform activity distribution (python 3.8): 
```
python activity_driven_network_generation_uniform.py
```
2.   Generating temporal network with powerlaw activity distribution (python 2.7, powerlaw package 1.4.5): 
```
python activity_driven_network_generation_power_law.py
```

### Temporal network dissimilarity:
1. Fastest arrival distance calculation (python 3.8):
```
python single_network_Node_fastest_arrival_distance.py --dataset EEU1
```

2. temporal network dissimilarity (python 3.8): 

```
python Temporal_dissimilarity_value.py --dataset_name1 gallery1 --dataset_name2 gallery2
```

If you have any question about the paper or the code, please contact us. **Xiu-Xiu zhan**, **zhanxxiu@gmail.com**

Please cite that paper if you use the code. Thanks!
