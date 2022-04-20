# coding=utf-8
# 基于FAD的D_measure
import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot
import matplotlib.pyplot as plt
import argparse
def parse_args():

    parser = argparse.ArgumentParser(description="Run FAD for a single network.")
    parser.add_argument('--dataset_name1', type=str, default='network_name')
    parser.add_argument('--dataset_name2', type=str, default='network_name')
    return parser.parse_args()

# 首先计算每个节点的距离分布P_i
#dataset_name = 'Gallery1_new'

def distance_distribution_cal(dataset_name):

    print(dataset_name)
    data_distance = 'network_distance/' + dataset_name + '_Distance_all_nodes.csv'
    node_distance = pd.read_csv(data_distance, sep=',', names=['node1', 'node2', 'distance'])
    distance_list = list(set(node_distance['distance']))
    # 对距离里面的值进行排序
    distance_list.sort()
    node_distance_seq_dict = {}
    for row in range(0, len(node_distance)):
        key = node_distance.iloc[row]['node1']
        distance_value = node_distance.iloc[row]['distance']
        if key not in node_distance_seq_dict:
            node_distance_seq_dict[key] = [distance_value]
        else:
            node_distance_seq_dict[key].append(distance_value)
    # 对每个节点，计算节点的距离分布
    Node_Distance_Distribution = {}  # 每个节点的距离分布，距离值由distance_list决定
    for key in node_distance_seq_dict:
        distance_seq = node_distance_seq_dict[key]
        # 计算分布
        dist_x = list(Counter(distance_seq).keys())
        dist_y = list(Counter(distance_seq).values())
        dist_y = np.array(dist_y)
        dist_y = 1.0 * dist_y / np.sum(dist_y)
        diff = list(set(distance_list) - set(dist_x))  # 距离不在dist_x中的
        dist_x = dist_x + diff
        dist_y = list(dist_y) + [0] * len(diff)
        dist_xy_dict = dict(zip(dist_x, dist_y))
        single_node_distance_distri = []
        for key_d in distance_list:
            single_node_distance_distri.append(dist_xy_dict[key_d])
        Node_Distance_Distribution[key] = single_node_distance_distri
    # 将节点的distance distribution变成dataframe, 每一列表示一个节点的距离分布
    Node_Distance_Distribution_df = pd.DataFrame.from_dict(Node_Distance_Distribution)
    # 将distance_list和节点的distance distribution存成csv
    np.savetxt('distance_distribution/' + dataset_name + '_distance_list.csv', distance_list, delimiter=',')
    Node_Distance_Distribution_df.to_csv('distance_distribution/' + dataset_name + '_distance_distribution.csv', float_format=None,
                            index=True)
    return Node_Distance_Distribution_df

'''

计算每个网络的distance distribution
from FAD_measure import distance_distribution_cal
dataset_name = ['EEU1', 'EEU2', 'EEU3', 'EEU4', 'ME', 'gallery1', 'gallery2', 'gallery3', 'gallery4', 'gallery5', 'gallery6',
                'HS2011', 'HS2012', 'HS2013', 'HT2009', 'PS', 'SFHH']
for data in dataset_name:
    distance_distribution_cal(data)
    
'''

# 计算网络的NND和miu_G

def NND(dataset_name):

    distance_dist = pd.read_csv('distance_distribution/' + dataset_name + '_distance_distribution.csv', sep=',', index_col=0)
    distance_list = pd.read_csv('distance_distribution/' + dataset_name + '_distance_list.csv', header=None)
    tmp_list = sorted(list(distance_list[0][:len(distance_list)]))
    if tmp_list[-1] == np.inf:
        diameter = tmp_list[-2] + 1
    else:
        diameter = tmp_list[-1]
    #diameter = max(list(distance_list[0][:len(distance_list)]))
    Node_list = list(distance_dist.columns)
    J = len(distance_dist[Node_list[0]])
    N = len(Node_list)   # 假设节点的序列是1~N的
    distance_matrix = distance_dist.values
    miu = (distance_matrix.sum(axis=1))/N   # miu_j
    print(distance_matrix.shape)  # 矩阵的大小，(d+1) * N, d表示直径
    distance_matrix_new = distance_matrix.T
    print(distance_matrix_new.shape) # 矩阵的大小，N * (d+1), d表示直径
    curlicue_J = 0
    for i in range(N):
        for j in range(J):
            p_ij = distance_matrix_new[i][j]
            miu_j = miu[j]
            if p_ij != 0 and miu_j != 0:
                curlicue_J = curlicue_J + p_ij * np.log(p_ij / miu_j)
    curlicue_J_Normalized = curlicue_J / N
    NND = curlicue_J_Normalized / np.log(diameter + 1)
    miu_G = miu
    return NND, miu_G
# 计算一个网络的NND 值
'''
dataset_name = 'gallery1'
from FAD_measure import NND
NND, miu_G = NND(dataset_name)
'''
def entropy(a):
    b = np.where(a > 0)
    return -np.sum(a[b]*np.log(a[b]))

# dataset_name1 = 'gallery1'
# dataset_name2 = 'gallery2'

def FAD_dissimilarity(dataset_name1, dataset_name2):
    distance_distribution_cal(dataset_name1)
    distance_distribution_cal(dataset_name2)
    NND1, miu_G1 = NND(dataset_name1)
    NND2, miu_G2 = NND(dataset_name2)
    w1 = 0.5
    w2 = 0.5
    miu_G1 = sorted(miu_G1)
    miu_G2 = sorted(miu_G2)
    # 计算miu_G1和miu_G2是否维度相同，不同要变成相同
    if len(miu_G1) > len(miu_G2):
        Complement_len = len(miu_G1) - len(miu_G2)
        miu_G2 = [0]*Complement_len + miu_G2
    elif len(miu_G1) < len(miu_G2):
        Complement_len = len(miu_G2) - len(miu_G1)
        miu_G1 = [0] * Complement_len + miu_G1
    miu_new = (np.array(miu_G1) + np.array(miu_G2)) / 2
    miu_G1 = np.array(miu_G1)
    miu_G2 = np.array(miu_G2)
    first_term = np.sqrt(max((entropy(miu_new) - (entropy(miu_G1) + entropy(miu_G2)) / 2) / np.log(2), 0))
    second_term = abs(np.sqrt(NND1) - np.sqrt(NND2))
    FAD_dis = w1 * first_term + w2 * second_term
    return FAD_dis, NND1, NND2

def main(args):
    FAD_dis, NND1, NND2 = FAD_dissimilarity(args.dataset_name1, args.dataset_name2)
    print('TD between two networks:', FAD_dis)



if __name__ == '__main__':
    args = parse_args()
    main(args)


#python Temporal_dissimilarity_value.py --dataset_name1 gallery1 --dataset_name2 gallery2