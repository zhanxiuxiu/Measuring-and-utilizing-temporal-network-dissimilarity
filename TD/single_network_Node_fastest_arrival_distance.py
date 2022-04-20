import pandas as pd
import numpy as np
import random
from joblib import Parallel, delayed
import argparse
def parse_args():

    parser = argparse.ArgumentParser(description="Run FAD for a single network.")
    parser.add_argument('--dataset', type=str, default='network_name')
    return parser.parse_args()

def process(i0, N, max_time, CT):

    path_len = np.ones(shape=N + 1, dtype=np.int32) * np.inf
    state = np.zeros(N + 1, dtype=np.int32)
    state[i0] = 1
    path_len[i0] = 0
    CT_arr = CT.values
    #print('i0', i0)
    t_list = CT_arr[np.logical_or(CT_arr[:, 0] == i0, CT_arr[:, 1] == i0)][:, 2]
    #print('t_list', t_list)
    if len(t_list) == 0:
        distance_all_nodes = [[i0, i, path_len[i]] for i in range(1, len(path_len))]
    else:
        min_t = t_list.min()
    #min_t = CT_arr[np.logical_or(CT_arr[:, 0] == i0, CT_arr[:, 1] == i0)][:, 2].min()
        #print('min_t', min_t)
        for t in range(min_t, max_time + 1):
            #print('spreading t', t)
            all_infected_t = np.where(state == 1)[0]
            Ct = CT_arr[CT_arr[:, 2] == t]

            for infected in all_infected_t:
                nei_right = Ct[Ct[:, 0] == infected][:, 1]
                nei_left = Ct[Ct[:, 1] == infected][:, 0]
                nei = np.concatenate([nei_left, nei_right], axis=0)
                nei_size = nei.shape[0]
                if nei_size > 0:
                    for each_nei in nei:
                        if state[each_nei] == 0:
                            state[each_nei] = 1
                            new_len = (path_len[infected] + 1)
                            path_len[each_nei] = np.min(np.array([path_len[each_nei], new_len]), axis=0)
        distance_all_nodes = [[i0, i, path_len[i]] for i in range(1, len(path_len))]
    # else:
    #     distance_all_nodes = []
    return distance_all_nodes


def Node_fastest_arrival_distance(path):
    ET = pd.read_csv(path, header=None)
    Node = set(ET[0]) | set(ET[1])
    N = max(Node)
    max_time = max(ET[2])
    CT = ET.sort_values(by=2, axis=0, ascending=True)
    CT.reset_index(drop=True, inplace=True)
    #并行执行每个节点，n_jobs 表示线程数
    results = Parallel(n_jobs=4)(delayed(process)(i, N, max_time, CT) for i in range(1, N+1))
    #print(type(results))
    results = np.array(results)
    #print(results)
    #print(results)
    #results = np.reshape(np.array(results), (len(results) * len(results[0]), len(results[0][0])))
    results = np.reshape(results, (len(results) * len(results[0]), len(results[0][0])))
    return results

def main(args):
        network_name = args.dataset
        results = Node_fastest_arrival_distance('../data/' + network_name + '.csv')
        save_distance_path = 'network_distance/' + network_name + '_Distance_all_nodes.csv'
        pd.DataFrame(results).to_csv(save_distance_path, header=None, index=None)

if __name__ == '__main__':
    args = parse_args()
    main(args)

#python single_network_Node_fastest_arrival_distance.py --dataset EEU1
#python single_network_Node_fastest_arrival_distance.py --dataset gallery1