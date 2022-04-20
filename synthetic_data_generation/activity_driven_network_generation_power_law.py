# coding=utf-8
from collections import Counter
import powerlaw
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats
from scipy.linalg import expm
import codecs
import argparse
def parse_args():

    parser = argparse.ArgumentParser(description="Run power law activity temporal network.")
    parser.add_argument('--N', type=int, default=100, help='the number of nodes')
    parser.add_argument('--m', type=int, default=2, help='number of contacts in each step')
    parser.add_argument('--T', type=int, default=1000, help='time window of a temporal network')
    parser.add_argument('--a', type=float, default=3.0, help='power law slope')
    return parser.parse_args()

# activity driven model
def main(args):
    for iteration in range(1, 21):
        print(iteration)
        xmin = 1
        random_numbers = powerlaw.Power_Law(xmin=xmin, parameters=[args.a]).generate_random(args.N)
        random_numbers = list(random_numbers)
        P = random_numbers / sum(random_numbers)
        Node_list = range(1, args.N + 1)
        Node_activity_prob = dict(zip(Node_list, P))
        G = {}
        for t in range(1, args.T + 1):
            if t % 10000 == 0:
                print(t)
            G_t = []
            for node in Node_list:
                if np.random.rand() < Node_activity_prob[node]:
                    m1 = 0
                    while m1 < args.m:
                        current_node_list = [node]
                        Node_list1 = list(set(Node_list) - set(current_node_list))
                        node_inactive = np.random.choice(Node_list1)
                        contact = (node, node_inactive)
                        if node < node_inactive:
                            contact = (node, node_inactive)
                        elif node > node_inactive:
                            contact = (node_inactive, node)
                        if contact not in G_t:
                            G_t.append(contact)
                            m1 = m1 + 1
            G[t] = G_t  # temporal network G
        G_dataframe = pd.DataFrame(columns=('node1', 'node2', 'timestamp'))
        i = 0
        for key in G:
            if key % 10000 == 0:
                print(key)
            for contact in G[key]:
                node1 = contact[0]
                node2 = contact[1]
                timestamp = key
                G_dataframe.loc[i] = [node1, node2, timestamp]
                i = i + 1

        node_sequence = list(G_dataframe['node1']) + list(G_dataframe['node2'])
        node_activity_count = Counter(node_sequence)
        activity_sequence = node_activity_count.values()
        G_dataframe.to_csv('powerlaw/' + 'powerlaw_acti_' + 'N' + str(args.N) + 'm' + str(args.m) + 'T' + str(args.T) + 'a' + str(args.a) + 'x_min' + str(
                xmin) + 'network' + 'iter' + str(iteration) + '.csv', index=None)
if __name__ == "__main__":
    args = parse_args()
    main(args)
