import os
import pandas as pd
import networkx as nx
import pickle as pkl
import numpy as np
import time, torch
from tools import date_range, to_sparse_tensor
from sklearn.preprocessing import StandardScaler

def normalize_adj(adj, mode='random walk'):
    # mode: 'random walk', 'aggregation'
    if mode == 'random walk': # for T. avg weight for sending node
        deg = np.sum(adj, axis=1).astype(np.float32)
        inv_deg = np.reciprocal(deg, out=np.zeros_like(deg), where=deg!=0)
        D_inv = np.diag(inv_deg)
        normalized_adj = np.matmul(D_inv, adj)
    if mode == 'aggregation': # for W. avg weight for receiving node
        deg = np.sum(adj, axis=0).astype(np.float32)
        inv_deg = np.reciprocal(deg, out=np.zeros_like(deg), where=deg!=0)
        D_inv = np.diag(inv_deg)
        normalized_adj = np.matmul(adj, D_inv)
    return normalized_adj

def get_road_list(road_df=None, out_path='fastdatasf/road_list.csv', update=False):
    # road_list: the mapping of road_id. for adjacency matrix
    if not update and os.path.exists(out_path):
        print('Road list exists')
        road_list = pd.read_csv(out_path)
    else:
        print('Generating new road list from road df')
        road_list = road_df[['road_id']].reset_index().drop('index', axis=1)
        road_list.to_csv(out_path, index=False)
    return road_list

def extract_trajectory_transition(start_date, end_date, interval=15):
    # start_date, end_date = '20160401', '20160421'
    # interval is in minutes, and should divide 60.
    
    total_file_path = 'fastdatasf/trajectory_transition_%s_%s.pkl'%(start_date, end_date)
    if os.path.exists(total_file_path):
        with open(total_file_path, 'rb') as f:
            print('Total file exists')
            total_trajectory_transition = pkl.load(f)
    else:
        result_train20 = pd.read_csv('fastdatasf/sf_train20.csv')
        result_train20.columns = ['vehicle_id', 'trajectory_id', 'time', 'road_id', 'scenario']
        
        interval = 15
        outputPath = 'fastdatasf/road_list.csv'
        edges_path = 'fastdatasf/edges_for_graph.csv'

        print('Reading road list')

        edges_df = pd.read_csv(edges_path)
        edges_df['length']/= 1000 # convert to km

        road_list = get_road_list(edges_df, outputPath)
        road_list = road_list[road_list['road_id'].isin(result_train20['road_id'].unique())].reset_index(drop=True)
        road_list = road_list.reset_index().rename(columns={'index':'road_index'})
        
        print('Creating empty total_trajectory_transition')
        total_trajectory_transition = np.zeros((60//interval*24, len(road_list), len(road_list)), dtype=np.int8)
        
        date_list = date_range(start_date, end_date)
        
        for current_date in date_list:
            
            print('Date %s'%(current_date))
            file_path = 'fastdatasf/trajectory_transition_%s_%s.pkl'%(current_date, current_date)
            if os.path.exists(file_path):
                print('File exists')
                with open(file_path, 'rb') as f:
                    trajectory_transition = pkl.load(f)
                
            else:
                start_time = time.time()
                
                print('Reading recovered_trajectory_df')
                trajectory_path = 'fastdatasf/recovered_trajectory_df_%s_%s.csv'%(current_date, current_date)
                recovered_trajectory_df = pd.read_csv(trajectory_path)
                print('Time spent till now: %.2f seconds'%(time.time() - start_time))

                print('Extracting road index')
                recovered_trajectory_df = recovered_trajectory_df.merge(road_list, on='road_id', how='left')
                print('Time spent till now: %.2f seconds'%(time.time() - start_time))

                print('Extracting time_index')
                def extract_time_index(time):
                    hour = int(time[-8:-6])
                    minute = int(time[-5:-3])
                    time_index = hour * 4 + minute // 15
                    return time_index
                recovered_trajectory_df['time_index'] = recovered_trajectory_df['time'].apply(extract_time_index)
                print('Time spent till now: %.2f seconds'%(time.time() - start_time))

                print('Creating empty trajectory_transition')
                trajectory_transition = np.zeros((60//interval*24, len(road_list), len(road_list)), dtype=np.int16)
                print('Time spent till now: %.2f seconds'%(time.time() - start_time))

                print('Calculating trajectory_transition')
                for i, row in recovered_trajectory_df.iterrows():
                    if i != 0:
                        if previous_row['vehicle_id'] == row['vehicle_id'] and \
                        previous_row['trajectory_id'] == row['trajectory_id'] and \
                        previous_row['road_id'] != row['road_id']:
                            trajectory_transition[previous_row['time_index'], previous_row['road_index'], row['road_index']] += 1
                    if i % 100000 == 0:
                        print(i, 'at %.2f seconds'%(time.time() - start_time))
                    previous_row = row
                print('Time spent till now: %.2f seconds'%(time.time() - start_time))

                print('Saving trajectory_transition')
                with open(file_path, 'wb') as f:
                    pkl.dump(trajectory_transition, f)
                print('Time spent till now: %.2f seconds'%(time.time() - start_time))
            
            print('Merging trajectory transition')
            total_trajectory_transition = total_trajectory_transition + trajectory_transition
            print('Total count: %d'%(total_trajectory_transition.sum()))
        
        print('Saving total_trajectory_transition')
        with open(total_file_path, 'wb') as f:
            pkl.dump(total_trajectory_transition, f)
        
    return total_trajectory_transition

def road_graph(road_df=None, out_path='fastdatasf/road_graph.gml', update=False):
    if not update and os.path.exists(out_path):
        print('Graph exists')
        G = nx.read_gml(out_path)
        G = nx.relabel_nodes(G, int)
    else:
        print('Generating new graph from road df')
        G = nx.DiGraph()

        node_list = list(road_df['road_id'])
        G.add_nodes_from(node_list)
        lengths = dict(zip(road_df['road_id'], road_df['length']))
        nx.set_node_attributes(G, lengths, 'length')

        road_df = road_df.copy()
        road_df = road_df[['road_id', 'start', 'end', 'length']]

        # Merge only where end of one matches start of another
        adj_df = pd.merge(
            road_df,
            road_df,
            left_on='end',
            right_on='start',
            suffixes=('_x', '_y')
        )

        # Remove self-loops except for explicit self-edges
        adj_df = adj_df[adj_df['road_id_x'] != adj_df['road_id_y']]

        adj_df['distance'] = (adj_df['length_x'] + adj_df['length_y']) / 2
        adj_df['edge'] = adj_df.apply(lambda row: (row['road_id_x'], row['road_id_y'], {'weight': row['distance']}), axis=1)
        edge_list = list(adj_df['edge'])

        # Add self-loops
        self_loops = [(row['road_id'], row['road_id'], {'weight': 0}) for _, row in road_df.iterrows()]
        edge_list.extend(self_loops)

        G.add_edges_from(edge_list)

        nx.write_gml(G, out_path)
        
    return G

def extract_road_adj(G=None, road_list=None):
    
    file_path = 'fastdatasf/road_adj.pkl'
    if os.path.exists(file_path):
        print('Road adj exists')
        with open(file_path, 'rb') as f:
            road_adj = pkl.load(f)
    else:
        print('Extracting road adj from graph')
        
        G = road_graph(road_df=None, out_path='fastdatasf/road_graph.gml', update=False)
        road_adj = np.zeros((len(G.nodes), len(G.nodes)), dtype=np.float32)
        
        road_list = get_road_list()
        def road_index(road_id):
            return road_list[road_list['road_id']==road_id].index[0]

        # masked exponential kernel. Set lambda = 1.
        lambda_ = 1
        # lambda: for future consideration
        # total_weight = 0
        # total_count = 0

        # for O in list(G.nodes):
        #     for D in list(G.successors(O)):
        #         total_weight += G.edges[O, D]['weight']
        #         total_count += 1

        # lambda_ = total_weight / total_count
        for O in list(G.nodes):
            for D in list(G.successors(O)):
                road_adj[road_index(O), road_index(D)] = lambda_ * np.exp(- lambda_ * G.edges[O, D]['weight'])

        with open(file_path, 'wb') as f:
            pkl.dump(road_adj, f)
    
    return road_adj

def preprocess_data(root_path, start_date, end_date, preprocess_path='data/sf_data/TrGNN/preprocess.pkl'):

    if os.path.exists(preprocess_path):
        print('Loading preprocessed data...')
        with open(preprocess_path, 'rb') as f:
            normalized_flows, transitions_ToD, W, W_norm = pkl.load(f)

    else:
        dates = date_range(start_date, end_date)
        preprocess_path = os.path.join(root_path, 'cache/preprocess_TrGNNsf_20.pkl')

        # weekdays scaler都要有 
        flow_df = pd.concat([pd.read_csv('fastdatasf/flow_%s_%s.csv'%(date, date), index_col=0) for date in dates])
        flow_df.columns = pd.Index(int(road_id) for road_id in flow_df.columns)

        N_len = int(len(flow_df) * 23 / 24)  # 只保留23小时的数据
        train_n = int(N_len * 0.7)
        val_n   = int(N_len * 0.1)
        test_n  = N_len - train_n - val_n
        segments = [('train', train_n),
                    ('val',   val_n),
                    ('test',  test_n)]
        idx_map = {}
        start = 0
        for name, length in segments:
            idx_map[name] = list(range(start, start + length))
            start += length
        scaler = StandardScaler().fit(
            flow_df.iloc[idx_map['train'] + idx_map['val']].values
            ) # normalize flow

        road_adj = extract_road_adj() 
        start_date, end_date = '20080517', '20080604'
        trajectory_transition = extract_trajectory_transition(start_date, end_date)
        road_adj_mask = np.zeros(road_adj.shape)
        road_adj_mask[road_adj > 0] = 1
        np.fill_diagonal(road_adj_mask, 0)
        for i in range(len(trajectory_transition)):
            trajectory_transition[i] = trajectory_transition[i] + road_adj_mask

        normalized_flows = torch.from_numpy(scaler.transform(flow_df.values)).float()
        transitions_ToD = [to_sparse_tensor(normalize_adj(trajectory_transition[i])) 
                            for i in range(len(trajectory_transition))]
        W = torch.from_numpy(road_adj)
        W_norm = torch.from_numpy(normalize_adj(road_adj, mode='aggregation'))
        with open(preprocess_path, 'wb') as f:
            pkl.dump([normalized_flows, transitions_ToD, W, W_norm], f)  
                      
    return normalized_flows, transitions_ToD, W, W_norm