#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic snapshot construction script
  1) Map time_stamp (with timezone) to continuous days from the start
  2) Group into fixed-day windows
  3) Build and save a global dense adjacency matrix for each window

New: optional start_time / end_time arguments to restrict processing to interactions within the interval
"""

import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp

def build_dynamic_snapshots(
    df: pd.DataFrame,
    time_col: str = 'time_stamp',
    window_size_days: int = 30,
    save_dir: str = 'snapshots',
    start_time: str = None,
    end_time: str = None
):
    os.makedirs(save_dir, exist_ok=True)

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_localize(None)
    if start_time:
        df = df[df[time_col] >= pd.to_datetime(start_time)]
    if end_time:
        df = df[df[time_col] <= pd.to_datetime(end_time)]
    if df.empty:
        raise ValueError("check start_time/end_time")

    t0 = df[time_col].min()
    df['days_from_start'] = (df[time_col] - t0).dt.days


    df['user_key'] = 'u' + df['user_id'].astype(str)
    df['item_key'] = 'i' + df['item_id'].astype(str)
    user_keys = sorted(df['user_key'].unique())
    item_keys = sorted(df['item_key'].unique())
    user2gid = {k: idx for idx, k in enumerate(user_keys)}
    item2gid = {k: idx + len(user2gid) for idx, k in enumerate(item_keys)}
    max_nodes = len(user2gid) + len(item2gid)


    mapping = {**user2gid, **item2gid}
    mapping_file = os.path.join(save_dir, 'global_id_mapping.pkl')
    with open(os.path.join(mapping_file), 'wb') as f:
        pickle.dump(mapping, f)


    df['window_id'] = df['days_from_start'] // window_size_days


    for wid, df_win in df.groupby('window_id'):
        if df_win.empty:
            continue


        start_date = t0 + pd.Timedelta(days=int(wid * window_size_days))
        end_date   = start_date + pd.Timedelta(days=window_size_days)
        if end_time and end_date > pd.to_datetime(end_time):
            end_date = pd.to_datetime(end_time)


        rows = df_win['user_key'].map(user2gid).to_numpy()
        cols = df_win['item_key'].map(item2gid).to_numpy()


        data = np.ones(len(rows), dtype=np.uint8)
        adj_sp = sp.coo_matrix((data, (rows, cols)), shape=(max_nodes, max_nodes))
        adj_sp = adj_sp + adj_sp.T


        adj = adj_sp.toarray()
        edge_count = int(adj_sp.nnz // 2)
        node_count = int(np.count_nonzero(adj.sum(axis=1)))

        print(f"[window {wid:03d}] {start_date.date()} → {end_date.date()}  "
              f"edges={edge_count}, nodes={node_count}, shape={adj.shape}")


        np.save(os.path.join(save_dir, f'snapshot_{wid:03d}.npy'), adj)

    print(f"save snapshot:{os.path.abspath(save_dir)}")
    
    return mapping_file

def generate_user_feature_csv(
    mapping_file: str,
    df: pd.DataFrame,
    save_dir: str = 'snapshots',
    output_filename: str = 'user_feature.csv'
):
    os.makedirs(save_dir, exist_ok=True)
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
    user_mapping = {k: v for k, v in mapping.items() if k.startswith('u')}
    df_users = df[['user_id', 'user_attr']].drop_duplicates().copy()
    df_users['user_key'] = 'u' + df_users['user_id'].astype(str)
    label_map = dict(zip(
        df_users['user_key'],
        df_users['user_attr'].map({'Small': 0, 'Large': 1}).fillna(-1).astype(int)
    ))
    records = [(gid, label_map.get(user_key, -1)) for user_key, gid in user_mapping.items()]
    df_feat = pd.DataFrame(records, columns=['gid', 'body_label'])
    df_feat.sort_values('gid', inplace=True)
    df_feat.reset_index(drop=True, inplace=True)
    out_path = os.path.join(save_dir, output_filename)
    df_feat.to_csv(out_path, index=False)
    print(f"User features saved to '{out_path}', shape={df_feat.shape}")
    
    
if __name__ == '__main__':
    CSV_PATH         = '/Users/mikegu/Library/CloudStorage/Dropbox/25-AAAI-code-new/Cloth_large_new/df_modcloth_new.csv'    
    SAVE_DIR         = '/Users/mikegu/Library/CloudStorage/Dropbox/25-AAAI-code-new/Cloth_large_new/snapshots'          
    WINDOW_SIZE_DAYS = 30*3                  
    START_TIME       = '2016-09-01'           
    END_TIME         = '2018-08-22'           


    df = pd.read_csv(
        CSV_PATH,
        parse_dates=['time_stamp'],
        date_parser=lambda col: pd.to_datetime(col, utc=True).tz_localize(None)
    )


    mapping_file = build_dynamic_snapshots(
        df,
        time_col='time_stamp',
        window_size_days=WINDOW_SIZE_DAYS,
        save_dir=SAVE_DIR,
        start_time=START_TIME,
        end_time=END_TIME        
    )
    generate_user_feature_csv(
    mapping_file=mapping_file,
    df=df,
    save_dir=SAVE_DIR,
    output_filename='user_feature.csv'
    )
    