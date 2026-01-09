#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ELEC = REPO_ROOT / "Electronics"

def build_dynamic_snapshots(
    df: pd.DataFrame,
    time_col: str = 'time_stamp',
    window_size_days: int = 30,
    save_dir: str = 'snapshots',
    start_time: str = None,
    end_time: str = None
) -> str:
    os.makedirs(save_dir, exist_ok=True)

    # 0. time interval
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    if start_time:
        df = df[df[time_col] >= pd.to_datetime(start_time)]
    if end_time:
        df = df[df[time_col] <= pd.to_datetime(end_time)]
    if df.empty:
        raise ValueError("check")

    # 1. time mapping
    t0 = df[time_col].min()
    df['days_from_start'] = (df[time_col] - t0).dt.days

    # 2. id mapping
    df['user_key'] = 'u' + df['user_id'].astype(str)
    df['item_key'] = 'i' + df['item_id'].astype(str)
    user_keys = sorted(df['user_key'].unique())
    item_keys = sorted(df['item_key'].unique())
    user2gid = {k: idx for idx, k in enumerate(user_keys)}
    item2gid = {k: idx + len(user2gid) for idx, k in enumerate(item_keys)}
    max_nodes = len(user2gid) + len(item2gid)

    mapping = {**user2gid, **item2gid}
    mapping_file = os.path.join(save_dir, 'global_id_mapping.pkl')
    with open(mapping_file, 'wb') as f:
        pickle.dump(mapping, f)

    # 3. divide into windows
    df['window_id'] = df['days_from_start'] // window_size_days

    # 4. save adjacency matrices
    for wid, df_win in df.groupby('window_id'):
        if df_win.empty:
            continue

        start_date = t0 + pd.Timedelta(days=int(wid * window_size_days))
        end_date = start_date + pd.Timedelta(days=window_size_days)
        if end_time and end_date > pd.to_datetime(end_time):
            end_date = pd.to_datetime(end_time)

        user_nodes = df_win['user_key'].unique()
        item_nodes = df_win['item_key'].unique()
        user_count = len(user_nodes)
        item_count = len(item_nodes)


        rows = np.array([user2gid[u] for u in user_nodes for _ in (0,)], dtype=np.int64)

        rows = df_win['user_key'].map(user2gid).to_numpy()
        cols = df_win['item_key'].map(item2gid).to_numpy()
        data = np.ones(len(rows), dtype=np.uint8)
        adj_sp = sp.coo_matrix((data, (rows, cols)), shape=(max_nodes, max_nodes))
        adj_sp = adj_sp + adj_sp.T

        edge_count = int(adj_sp.nnz // 2)
        total_node_count = int(np.count_nonzero(adj_sp.sum(axis=1)))
        print(f"[window {wid:03d}] {start_date.date()} → {end_date.date()}  "
              f"edges={edge_count}, total_nodes={total_node_count}, "
              f"users={user_count}, items={item_count}")


        adj = adj_sp.toarray()
        fname = os.path.join(save_dir, f'snapshot_{wid:03d}.npy')
        np.save(fname, adj)
    print(adj.shape)
    print(f"save{os.path.abspath(save_dir)}")
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
        df_users['user_attr'].map({'Male': 0, 'Female': 1}).fillna(-1).astype(int)
    ))
    records = [(gid, label_map.get(user_key, -1)) for user_key, gid in user_mapping.items()]
    df_feat = pd.DataFrame(records, columns=['gid', 'gender_label'])
    df_feat.sort_values('gid', inplace=True)
    df_feat.reset_index(drop=True, inplace=True)
    out_path = os.path.join(save_dir, output_filename)
    df_feat.to_csv(out_path, index=False)
    print(f"User features saved to '{out_path}', shape={df_feat.shape}")


if __name__ == '__main__':
    CSV_PATH = ELEC / "df_electronics_new.csv"
    SAVE_DIR = ELEC / "snapshots"
    WINDOW_SIZE_DAYS = 60
    START_TIME = '2012-01-01'
    END_TIME = '2013-01-01'

    df = pd.read_csv(CSV_PATH, dtype={'user_id': str})
    mapping_file = build_dynamic_snapshots(
        df, time_col='time_stamp', window_size_days=WINDOW_SIZE_DAYS,
        save_dir=SAVE_DIR, start_time=START_TIME, end_time=END_TIME
    )
    generate_user_feature_csv(
        mapping_file=mapping_file,
        df=df,
        save_dir=SAVE_DIR,
        output_filename='user_feature.csv'
    )