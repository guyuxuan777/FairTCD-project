import pandas as pd
import numpy as np
from pandas import Timestamp
import pickle
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ML1M = REPO_ROOT / "Movielens_1M"
SNAPSHOT_OUT = ML1M / "snapshots_new"
#data preprocessing
def load_data(inter_path, user_attr_path):

    df_inter = pd.read_csv(
        inter_path, sep='::', engine='python',
        names=['user_id','item_id','rating','timestamp']
    )
    df_inter.columns = df_inter.columns.str.strip()
    df_user = pd.read_csv(
        user_attr_path, sep='::', engine='python',
        names=['user_id','gender','age','job','post']
    )
    df_user.columns = df_user.columns.str.strip()
    return df_inter, df_user

def process_time_range(df_inter, time_col, start_time=None, end_time=None):

    min_time = df_inter[time_col].min()
    max_time = df_inter[time_col].max()
    if start_time:
        if isinstance(start_time, str):
            start_time = Timestamp(start_time).value // 10**9
        start_ts = max(start_time, min_time)
    else:
        start_ts = min_time
    if end_time:
        if isinstance(end_time, str):
            end_time = Timestamp(end_time).value // 10**9
        end_ts = min(end_time, max_time)
    else:
        end_ts = max_time
    if start_ts >= end_ts:
        raise ValueError("start_time must smaller than end_time")
    return start_ts, end_ts


def generate_time_windows(start_ts, end_ts, window_size, time_unit='s'):

    if time_unit == 'd':
        ws = window_size * 86400
    else:
        ws = window_size
    num_snapshots = (end_ts - start_ts) // ws
    windows = []
    for i in range(int(num_snapshots)):
        ws_i = start_ts + i * ws

        we_i = ws_i + ws if i < num_snapshots - 1 else end_ts
        windows.append((ws_i, we_i))
    return windows


def save_np(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

def save_pkl(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def build_dynamic_graph_dataset(
    inter_path,
    user_attr_path,
    save_dir='snapshots',
    start_time=None,
    end_time=None,
    window_size=3*30*86400,
    time_unit='s'
):
    os.makedirs(save_dir, exist_ok=True)


    df_inter, _ = load_data(inter_path, user_attr_path)


    start_ts, end_ts = process_time_range(df_inter, 'timestamp', start_time, end_time)

    windows = generate_time_windows(start_ts, end_ts, window_size, time_unit)

    # id mapping
    df_seg = df_inter[(df_inter['timestamp'] >= start_ts) & (df_inter['timestamp'] <= end_ts)]
    users = sorted(df_seg['user_id'].unique())
    items = sorted(df_seg['item_id'].unique())
    global_nodes = ['u' + str(u) for u in users] + ['i' + str(i) for i in items]
    global_map = {node: idx for idx, node in enumerate(global_nodes)}
    N = len(global_nodes)

    save_pkl(global_map, os.path.join(save_dir, 'global_map.pkl'))


    snapshots = []
    for idx, (ws, we) in enumerate(windows):
        df_win = df_inter[(df_inter['timestamp'] >= ws) & (df_inter['timestamp'] <= we)]
        A = np.zeros((N, N), dtype=np.float32)
        for _, row in df_win.iterrows():
            u_node = 'u' + str(row['user_id'])
            i_node = 'i' + str(row['item_id'])
            gi = global_map[i_node]
            gu = global_map[u_node]
            A[gu, gi] = 1.0
            A[gi, gu] = 1.0

        adj_path = os.path.join(save_dir, f'snapshot_{idx}.npy')
        save_np(A, adj_path)
        snapshots.append({
            'adj': adj_path,
            'window_start': pd.to_datetime(ws, unit='s'),
            'window_end': pd.to_datetime(we, unit='s')
        })

    return {
        'snapshots': snapshots,
        'global_map': global_map,
        'max_nodes': N,
        'save_dir': os.path.abspath(save_dir)
    }


if __name__ == '__main__':
    ratings = ML1M / "ml-1m" / "ratings.dat"
    users   = ML1M / "ml-1m" / "users.dat"
    ds = build_dynamic_graph_dataset(
        str(ratings), str(users),
        save_dir=str(SNAPSHOT_OUT),
        start_time='2001-09',
        end_time='2003-01',
        window_size=3*30*86400,
        time_unit='s'
    )
    print('time window:', len(ds['snapshots']))
    for s in ds['snapshots']:
        A = np.load(s['adj'])
        print(s['window_start'], 'interact:', np.count_nonzero(A)//2)
