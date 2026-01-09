import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  
def load_id_mapping_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def group_users_by_gender_per_snapshot(adj_dir, id_mapping, user_features_csv):

    # 1) read CSV
    df_feat = pd.read_csv(user_features_csv, dtype={'gid': int, 'gender_label': int})
    gid_to_gender = df_feat.set_index('gid')['gender_label'].to_dict()

    N = len(id_mapping)  
    labels_list = []
    file_list = sorted(f for f in os.listdir(adj_dir) if f.endswith('.npy'))

    for fname in file_list:
        adj_path = os.path.join(adj_dir, fname)
        adj = np.load(adj_path)
        assert adj.shape == (N, N), f"{fname} 大小 {adj.shape} 与 N={N} 不匹配"

        labels = np.full((N,), -1, dtype=int)

        for key, idx in id_mapping.items():
            if key.startswith('u') and idx in gid_to_gender:
                labels[idx] = gid_to_gender[idx]

        labels_list.append(labels)

    return labels_list, file_list

def save_labels(labels_list, file_list, out_dir):

    os.makedirs(out_dir, exist_ok=True)
    for labels, fname in zip(labels_list, file_list):
        base = os.path.splitext(fname)[0]  
        save_name = f"{base}_attribute_groups.npy"
        save_path = os.path.join(out_dir, save_name)
        np.save(save_path, labels)
        print(f"Saved {save_path}")

if __name__ == "__main__":

    data_root         = PROJECT_ROOT / "Electronics"
    adj_folder        = data_root / "snapshots_ablation_7"
    id_mapping_path   = adj_folder / "global_id_mapping.pkl"
    user_features_csv = adj_folder / "user_feature.csv"
    out_folder        = data_root / "attribute_groups_ablation_7"


    # 1) load id_mapping
    id_mapping = load_id_mapping_pickle(id_mapping_path)
    print("Loaded id_mapping, total nodes:", len(id_mapping))

    # 2) label
    labels_list, file_list = group_users_by_gender_per_snapshot(
        adj_folder,
        id_mapping,
        user_features_csv
    )

    # 3) save .npy
    save_labels(labels_list, file_list, out_folder)

    # 4) check
    lbl0 = labels_list[0]
    print("Snapshot 0 label counts → ",
          "-1:", np.sum(lbl0 == -1),
          " 0:",  np.sum(lbl0 == 0),
          " 1:",  np.sum(lbl0 == 1))