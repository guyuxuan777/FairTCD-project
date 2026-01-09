import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, GATConv
import time
import numpy as np
from collections import defaultdict
import pickle

from model.structural_teacher.structural_adversarial import StructuralAdversarialModel
from model.attribute_teacher.attribute_adversarial import AttributeAdversarialModel
from model.structural_teacher.data_loader import load_structural_adversarial_data
from model.attribute_teacher.data_loader import load_attribute_adversarial_data
from model.evaluation.downstream_hr import calculate_hr_at_k
from model.evaluation.fairness_hr import load_and_split_active_users, calculate_hr_at_k_fair
from model.evaluation.downstream_ndcg import calculate_ndcg_at_k

from model.temporal_agg import temporal_decay_aggregate  # 虽不再使用，可保留
from sklearn.metrics import f1_score

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class StudentModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)


def create_test_edges_labels(graph):
    pos = graph.edge_index
    pos_labels = torch.ones(pos.size(1), dtype=torch.float, device=pos.device)
    neg = negative_sampling(
        edge_index=pos,
        num_nodes=graph.num_nodes,
        num_neg_samples=pos.size(1),
    )
    neg_labels = torch.zeros(neg.size(1), dtype=torch.float, device=neg.device)
    ei = torch.cat([pos, neg], dim=1)
    labels = torch.cat([pos_labels, neg_labels], dim=0)
    return ei, labels


def compute_hr_user(ei, labels, scores, k):
    users = ei[0].tolist()
    idxs_per_user = defaultdict(list)
    for i, u in enumerate(users):
        idxs_per_user[u].append(i)
    hits = 0
    for u, idxs in idxs_per_user.items():
        user_scores = scores[idxs]
        user_labels = labels[idxs]
        topk = torch.topk(user_scores, min(k, len(idxs))).indices
        if user_labels[topk].sum() > 0:
            hits += 1
    return hits / len(idxs_per_user), {
        u: float((labels[idxs][torch.topk(scores[idxs], min(k, len(idxs))).indices].sum() > 0).item())
        for u, idxs in idxs_per_user.items()
    }


def tem_agg(hlist, decays):
    out = torch.zeros_like(hlist[0])
    for h, d in zip(hlist, decays):
        out += d * h
    return out


# gamma: lp, alpha: structure, beta: attribute 
def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


    epochs, lr = 50, 1e-3
    alpha = 0.1
    beta = 1 - alpha
    gamma = 0.4
    K = 50
    decays = 0.1 


    data_root  = PROJECT_ROOT / 'Electronics'
    adj_dir    = data_root / 'snapshots'
    struct_dir = data_root / 'structural_groups'
    attr_dir   = data_root / 'attribute_groups'
    struct_fp  = PROJECT_ROOT / 'model_file_07_12_temporal_backbone' / 'structural_el.pth'
    attr_fp    = PROJECT_ROOT / 'model_file_07_12_temporal_backbone' / 'attribute_el.pth'


    struct_graphs, struct_labels = load_structural_adversarial_data(adj_dir, struct_dir, device=device)
    attr_graphs, attr_labels = load_attribute_adversarial_data(adj_dir, attr_dir, device=device)
    train_s, test_s = struct_graphs[:-1], struct_graphs[-1]
    train_a, test_a = attr_graphs[:-1], attr_graphs[-1]
    print(f"train snapshots: {len(train_s)}")

    # teacher model
    struct_teacher = torch.load(struct_fp, map_location=device, weights_only=False)
    attr_teacher = torch.load(attr_fp, map_location=device, weights_only=False)
    struct_teacher.eval(); attr_teacher.eval()
    for p in struct_teacher.parameters(): p.requires_grad = False
    for p in attr_teacher.parameters(): p.requires_grad = False

    # student model + GRU + optimizer
    in_ch = train_s[0].x.size(1)
    student = StudentModel(in_ch, 16, 64).to(device)
    # GRU 
    gru = torch.nn.GRU(input_size=64, hidden_size=64).to(device)
    opt = Adam(list(student.parameters()) + list(gru.parameters()), lr=lr)

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        student.train(); opt.zero_grad()


        with torch.no_grad():
            te_s, _ = struct_teacher(train_s)
            te_a, _ = attr_teacher(train_a)

        stu = torch.stack([student(g) for g in train_s], dim=0)  
        # 2) GRU 
        out, _ = gru(stu)  
        stu_emd = out      

        # loss
        loss_s = F.mse_loss(stu_emd, te_s)
        loss_a = F.mse_loss(stu_emd, te_a)

        # BPR 
        loss_lp = 0.0
        global_map_path = adj_dir / 'global_id_mapping.pkl'
        with open(global_map_path, 'rb') as f:
            mapping = pickle.load(f)
        num_users = sum(1 for k in mapping if k.startswith('u'))
        num_nodes = struct_graphs[0].num_nodes
        num_items = num_nodes - num_users

        for t in range(stu_emd.size(0) - 1):
            z = stu_emd[t]
            pos_global = struct_graphs[t + 1].edge_index.to(device)
            pos_bi = pos_global.clone()
            pos_bi[1] = pos_bi[1] - num_users
            neg_bi = negative_sampling(
                edge_index=pos_bi,
                num_nodes=(num_users, num_items),
                num_neg_samples=pos_bi.size(1),
            ).to(device)
            neg_global = neg_bi.clone()
            neg_global[1] = neg_bi[1] + num_users
            pos_score = (z[pos_global[0]] * z[pos_global[1]]).sum(dim=1)
            neg_score = (z[neg_global[0]] * z[neg_global[1]]).sum(dim=1)
            bpr = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
            loss_lp += bpr

        loss_lp = loss_lp / (stu_emd.size(0) - 1)
        (gamma * loss_lp + alpha * loss_s + beta * loss_a).backward()
        opt.step()

        print(f"[{ep:03d}] lp={loss_lp:.4f} s={loss_s:.4f} a={loss_a:.4f} time={time.perf_counter() - t0:.2f}s")


    student.eval()
    with torch.no_grad():
        stu = torch.stack([student(g) for g in train_s], dim=0)  
        out, _ = gru(stu)                                      
        emb_student = out[-1]                                  

    print(f"final embedding dimension: {emb_student.shape}")
    embeddings_numpy = emb_student.cpu().numpy()


    with open(os.path.join(adj_dir, 'global_id_mapping.pkl'), 'rb') as f:
        id_mapping = pickle.load(f)
    user_idxs = sorted(idx for key, idx in id_mapping.items() if key.startswith('u'))
    item_idxs = sorted(idx for key, idx in id_mapping.items() if key.startswith('i'))

    user_embeddings = emb_student[user_idxs, :]
    item_embeddings = emb_student[item_idxs, :]

    # HR@K & NDCG@K
    k_list = [20]
    all_hr = []
    for k in k_list:
        hr_at_k = calculate_hr_at_k(user_embeddings, item_embeddings, test_s, k=k, return_details=False)
        all_hr.append(hr_at_k)
    print(f"Downstream task HR@K: {all_hr}")

    all_ndcg = []
    for k in k_list:
        ndcg_at_k = calculate_ndcg_at_k(user_embeddings, item_embeddings, test_s, k=k)
        all_ndcg.append(ndcg_at_k)
    print(f"Downstream NDCG@K: {all_ndcg}")


    # attribute fairness
    attribute_group_path = attr_dir / 'snapshot_004_attribute_groups.npy'
    _, label0_users, label1_users = load_and_split_active_users(attribute_group_path, test_s, user_embeddings)
    all_att_fair = []
    for k in k_list:
        hr0 = calculate_hr_at_k_fair(user_embeddings, item_embeddings, test_s, label0_users, k)
        hr1 = calculate_hr_at_k_fair(user_embeddings, item_embeddings, test_s, label1_users, k)
        all_att_fair.append(abs(hr0 - hr1))
    print(f"Attribute fairness HR gaps: {all_att_fair}")

    # structural fairness
    struct_group_path = struct_dir / 'snapshot_004_structural_groups.npy'
    _, label0_users_s, label1_users_s = load_and_split_active_users(struct_group_path, test_s, user_embeddings)
    all_str_fair = []
    for k in k_list:
        hr0 = calculate_hr_at_k_fair(user_embeddings, item_embeddings, test_s, label0_users_s, k)
        hr1 = calculate_hr_at_k_fair(user_embeddings, item_embeddings, test_s, label1_users_s, k)
        all_str_fair.append(abs(hr0 - hr1))
    print(f"Structure fairness HR gaps: {all_str_fair}")


if __name__ == '__main__':
    main()
