import os
import pickle
import torch
import torch.nn.functional as F
from torch.optim import Adam
import time
import matplotlib.pyplot as plt
from torch_geometric.utils import negative_sampling
import pandas as pd

from model.structural_teacher.structural_adversarial import StructuralAdversarialModel
from model.attribute_teacher.attribute_adversarial import AttributeAdversarialModel
from model.structural_teacher.data_loader import load_structural_adversarial_data
from model.attribute_teacher.data_loader import load_attribute_adversarial_data
from model.temporal_agg import temporal_decay_aggregate
from model.dynamic_contrastive_distillation import load_contrastive_pairs, dynamic_contrastive_loss
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main():

    device = torch.device('mps' if torch.backends.mps.is_built() and torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")


    epochs      = 50
    lr_b_s      = 1e-3   
    lr_d_s      = 1e-3   
    lr_b_a      = 1e-3
    lr_d_a      = 1e-3
    beta_cd     = 0.5    
    gamma_lp    = 0.2    
    temperature = 0.1

    # data path
    data_root       = PROJECT_ROOT / 'Electronics'
    adj_dir         = data_root / 'snapshots'
    struct_lbl_dir  = data_root / 'structural_groups'
    attr_lbl_dir    = data_root / 'attribute_groups'
    pairs_path      = data_root / 'contrastive_pairs_new' / 'contrastive_pairs_global.pkl'


    struct_graphs, struct_labels = load_structural_adversarial_data(adj_dir, struct_lbl_dir, device=device)
    attr_graphs,   attr_labels   = load_attribute_adversarial_data(adj_dir, attr_lbl_dir,   device=device)
    pos_pairs, neg_pairs         = load_contrastive_pairs(pairs_path)

    # train/test
    train_graphs_struct = struct_graphs[:-1]
    test_graph_struct   = struct_graphs[-1]
    struct_labels_train = struct_labels[:-1]
    struct_labels_test  = struct_labels[-1]

    train_graphs_attr = attr_graphs[:-1]
    test_graph_attr   = attr_graphs[-1]
    attr_labels_train = attr_labels[:-1]
    attr_labels_test  = attr_labels[-1]

    # positive/negative edges
    train_pos_edges = torch.cat([g.edge_index for g in train_graphs_attr], dim=1)
    num_nodes = train_graphs_attr[0].num_nodes
    train_neg_edges = negative_sampling(
        edge_index=train_pos_edges,
        num_nodes=num_nodes,
        num_neg_samples=train_pos_edges.size(1)
    )

    # optimizers
    in_ch = train_graphs_struct[0].x.size(1)
    struct_model = StructuralAdversarialModel(in_ch,32,64,3,0.3,True,True,64,2).to(device)
    attr_model   = AttributeAdversarialModel(in_ch,32,64,3,0.3,True,True,64,2).to(device)

    # GRU
    gru = torch.nn.GRU(input_size=64, hidden_size=64).to(device)

    opt_d_s = Adam(struct_model.discriminator.parameters(), lr=lr_d_s)
    opt_b_s = Adam(struct_model.backbone.parameters(),    lr=lr_b_s)
    opt_d_a = Adam(attr_model.discriminator.parameters(),   lr=lr_d_a)

    opt_b_a = Adam(list(attr_model.backbone.parameters()) + list(gru.parameters()), lr=lr_b_a)


    loss_ds_list, loss_da_list = [], []
    loss_adv_s_list, loss_adv_a_list = [], []
    loss_cd_list, loss_lp_list = [], []

    for epoch in range(1, epochs+1):
        t0 = time.perf_counter()
        struct_model.train(); attr_model.train()


        # structural discriminator
        for p in struct_model.backbone.parameters(): p.requires_grad=False
        for p in struct_model.discriminator.parameters(): p.requires_grad=True
        _, logits_s_d = struct_model(train_graphs_struct)
        fs = logits_s_d.view(-1, 2)
        mask = (struct_labels_train.view(-1) >= 0)
        valid_idx = mask.nonzero(as_tuple=False).squeeze()
        fs_valid = fs[valid_idx]
        labels_valid = struct_labels_train.view(-1)[valid_idx]
        loss_ds = F.cross_entropy(fs_valid, labels_valid)
        opt_d_s.zero_grad(); loss_ds.backward(); opt_d_s.step()
        
        # attribute discriminator
        for p in attr_model.backbone.parameters(): p.requires_grad=False
        for p in attr_model.discriminator.parameters(): p.requires_grad=True
        _, logits_a_d = attr_model(train_graphs_attr)
        fa = logits_a_d.view(-1,2); la = attr_labels_train.view(-1)
        mask_u = la >= 0
        loss_da = F.cross_entropy(fa[mask_u], la[mask_u])
        opt_d_a.zero_grad(); loss_da.backward(); opt_d_a.step()

        # —— Backbone + tcd + lp ——
        for p in struct_model.backbone.parameters(): p.requires_grad=True
        for p in struct_model.discriminator.parameters(): p.requires_grad=False
        for p in attr_model.backbone.parameters(): p.requires_grad=True
        for p in attr_model.discriminator.parameters(): p.requires_grad=False

        emb_s, logits_s = struct_model(train_graphs_struct)
        emb_a, logits_a = attr_model(train_graphs_attr)

        # structural adversarial loss, attribute adversarial loss
        ls_i = (1 - struct_labels_train).view(-1); mask_si = ls_i >= 0
        loss_adv_s = F.cross_entropy(logits_s.view(-1,2)[mask_si], ls_i[mask_si])

        la_i = (1 - attr_labels_train).view(-1); mask_ai = (la_i >= 0)
        loss_adv_a = F.cross_entropy(logits_a.view(-1,2)[mask_ai], la_i[mask_ai])
        # tcd
        loss_cd = dynamic_contrastive_loss(emb_s, emb_a, pos_pairs, neg_pairs, temperature)
        
        # GRU 
        agg_emb_a_seq, _ = gru(emb_a)  # [T, N, 64]

        # link prediction loss
        global_map_path = adj_dir / 'global_id_mapping.pkl'
        with open(global_map_path, 'rb') as f:
            mapping = pickle.load(f)
        num_users = sum(1 for k in mapping if k.startswith('u'))
        num_nodes = train_graphs_attr[0].num_nodes
        num_items = num_nodes - num_users
        
        loss_lp = 0.0
        for t in range(agg_emb_a_seq.size(0) - 1):
            z = agg_emb_a_seq[t]  

            pos_global = train_graphs_attr[t+1].edge_index.to(device)
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

        loss_lp = loss_lp / (agg_emb_a_seq.size(0) - 1)


        opt_b_s.zero_grad()
        opt_b_a.zero_grad()

        total_struct_loss = loss_adv_s + beta_cd * loss_cd
        total_attr_loss  = loss_adv_a + beta_cd * loss_cd + gamma_lp * loss_lp

        total_struct_loss.backward(retain_graph=True)
        total_attr_loss.backward()

        opt_b_s.step()
        opt_b_a.step()

        loss_ds_list.append(loss_ds.item())
        loss_da_list.append(loss_da.item())
        loss_adv_s_list.append(loss_adv_s.item())
        loss_adv_a_list.append(loss_adv_a.item())
        loss_cd_list.append(loss_cd.item())
        loss_lp_list.append(loss_lp.item())
        print(f"[{epoch:03d}] D_s={loss_ds:.4f} D_a={loss_da:.4f} adv_s={loss_adv_s:.4f} adv_a={loss_adv_a:.4f} cd={loss_cd:.4f} lp={loss_lp:.4f}")
        print(f"    Epoch time: {time.perf_counter()-t0:.2f}s")


    # out_model_dir = PROJECT_ROOT / 'model_file_07_12_temporal_backbone'
    # out_model_dir.mkdir(parents=True, exist_ok=True)
    # torch.save(struct_model, out_model_dir / 'structural_el.pth')
    # torch.save(attr_model,   out_model_dir / 'attribute_el.pth')
    


if __name__ == '__main__':
    main()
