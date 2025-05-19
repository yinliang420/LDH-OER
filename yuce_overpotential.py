import copy
import os
import time
import torch.distributed as dist
import random
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from ocpmodels.datasets import LmdbDataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from ocpmodels.models.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_f, EquiformerV2_OC20


def evaluate(model, data_loader, normalizers, device, index, z):
    model.eval()
    running_loss = 0.0
    num_samples = 0
    # with open("D:\pre_train\checkpoint\equiformerV2\\train_record\\siyuanyuce_overpotential.txt", 'a') as f:
        # f.write(f"nums:{nums}, i:{i} \n")
    for batch_data in data_loader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            energy = model(batch_data)

        energy = energy.detach() * normalizers["target"]["std"] + normalizers["target"]["mean"]
        # print(str(energy.item()))
        # batch_data.y_relaxed = (batch_data.y_relaxed.detach() * normalizers["target"]["std"] + normalizers["target"]["mean"]).to(device)
        # with open("D:\pre_train\checkpoint\equiformerV2\\train_record\doping\\yuce.txt", 'a') as f:
        with open(f"D:\pre_train\checkpoint\equiformerV2\\train_record\预测\\eryuan\\siyuanyuce_overpotential_{index}_{z}.txt", 'a') as f:
            for i in range(len(energy)):

                f.write(str(str(energy.item()) + "\t" + str(batch_data.sid) + '\n'))
            # print(energy.item(), batch_data.y_relaxed.item())
            # loss = criterion(energy, batch_data.y_relaxed)
            # running_loss += loss.item() * len(batch_data)
            # num_samples += len(batch_data)

    return running_loss


if __name__ == '__main__':
    nums_list = [0, ]
    for nums in nums_list:
        nums_loss_list = []
        for i in range(5):

            best_model_path = (f"{nums}-{i}-over.pt")

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model_config = {
                "use_pbc": True,
                "regress_forces": True,
                "otf_graph": True,
                "enforce_max_neighbors_strictly": False,
                "max_neighbors": 20,
                "max_radius": 12.0,
                "max_num_elements": 90,
                "num_layers": 18,
                "sphere_channels": 128,
                "attn_hidden_channels": 64,
                # [64, 96] This determines the hidden size of message passing. Do not necessarily use 96.
                "num_heads": 8,
                "attn_alpha_channels": 64,  # Not used when `use_s2_act_attn` is True.
                "attn_value_channels": 16,
                "ffn_hidden_channels": 128,
                "norm_type": "layer_norm_sh",  # ['rms_norm_sh', 'layer_norm', 'layer_norm_sh']
                "lmax_list": [6],
                "mmax_list": [2],
                "grid_resolution": 18,  # [18, 16, 14, None] For `None`, simply comment this line.
                "num_sphere_samples": 128,
                "edge_channels": 128,
                "use_atom_edge_embedding": True,
                "share_atom_edge_embedding": False,
                # If `True`, `use_atom_edge_embedding` must be `True` and the atom edge embedding will be shared across all blocks.
                "use_m_share_rad": False,
                "distance_function": "gaussian",
                "num_distance_basis": 512,  # not used
                "attn_activation": "silu",
                "use_s2_act_attn": False,
                # [False, True] Switch between attention after S2 activation or the original EquiformerV1 attention.
                "use_attn_renorm": True,  # Attention re-normalization. Used for ablation study.
                "ffn_activation": "silu",  # ['silu', 'swiglu']
                "use_gate_act": False,  # [True, False] Switch between gate activation and S2 activation
                "use_grid_mlp": True,  # [False, True] If `True`, use projecting to grids and performing MLPs for FFNs.
                "use_sep_s2_act": True,  # Separable S2 activation. Used for ablation study.
                "alpha_drop": 0.1,  # [0.0, 0.1]
                "drop_path_rate": 0.1,  # [0.0, 0.05]
                "proj_drop": 0.0,
                "weight_init": "uniform",  # ['uniform', 'normal']
                "load_energy_lin_ref": True,
                # Set to `True` for the test set or when loading a checkpoint that has `energy_lin_ref` parameters, `False` for training and val.
                "use_energy_lin_ref": True,  # Set to `True` for the test set, `False` for training and val.
            }

            # 加载EquiformeV2-OC20的模型初始化和预训练参数加载
            equiformer = EquiformerV2_OC20(num_atoms=-1, bond_feat_dim=-1, num_targets=-1, **model_config)
            equiformer.regress_forces = False
            # 定义自己的model，修改了两个线性层
            layer_output = 160
            lr_dropout1 = 0.26489566641840434
            lr_dropout2 = 0.13752588542238575


            my_model = EquiformerV2_f(equiformer, layer_output, lr_dropout1, lr_dropout2)

            ckpt = torch.load(best_model_path)
            state_dict = ckpt["state_dict"]

            my_model.load_state_dict(state_dict)
            criterion = nn.L1Loss()
            judge_force = equiformer.regress_forces
            checkpoint_path = "D:\\eq2_121M_e4_f100_oc22_s2ef.pt"
            ckpt = torch.load(checkpoint_path)
            normalizers = ckpt.get("normalizers")

            batch_size = 1
            num_workers = 1

            best_model = my_model.to(device)
            # for i in range(2, 10):
            for j in tqdm(range(100), desc="Processing datasets"):
                print(j)
                dir_2 = f'\\yuce_overpotential_{j}.lmdb'
                dataset = LmdbDataset({"src": dir_2})

                all_data = [data for data in dataset]

                test_batch = Batch.from_data_list(all_data)
                test_dataloader = DataLoader(test_batch, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                             pin_memory=True)
                test_loss_2 = evaluate(best_model, test_dataloader, normalizers, device, nums, i)

