import torch
import torch.nn as nn
import numpy as np
from torch.nn.parallel import DataParallel
from ocpmodels.datasets import LmdbDataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from ocpmodels.models.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_f, EquiformerV2_OC20

def set_para_requires_grad(model, feature_frozen):
    if feature_frozen:
        for para in model.parameters():
            para.requires_grad = False


def train(model, data_loader, num_epochs, criterion, normalizers, device, save_best: bool,
          judge_force: bool, i):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.eval()
        for batch_data in data_loader:
            batch_data = batch_data.to(device)
            # optimizer.zero_grad()
            outputs = []

            def hook(module, input, output):
                outputs.append(output)

            hook_handle = model.layer2.register_forward_hook(hook)

            energy = model(batch_data)

            # CAM 值
            import numpy as np
            output1 = outputs[0].detach().cpu().numpy()
            sum_act = np.sum(output1, axis=1) / len(output1[1])

            layer_1_weight = model.layer2.weight.data.detach().cpu().numpy()
            wij = np.sum(layer_1_weight, axis=1) / len(layer_1_weight[1])

            aij = sum_act * wij
            aij = np.sum(aij, axis=1) / len(aij[1])

            with open(f"/path/your_dic", 'a') as f:
                f.write(str(batch_data.sid) + ' ')
                f.write(' '.join(map(str, np.sum(sum_act, axis=1) / len(sum_act[1]))) + '\n')


if __name__ == "__main__":
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

    equiformer = EquiformerV2_OC20(num_atoms=-1, bond_feat_dim=-1, num_targets=-1, **model_config)
    equiformer.regress_forces = False
    layer_output = 160
    lr_dropout1 = 0.26489566641840434
    lr_dropout2 = 0.13752588542238575
    my_model = EquiformerV2_f(equiformer, layer_output, lr_dropout1, lr_dropout2)
    for i in range(5):

        checkpoint_path = (f"path.pt")

        ckpt = torch.load(checkpoint_path)
        state_dict = ckpt["state_dict"]
        my_model.load_state_dict(state_dict)

        set_para_requires_grad(my_model, True)

        checkpoint_path = "eq2_121M_e4_f100_oc22_s2ef.pt"
        ckpt = torch.load(checkpoint_path)
        normalizers = ckpt.get("normalizers")

        input_data = LmdbDataset({"src": 'your.lmdb'})
        batch = []
        for ele in input_data:
            batch.append(ele)
        batch_s = Batch.from_data_list(batch)
        test_dataloader = DataLoader(
            batch_s,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )

        criterion = nn.L1Loss()

        train(my_model, test_dataloader, 1, criterion, normalizers, device, True, False, i)

