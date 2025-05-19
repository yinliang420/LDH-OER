import copy
import os
import time
import torch.distributed as dist
import random
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.nn.parallel import DataParallel
from ocpmodels.datasets import LmdbDataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from ocpmodels.models.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_f, EquiformerV2_OC20


class EarlyStopping:
    def __init__(self, patience=7, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# 参数微调是否 - 设置
def set_para_requires_grad(model, feature_frozen):
    if feature_frozen:
        for para in model.parameters():
            para.requires_grad = False


def train(model, data_loader, optimizer, num_epochs, criterion, normalizers, device, save_best: bool,
          judge_force: bool, fold, nums, patience=7, delta=0, verbose=False):
    train_losses = []
    valid_losses = []
    best_loss = 1000000.0

    model = model.to(device)


    if not os.path.exists(f'path_to_save'):
        os.makedirs(f'path_to_save')

    for epoch in range(num_epochs):
        for phase in ['train', 'valid']:
        # for phase in ['train']:
            if phase == 'train':
                with open(f'path_to_save\\record{epoch}.txt', 'a') as f:
                    f.write(f'train_epoch: {epoch} start \n')
                model.train()  # 训练
            else:
                with open(f'path_to_save\\record{epoch}.txt', 'a') as f:
                    f.write(f'vaild:  start   \n')
                model.eval()  # 验证
            running_loss = 0.0
            num_samples = 0

            for batch_data in data_loader[phase]:
                batch_data = batch_data.to(device)
                optimizer.zero_grad()  # 梯度清零

                if judge_force:
                    # 将数据批次传递给模型进行前向传播
                    energy, forces = model(batch_data)
                    energy = (
                            energy.detach() * normalizers["target"]["std"]
                            + normalizers["target"]["mean"]
                    )
                else:
                    energy = model(batch_data)
                    # print(energy)

                x = (energy.detach() * normalizers["target"]["std"] + normalizers["target"]["mean"])
                for i in range(len(energy)):
                    if phase == 'train':
                        with open(f'path_to_save\\record{epoch}.txt', 'a') as f:
                            f.write(f'{batch_data[i].sid.item()}  Train Energy: {x[i].item()}, Y_relaxed: {batch_data[i].y_relaxed.item()}\n')
                    if phase == 'valid':
                        with open(f'path_to_save\\record{epoch}.txt', 'a') as f:
                            f.write(f'{batch_data[i].sid.item()}  Valid Energy: {x[i].item()}, Y_relaxed: {batch_data[i].y_relaxed.item()}\n')

                # 计算损失函数
                batch_data.y_relaxed = (
                        (batch_data.y_relaxed.detach() - normalizers["target"]["mean"]) / normalizers["target"]["std"]
                ).to(device)
                loss = criterion(energy, batch_data.y_relaxed)

                if phase == 'train':
                    # 反向传播和参数更新
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * len(batch_data)
                num_samples += len(batch_data)
            epoch_loss = running_loss / num_samples
            epoch_loss = epoch_loss * normalizers["target"]["std"] + normalizers["target"]["mean"]
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                if save_best is True:
                    # best_model_wts = copy.deepcopy(model.state_dict())
                    state = {
                        'state_dict': model.state_dict(),  # 字典里key就是各层的名字，值就是训练好的权重
                        'best_loss': best_loss,
                        'optimizer': optimizer.state_dict(),  # 优化器的状态信息
                    }

                    filename = f'path_to_save\\{nums}-{fold}-over.pt'
                    torch.save(state, filename)

            print(f"{phase}  Epoch [{epoch + 1}/{num_epochs}], Epoch Loss: {epoch_loss}")

            if phase == 'valid':
                valid_losses.append(epoch_loss)
            if phase == 'train':
                train_losses.append(epoch_loss)
    if save_best is True:
        print("Train_loss: ", train_losses)
        print("Valid_losses: ", valid_losses)
        print("Best_loss: ", best_loss, "\t", valid_losses.index(best_loss))
    else:
        print("Best_loss: ", best_loss)
    return train_losses, valid_losses, best_loss, False


def evaluate(model, data_loader, normalizers, device, criterion):
    model.eval()
    running_loss = 0.0
    num_samples = 0

    for batch_data in data_loader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            energy = model(batch_data)

        energy = energy.detach() * normalizers["target"]["std"] + normalizers["target"]["mean"]
        loss = criterion(energy, batch_data.y_relaxed)
        running_loss += loss.item() * len(batch_data)
        num_samples += len(batch_data)

    return running_loss / num_samples


if __name__ == "__main__":
    num_epochs = 100
    dir = "over.lmdb"
    batch_size = 10
    num_workers = 1
    num_folds = 5
    numlist = [1000, 2000, 3000, 4000, 0]

    dataset = LmdbDataset({"src": dir})
    all_data = [data for data in dataset]

    for nums in numlist:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        split_data = random.sample(all_data, int(nums)) if nums != 0 else all_data

        fold_results = []

        for fold, (train_index, valid_index) in enumerate(kf.split(split_data)):
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
            checkpoint_path = "eq2_121M_e4_f100_oc22_s2ef.pt"
            ckpt = torch.load(checkpoint_path)
            normalizers = ckpt.get("normalizers")
            state_dict = ckpt["state_dict"]
            state_dict = {k[2 * len("module."):]: v for k, v in state_dict.items()}
            equiformer.load_state_dict(state_dict)

            # 设置不输出力的训练, 或者设置为True让其参与训练
            equiformer.regress_forces = False
            judge_force = equiformer.regress_forces
            # 设置equiformerV2的模型预训练参数不参与反向传播更新
            set_para_requires_grad(equiformer, True)

            # 定义自己的model，修改了两个线性层
            layer_output = 160
            lr_dropout1 = 0.26489566641840434
            lr_dropout2 = 0.13752588542238575
            my_model = EquiformerV2_f(equiformer, layer_output, lr_dropout1, lr_dropout2)

            # filter(lambda p : p.requires_grad, model.parameters()), lr=1e-2
            lr_layer1 = 0.0008183145107957354
            lr_layer2 = 0.005590396725491561

            optimizer = torch.optim.AdamW([
                {'params': my_model.layer1.parameters(), 'lr': lr_layer1},
                {'params': my_model.layer2.parameters(), 'lr': lr_layer2}
            ])

            criterion = nn.L1Loss()

            print(f"Fold [{fold + 1}/{num_folds}]")

            train_data = [split_data[idx] for idx in train_index]
            valid_data = [split_data[idx] for idx in valid_index]

            train_batch = Batch.from_data_list(train_data)
            valid_batch = Batch.from_data_list(valid_data)

            train_dataloader = DataLoader(train_batch, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                          pin_memory=True)
            valid_dataloader = DataLoader(valid_batch, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                          pin_memory=True)

            dataloader = {'train': train_dataloader, 'valid': valid_dataloader}

            valid_loss = train(my_model, dataloader, optimizer, num_epochs, criterion, normalizers, device, False,
            False, fold, nums, patience=7, delta=0, verbose=True)

            print(valid_loss)
