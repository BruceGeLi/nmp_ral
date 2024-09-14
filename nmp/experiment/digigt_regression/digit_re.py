
import random

import numpy as np
import torch
from addict import Dict
import yaml
import wandb

from mp_pytorch.mp import MPFactory
from nmp import util
from nmp import get_data_loaders_and_normalizer
from nmp import mse_loss


class OneDigitRegression:
    def __init__(self, cfg):

        self.cfg = cfg
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        self.device = "cuda"
    @torch.no_grad()
    def run(self):

        wandb.init(**self.cfg["wandb"])
        wandb.config.update(self.cfg)

        # res = dict()
        res = Dict()

        # test different num of basis
        for num_basis in range(10, 26):

            # res[f"{num_basis}_bases"] = dict()

            # init mp
            mp_cfg = self.cfg["mp"]
            mp_cfg["mp_args"]["num_basis"] = num_basis

            # loop over different number
            for key, data_dic in self.cfg["data"]["datasets"].items():

                dataset = util.load_npz_dataset(data_dic)
                train_loader, _, _, _ = get_data_loaders_and_normalizer(
                    dataset, **self.cfg["data"]["data_param"],
                    seed=self.cfg["seed"])

                temp = []
                # for bw in torch.linspace(0.5, 4, 36, device=self.device):
                for dp in range(4, 8):
                    for batch in train_loader:
                        gt = batch["trajs"]["value"]
                        # mp_cfg["mp_args"]["basis_bandwidth_factor"] = bw
                        mp_cfg["mp_args"]["degree_p"] = dp
                        mp = MPFactory.init_mp(device=self.device,
                                           dtype=torch.float64, **mp_cfg)
                        mp.learn_mp_params_from_trajs(torch.as_tensor(batch["trajs"]["time"], dtype=torch.float64, device=self.device),
                                                      torch.as_tensor(gt, dtype=torch.float64, device=self.device),
                                                      reg=1e-9)
                        pos = mp.get_traj_pos()
                        loss = mse_loss(pos, gt.to(device=self.device))
                        temp.append(loss)
                temp = min(temp)
                res[f"{num_basis}_bases"][key] = temp.item()

            # res[f"{num_basis}_bases"]["total"] = torch.mean(
            #     torch.cat([value for _, value in res[f"{num_basis}_bases"]]))
            total_mean = sum(
                [value for _, value in res[f"{num_basis}_bases"].items()])/len(res[f"{num_basis}_bases"])
            res[f"{num_basis}_bases"]["total"] = total_mean
            wandb.log({"num_basis": num_basis}, commit=False)
            wandb.log(res[f"{num_basis}_bases"].to_dict())

        wandb.finish()


def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":

    cfg = read_config("./purereg.yaml")
    exp = OneDigitRegression(cfg)
    exp.run()

