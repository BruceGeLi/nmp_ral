
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

        # res = dict()
        res = Dict()

        # test different num of basis
        for num_basis in range(10, 26):

            # res[f"{num_basis}_bases"] = dict()

            # init mp
            mp_cfg = self.cfg["mp"]
            mp_cfg["mp_args"]["num_basis"] = num_basis
            mp = MPFactory.init_mp(device=self.device, **mp_cfg)

            # loop over different number
            for key, data_dic in self.cfg["data"]["datasets"].items():

                dataset = util.load_npz_dataset(data_dic)
                train_loader, _, _, _ = get_data_loaders_and_normalizer(
                    dataset, **self.cfg["data"]["data_param"],
                    seed=self.cfg["seed"])

                temp = []
                for batch in train_loader:

                    gt = batch["trajs"]["value"]
                    mp.learn_mp_params_from_trajs(batch["trajs"]["time"], gt)
                    pos = mp.get_traj_pos()
                    loss = mse_loss(pos, gt.to(device=self.device))
                    temp.append(loss)

                res[f"{num_basis}_bases"][key] = torch.mean(torch.stack(temp)).item()

            # res[f"{num_basis}_bases"]["total"] = torch.mean(
            #     torch.cat([value for _, value in res[f"{num_basis}_bases"]]))
            total_mean = sum(
                [value for _, value in res[f"{num_basis}_bases"].items()])/len(res[f"{num_basis}_bases"])
            res[f"{num_basis}_bases"]["total"] = total_mean
            wandb.log({"num_basis": num_basis})
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

