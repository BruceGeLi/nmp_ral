
import random

import numpy as np
import torch
from addict import Dict
import yaml
import wandb
import matplotlib.pyplot as plt

from mp_pytorch.mp import MPFactory
from nmp import util
from nmp import get_data_loaders_and_normalizer


# def avg_squared_acc(pos):
#     # Shape: [num_times]
#     vel = np.diff(pos[:, ::10], n=1, axis=-1)[:, 5:] * 100
#     acc = np.diff(vel, n=1, axis=-1) * 100
#     avg_sq_acc = np.power(acc, 2).mean()
#     return avg_sq_acc
#
#
# def dlj_smooth(pos):
#     vel = np.diff(pos[:, ::10], n=1, axis=-1)[:, 5:] * 100
#     acc = np.diff(vel, n=1, axis=-1) * 100
#     jerk = np.diff(acc, n=1, axis=-1) * 100
#     dlj = -np.power(2.9, 5) / np.power(np.amax(vel, axis=-1), 2) * np.trapz(
#         np.power(np.abs(jerk), 2), dx=0.01, axis=-1)
#     # dlj = dlj.mean()
#     return -np.log(np.abs(dlj)).mean(axis=0)


def calculate_acceleration_and_jerk(trajectories):
    # Calculate acceleration
    # Using np.diff with n=1 along axis=1
    acceleration = np.diff(trajectories, n=1, axis=1)

    # Append a zero column at the end to maintain the shape
    # Alternatively, you could prepend a zero column
    acceleration = np.pad(acceleration, ((0, 0), (0, 1)), mode='constant',
                          constant_values=0)

    # Calculate jerk
    # Using np.diff on the acceleration
    jerk = np.diff(acceleration, n=1, axis=1)

    # Append a zero column at the end to maintain the shape
    jerk = np.pad(jerk, ((0, 0), (0, 1)), mode='constant', constant_values=0)

    return acceleration, jerk

def mean_squred_acc_jerk(trajs):

    x = trajs[..., 0]
    y = trajs[..., 1]

    acc_x, jerk_x = calculate_acceleration_and_jerk(x)
    acc_y, jerk_y = calculate_acceleration_and_jerk(y)

    sq_acc = np.square(acc_x) + np.square(acc_y)
    sq_jerk = np.square(jerk_x) + np.square(jerk_y)

    acc_sum = np.sum(sq_acc, axis=-1)
    jerk_sum = np.sum(sq_jerk, axis=-1)

    acc_sum_mean = np.mean(acc_sum, axis=0)
    acc_sum_var = np.var(acc_sum, axis=0)
    jerk_sum_mean = np.mean(jerk_sum, axis=0)
    jerk_sum_var = np.var(jerk_sum, axis=0)

    res = Dict()
    res.acc_mean = acc_sum_mean
    res.acc_var = acc_sum_var
    res.jerk_mean = jerk_sum_mean
    res.jerk_var = jerk_sum_var

    return res

class OneDigitRegression:
    def __init__(self, cfg):

        self.cfg = cfg
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        self.device = "cuda"
    @torch.no_grad()
    def run(self):

        # wandb.init(**self.cfg["wandb"])

        # res = dict()
        res = Dict()

        # test different num of basis
        for num_basis in range(10, 26):

            # res[f"{num_basis}_bases"] = dict()

            # init mp
            prodmp_cfg = self.cfg["prodmp"]
            prodmp_cfg["mp_args"]["num_basis"] = num_basis
            prodmp = MPFactory.init_mp(device=self.device, **prodmp_cfg)

            prodmpp_cfg = self.cfg["prodmp+"]
            prodmpp_cfg["mp_args"]["num_basis"] = num_basis
            prodmpp = MPFactory.init_mp(device=self.device, **prodmpp_cfg)



            # loop over different number
            for key, data_dic in self.cfg["data"]["datasets"].items():

                dataset = util.load_npz_dataset(data_dic)
                train_loader, _, _, _ = get_data_loaders_and_normalizer(
                    dataset, **self.cfg["data"]["data_param"],
                    seed=self.cfg["seed"])

                temp = []
                for batch in train_loader:

                    gt = batch["trajs"]["value"]
                    prodmp.learn_mp_params_from_trajs(batch["trajs"]["time"], gt)
                    prod = prodmp.get_traj_pos().to("cpu").numpy()
                    prodmpp.learn_mp_params_from_trajs(batch["trajs"]["time"], gt)
                    prodp = prodmp.get_traj_pos().to("cpu").numpy()
                    if res.ground_truth.get(key) is None:
                        rs_gt = mean_squred_acc_jerk(gt)
                        res.ground_truth[f"{key}"] = rs_gt
                    rs_prod = mean_squred_acc_jerk(prod)
                    rs_prodp = mean_squred_acc_jerk(prodp)

                res[f"{num_basis}_basis"].prodmp[f"{key}"] = rs_prod
                res[f"{num_basis}_basis"].prodmpp[f"{key}"] = rs_prodp

            for k, v in res[f"{num_basis}_basis"].items():
                acc_mean = np.mean([val.acc_mean  for _, val in v.items()])
                acc_var = np.mean([val.acc_var for _, val in v.items()])
                jerk_mean = np.mean([val.jerk_mean for _, val in v.items()])
                jerk_var = np.mean([val.jerk_var for _, val in v.items()])
                res[f"{num_basis}_basis"][f"{k}"].total.acc_mean = acc_mean
                res[f"{num_basis}_basis"][f"{k}"].total.acc_var = acc_var
                res[f"{num_basis}_basis"][f"{k}"].total.jerk_mean = jerk_mean
                res[f"{num_basis}_basis"][f"{k}"].total.jerk_var = jerk_var

            v = res.ground_truth
            acc_mean = np.mean([val.acc_mean for _, val in v.items()])
            acc_var = np.mean([val.acc_var for _, val in v.items()])
            jerk_mean = np.mean([val.jerk_mean for _, val in v.items()])
            jerk_var = np.mean([val.jerk_var for _, val in v.items()])
            res.ground_truth.total.acc_mean = acc_mean
            res.ground_truth.total.acc_var = acc_var
            res.ground_truth.total.jerk_mean = jerk_mean
            res.ground_truth.total.jerk_var = jerk_var

        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
        axes = axes.flatten()
        for i, numb in enumerate(['num_0', 'num_1', 'num_2', 'num_3', 'num_4',
                                  'num_5', 'num_6', 'num_7', 'num_8', 'num_9',
                                  'total']):
            # ax = axes[i]
            # x_values = [key for key in res.keys() if key != "ground_truth"]
            # x_pos = np.arange(len(x_values))
            # means_prodmp = [nb.prodmp[f"{numb}"].acc_mean for key, nb in res.items() if key != "ground_truth"]
            # std_prodmp = np.sqrt([nb.prodmp[f"{numb}"].acc_var for key, nb in res.items() if key != "ground_truth"])
            # means_prodmpp = [nb.prodmpp[f"{numb}"].acc_mean for key, nb in res.items() if key != "ground_truth"]
            # std_prodmpp = np.sqrt(
            #     [nb.prodmpp[f"{numb}"].acc_var for key, nb in res.items() if key != "ground_truth"])
            #
            # ax.errorbar(x_pos - 0.1, means_prodmp, yerr=std_prodmp, fmt='o',
            #             label='prodmp', color='blue', capsize=5)
            # # ax.errorbar(x_pos, means_prodmp, yerr=std_prodmp, fmt='o',
            # #             label='prodmp', color='blue', capsize=5)
            #
            # ax.errorbar(x_pos + 0.1, means_prodmpp, yerr=std_prodmpp, fmt='s',
            #             label='prodmpp', color='green', capsize=5)
            # # ax.errorbar(x_pos, means_prodmpp, yerr=std_prodmpp, fmt='s',
            # #             label='prodmpp', color='green', capsize=5)
            #
            # gt_mean = res.ground_truth[f"{numb}"].acc_mean
            # gt_std = np.sqrt(res.ground_truth[f"{numb}"].acc_var)
            #
            # mean_line = [gt_mean] * len(x_pos)
            # upper_bound = [gt_mean + gt_std] * len(x_pos)
            # lower_bound = [gt_mean - gt_std] * len(x_pos)
            #
            # ax.plot(x_pos, mean_line, label='ground_truth', color='red',
            #         linestyle='--')
            # ax.fill_between(x_pos, lower_bound, upper_bound, color='red',
            #                 alpha=0.2,)
            #
            # # Adding labels and title
            #
            # ax.set_xticks(x_pos)
            # ax.set_xticklabels(x_values, rotation=45)
            # ax.set_xlabel('num_basis')
            # ax.set_ylabel('smoothness index')
            # ax.set_title(numb)
            # ax.legend()


            ax = axes[i]
            x_values = [key for key in res.keys() if key != "ground_truth"]
            x_pos = np.arange(len(x_values))
            means_prodmp = [nb.prodmp[f"{numb}"].jerk_mean for key, nb in
                            res.items() if key != "ground_truth"]
            std_prodmp = np.sqrt(
                [nb.prodmp[f"{numb}"].jerk_var for key, nb in res.items() if
                 key != "ground_truth"])
            means_prodmpp = [nb.prodmpp[f"{numb}"].jerk_mean for key, nb in
                             res.items() if key != "ground_truth"]
            std_prodmpp = np.sqrt(
                [nb.prodmpp[f"{numb}"].jerk_var for key, nb in res.items() if
                 key != "ground_truth"])

            ax.errorbar(x_pos - 0.1, means_prodmp, yerr=std_prodmp, fmt='o',
                        label='prodmp', color='blue', capsize=5)
            # ax.errorbar(x_pos, means_prodmp, yerr=std_prodmp, fmt='o',
            #             label='prodmp', color='blue', capsize=5)

            ax.errorbar(x_pos + 0.1, means_prodmpp, yerr=std_prodmpp, fmt='s',
                        label='prodmpp', color='green', capsize=5)
            # ax.errorbar(x_pos, means_prodmpp, yerr=std_prodmpp, fmt='s',
            #             label='prodmpp', color='green', capsize=5)

            # gt_mean = res.ground_truth[f"{numb}"].jerk_mean
            # gt_std = np.sqrt(res.ground_truth[f"{numb}"].jerk_var)
            #
            # mean_line = [gt_mean] * len(x_pos)
            # upper_bound = [gt_mean + gt_std] * len(x_pos)
            # lower_bound = [gt_mean - gt_std] * len(x_pos)
            #
            # ax.plot(x_pos, mean_line, label='ground_truth', color='red',
            #         linestyle='--')
            # ax.fill_between(x_pos, lower_bound, upper_bound, color='red',
            #                 alpha=0.2, )

            # Adding labels and title

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_values, rotation=45)
            ax.set_xlabel('num_basis')
            ax.set_ylabel('jerk')
            ax.set_title(numb)
            ax.legend()


        for j in range(11, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()



                # res[f"{num_basis}_bases"][key] = torch.mean(torch.stack(temp)).item()

            # res[f"{num_basis}_bases"]["total"] = torch.mean(
            #     torch.cat([value for _, value in res[f"{num_basis}_bases"]]))
            # total_mean = sum(
            #     [value for _, value in res[f"{num_basis}_bases"].items()])/len(res[f"{num_basis}_bases"])
            # res[f"{num_basis}_bases"]["total"] = total_mean
            # wandb.log({"num_basis": num_basis})
            # wandb.log(res[f"{num_basis}_bases"].to_dict())

        # wandb.finish()


def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":

    cfg = read_config("./smoothness.yaml")
    exp = OneDigitRegression(cfg)
    exp.run()

