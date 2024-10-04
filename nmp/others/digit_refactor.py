from pathlib import Path

import numpy as np
import torch
from addict import Dict
import yaml

import nmp
from mp_pytorch.mp import MPFactory
from nmp import util


def refactor_image_dataset(cfg):

    dataset_name = cfg["dataset"]
    mp_cfg = cfg["mp"]

    mp = MPFactory.init_mp(**mp_cfg)

    data_dict = nmp.util.load_npz_dataset(dataset_name)
    gt_trajs = data_dict["trajs"]["value"]
    times = data_dict["trajs"]["time"]

    init_pos = torch.as_tensor(gt_trajs[..., 0, :])
    init_vel = torch.zeros(init_pos.shape)
    init_time = torch.as_tensor(times[..., 0])

    params_dict = mp.learn_mp_params_from_trajs(times, gt_trajs,
                                                init_time=init_time,
                                                init_pos=init_pos,
                                                init_vel=init_vel)

    init_pos = params_dict["init_pos"].cpu().numpy()
    # init_vel = params_dict["init_vel"].cpu().numpy()
    params = params_dict["params"].cpu().numpy()

    data_dict["init_x_y_dmp_w_g"] = \
        {"value": np.concatenate([init_pos, params], axis=-1)}

    # nmp.util.save_npz_dataset(dataset_name + "_+", overwrite=True, **data_dict)
    path = "../../dataset/" + dataset_name+"_bsp/"
    Path(path).mkdir()
    np.savez(util.join_path(path, dataset_name+ "_bsp.npz"),#
             **data_dict)

def refactor_image_dataset_big(cfg):

    dataset_name = cfg["dataset"]
    mp_cfg = cfg["mp"]

    mp = MPFactory.init_mp(**mp_cfg)

    data_dict = nmp.util.load_npz_dataset(dataset_name)
    gt_trajs = data_dict["trajs"]["value"]
    times = data_dict["trajs"]["time"]
    batch = 2000

    for b in range(10):
        init_pos = torch.as_tensor(gt_trajs[b*batch:(b+1)*batch, 0, :])
        init_vel = torch.zeros(init_pos.shape)
        init_time = torch.as_tensor(times[b*batch:(b+1)*batch, 0])
        params_dict = mp.learn_mp_params_from_trajs(times[b*batch:(b+1)*batch, ...],
                                                    gt_trajs[b*batch:(b+1)*batch, ...],
                                                    init_time=init_time,
                                                    init_pos=init_pos,
                                                    init_vel=init_vel)

        init_pos = params_dict["init_pos"].cpu().numpy()
        params = params_dict["params"].cpu().numpy()
        params_temp = np.concatenate([init_pos, params], axis=-1)
        data_dict["init_x_y_dmp_w_g"]["value"][b*batch:(b+1)*batch, ...] = params_temp

    # data_dict["init_x_y_dmp_w_g"] = \
    #     {"value": np.stack(params_temp, axis=0)}

    # nmp.util.save_npz_dataset(dataset_name + "_+", overwrite=True, **data_dict)
    path = "../../dataset/" + dataset_name  # +"_+/"
    # Path(path).mkdir()
    np.savez(util.join_path(path, dataset_name + ".npz"),
             **data_dict)


def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":

    cfg = read_config("./refactor.yaml")
    # refactor_image_dataset(cfg)
    refactor_image_dataset_big(cfg)