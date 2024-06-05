import random
import numpy as np
import torch
from addict import Dict
from matplotlib import pyplot as plt

from nmp import select_ctx_pred_pts
from nmp import util
from nmp.data_process import NormProcess
from nmp.experiment.digit.digit import OneDigit
from nmp.logger import WandbLogger


class OneDigitPostProcess:
    def __init__(self, logger, model_api, epoch):

        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)

        # Set logger
        self.logger = logger

        # Download model using logger
        model_dir = logger.load_model(model_api=model_api)

        # Load config and create network
        config_path = util.join_path(model_dir, "config.yaml")
        cfg = util.parse_config(config_path)
        cfg = Dict(cfg)
        cfg.assign_config.num_select = None
        self.exp = OneDigit(cfg)

        # Load network parameters
        self.exp.net.load_weights(model_dir, epoch)

    def test(self):
        mnist_post_processing(self.exp, self.logger)


@torch.no_grad()
def mnist_post_processing(exp, logger):
    use_test_data_list = torch.linspace(0, 9, 10).long()
    batch = None
    for test_batch in exp.test_loader:
        batch = test_batch
        break

    _, pred_index = select_ctx_pred_pts(**exp.assign_config)

    # Get encoder input
    ctx = {"cnn": batch["images"]["value"]}

    # Reconstructor input
    num_traj = batch["images"]["value"].shape[0]
    num_agg = batch["images"]["value"].shape[1] + 1

    bc_time = torch.zeros(num_traj, num_agg)
    bc_vel = torch.zeros(num_traj, num_agg, exp.mp.num_dof)
    times = util.add_expand_dim(batch["trajs"]["time"],
                                add_dim_indices=[1],
                                add_dim_sizes=[num_agg])

    # Ground-truth
    gt = util.add_expand_dim(batch["trajs"]["value"],
                             add_dim_indices=[1], add_dim_sizes=[num_agg])

    # Make the time and dof dimensions flat
    gt = gt.reshape(*gt.shape[:-2], -1)

    # Predict
    mean, diag, off_diag = exp.net.predict(num_traj=num_traj,
                                           enc_inputs=ctx,
                                           dec_input=None)

    # Denormalize prediction
    mean, L = NormProcess.distribution_denormalize(exp.normalizer,
                                                   "init_x_y_dmp_w_g",
                                                   mean, diag, off_diag)

    # Split initial position and DMP weights
    bc_pos = mean[..., 0, :exp.mp.num_dof]
    mean = mean[..., exp.mp.num_dof:].squeeze(-2)
    L = L[..., exp.mp.num_dof:, exp.mp.num_dof:].squeeze(-3)
    assert mean.ndim == 3

    # Reconstruct predicted trajectories
    # exp.mp.update_inputs(times=times, params=mean, params_L=L,
    #                         init_time=bc_time, init_conds=[bc_pos, bc_vel])
    exp.mp.update_inputs(times=times, params=mean, params_L=L,
                            init_time=bc_time, init_pos=bc_pos, init_vel=bc_vel)

    traj_pos_mean = exp.mp.get_traj_pos(flat_shape=True)
    traj_pos_L = torch.linalg.cholesky(exp.mp.get_traj_pos_cov())

    # Select 0-9
    traj_pos_mean = traj_pos_mean[use_test_data_list, -1].reshape(10, 2, -1)
    images = batch["images"]["value"][use_test_data_list, -1, 0]

    # Get ground truth trajs
    traj_x_y = batch["trajs"]["value"][use_test_data_list, :]

    # Plot and log mean prediction
    fig = mean_plot(images, traj_x_y, traj_pos_mean)
    logger.log_figure(fig, figure_name="Mean prediction")

    # Samples
    num_smp = 10
    samples = exp.mp.sample_trajectories(num_smp=num_smp)
    samples = samples[0][use_test_data_list, -1]
    # Plot and log samples
    fig = sample_plot(images, samples)
    logger.log_figure(fig, figure_name="Samples")


def mean_plot(images, traj_x_y, pred_dmp_traj):
    num_images = len(images)

    traj_x_y = util.to_np(traj_x_y)
    x = traj_x_y[..., 0]
    y = traj_x_y[..., 1]

    pred_dmp_traj = util.to_np(pred_dmp_traj)
    pred_dmp_x = pred_dmp_traj[:, 0]
    pred_dmp_y = pred_dmp_traj[:, 1]

    img_size = images.shape[-1]
    if img_size == 40:
        img_size += 1

    fig = plt.figure(figsize=(100, 10), dpi=200, tight_layout=True)
    for i in range(num_images):
        plt.subplot(1, (num_images + 1), i + 1)

        # plt.imshow(images[i].cpu().numpy(),
        #            extent=[0, img_size, img_size, 0])
        plt.plot(x[i], y[i], 'k', label="ground_truth")
        plt.plot(pred_dmp_x[i], pred_dmp_y[i], 'g--', label="pred_dmp")
        # plt.gca().axis("off")
    # plt.show()
    util.savefig(fig, "digits", "pdf", overwrite=True)
    return fig


def sample_plot(images, samples):
    num_images = len(images)
    samples = util.to_np(samples)
    img_size = images.shape[-1]
    if img_size == 40:
        img_size += 1
    fig = plt.figure(figsize=(40, 6), dpi=100, tight_layout=True)
    for i in range(num_images):
        plt.subplot(1, (num_images + 1), i + 1)
        plt.plot(samples[i, :, :, 0].T, samples[i, :, :, 1].T,
                 linewidth=2,
                 label="pred_dmp_sample")
        plt.gca().invert_yaxis()
        plt.gca().axis("off")

        ##### save 3 or 8
        if i == 8:
            fig = plt.figure(figsize=(3,5), dpi=200, tight_layout=True)
            plt.plot(samples[i, :5, :, 0].T, samples[i, :5, :, 1].T, linewidth=3.0)
            # plt.plot(pred_dmp_x[i], pred_dmp_y[i], 'b', linewidth=3.0)
            # plt.plot(mean_x[idx], mean_y[idx], 'r--', linewidth=3.0)
            plt.axis('off')
            # eight
            # plt.xlim([12.5,27.5])
            # plt.ylim([11.5,36.5])
            # three
            # plt.xlim([9.5,24.5])
            # plt.ylim([7.5,32.5])

            plt.gca().invert_yaxis()
            util.savefig(fig, "eight_digit", "pdf", overwrite=True)

            exit()



    # plt.show()
    util.savefig(fig, "digits", "pdf", overwrite=True)
    return fig


if __name__ == "__main__":

    # Logger
    logger_config = Dict()
    logger_config.logger.log_name = "pronmp_one_digit"
    logger_config.logger.entity = "upjtr"
    logger_config.logger.group = "testing"
    logger_config.logger.run_name = "show_plot__prodmp+_25_nll"
    wb_logger = WandbLogger(logger_config.to_dict())
    model_api = "artifact = run.use_artifact('upjtr/pronmp_one_digit/test_model_name:v11', type='model')"
    epoch = 1600
    OneDigitPostProcess(wb_logger, model_api, epoch).test()