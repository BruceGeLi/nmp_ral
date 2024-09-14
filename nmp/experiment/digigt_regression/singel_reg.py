
import numpy as np
import torch
from matplotlib import pyplot as plt
from addict import Dict

from mp_pytorch.mp import MPFactory


data = np.load('/home/weiran/MP_Thesis/experiments/nmp_ral/dataset/s_mnist_25_new/s_mnist_25_new.npz',
               allow_pickle=True)
references = data['trajs'].item()['value']
times = data['trajs'].item()['time']

i = 8
ref = references[i, ...]
time = times[i, ...]
num_b = 10
# prodmp+
# mp_type = 'prodmp+'
# mp_args = Dict()
# mp_args.order = 2
# mp_args.num_basis = num_b
# mp_args.basis_bandwidth_factor = 1
# mp_args.num_basis_outside = 0
# mp_args.alpha = 25
# mp1 = MPFactory().init_mp(mp_type=mp_type, num_dof=2, tau=3.0,
#                          mp_args=mp_args.to_dict())

# prodmp
mp_type = 'prodmp'
mp_args = Dict()
mp_args.alpha_phase = 2.0
mp_args.num_basis = num_b
mp_args.basis_bandwidth_factor = 2
mp_args.num_basis_outside = 0
mp_args.alpha = 25
mp_args.dt = 0.01
mp2 = MPFactory().init_mp(mp_type=mp_type, num_dof=2, tau=3.0,
                         mp_args=mp_args.to_dict())



# mp_type = 'uni_bspline'
# mp_args = Dict()
# mp_args.num_basis = num_b
# mp_args.degree = 4
# mp_args.init_condtion_order = 0
# mp1 = MPFactory().init_mp(mp_type=mp_type, num_dof=2, tau=3.0,
#                          mp_args=mp_args.to_dict())

mp_type = 'uni_bspline'
mp_args = Dict()
mp_args.num_basis = num_b
mp_args.degree_p = 5
mp_args.init_condition_order = 2
mp_args.end_condition_order = 2
mp_args.goal_basis = True
mp1 = MPFactory().init_mp(mp_type=mp_type, num_dof=2, tau=3.0,
                         mp_args=mp_args.to_dict())



# cfg = Dict()
# cfg.mp_type = "uni_bspline"
# cfg.num_dof = 2
# cfg.tau = 3
# cfg.learn_tau = True
# cfg.learn_delay = True
# cfg.mp_args.num_basis = 10
# cfg.mp_args.degree_p = 4
# cfg.mp_args.init_condition_order = 2
# cfg.mp_args.end_condition_order = 2
# mp1 = MPFactory().init_mp(cfg.to_dict())

params = mp1.learn_mp_params_from_trajs(time, ref, reg=1e-5)
params2 = mp2.learn_mp_params_from_trajs(torch.tensor(time), torch.tensor(ref))
pos1 = mp1.get_traj_pos()
pos2 = mp2.get_traj_pos()

from mp_pytorch.util import debug_plot
debug_plot(x=None, y=[ref[:, 0], pos1[:, 0]], labels=['gt', 'bsp'], title='y1')
debug_plot(x=None, y=[ref[:, 1], pos1[:, 1]], labels=['gt', 'bsp'], title='y2')

fig = plt.figure(figsize=(3, 5), dpi=200, tight_layout=True)
plt.plot(ref[:, 0], ref[:, 1], 'b', linewidth=3.0, label='gt')


plt.plot(pos1[:, 0], pos1[:, 1], 'r--', linewidth=3.0, label='bsp')
# plt.plot(pos2[:, 0], pos2[:, 1], 'g--', linewidth=3.0, label='prodmp')
plt.legend()
plt.title(f'{mp_args.num_basis} bases')
# plt.xlim([12.5, 27.5])
# plt.ylim([11.5, 36.5])
plt.show()



