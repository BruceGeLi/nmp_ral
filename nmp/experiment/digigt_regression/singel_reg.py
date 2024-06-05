
import numpy as np
import torch
from matplotlib import pyplot as plt
from addict import Dict

from mp_pytorch.mp import MPFactory


data = np.load('/home/weiran/MP_Thesis/experiments/nmp_ral/dataset/s_mnist_25_new/s_mnist_25_new.npz',
               allow_pickle=True)
references = data['trajs'].item()['value']
times = data['trajs'].item()['time']

i = 3
ref = references[i, ...]
time = times[i, ...]
num_b = 15

# prodmp+
mp_type = 'prodmp+'
mp_args = Dict()
mp_args.order = 2
mp_args.num_basis = num_b
mp_args.basis_bandwidth_factor = 1
mp_args.num_basis_outside = 0
mp_args.alpha = 25
mp1 = MPFactory().init_mp(mp_type=mp_type, num_dof=2, tau=3.0,
                         mp_args=mp_args.to_dict())

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

params = mp1.learn_mp_params_from_trajs(time, ref)
params2 = mp2.learn_mp_params_from_trajs(time, ref)
pos1 = mp1.get_traj_pos()
pos2 = mp2.get_traj_pos()

fig = plt.figure(figsize=(3, 5), dpi=200, tight_layout=True)
plt.plot(ref[:, 0], ref[:, 1], 'b', linewidth=3.0, label='gt')
plt.plot(pos1[:, 0], pos1[:, 1], 'r--', linewidth=3.0, label='prodmp+')
plt.plot(pos2[:, 0], pos2[:, 1], 'g--', linewidth=3.0, label='prodmp')
plt.legend()
plt.title(f'{mp_args.num_basis} bases')
# plt.xlim([12.5, 27.5])
# plt.ylim([11.5, 36.5])
plt.show()



