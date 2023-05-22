import numpy as np
import torch
from torch.autograd import Variable
import logging
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
import numpy as np
logger = logging.getLogger(__name__)


def unravel_idx(indices, shape):
    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = np.stack(coord[::-1], axis=-1)
    return coord


def calc_cnn_out_size(in_size, k, p=0, s=1):
    out_size = ((in_size + 2 * p - k) / s) + 1
    return int(out_size)


def np_quat_to_scipy_quat(quat):
    """wxyz to xyzw"""
    return np.array([quat.x, quat.y, quat.z, quat.w])


def pos_orn_to_matrix(pos, orn):
    """
    :param pos: np.array of shape (3,)
    :param orn: np.array of shape (4,) -> quaternion xyzw
                np.quaternion -> quaternion wxyz
                np.array of shape (3,) -> euler angles xyz
    :return: 4x4 homogeneous transformation
    """
    mat = np.eye(4)
    if isinstance(orn, np.quaternion):
        orn = np_quat_to_scipy_quat(orn)
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 4:
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 3:
        mat[:3, :3] = R.from_euler('xyz', orn).as_matrix()
    mat[:3, 3] = pos
    return mat


def tt(x, device):
    if isinstance(x, dict):
        dict_of_list = {}
        for key, val in x.items():
            dict_of_list[key] = Variable(torch.from_numpy(val).float().to(device),
                                         requires_grad=False)
        return dict_of_list
    else:
        return Variable(torch.from_numpy(x).float().to(device),
                        requires_grad=False)


def torch_to_numpy(x):
    return x.detach().cpu().numpy()
