import numpy as np
from typing import Any
from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def __init__(self) -> None:
        ...

    @abstractmethod
    def act(self, obs, info) -> Any:
        ...

    def calculate_rot_error(self, gt_quat, quat):
        gt_quat = np.array(gt_quat)
        quat = np.array(quat) # x y z w
        inv_q = quat_inverse(quat)
        diff_q = quat_mult(gt_quat, inv_q)

        diff_i = diff_q[:3]
        norm_i = normalize(diff_i)
        diff_r = diff_q[3]
        return 2 * np.arctan2(norm_i, diff_r)
        # return 2 * np.arccos(diff_r) # Not as stable?

    def calculate_trans_error(self, gt_trans, trans):
        gt_trans = np.array(gt_trans)
        tranas = np.array(trans)
        diff = gt_trans - trans
        magn = normalize(diff)
        return magn

    def calculate_error(self, gt_pose, pose):
        # Not possible at the moment, because of random nature and the
        # returned poses, that are not validated to actually pick objects
        # gt_action = self.act(None, None)
        
        trans, rot = pose
        gt_trans, gt_rot = gt_pose
        t_error = self.calculate_trans_error(gt_trans, trans)
        r_error = self.calculate_rot_error(gt_rot, rot)
        error = [t_error, r_error]
        return error
    
def quat_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.array([x, y, z, w])

def quat_inverse(q1):
    conj = q1 * np.array([-1.0, -1.0, -1.0, 1.0])
    # expected to be 1 -> unit quaternion
    norm = normalize(q1)
    return conj / norm

def normalize(m1):
    return np.sqrt(np.dot(m1, m1))

    
