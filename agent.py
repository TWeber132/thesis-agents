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
        conj_gtq = np.array(gt_quat) * np.array([-1.0, -1.0, -1.0, 1.0])
        # expected to be 1 - unit quaternion
        norm_gtq = np.sqrt(np.dot(gt_quat, gt_quat))
        print("norm_gtq: ", norm_gtq)
        inv_gtq = conj_gtq / norm_gtq
        diff_q = np.array(quat) * inv_gtq

        # diff_q = p.getDifferenceQuaternion(gt_rot, rot)
        norm_q = np.sqrt(np.dot(diff_q, diff_q))
        print("norm_q: ", norm_q)
        if abs(norm_q - 1) > 0.00005:  # How far from being a unit quaternion
            diff_q = diff_q / norm_q

        diff_i = diff_q[:3]
        norm_i = np.sqrt(np.dot(diff_i, diff_i))
        diff_r = diff_q[3]
        return 2 * np.arctan2(norm_i, diff_r)
        # return 2 * np.arccos(diff_r) # Not as stable?

    def calculate_trans_error(self, gt_trans, trans):
        diff = gt_trans - trans
        magn = np.sqrt(np.dot(diff, diff))
        return magn

    def calculate_errors(self, gt_action, action):
        t_errors = []
        r_errors = []
        # Not possible at the moment, because of random nature and the
        # returned poses, that are not validated to actually pick objects
        # gt_action = self.act(None, None)
        assert len(gt_action) == len(action)

        for gt_pose, pose in zip(gt_action, action):
            trans, rot = pose
            gt_trans, gt_rot = gt_pose
            t_error = self.calculate_trans_loss(gt_trans, trans)
            r_error = self.calculate_rot_loss(gt_rot, rot)
            t_errors.append(t_error)
            r_errors.append(r_error)

        return t_errors, r_errors
