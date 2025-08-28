import os
import time
import argparse
import pathlib

import torch
import numpy as np
import smplx
import viser
import ipdb

from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES
from rich import print

from general_motion_retargeting.utils.smpl import (
    FINGERTIP_NAMES, FINGER_JOINT_NAMES, 
    get_joints_from_names, rotation_from_two_points
)



JOINTS_TO_TRACK = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3'
]


def load_smplx_data(smplx_file, smplx_body_model_dir):
    smplx_data = np.load(smplx_file, allow_pickle=True)
    body_model = smplx.create(
        smplx_body_model_dir,
        "smplx",
        gender=str(smplx_data["gender"]),
        use_pca=False,
        ext="pkl",
        # num_pca_comps=24,
        flat_hand_mean=True
    )
    num_frames = smplx_data["pose_body"].shape[0]
    smplx_output = body_model(
        # Global orientation of the root in axis-angle representation
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),      # (T, 3)

        # Body shape parameters (betas)
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1),        # (1, 16)
        # betas=torch.tensor(betas).float().view(1, -1),        # (1, 16)

        # Orientations of body joints in axis-angle representation
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),            # (T, 63)

        # Global translation of the root
        transl=torch.tensor(smplx_data["trans"]).float(),                   # (T, 3)

        # Orientations of hands in axis-angle representation
        left_hand_pose=torch.tensor(smplx_data["left_hand_pose"]).float(),  # (T, 45)
        right_hand_pose=torch.tensor(smplx_data["right_hand_pose"]).float(), # (T, 45)
        # left_hand_pose=torch.zeros(num_frames, 45).float(),  # (T, 45)
        # right_hand_pose=torch.zeros(num_frames, 45).float(), # (T, 45)

        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        return_full_pose=True,
    )
    return smplx_output, body_model, smplx_data


def FK(global_orient, full_pose, parents):
    all_rots = []
    T, J, _ = full_pose.shape
    for t in range(T):
        cur_pose = full_pose[t]
        orientations = [R.from_rotvec(global_orient[t])]
        for j in range(1, J):
            rot = orientations[parents[j]] * R.from_rotvec(cur_pose[j])
            orientations.append(rot)
        
        orientations = np.stack([o.as_quat(scalar_first=True) for o in orientations])
        all_rots.append(orientations)

    all_rots = np.stack(all_rots)
    return all_rots


def correct_finger_joint_rots(joints, joint_rots, joint_names, parents):
    # Augment parents to include fingertips
    fingertip_parents = [JOINT_NAMES.index(name + "3") for name in FINGERTIP_NAMES]
    parents = np.concatenate([parents, fingertip_parents], axis=0)
    assert joints.shape[1] == len(parents) and joints.shape[1] == len(joint_names)

    # This array only returns the first child for each joint. Make this a dictionary if you want all children
    children = []
    for j in range(len(parents)):
        if j in parents:
            children.append(np.where(parents == j)[0][0])
        else:
            children.append(-1)
    children = np.array(children)

    finger_idxs = [joint_names.index(name) for name in FINGER_JOINT_NAMES]
    assert all([idx != -1 for idx in children[finger_idxs]]), "All joints computing rotations must have children"
    finger_joint_rots = []
    for t in range(joints.shape[0]):
        rots = []
        for name, j in zip(FINGER_JOINT_NAMES, finger_idxs):
            cur_pos = joints[t, j]
            child_pos = joints[t, children[j]]
            cur_rot = R.from_quat(joint_rots[t, j], scalar_first=True)
            negate_x = 'right' in name
            rot = rotation_from_two_points(
                cur_pos, child_pos,
                up=cur_rot.as_matrix()[:, 1], # Keep y axis as up direction
                negate_x=negate_x
            ).as_quat(scalar_first=True)
            rots.append(rot)

        rots = np.stack(rots)
        finger_joint_rots.append(rots)

    finger_joint_rots = np.stack(finger_joint_rots)
    joint_rots[:, finger_idxs] = finger_joint_rots
    return joint_rots


def append_fingertip_rots(joint_rots):
    fingertip_parents = [JOINT_NAMES.index(name + "3") for name in FINGERTIP_NAMES]
    joint_rots = np.concatenate([
        joint_rots,
        joint_rots[:, fingertip_parents]
    ], axis=1)
    return joint_rots


def main(args):
    HERE = pathlib.Path(__file__).parent
    smplx_body_model_dir = HERE / ".." / "assets" / "body_models"
    
    smplx_output, body_model, smplx_data = load_smplx_data(args.smplx_file, smplx_body_model_dir)
    vertices = smplx_output.vertices.detach().cpu().numpy().squeeze()

    # joints are 3D positions in global coordinates
    joint_names = JOINT_NAMES[:len(body_model.parents)] + FINGERTIP_NAMES
    joints = get_joints_from_names(smplx_output, joint_names)

    # Do FK for global joint orientations
    global_orient = smplx_output.global_orient.detach().cpu().numpy()
    full_pose = smplx_output.full_pose.detach().cpu().numpy()
    full_pose = full_pose.reshape(full_pose.shape[0], -1, 3)
    parents = body_model.parents
    joint_rots = FK(global_orient=global_orient, full_pose=full_pose, parents=parents)
    joint_rots = correct_finger_joint_rots(joints, joint_rots, joint_names, parents)
    joint_rots = append_fingertip_rots(joint_rots)

    # finger_pos, finger_rots = modify_hand_joint_angles(full_pose, joints, joint_names, parents)
    assert joint_rots.shape[:2] == joints.shape[:2], (
        f"""Joint rotations and joints must have the same time dimension but got shapes """
        f"""joint_rots: {joint_rots.shape}, joints: {joints.shape}"""
    )
    assert joint_rots.shape[1] == len(joint_names), "Number of joints must match the number of joint names"

    t = 0
    src_fps, tgt_fps = smplx_data["mocap_frame_rate"].item(), 30
    frame_skip = int(src_fps / tgt_fps)
    T, J = joints.shape[:2]

    '''
    Viser axes are (x, y, z) = (red, green, blue)
    '''
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", plane="xy")

    body_handle = server.scene.add_mesh_simple(
        "/human",
        vertices=vertices[0],
        faces=body_model.faces,
        color=(90, 200, 255),
        opacity=0.8,
    )
    frame_handles = {}
    for name, j in zip(joint_names, range(J)):
        # if name not in JOINTS_TO_TRACK: continue
        frame_handle = server.scene.add_frame(
            f"/frames/{name}",
            show_axes=True,
            axes_length=0.02,
            axes_radius=0.002,
            origin_radius=0.005,
            origin_color=(255, 255, 0),
            wxyz=joint_rots[0, j],
            position=joints[0, j],
        )
        frame_handles[name] = frame_handle

    # Add a play button for the motion
    playing = False
    play_button = server.gui.add_button(label="Play")
    play_button.value = False
    @play_button.on_click
    def _(_) -> None:
        global playing
        playing = not playing
        if playing:
            play_button.label = "Pause"
        else:
            play_button.label = "Play"

    # Slider to control the current frame
    gui_slider = server.gui.add_slider(
        "Frame Slider",
        min=0, max=T,
        step=frame_skip, initial_value=0,
    )

    while True:
        start = time.time()
        t = gui_slider.value
        if playing:
            nxt = (t + frame_skip) % T
            gui_slider.value = nxt
        
        body_handle.vertices = vertices[t]
        for name, j in zip(joint_names, range(J)):
            # if name not in JOINTS_TO_TRACK: continue
            frame_handles[name].position = joints[t, j]
            frame_handles[name].wxyz = joint_rots[t, j]

        end = time.time()
        sleep_time = max(0, (1 / tgt_fps) - (end - start))
        time.sleep(sleep_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_file", type=str, required=True)
    args = parser.parse_args()
    main(args)


'''
python scripts/vis_human_motion.py --smplx_file data/GRAB_smplx/s1/apple_lift.pkl
'''
    