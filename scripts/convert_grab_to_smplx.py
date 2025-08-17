import os
import numpy as np
import pickle
from pathlib import Path
import ipdb


# these paths are from the GRAB dataset npz files (not the pt files you get from their preprocessing code)
motion_path1 = "/move/u/mpiseno/data/GRAB/grab/"

# find all npz files in this subdir recursively
motion_files = [
    os.path.join(root, file)
    for root, _, files in os.walk(motion_path1)
    for file in files if file.endswith('.npz')
]

motion_files = [
    'data/GRAB/s1/apple_lift.npz',
    # '/move/u/mpiseno/data/GRAB/grab/s1/banana_pass_1.npz',
    # '/move/u/mpiseno/data/GRAB/grab/s1/binoculars_see_1.npz',
    # '/move/u/mpiseno/data/GRAB/grab/s1/cubelarge_lift.npz',
    # '/move/u/mpiseno/data/GRAB/grab/s1/cubesmall_lift.npz',
    # '/move/u/mpiseno/data/GRAB/grab/s1/duck_inspect_1.npz',
    # '/move/u/mpiseno/data/GRAB/grab/s1/elephant_offhand_1.npz',
    # '/move/u/mpiseno/data/GRAB/grab/s1/mug_drink_1.npz',
    # '/move/u/mpiseno/data/GRAB/grab/s1/teapot_pour_1.npz',
    # '/move/u/mpiseno/data/GRAB/grab/s1/waterbottle_open_1.npz',
]

# save as individual files
target_dir = "./data/GRAB_smplx"
os.makedirs(target_dir, exist_ok=True)
for motion_file in motion_files:
    subject = motion_file.split('/')[-2]  # e.g., 's1'
    seq_name = motion_file.split('/')[-1][:-len('.npz')] # e.g., 'apple_lift'
    data = dict(np.load(motion_file, allow_pickle=True))

    motion_file.split('/')[:-3]
    body_dict = data["body"].item()["params"]
    lhand_dict = data["lhand"].item()["params"]
    rhand_dict = data["rhand"].item()["params"]
    num_frames = body_dict["body_pose"].shape[0]

    # Create new dict and just save the stuff we need. Also rename some stuff to fit this codebase
    smplx_data = {}
    smplx_data["mocap_frame_rate"] = data["framerate"]
    smplx_data["gender"] = data["gender"]
    # smplx_data["betas"] = betas[subject]
    smplx_data["betas"] = np.zeros((1, 16))
    smplx_data["pose_body"] = body_dict["body_pose"]
    smplx_data["root_orient"] = body_dict["global_orient"]
    smplx_data["trans"] = body_dict["transl"]

    # The fullpose key contains the (T, 45) dim axis-angle rotations
    # The hand_pose key (not used here) contains the (T, 24) dim betas
    smplx_data["left_hand_pose"] = lhand_dict["fullpose"]
    smplx_data["right_hand_pose"] = rhand_dict["fullpose"]

    # use pickle to save
    out_dir = f"{target_dir}/{subject}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{seq_name}.pkl", "wb") as f:
        pickle.dump(smplx_data, f)
    print(f"saved {seq_name}")
