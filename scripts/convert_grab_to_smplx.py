import os
import joblib
import numpy as np
import pickle
from pathlib import Path


# these paths are from the GRAB dataset npz files (not the pt files you get from their preprocessing code)
motion_path1 = "/move/u/mpiseno/data/GRAB/grab/"

# find all npz files in this subdir recursively
motion_files = [
    os.path.join(root, file)
    for root, _, files in os.walk(motion_path1)
    for file in files if file.endswith('.npz')
]

motion_files = [
    '/move/u/mpiseno/data/GRAB/grab/s1/apple_lift.npz',
    '/move/u/mpiseno/data/GRAB/grab/s1/banana_pass_1.npz',
    '/move/u/mpiseno/data/GRAB/grab/s1/binoculars_see_1.npz',
    '/move/u/mpiseno/data/GRAB/grab/s1/cubelarge_lift.npz',
    '/move/u/mpiseno/data/GRAB/grab/s1/cubesmall_lift.npz',
    '/move/u/mpiseno/data/GRAB/grab/s1/duck_inspect_1.npz',
    '/move/u/mpiseno/data/GRAB/grab/s1/elephant_offhand_1.npz',
    '/move/u/mpiseno/data/GRAB/grab/s1/mug_drink_1.npz',
    '/move/u/mpiseno/data/GRAB/grab/s1/teapot_pour_1.npz',
    '/move/u/mpiseno/data/GRAB/grab/s1/waterbottle_open_1.npz',
]
# betas_dir = Path(motion_files[0]).parent.parent.parent / "tools" / "subject_meshes"
# betas = {
#     # male
#     's1': np.load(betas_dir / "male" / "s1_betas.npy"),
#     's2': np.load(betas_dir / "male" / "s2_betas.npy"),
#     's9': np.load(betas_dir / "male" / "s8_betas.npy"),
#     's9': np.load(betas_dir / "male" / "s9_betas.npy"),
#     's10': np.load(betas_dir / "male" / "s10_betas.npy"),
#     # female
#     's3': np.load(betas_dir / "female" / "s3_betas.npy"),
#     's4': np.load(betas_dir / "female" / "s4_betas.npy"),
#     's5': np.load(betas_dir / "female" / "s5_betas.npy"),
#     's6': np.load(betas_dir / "female" / "s6_betas.npy"),
#     's7': np.load(betas_dir / "female" / "s7_betas.npy"),
# }

# save as individual files
target_dir = "./data/GRAB_smplx"
os.makedirs(target_dir, exist_ok=True)
for motion_file in motion_files:
    subject = motion_file.split('/')[-2]  # e.g., 's1'
    seq_name = motion_file.split('/')[-1][:-len('.npz')] # e.g., 'apple_lift'
    data = dict(np.load(motion_file, allow_pickle=True))

    motion_file.split('/')[:-3]
    body_dict = data["body"].item()["params"]
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

    poses = np.concatenate([body_dict["body_pose"], 
                            np.zeros((num_frames, 102))],
                            axis=1)
    smplx_data["poses"] = poses

    # use pickle to save
    out_dir = f"{target_dir}/{subject}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{seq_name}.pkl", "wb") as f:
        pickle.dump(smplx_data, f)
    print(f"saved {seq_name}")
