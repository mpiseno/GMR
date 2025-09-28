import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco as mj
import mujoco.viewer as mjv
import imageio
import numpy as np

from mujoco import _structs
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from loop_rate_limiters import RateLimiter
from rich import print


def add_object_assets_to_xml(
    base_xml_path: str,
    mesh_path: str,
    mesh_name: str = "object_viz_mesh",
    pos=(0.0, 0.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),  # w, x, y, z
    rgba=(0.8, 0.8, 0.8, 1.0),
):
    base_xml_path = Path(base_xml_path).expanduser().resolve()
    mesh_path = Path(mesh_path).expanduser().resolve()
    base_dir = base_xml_path.parent.as_posix()

    # --- load and parse the base XML (must have <mujoco> root)
    tree = ET.parse(base_xml_path)
    root = tree.getroot()

    asset = root.find("asset")
    mesh_el = ET.SubElement(asset, "mesh", {
        "name": mesh_name,
        "file": mesh_path.as_posix(),
        "scale": "0.001 0.001 0.001",  # assuming input mesh is in mm
    })

    # --- ensure <worldbody> exists; add a visual-only body/geom
    worldbody = root.find("worldbody")
    body = ET.SubElement(worldbody, "body", {
        "name": "object_viz",
        "pos": f"{pos[0]} {pos[1]} {pos[2]}",
        "quat": f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}",
        "mocap": "true",  # make sure the body does not affect physics
    })
    ET.SubElement(body, "geom", {
        "type": "mesh",
        "mesh": mesh_name,
        "contype": "0",
        "conaffinity": "0",
        "group": "1",
        "rgba": f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}",
    })

    # write out modified XML to a temporary file
    wrapper_xml = base_dir + f"/{base_xml_path.stem}_temp.xml"
    tree.write(wrapper_xml)
    return wrapper_xml


def draw_frame(
    pos,
    mat,
    scene: mj.MjvScene,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if joint_name is not None:
            geom.label = joint_name  # 这里赋名字
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            scene.geoms[scene.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        scene.ngeom += 1

class RobotMotionViewer:
    def __init__(self,
                robot_type,
                object_mesh_path=None,
                camera_follow=True,
                motion_fps=30,
                transparent_robot=0,
                # video recording
                record_video=False,
                video_path=None,
                video_width=640,
                video_height=480):
        
        self.robot_type = robot_type
        xml_path = ROBOT_XML_DICT[robot_type]
        if object_mesh_path is not None:
            xml_path = add_object_assets_to_xml(
                xml_path,
                mesh_path=object_mesh_path,
            )
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        mj.mj_step(self.model, self.data)
        
        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.record_video = record_video

        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False)      
        
        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot

        # Create separate camera for video recording so we have the option to change the viewer camera separate
        # from video recording camera
        self.camera = _structs.MjvCamera()
        self.camera.fixedcamid = 0
        
        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)
            
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
            print(f"Recording video to {self.video_path}")
            
            # Initialize renderer for video recording
            self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)
        
    def step(self, 
            # robot data
            root_pos, root_rot, dof_pos,
            # human data
            human_motion_data=None, 
            show_human_body_name=False,
            # scale for human point visualization
            human_point_scale=0.1,
            # human pos offset add for visualization    
            human_pos_offset=np.array([0.0, 0.0, 0]),
            object_pose=None,
            # rate limit
            rate_limit=True, 
        ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.
        
        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """
        
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot # quat need to be scalar first! for mujoco
        self.data.qpos[7:] = dof_pos

        if object_pose is not None:
            object_pos, object_rot = object_pose[:3], object_pose[3:]
            bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "object_viz")
            mocap_id = self.model.body_mocapid[bid]
            self.data.mocap_pos[mocap_id] = object_pos
            self.data.mocap_quat[mocap_id] = object_rot
            
        mj.mj_forward(self.model, self.data)

        if self.record_video:
            self.camera.lookat = self.data.xpos[self.model.body(self.robot_base).id]
            self.camera.distance = self.viewer_cam_distance
            self.camera.elevation = -10
        
        if human_motion_data is not None:
            # Clean custom geometry
            self.viewer.user_scn.ngeom = 0

            # Draw the task targets for reference
            for human_body_name, (pos, rot) in human_motion_data.items():
                draw_frame(
                    pos,
                    R.from_quat(rot, scalar_first=True).as_matrix(),
                    self.viewer.user_scn,
                    size=human_point_scale,
                    joint_name=human_body_name if show_human_body_name else None,
                    pos_offset=human_pos_offset,
                )

        if object_pose is not None:
            # Draw object frame
            draw_frame(
                object_pos,
                R.from_quat(object_rot, scalar_first=True).as_matrix(),
                self.viewer.user_scn,
                size=human_point_scale,
                joint_name="object" if show_human_body_name else None,
            )

        self.viewer.sync()
        if rate_limit is True:
            self.rate_limiter.sleep()

        if self.record_video:
            # Use renderer for proper offscreen rendering
            self.renderer.update_scene(self.data, camera=self.camera)
            img = self.renderer.render()
            self.mp4_writer.append_data(img)
    
    def close(self):
        self.viewer.close()
        time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")
