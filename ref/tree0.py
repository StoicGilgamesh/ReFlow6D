# encoding: utf-8
"""This file includes necessary params, info."""
import os
import os.path as osp
import mmcv
import numpy as np

# ---------------------------------------------------------------- #
# ROOT PATH INFO
# ---------------------------------------------------------------- #
cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
output_dir = osp.join(root_dir, "output")  # directory storing experiment data (result, model checkpoints, etc).

data_root = osp.join(root_dir, "datasets")
bop_root = osp.join(data_root, "BOP_DATASETS/")
# ---------------------------------------------------------------- #
# TRACEBOT
# ---------------------------------------------------------------- #
dataset_root = osp.join(bop_root, "tree0")
train_dir = osp.join(dataset_root, "train_pbr")
test_dir = osp.join(dataset_root, "test")

model_dir = osp.join(dataset_root, "models")
model_eval_dir = osp.join(dataset_root, "models_eval")
model_scaled_simple_dir = osp.join(dataset_root, "models_scaled_f5k")
#vertex_scale = 0.001
vertex_scale = 1.00

id2obj = {
    1: "tree0.stl",
}
objects = list(id2obj.values())
obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}

model_paths = [osp.join(model_dir, "obj_{:06d}.ply").format(_id) for _id in id2obj]
texture_paths = None
model_colors = [((i + 1) * 10, (i + 1) * 10, (i + 1) * 10) for i in range(obj_num)]  # for renderer

#diameters = (
#        np.array(
#        [
#            91.56727218530786,  # 1
#        ]
#    )
#    / 1000.0
#)

diameters = (
        np.array(
        [
            0.11087263584819747,  # 1
        ]
    )
)


# Camera info
width = 1280
height = 720
zNear = 0.25
zFar = 6.0
center = (height / 2, width / 2)
camera_matrix = np.array([[675.6171264648438, 0.0, 632.1181030273438], [0.0, 675.6171264648438, 338.2853698730469], [0, 0, 1]])


def get_models_info():
    """key is str(obj_id)"""
    models_info_path = osp.join(model_dir, "models_info.json")
    assert osp.exists(models_info_path), models_info_path
    models_info = mmcv.load(models_info_path)  # key is str(obj_id)
    return models_info


def get_fps_points():
    """key is str(obj_id) generated by
    core/gdrn_modeling/tools/lmo/lmo_1_compute_fps.py."""
    fps_points_path = osp.join(model_dir, "fps_points.pkl")
    assert osp.exists(fps_points_path), fps_points_path
    fps_dict = mmcv.load(fps_points_path)
    return fps_dict


def get_keypoints_3d():
    """key is str(obj_id) generated by
    core/roi_pvnet/tools/lmo/lmo_1_compute_keypoints_3d.py."""
    keypoints_3d_path = osp.join(model_dir, "keypoints_3d.pkl")
    assert osp.exists(keypoints_3d_path), keypoints_3d_path
    kpts_dict = mmcv.load(keypoints_3d_path)
    return kpts_dict
