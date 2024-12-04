"""Register datasets in this file will be imported in project root to register
the datasets."""
import logging
import os
import os.path as osp
import mmcv
import detectron2.utils.comm as comm
import ref
from detectron2.data import DatasetCatalog, MetadataCatalog
from core.gdrn_modeling.datasets import (
    lm_pbr,
    lmo_bop_test,
    ycbv_pbr,
    ycbv_d2,
    ycbv_bop_test,
    hb_pbr,
    hb_bop_val,
    hb_bop_test,
    tudl_pbr,
    tudl_d2,
    tudl_bop_test,
    tless_pbr,
    tless_d2,
    tless_bop_test,
    icbin_pbr,
    icbin_bop_test,
    itodd_pbr,
    itodd_bop_test,
    itodd_d2,
    bottlearch_pbr,
    bottlearch_bop_test,
    bottle1_pbr,
    bottle1_bop_test,
    bottle2_pbr,
    bottle2_bop_test,
    tree0_pbr,
    tree0_bop_test,
    heart0_pbr,
    heart0_bop_test,
    mug6_pbr,
    mug6_bop_test,
    mug1_pbr,
    mug1_bop_test,
    mug4_pbr,
    mug4_bop_test,
    mug5_pbr,
    mug5_bop_test,
    mug3_pbr,
    mug3_bop_test,
    mugarch_pbr,
    mugarch_bop_test,
    cupzeroarch_pbr,
    cupzeroarch_bop_test,
    bigcup_pbr,
    bigcup_bop_test,
    budui1_pbr,
    budui1_bop_test,
    budui2_pbr,
    budui2_bop_test,
    budui3_pbr,
    budui3_bop_test,
    budui4_pbr,
    budui4_bop_test,
    budui5_pbr,
    budui5_bop_test,
    mmmmm2_pbr,
    mmmmm2_bop_test,
    mmmmm1_pbr,
    mmmmm1_bop_test,
    zhuixin_pbr,
    zhuixin_bop_test,
    mcup31_pbr,
    mcup31_bop_test,
    tracebotstripecanister_pbr,
    tracebotstripecanister_bop_test,
    tracebotcanister_pbr,
    tracebotcanister_bop_test,
    fluidcontainer_pbr,
    fluidcontainer_bop_test,
)


cur_dir = osp.dirname(osp.abspath(__file__))
# from lib.utils.utils import iprint
__all__ = [
    "register_dataset",
    "register_datasets",
    "register_datasets_in_cfg",
    "get_available_datasets",
]
_DSET_MOD_NAMES = [
    "lm_pbr",
    "lmo_bop_test",
    "ycbv_pbr",
    "ycbv_d2",
    "ycbv_bop_test",
    "hb_pbr",
    "hb_bop_val",
    "hb_bop_test",
    "tudl_pbr",
    "tudl_d2",
    "tudl_bop_test",
    "tless_pbr",
    "tless_d2",
    "tless_bop_test",
    "icbin_pbr",
    "icbin_bop_test",
    "itodd_pbr",
    "itodd_bop_test",
    "itodd_d2",    
    "bottlearch_pbr",
    "bottlearch_bop_test",
    "bottle1_pbr",
    "bottle1_bop_test",
    "bottle2_pbr",
    "bottle2_bop_test",
    "tree0_pbr",
    "tree0_bop_test",
    "heart0_pbr",
    "heart0_bop_test",
    "mug6_pbr",
    "mug6_bop_test",
    "mug1_pbr",
    "mug1_bop_test",
    "mug4_pbr",
    "mug4_bop_test",
    "mug5_pbr",
    "mug5_bop_test",
    "mug3_pbr",
    "mug3_bop_test",
    "mugarch_pbr",
    "mugarch_bop_test",
    "cupzeroarch_pbr",
    "cupzeroarch_bop_test",
    "bigcup_pbr",
    "bigcup_bop_test",
    "budui1_pbr",
    "budui1_bop_test",
    "budui2_pbr",
    "budui2_bop_test",
    "budui3_pbr",
    "budui3_bop_test",
    "budui4_pbr",
    "budui4_bop_test",
    "budui5_pbr",
    "budui5_bop_test",
    "mmmmm2_pbr",
    "mmmmm2_bop_test",
    "mmmmm1_pbr",
    "mmmmm1_bop_test",
    "zhuixin_pbr",
    "zhuixin_bop_test",
    "mcup31_pbr",
    "mcup31_bop_test",
    "tracebotstripecanister_pbr",
    "tracebotstripecanister_bop_test",
    "tracebotcanister_pbr",
    "tracebotcanister_bop_test",
    "fluidcontainer_pbr",
    "fluidcontainer_bop_test",
]

logger = logging.getLogger(__name__)


def register_dataset(mod_name, dset_name, data_cfg=None):
    """
    mod_name: a module under core.datasets or other dataset source file imported here
    dset_name: dataset name
    data_cfg: dataset config
    """
    register_func = eval(mod_name)
    register_func.register_with_name_cfg(dset_name, data_cfg)


def get_available_datasets(mod_name):
    return eval(mod_name).get_available_datasets()


def register_datasets_in_cfg(cfg):
    for split in [
        "TRAIN",
        "TEST",
        "SS_TRAIN",
        "TEST_DEBUG",
        "TRAIN_REAL",
        "TRAIN2",
        "TRAIN_SYN_SUP",
    ]:
        for name in cfg.DATASETS.get(split, []):
            if name in DatasetCatalog.list():
                continue
            registered = False
            # try to find in pre-defined datasets
            # NOTE: it is better to let all datasets pre-refined
            for _mod_name in _DSET_MOD_NAMES:
                if name in get_available_datasets(_mod_name):
                    register_dataset(_mod_name, name, data_cfg=None)
                    registered = True
                    break
            # not in pre-defined; not recommend
            if not registered:
                # try to get mod_name and data_cfg from cfg
                """load data_cfg and mod_name from file
                cfg.DATA_CFG[name] = 'path_to_cfg'
                """
                assert "DATA_CFG" in cfg and name in cfg.DATA_CFG, "no cfg.DATA_CFG.{}".format(name)
                assert osp.exists(cfg.DATA_CFG[name])
                data_cfg = mmcv.load(cfg.DATA_CFG[name])
                mod_name = data_cfg.pop("mod_name", None)
                assert mod_name in _DSET_MOD_NAMES, mod_name
                register_dataset(mod_name, name, data_cfg)


def register_datasets(dataset_names):
    for name in dataset_names:
        if name in DatasetCatalog.list():
            continue
        registered = False
        # try to find in pre-defined datasets
        # NOTE: it is better to let all datasets pre-refined
        for _mod_name in _DSET_MOD_NAMES:
            if name in get_available_datasets(_mod_name):
                register_dataset(_mod_name, name, data_cfg=None)
                registered = True
                break

        # not in pre-defined; not recommend
        if not registered:
            raise ValueError(f"dataset {name} is not defined")