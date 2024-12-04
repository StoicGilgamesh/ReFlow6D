# about 3 days
_base_ = ["../../_base_/gdrn_base.py"]

OUTPUT_DIR = "output/gdrn/cuparch/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_cuparch"
INPUT = dict(
    DZI_PAD_SCALE=1.5,
    TRUNCATE_FG=True,
    CHANGE_BG_PROB=0.0,
    #COLOR_AUG_PROB=0.8,
    COLOR_AUG_PROB=0.0,
    IMG_AUG_RESIZE=False,
    COLOR_AUG_TYPE="code",
    COLOR_AUG_CODE=(
        "Sequential(["
        # Sometimes(0.5, PerspectiveTransform(0.05)),
        # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
        # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
        "Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),"
        "Sometimes(0.4, GaussianBlur((0., 3.))),"
        "Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),"
        "Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),"
        "Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),"
        "Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),"
        "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
        "Sometimes(0.3, Invert(0.2, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
        "Sometimes(0.5, Multiply((0.6, 1.4))),"
        "Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),"
        "Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),"
        "Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),"  # maybe remove for det
        "], random_order=True)"
        # cosy+aae
    ),
)

SOLVER = dict(
    IMS_PER_BATCH=8,
    TOTAL_EPOCHS=600,  # 10
    # 40 epochs original
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",  # "cosine"
    ANNEAL_POINT=0.72,
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=8e-4, weight_decay=0.01),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
)

DATASETS = dict(
    TRAIN = ["cuparch_train_pbr"],
#    TRAIN = ["train"],
    TEST = ["cuparch_bop_test"],
    DET_FILES_TEST=("datasets/BOP_DATASETS/cuparch/test/test_bboxes/yolox_x_640_cuparch.json",),
    DET_TOPK_PER_OBJ=100,
)

DATALOADER = dict(
    # Number of data loading threads
    NUM_WORKERS=1,
    FILTER_VISIB_THR=0.3,
)

MODEL = dict(
    LOAD_DETS_TEST=True,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    BBOX_TYPE="AMODAL_CLIP",  # VISIB or AMODAL
    POSE_NET=dict(
        NAME="GDRN_Rho_flow",
        XYZ_ONLINE=True,
        XYZ_BP=False,
        NUM_CLASSES=1,
        BACKBONE=dict(
            FREEZE=False,
            PRETRAINED="timm",
            INIT_CFG=dict(
                type="timm/convnext_base",
                pretrained=True,
                in_chans=3,
                features_only=True,
                out_indices=(3,),
            ),
        ),
        ## geo head: Mask, XYZ, Region
        GEO_HEAD=dict(
            FREEZE=False,
            INIT_CFG=dict(
                type="TopDownMaskFlowRhoRegionHead",
                #type="TopDownMaskFlowRhoHead",
                in_dim=1024,  # this is num out channels of backbone conv feature
                out_layer_shared=False,
            ),
            NUM_REGIONS=64,
            # XYZ_CLASS_AWARE=True,
            MASK_CLASS_AWARE=True,
            REGION_CLASS_AWARE=True,
            FLOW_CLASS_AWARE=True,
            RHO_CLASS_AWARE=True,
        ),
        PNP_NET=dict(
            INIT_CFG=dict(norm="GN", act="gelu"),
            REGION_ATTENTION=True,
            # WITH_2D_COORD=True,
            WITH_2D_COORD=False,
            ROT_TYPE="allo_rot6d",
            TRANS_TYPE="centroid_z",
        ),
        LOSS_CFG=dict(
            # flow loss ----------------------------
            FLOW_LOSS_TYPE="L1",  # L1 | CE_coor
            FLOW_LOSS_MASK_GT="visib",  # trunc | visib | obj
            FLOW_LW=1.0,
            # rho loss ----------------------------
            RHO_LOSS_TYPE="L1",  # L1 | CE_coor
            RHO_LOSS_MASK_GT="visib",  # trunc | visib | obj
            RHO_LW=1.0,
            # mask loss ---------------------------
            MASK_LOSS_TYPE="L1",  # L1 | BCE | CE
            MASK_LOSS_GT="trunc",  # trunc | visib | gt
            MASK_LW=1.0,
            # full mask loss ---------------------------
            FULL_MASK_LOSS_TYPE="L1",  # L1 | BCE | CE
            FULL_MASK_LW=1.0,
            # region loss -------------------------
            REGION_LOSS_TYPE="CE",  # CE
            REGION_LOSS_MASK_GT="visib",  # trunc | visib | obj
            REGION_LW=1.0,
            # pm loss --------------
            PM_LOSS_SYM=True,  # NOTE: sym loss
            PM_R_ONLY=True,  # only do R loss in PM
            PM_LW=1.0,
            # centroid loss -------
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=1.0,
            # z loss -----------
            Z_LOSS_TYPE="L1",
            Z_LW=1.0,
        ),
    ),
)

VAL = dict(
    DATASET_NAME="cuparch",
    SCRIPT_PATH="lib/pysixd/scripts/eval_pose_results_more.py",
    TARGETS_FILENAME="test_targets_bop19.json",
    ERROR_TYPES="vsd,mspd,mssd",
    #RENDERER_TYPE="cpp",  # cpp, python, egl
    RENDERER_TYPE="python",  # cpp, python, egl
    SPLIT="test",
    SPLIT_TYPE="",
    N_TOP=-1,  # SISO: 1, VIVO: -1 (for LINEMOD, 1/-1 are the same)
    EVAL_CACHED=False,  # if the predicted poses have been saved
    SCORE_ONLY=False,  # if the errors have been calculated
    EVAL_PRINT_ONLY=False,  # if the scores/recalls have been saved
    EVAL_PRECISION=False,  # use precision or recall
    USE_BOP=True,  # whether to use bop toolkit
)

TEST = dict(EVAL_PERIOD=0, VIS=False, TEST_BBOX_TYPE="est")  # gt | est
