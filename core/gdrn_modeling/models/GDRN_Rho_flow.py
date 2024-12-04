import copy
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.solver_utils import build_optimizer_with_params
from detectron2.utils.events import get_event_storage
from mmcv.runner import load_checkpoint

from ..losses.coor_cross_entropy import CrossEntropyHeatmapLoss
from ..losses.l2_loss import L2Loss
from ..losses.mask_losses import weighted_ex_loss_probs, soft_dice_loss
from ..losses.pm_loss import PyPMLoss
from ..losses.rot_loss import angular_distance, rot_l2_loss
from .model_utils import (
    compute_mean_re_te,
    get_neck,
    get_geo_head,
    get_mask_prob,
    get_pnp_net,
    get_rot_mat,
    get_xyz_mask_region_out_dim,
    get_xyz_mask_flow_rho_region_out_dim,
)
from .pose_from_pred import pose_from_pred
from .pose_from_pred_centroid_z import pose_from_pred_centroid_z
from .pose_from_pred_centroid_z_abs import pose_from_pred_centroid_z_abs
from .net_factory import BACKBONES
from core.utils.my_checkpoint import load_timm_pretrained

logger = logging.getLogger(__name__)


def denormalize_image(image):
    # CHW
    pixel_mean = np.array([0.0, 0.0, 0.0]).reshape(-1, 1, 1)
    pixel_std = np.array([255.0, 255.0, 255.0]).reshape(-1, 1, 1)
    return image * pixel_std + pixel_mean

def denormalize_mask(image):
    # CHW
    pixel_mean = np.array([0.0]).reshape(-1, 1, 1)
    pixel_std = np.array([255.0]).reshape(-1, 1, 1)
    return image * pixel_std[0] + pixel_mean[0]


def normalize_image(image):
    # CHW
    pixel_mean = np.array([0.0, 0.0, 0.0]).reshape(-1, 1, 1)
    pixel_std = np.array([255.0, 255.0, 255.0]).reshape(-1, 1, 1)
    return (image - pixel_mean) / pixel_std



def compute_matting(input_mask, input_flow, input_rho, background, debug=False):
    import cv2
    import matplotlib
    
    final_tensor = torch.empty((input_mask.shape[0], 3, 64, 64))
    # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_input/output_mask.npy",input_mask.cpu().detach().numpy())
    # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_input/output_flow.npy",input_flow.cpu().detach().numpy())
    # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_input/output_rho.npy",input_rho.cpu().detach().numpy())
    # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_input/background.npy",background)
    
    for i in range(len(input_flow)):
        
        bg = background[i]
        rf = input_flow[i].cpu().detach().numpy()
        rho = input_rho[i].cpu().detach().numpy()
        mask = input_mask[i].cpu().detach().numpy()


        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_input/mask"+str(i)+".npy",mask)
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_input/rf"+str(i)+".npy",rf)
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_input/rho"+str(i)+".npy",rho)
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_input/bg"+str(i)+".npy",bg)
        
        rho = denormalize_image(rho).transpose(1,2,0).astype("uint8")
        rf = denormalize_image(rf).transpose(1,2,0).astype("uint8")
        mask = denormalize_mask(mask)
        # mask = np.invert(mask)
        if debug == True:
            mask = mask[0]
            mask[mask<0]=0
        mask = np.invert(mask.astype(np.uint8))
        rf = rf.astype(np.uint8)
        rho = rho.astype(np.uint8)

        
        y_shape, x_shape, _ = rf.shape
        bg = cv2.resize(bg[:480, :480, :], (x_shape, y_shape))
        x_grid = np.tile(np.linspace(0, x_shape-1, x_shape), (x_shape, 1)).astype(int)
        y_grid = np.tile(np.linspace(0, y_shape-1, y_shape), (y_shape, 1)).T.astype(int)
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_input/bg_resize"+str(i)+".npy",bg)
        #with open("/hdd/test_matting/000127_4_flow.flo", 'r') as f:
        # 	rf_bin = np.fromfile(f, dtype=np.int16)
    	
        #rf_bin = rf_bin[6:].reshape([512, 512, 2])
        #x_off = rf_bin[:, :, 1]
        #y_off = rf_bin[:, :, 0]

        # invert flowwithRho
        rho = 255 - rho
        rf = (rf / (rho)) 
        rho = rho/255

        #rf = cv2.cvtColor(rf, cv2.COLOR_BGR2HSV)
        rf = matplotlib.colors.rgb_to_hsv(rf*255)

        F_dir = rf[:, :, 0]
        F_mag = rf[:, :, 1]
        
        F_mag[np.isnan(F_mag)] = 0.0
        F_dir[np.isnan(F_dir)] = 0.0
    
        #F_mag = np.ones_like(F_mag)
    
        # invert flow_color[:,:,1] = F_mag / (F_mag.shape[0]*0.5)
        F_mag = F_mag * (y_shape * 0.5) # already pixel distance?
    
        # invert flow_color[:,:,0] = (F_dir+np.pi) / (2 * np.pi)
        F_dir = (F_dir * (2.0 * np.pi))  - np.pi
    
        # invert F_dir = np.arctan2(F_dy, F_dx)
        # https://stackoverflow.com/questions/11394706/inverse-of-math-atan2
        # local dx = len * cos(theta)
        # local dy = len * sin(theta)
        x_off = F_mag * np.sin(F_dir)
        y_off = F_mag * np.cos(F_dir)
        
        
        # print("!!!!!!!! x_off: ", x_off.shape)
        # print("\n !!!!!!!! mask: ", mask.shape)

        
        x_off[mask >= 200] = 0
        y_off[mask >= 200] = 0
        # x_off[mask >= 1] = 0
        # y_off[mask >= 1] = 0
    
        x_off += x_grid
        y_off += y_grid * y_shape
    
        corr = x_off + y_off
    
        corr = corr.astype(np.int32)
    
        np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_input/corr"+str(i)+".npy",corr)
        bg_flat = bg.reshape((y_shape * x_shape, 3))
        
        # refractive flow
        
        if np.max(corr.flatten()) >= len(bg_flat):
            
            corr_flatten = np.clip(corr.flatten(),0, len(bg_flat)-1)
            comp = bg_flat[corr_flatten, :]
            
        else:
            
            comp = bg_flat[corr.flatten(), :]
            
        # try:
        #     comp = bg_flat[corr.flatten(), :]
        # except IndexError as e:
        #     print(f"{e}")
        #     print("\n ********************************************************** INDEX: ", i)
        #     print("\n ********************************************************** DEBUG: ", debug)


        # comp = bg_flat[corr.flatten(), :] ########################################## ORIGINAL ONE #####################
        comp = comp.reshape([y_shape, x_shape, 3])
        mask_rep = np.repeat((mask)[:, :, np.newaxis], axis=2, repeats=3)
        comp = np.where(mask_rep <= 200, comp, bg)
    
        #cv2.imshow('rf applied', comp)
        #cv2.waitKey(0)
    
        #attenuation
        #rho[mask_rep >= 200] = 0 
        comp = comp * (1 - rho) + rho * 255 * (1 - rho) # somehow
        
        np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/comp_pre255.npy",comp)
        # comp = comp/255
        
        comp = comp.astype(np.uint8)

        # np.save("/PhD/WORK/gdrn/debug_output/comp.npy",comp)
        comp = (normalize_image(comp.transpose(2,1,0))).transpose(0,2,1)
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/comp.npy",comp)
        # print("COMP: ",comp.shape)
        # print("final_tensor: ",final_tensor.shape)

        
        final_tensor[i] = torch.tensor(comp)
        
        
    # torch.stack([d["flow_img"] for d in data], dim=0).to(device, non_blocking=True)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! GDRN_RHO compute_matting !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # import sys
    # sys.exit()
    return final_tensor.to("cuda", non_blocking=True)


class GDRN_Rho_flow(nn.Module):
    def __init__(self, cfg, backbone, geo_head_net, neck=None, pnp_net=None):
        super().__init__()
        assert cfg.MODEL.POSE_NET.NAME == "GDRN_Rho_flow", cfg.MODEL.POSE_NET.NAME
        self.backbone = backbone
        self.neck = neck
        
        self.count = 1
        
        self.geo_head_net = geo_head_net
        self.pnp_net = pnp_net

        self.cfg = cfg
        # self.xyz_out_dim, self.mask_out_dim, self.region_out_dim = get_xyz_mask_region_out_dim(cfg)
        ####################################################### FOR ABALATION STUDIES ######################################################################################
        self.flow_out_dim, self.rho_out_dim, self.mask_out_dim, self.region_out_dim = get_xyz_mask_flow_rho_region_out_dim(cfg)
        # self.mask_out_dim, self.region_out_dim = get_xyz_mask_flow_rho_region_out_dim(cfg)
        ####################################################### FOR ABALATION STUDIES ######################################################################################
        
        # uncertainty multi-task loss weighting
        # https://github.com/Hui-Li/multi-task-learning-example-PyTorch/blob/master/multi-task-learning-example-PyTorch.ipynb
        # a = log(sigma^2)
        # L*exp(-a) + a  or  L*exp(-a) + log(1+exp(a))
        # self.log_vars = nn.Parameter(torch.tensor([0, 0], requires_grad=True, dtype=torch.float32).cuda())
        # yapf: disable
        if cfg.MODEL.POSE_NET.USE_MTL:
            # self.loss_names = [
            #     "mask", "coor_x", "coor_y", "coor_z", "coor_x_bin", "coor_y_bin", "coor_z_bin", "region",
            #     "PM_R", "PM_xy", "PM_z", "PM_xy_noP", "PM_z_noP", "PM_T", "PM_T_noP",
            #     "centroid", "z", "trans_xy", "trans_z", "trans_LPnP", "rot", "bind",
            # ]
            
            self.loss_names = [
                "mask", "rho", "flow", "region",
                "PM_R", "PM_xy", "PM_z", "PM_xy_noP", "PM_z_noP", "PM_T", "PM_T_noP",
                "centroid", "z", "trans_xy", "trans_z", "trans_LPnP", "rot", "bind",
            ]
            ####################################################### FOR ABALATION STUDIES ######################################################################################
            # self.loss_names = [
            #     "mask", "region",
            #     "PM_R", "PM_xy", "PM_z", "PM_xy_noP", "PM_z_noP", "PM_T", "PM_T_noP",
            #     "centroid", "z", "trans_xy", "trans_z", "trans_LPnP", "rot", "bind",
            # ]
            ####################################################### FOR ABALATION STUDIES ######################################################################################
            for loss_name in self.loss_names:
                self.register_parameter(
                    f"log_var_{loss_name}", nn.Parameter(torch.tensor([0.0], requires_grad=True, dtype=torch.float32))
                )
        # yapf: enable
    count = 1
    def forward(
        self,
        x,
        roi_file_path = None,
        gt_flow_img=None,
        gt_rho_img=None,
        gt_mask_trunc=None,
        gt_mask_visib=None,
        gt_mask_obj=None,
        gt_mask_full=None,
        gt_region=None,
        gt_ego_rot=None,
        gt_points=None,
        sym_infos=None,
        gt_trans=None,
        gt_trans_ratio=None,
        roi_classes=None,
        roi_coord_2d=None,
        roi_coord_2d_rel=None,
        roi_cams=None,
        roi_centers=None,
        roi_whs=None,
        roi_extents=None,
        resize_ratios=None,
        do_loss=False,
        background=False

    ): 
        # # # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_debug/x"+str(self.count)+".npy",x.cpu().detach().numpy())
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_debug2/x.npy",x.cpu().detach().numpy())
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_debug2/gt_flow_img.npy",gt_flow_img.cpu().detach().numpy())
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_debug2/gt_rho_img.npy",gt_rho_img.cpu().detach().numpy())
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_debug2/gt_mask_trunc.npy",gt_mask_trunc.cpu().detach().numpy())
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_debug2/gt_mask_obj.npy",gt_mask_obj.cpu().detach().numpy())
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_debug2/gt_mask_visib.npy",gt_mask_visib.cpu().detach().numpy())
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_debug2/gt_mask_full.npy",gt_mask_full.cpu().detach().numpy())
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_debug2/gt_region.npy",gt_region.cpu().detach().numpy())
        # # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_debug/roi_file_path.npy",roi_file_path)
        # # #np.save("/RoadmapWithStefan/gdrnpp_bop2022/debug/gt_ego_rot.npy",gt_ego_rot.cpu().detach().numpy())
        # # #np.save("/RoadmapWithStefan/gdrnpp_bop2022/debug/gt_trans.npy",gt_trans.cpu().detach().numpy())
        # # #np.save("/RoadmapWithStefan/gdrnpp_bop2022/debug/roi_coord_2d.npy",roi_coord_2d.cpu().detach().numpy())
        # # # np.save("/RoadmapWithStefan/gdrnpp_bop2022/debug/roi_coord_2d_rel.npy",roi_coord_2d_rel)
        # # # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output/background.npy",background)
        # # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! GDRN_RHO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # # import sys
        # # sys.exit()

        cfg = self.cfg
        net_cfg = cfg.MODEL.POSE_NET
        g_head_cfg = net_cfg.GEO_HEAD
        pnp_net_cfg = net_cfg.PNP_NET

        device = x.device
        bs = x.shape[0]
        num_classes = net_cfg.NUM_CLASSES
        out_res = net_cfg.OUTPUT_RES

        # x.shape [bs, 3, 256, 256]
        conv_feat = self.backbone(x)  # [bs, c, 8, 8]
        if self.neck is not None:
            conv_feat = self.neck(conv_feat)
        mask, region, flow, rho = self.geo_head_net(conv_feat)
        ####################################################### FOR ABALATION STUDIES ######################################################################################
        # mask, region = self.geo_head_net(conv_feat)
        ####################################################### FOR ABALATION STUDIES ######################################################################################


        # if g_head_cfg.XYZ_CLASS_AWARE:
        #     assert roi_classes is not None
        #     coor_x = coor_x.view(bs, num_classes, self.xyz_out_dim // 3, out_res, out_res)
        #     coor_x = coor_x[torch.arange(bs).to(device), roi_classes]
        #     coor_y = coor_y.view(bs, num_classes, self.xyz_out_dim // 3, out_res, out_res)
        #     coor_y = coor_y[torch.arange(bs).to(device), roi_classes]
        #     coor_z = coor_z.view(bs, num_classes, self.xyz_out_dim // 3, out_res, out_res)
        #     coor_z = coor_z[torch.arange(bs).to(device), roi_classes]

        if g_head_cfg.MASK_CLASS_AWARE:
            assert roi_classes is not None
            mask = mask.view(bs, num_classes, self.mask_out_dim, out_res, out_res)
            mask = mask[torch.arange(bs).to(device), roi_classes]

        if g_head_cfg.REGION_CLASS_AWARE:
            assert roi_classes is not None
            region = region.view(bs, num_classes, self.region_out_dim, out_res, out_res)
            region = region[torch.arange(bs).to(device), roi_classes]

        ####################################################### FOR ABALATION STUDIES ######################################################################################
        if g_head_cfg.FLOW_CLASS_AWARE:
            assert roi_classes is not None
            flow = flow.view(bs, num_classes, self.flow_out_dim, out_res, out_res)
            flow = flow[torch.arange(bs).to(device), roi_classes]            
        
        if g_head_cfg.RHO_CLASS_AWARE:
            assert roi_classes is not None
            rho = rho.view(bs, num_classes, self.rho_out_dim, out_res, out_res)
            rho = rho[torch.arange(bs).to(device), roi_classes]
        ####################################################### FOR ABALATION STUDIES ######################################################################################

        # # -----------------------------------------------
        # # -----------------------------------------------
        # # get rot and trans from pnp_net
        # # NOTE: use softmax for bins (the last dim is bg)
        # if coor_x.shape[1] > 1 and coor_y.shape[1] > 1 and coor_z.shape[1] > 1:
        #     coor_x_softmax = F.softmax(coor_x[:, :-1, :, :], dim=1)
        #     coor_y_softmax = F.softmax(coor_y[:, :-1, :, :], dim=1)
        #     coor_z_softmax = F.softmax(coor_z[:, :-1, :, :], dim=1)
        #     coor_feat = torch.cat([coor_x_softmax, coor_y_softmax, coor_z_softmax], dim=1)
        # else:
        #     coor_feat = torch.cat([coor_x, coor_y, coor_z], dim=1)  # BCHW

        # -----------------------------------------------
        # if pnp_net_cfg.WITH_2D_COORD:
        #     if pnp_net_cfg.COORD_2D_TYPE == "rel":
        #         assert roi_coord_2d_rel is not None
        #         coor_feat = torch.cat([coor_feat, roi_coord_2d_rel], dim=1)
        #     else:  # default abs
        #         assert roi_coord_2d is not None
        #         coor_feat = torch.cat([coor_feat, roi_coord_2d], dim=1)

        # NOTE: for region, the 1st dim is bg
        region_softmax = F.softmax(region[:, 1:, :, :], dim=1)
        

        mask_atten = None
        if pnp_net_cfg.MASK_ATTENTION != "none":
            mask_atten = get_mask_prob(mask, mask_loss_type=net_cfg.LOSS_CFG.MASK_LOSS_TYPE)

        region_atten = None
        if pnp_net_cfg.REGION_ATTENTION:
            region_atten = region_softmax

        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_debug/flow.npy",flow.cpu().detach().numpy())
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_debug/rho.npy",rho.cpu().detach().numpy())
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_debug/mask.npy",mask.cpu().detach().numpy())
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_debug/mask_atten.npy",mask_atten.cpu().detach().numpy())
        self.count = self.count + 1
        
        pred_rot_, pred_t_ = self.pnp_net(
            region=region_atten, flow=flow, rho=rho, extents=roi_extents, mask_attention=mask_atten
        )
        
        ####################################################### FOR ABALATION STUDIES ######################################################################################
        # pred_rot_, pred_t_ = self.pnp_net(
        #     region=region_atten, extents=roi_extents, mask_attention=mask_atten
        # )
        ####################################################### FOR ABALATION STUDIES ######################################################################################

        # convert pred_rot to rot mat -------------------------
        rot_type = pnp_net_cfg.ROT_TYPE
        pred_rot_m = get_rot_mat(pred_rot_, rot_type)
        
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output/gt_region_x_mask_img_paper.npy",(( gt_region * gt_mask_obj.long() ).cpu()).detach().numpy())
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output/gt_flow_x_mask_img_paper.npy",((gt_flow_img*gt_mask_obj[:, None]).cpu()).detach().numpy()) 
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output/gt_region_x_mask_img_paper.npy",(( gt_region * gt_mask_obj.long() ).cpu()).detach().numpy())
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output/gt_region_x_mask_img_paper.npy",(( gt_region * gt_mask_obj.long() ).cpu()).detach().numpy())
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output/gt_flow_x_mask_img_paper.npy",((flow*gt_mask_obj[:, None]).cpu()).detach().numpy()) 
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output/gt_region_x_mask_img_paper.npy",(( gt_region * gt_mask_obj.long() ).cpu()).detach().numpy())
        np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/out_flow.npy",(flow.cpu()).detach().numpy())
        np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/out_rho.npy",(rho.cpu()).detach().numpy())
        np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/x.npy",x.cpu().detach().numpy())
        # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output/gt_flow_img.npy",gt_flow_img.cpu().detach().numpy())
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! GDRN_RHO LOSS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # import sys
        # sys.exit()  

        # convert pred_rot_m and pred_t to ego pose -----------------------------
        if pnp_net_cfg.TRANS_TYPE == "centroid_z":
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,
                roi_centers=roi_centers,
                resize_ratios=resize_ratios,
                roi_whs=roi_whs,
                eps=1e-4,
                is_allo="allo" in rot_type,
                z_type=pnp_net_cfg.Z_TYPE,
                # is_train=True
                is_train=do_loss,  # TODO: sometimes we need it to be differentiable during test
            )
        elif pnp_net_cfg.TRANS_TYPE == "centroid_z_abs":
            # abs 2d obj center and abs z
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z_abs(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,
                eps=1e-4,
                is_allo="allo" in rot_type,
                # is_train=True
                is_train=do_loss,  # TODO: sometimes we need it to be differentiable during test
            )
        elif pnp_net_cfg.TRANS_TYPE == "trans":
            pred_ego_rot, pred_trans = pose_from_pred(
                pred_rot_m, pred_t_, eps=1e-4, is_allo="allo" in rot_type, is_train=do_loss
            )
        else:
            raise ValueError(f"Unknown trans type: {pnp_net_cfg.TRANS_TYPE}")
            
        # DEBUG = True
        if not do_loss:  # test
        # if DEBUG:  # test
            out_dict = {"rot": pred_ego_rot, "trans": pred_trans}
            if cfg.TEST.USE_PNP or cfg.TEST.SAVE_RESULTS_ONLY:
                # TODO: move the pnp/ransac inside forward
        ####################################################### FOR ABALATION STUDIES ######################################################################################
                out_dict.update({"mask": mask, "flow": flow, "rho": rho, "region": region})
                # out_dict.update({"mask": mask, "region": region})
        ####################################################### FOR ABALATION STUDIES ######################################################################################
        else:
            out_dict = {}
        ####################################################### FOR ABALATION STUDIES ######################################################################################

            # assert (
            #     (gt_flow_img is not None)
            #     and (gt_rho_img is not None)
            #     and (gt_trans is not None)
            #     and (gt_trans_ratio is not None)
            #     and (gt_region is not None)
            # )
            
            assert (
                # (gt_flow_img is not None)
                # and (gt_rho_img is not None)
                (gt_trans is not None)
                and (gt_trans_ratio is not None)
                and (gt_region is not None)
            )
            
        ####################################################### FOR ABALATION STUDIES ######################################################################################

            mean_re, mean_te = compute_mean_re_te(pred_trans, pred_rot_m, gt_trans, gt_ego_rot)
            vis_dict = {
                "vis/error_R": mean_re,
                "vis/error_t": mean_te * 100,  # cm
                "vis/error_tx": np.abs(pred_trans[0, 0].detach().item() - gt_trans[0, 0].detach().item()) * 100,  # cm
                "vis/error_ty": np.abs(pred_trans[0, 1].detach().item() - gt_trans[0, 1].detach().item()) * 100,  # cm
                "vis/error_tz": np.abs(pred_trans[0, 2].detach().item() - gt_trans[0, 2].detach().item()) * 100,  # cm
                "vis/tx_pred": pred_trans[0, 0].detach().item(),
                "vis/ty_pred": pred_trans[0, 1].detach().item(),
                "vis/tz_pred": pred_trans[0, 2].detach().item(),
                "vis/tx_net": pred_t_[0, 0].detach().item(),
                "vis/ty_net": pred_t_[0, 1].detach().item(),
                "vis/tz_net": pred_t_[0, 2].detach().item(),
                "vis/tx_gt": gt_trans[0, 0].detach().item(),
                "vis/ty_gt": gt_trans[0, 1].detach().item(),
                "vis/tz_gt": gt_trans[0, 2].detach().item(),
                "vis/tx_rel_gt": gt_trans_ratio[0, 0].detach().item(),
                "vis/ty_rel_gt": gt_trans_ratio[0, 1].detach().item(),
                "vis/tz_rel_gt": gt_trans_ratio[0, 2].detach().item(),
            }

            loss_dict = self.gdrn_loss(
                cfg=self.cfg,
                out_mask=mask,
                gt_mask_trunc=gt_mask_trunc,
                gt_mask_visib=gt_mask_visib,
                gt_mask_obj=gt_mask_obj,
        ####################################################### FOR ABALATION STUDIES ######################################################################################
                out_flow=flow,
                out_rho=rho,
        ####################################################### FOR ABALATION STUDIES ######################################################################################
                gt_flow_img=gt_flow_img,
                gt_rho_img=gt_rho_img,
                background =  background,
                out_region=region,
                gt_region=gt_region,
                out_trans=pred_trans,
                gt_trans=gt_trans,
                out_rot=pred_ego_rot,
                gt_rot=gt_ego_rot,
                out_centroid=pred_t_[:, :2],  # TODO: get these from trans head
                out_trans_z=pred_t_[:, 2],
                gt_trans_ratio=gt_trans_ratio,
                gt_points=gt_points,
                sym_infos=sym_infos,
                extents=roi_extents,
                # roi_classes=roi_classes,
            )

            if net_cfg.USE_MTL:
                for _name in self.loss_names:
                    if f"loss_{_name}" in loss_dict:
                        vis_dict[f"vis_lw/{_name}"] = torch.exp(-getattr(self, f"log_var_{_name}")).detach().item()
            for _k, _v in vis_dict.items():
                if "vis/" in _k or "vis_lw/" in _k:
                    if isinstance(_v, torch.Tensor):
                        _v = _v.item()
                    vis_dict[_k] = _v
            storage = get_event_storage()
            storage.put_scalars(**vis_dict)

            return out_dict, loss_dict
        return out_dict

    def gdrn_loss(
        self,
        cfg,
        out_mask,
        gt_mask_trunc,
        gt_mask_visib,
        gt_mask_obj,
        ####################################################### FOR ABALATION STUDIES ######################################################################################
        out_flow,
        out_rho,
        ####################################################### FOR ABALATION STUDIES ######################################################################################
        gt_flow_img,
        gt_rho_img,
        background,
        out_region,
        gt_region,
        out_rot=None,
        gt_rot=None,
        out_trans=None,
        gt_trans=None,
        out_centroid=None,
        out_trans_z=None,
        gt_trans_ratio=None,
        gt_points=None,
        sym_infos=None,
        extents=None,
    ):
        net_cfg = cfg.MODEL.POSE_NET
        g_head_cfg = net_cfg.GEO_HEAD
        pnp_net_cfg = net_cfg.PNP_NET
        loss_cfg = net_cfg.LOSS_CFG

        loss_dict = {}

        gt_masks = {"trunc": gt_mask_trunc, "visib": gt_mask_visib, "obj": gt_mask_obj}

        # INTERMEDIATE REPRESENTATION REFLECTIVE ############################################

        # ####################################################### FOR ABALATION STUDIES ######################################################################################
        # flow loss ----------------------------------
        if not g_head_cfg.FREEZE:
            flow_loss_type = loss_cfg.FLOW_LOSS_TYPE
            gt_mask_flow = gt_masks[loss_cfg.FLOW_LOSS_MASK_GT]
            if flow_loss_type == "L1":
                loss_func = nn.L1Loss(reduction="sum")
                #loss_dict["loss_flow"] = loss_func(
                #    out_flow * gt_mask_flow[:, None], gt_flow_img[:, 0:1] * gt_mask_flow[:, None]
                #) / gt_mask_flow.sum().float().clamp(min=1.0)
                loss_dict["loss_flow"] = loss_func(
                        out_flow * gt_mask_flow[:, None], gt_flow_img*gt_mask_flow[:, None]
              ) / gt_mask_flow.sum().float().clamp(min=1.0)

            # elif flow_loss_type == "CE_coor":
            #     gt_xyz_bin = gt_xyz_bin.long()
            #     loss_func = CrossEntropyHeatmapLoss(reduction="sum", weight=None)  # g_head_cfg.XYZ_BIN+1
            #     loss_dict["loss_flow"] = loss_func(
            #         out_flow * gt_mask_flow[:, None], gt_flow_img[:, 0] * gt_mask_flow.long()
            #     ) / gt_mask_flow.sum().float().clamp(min=1.0)
            else:
                raise NotImplementedError(f"unknown flow loss type: {flow_loss_type}")
            loss_dict["loss_flow"] *= loss_cfg.FLOW_LW
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/out_flow.npy",(out_flow.cpu()).detach().numpy())
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/gt_mask_flow.npy",((gt_mask_flow[:, None]).cpu()).detach().numpy())
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/gt_flow_img.npy",((gt_flow_img.cpu()).detach().numpy()))
            # np.save("/RoadmapWithStefan/gdrnpp_bop2022/gt_mask_trunc_dictionary.npy",(gt_masks['trunc'].cpu()).detach().numpy())
            # np.save("/RoadmapWithStefan/gdrnpp_bop2022/gt_mask_visib_dictionary.npy",(gt_masks['visib'].cpu()).detach().numpy())
            # np.save("/RoadmapWithStefan/gdrnpp_bop2022/gt_mask_obj_dictionary.npy",(gt_masks['obj'].cpu()).detach().numpy())
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/out_flow_x_mask_img.npy",((( out_flow * gt_mask_flow[:, None]).cpu()).detach().numpy()))
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/gt_flow_x_mask_img.npy",((gt_flow_img*gt_mask_flow[:, None]).cpu()).detach().numpy()) 
        # ####################################################### FOR ABALATION STUDIES ######################################################################################
        
        # rho loss ----------------------------------
        if not g_head_cfg.FREEZE:
            rho_loss_type = loss_cfg.RHO_LOSS_TYPE
            gt_mask_rho = gt_masks[loss_cfg.RHO_LOSS_MASK_GT]
            if rho_loss_type == "L1":
                loss_func = nn.L1Loss(reduction="sum")
                #loss_dict["loss_rho"] = loss_func(
                #    out_rho * gt_mask_rho[:, None], gt_rho_img[:, 0:1] * gt_mask_rho[:, None]
                #) / gt_mask_rho.sum().float().clamp(min=1.0)

                loss_dict["loss_rho"] = loss_func(
                        out_rho * gt_mask_rho[:, None], gt_rho_img * gt_mask_rho[:, None]
                ) / gt_mask_rho.sum().float().clamp(min=1.0)

            # elif rho_loss_type == "CE_coor":
            #     gt_xyz_bin = gt_xyz_bin.long()
            #     loss_func = CrossEntropyHeatmapLoss(reduction="sum", weight=None)  # g_head_cfg.XYZ_BIN+1
            #     loss_dict["loss_coor_x"] = loss_func(
            #         out_x * gt_mask_xyz[:, None], gt_xyz_bin[:, 0] * gt_mask_xyz.long()
            #     ) / gt_mask_xyz.sum().float().clamp(min=1.0)
            else:
                raise NotImplementedError(f"unknown rho loss type: {rho_loss_type}")
            loss_dict["loss_rho"] *= loss_cfg.RHO_LW
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/out_rho.npy",(out_rho.cpu()).detach().numpy())
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/gt_rho_img.npy",(gt_rho_img.cpu()).detach().numpy())
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/gt_mask_rho.npy",(gt_mask_rho.cpu()).detach().numpy())
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/out_rho_x_mask_img.npy",(( out_rho * gt_mask_rho[:, None]).cpu()).detach().numpy())
        # # INTERMEDIATE REPRESENTATION REFLECTIVE ############################################

        # ####################################################### FOR ABALATION STUDIES ######################################################################################
        # mask loss ----------------------------------
        if not g_head_cfg.FREEZE:
            mask_loss_type = loss_cfg.MASK_LOSS_TYPE
            gt_mask = gt_masks[loss_cfg.MASK_LOSS_GT]

#            for i_ in range(len(gt_mask)): ################# ADDED BY ME #######################
#                gt_mask[i_,:] = (gt_mask[i_,:].clone()).T

            if mask_loss_type == "L1":
                loss_dict["loss_mask"] = nn.L1Loss(reduction="mean")(out_mask[:, 0, :, :], gt_mask)
            elif mask_loss_type == "BCE":
                loss_dict["loss_mask"] = nn.BCEWithLogitsLoss(reduction="mean")(out_mask[:, 0, :, :], gt_mask)
            elif mask_loss_type == "RW_BCE":
                loss_dict["loss_mask"] = weighted_ex_loss_probs(
                    torch.sigmoid(out_mask[:, 0, :, :]), gt_mask, weight=None
                )
            elif mask_loss_type == "dice":
                loss_dict["loss_mask"] = soft_dice_loss(
                    torch.sigmoid(out_mask[:, 0, :, :]), gt_mask, eps=0.002, reduction="mean"
                )
            elif mask_loss_type == "CE":
                loss_dict["loss_mask"] = nn.CrossEntropyLoss(reduction="mean")(out_mask, gt_mask.long())
            else:
                raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")
            loss_dict["loss_mask"] *= loss_cfg.MASK_LW
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/out_mask.npy",( (out_mask[:, 0, :, :]).cpu()).detach().numpy())
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/gt_mask.npy",(gt_mask.cpu()).detach().numpy())

        # roi region loss --------------------
        if not g_head_cfg.FREEZE:
            region_loss_type = loss_cfg.REGION_LOSS_TYPE
            gt_mask_region = gt_masks[loss_cfg.REGION_LOSS_MASK_GT]
            gt_mask_region = gt_mask_region.permute(0,2,1) ########################**************************############### CHANGE THIS ######################################################
            # # gt_mask_region.permute(0,2,1) ########################**************************############### CHANGE THIS ######################################################
            # np.save("/PhD/gdrnpp_bop2022_experimentation_2/debug_output/out_region.npy",(out_region.cpu()).detach().numpy())
            # np.save("/PhD/gdrnpp_bop2022_experimentation_2/debug_output/gt_region_img.npy",(gt_region.cpu()).detach().numpy())
            # np.save("/PhD/gdrnpp_bop2022_experimentation_2/debug_output/gt_mask_region.npy",(gt_mask_region.cpu()).detach().numpy())
            # np.save("/PhD/gdrnpp_bop2022_experimentation_2/debug_output/gt_region_x_mask_img.npy",(( gt_region * gt_mask_region.long() ).cpu()).detach().numpy())
            # np.save("/PhD/gdrnpp_bop2022_experimentation_2/debug_output/gt_region_x_mask_img_none.npy",(( gt_region * (gt_mask_region[:, None]).long() ).cpu()).detach().numpy())
            # np.save("/PhD/gdrnpp_bop2022_experimentation_2/debug_output/out_region_x_mask_img.npy",(( out_region * gt_mask_region[:, None] ).cpu()).detach().numpy())

            if region_loss_type == "CE":
                gt_region = gt_region.long()
                loss_func = nn.CrossEntropyLoss(reduction="sum", weight=None)  # g_head_cfg.XYZ_BIN+1
                loss_dict["loss_region"] = loss_func(
                    out_region * gt_mask_region[:, None], gt_region * gt_mask_region.long()
                ) / gt_mask_region.sum().float().clamp(min=1.0)
                # loss_dict["loss_region"] = loss_func(
                #     out_region * gt_mask_region[:, None], gt_region * (gt_mask_region[:, None]).long()
                # ) / gt_mask_region.sum().float().clamp(min=1.0)
            else:
                raise NotImplementedError(f"unknown region loss type: {region_loss_type}")
            loss_dict["loss_region"] *= loss_cfg.REGION_LW
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/out_region.npy",(out_region.cpu()).detach().numpy())
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/gt_region_img.npy",(gt_region.cpu()).detach().numpy())
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/gt_mask_region.npy",(gt_mask_region.cpu()).detach().numpy())
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/gt_region_x_mask_img.npy",(( gt_region * gt_mask_region.long() ).cpu()).detach().numpy())
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/out_region_x_mask_img.npy",(( out_region * gt_mask_region[:, None] ).cpu()).detach().numpy())
        # point matching loss ---------------
        if loss_cfg.PM_LW > 0:
            
            # print("loss_cfg.PM_LWloss_cfg.PM_LWloss_cfg.PM_LWloss_cfg.PM_LWloss_cfg.PM_LWloss_cfg.PM_LW: ", loss_cfg.PM_LW)
            assert (gt_points is not None) and (gt_trans is not None) and (gt_rot is not None)
            loss_func = PyPMLoss(
                loss_type=loss_cfg.PM_LOSS_TYPE,
                beta=loss_cfg.PM_SMOOTH_L1_BETA,
                reduction="mean",
                loss_weight=loss_cfg.PM_LW,
                norm_by_extent=loss_cfg.PM_NORM_BY_EXTENT,
                symmetric=loss_cfg.PM_LOSS_SYM,
                disentangle_t=loss_cfg.PM_DISENTANGLE_T,
                disentangle_z=loss_cfg.PM_DISENTANGLE_Z,
                t_loss_use_points=loss_cfg.PM_T_USE_POINTS,
                r_only=loss_cfg.PM_R_ONLY,
            )
            loss_pm_dict = loss_func(
                pred_rots=out_rot,
                gt_rots=gt_rot,
                points=gt_points,
                pred_transes=out_trans,
                gt_transes=gt_trans,
                extents=extents,
                sym_infos=sym_infos,
            )
            # print("loss_cfg.PM_LWloss_cfg.PM_LWloss_cfg.PM_LWloss_cfg.PM_LWloss_cfg.PM_LWloss_cfg.PM_LW: ", loss_pm_dict)
            loss_dict.update(loss_pm_dict)


        # roi region loss --------------------
        if not g_head_cfg.FREEZE:
            # ################################### MATTING LOSS ######################################################
            gt_matt = compute_matting(gt_mask, gt_flow_img, gt_rho_img, background)
            # print(" GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
            out_matt = compute_matting(out_mask, out_flow, out_rho, background, debug=True)
            
            # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/gt_matt.npy",(gt_matt.cpu()).detach().numpy())
            
            # np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/out_matt.npy",(out_matt.cpu()).detach().numpy())

        
        if not g_head_cfg.FREEZE:
            flow_loss_type = loss_cfg.FLOW_LOSS_TYPE
            gt_mask_flow = gt_masks[loss_cfg.FLOW_LOSS_MASK_GT]
            if flow_loss_type == "L1":
                loss_func = nn.L1Loss(reduction="sum")
                loss_dict["loss_matt"] = loss_func(
                        out_matt * gt_mask_flow[:, None], gt_matt*gt_mask_flow[:, None]
                ) / gt_mask_flow.sum().float().clamp(min=1.0)

            else:
                raise NotImplementedError(f"unknown matt loss type: {flow_loss_type}")
            loss_dict["loss_matt"] *= loss_cfg.FLOW_LW
            
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/out_matt.npy",(out_matt.cpu()).detach().numpy())
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/gt_matt.npy",(gt_matt.cpu()).detach().numpy())
            np.save("/PhD_Stuff/GDRN_TRANSPARENT/gdrn/debug_output2/gt_matt_x_mask_img.npy",( gt_matt*(gt_mask_flow[:, None]) ).cpu().detach().numpy())
        # ################################## MATTING LOSS ######################################################
        
        # rot_loss ----------
        if loss_cfg.ROT_LW > 0:
            if loss_cfg.ROT_LOSS_TYPE == "angular":
                loss_dict["loss_rot"] = angular_distance(out_rot, gt_rot)
            elif loss_cfg.ROT_LOSS_TYPE == "L2":
                loss_dict["loss_rot"] = rot_l2_loss(out_rot, gt_rot)
            else:
                raise ValueError(f"Unknown rot loss type: {loss_cfg.ROT_LOSS_TYPE}")

            loss_dict["loss_rot"] *= loss_cfg.ROT_LW

        # centroid loss -------------
        if loss_cfg.CENTROID_LW > 0:
            assert (
                pnp_net_cfg.TRANS_TYPE == "centroid_z"
            ), "centroid loss is only valid for predicting centroid2d_rel_delta"

            if loss_cfg.CENTROID_LOSS_TYPE == "L1":
                loss_dict["loss_centroid"] = nn.L1Loss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            elif loss_cfg.CENTROID_LOSS_TYPE == "L2":
                loss_dict["loss_centroid"] = L2Loss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            elif loss_cfg.CENTROID_LOSS_TYPE == "MSE":
                loss_dict["loss_centroid"] = nn.MSELoss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            else:
                raise ValueError(f"Unknown centroid loss type: {loss_cfg.CENTROID_LOSS_TYPE}")
            loss_dict["loss_centroid"] *= loss_cfg.CENTROID_LW

        # z loss ------------------
        if loss_cfg.Z_LW > 0:
            z_type = pnp_net_cfg.Z_TYPE
            if z_type == "REL":
                gt_z = gt_trans_ratio[:, 2]
            elif z_type == "ABS":
                gt_z = gt_trans[:, 2]
            else:
                raise NotImplementedError

            z_loss_type = loss_cfg.Z_LOSS_TYPE
            if z_loss_type == "L1":
                loss_dict["loss_z"] = nn.L1Loss(reduction="mean")(out_trans_z, gt_z)
            elif z_loss_type == "L2":
                loss_dict["loss_z"] = L2Loss(reduction="mean")(out_trans_z, gt_z)
            elif z_loss_type == "MSE":
                loss_dict["loss_z"] = nn.MSELoss(reduction="mean")(out_trans_z, gt_z)
            else:
                raise ValueError(f"Unknown z loss type: {z_loss_type}")
            loss_dict["loss_z"] *= loss_cfg.Z_LW

        # trans loss ------------------
        if loss_cfg.TRANS_LW > 0:
            if loss_cfg.TRANS_LOSS_DISENTANGLE:
                # NOTE: disentangle xy/z
                if loss_cfg.TRANS_LOSS_TYPE == "L1":
                    loss_dict["loss_trans_xy"] = nn.L1Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.L1Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif loss_cfg.TRANS_LOSS_TYPE == "L2":
                    loss_dict["loss_trans_xy"] = L2Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = L2Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif loss_cfg.TRANS_LOSS_TYPE == "MSE":
                    loss_dict["loss_trans_xy"] = nn.MSELoss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.MSELoss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                else:
                    raise ValueError(f"Unknown trans loss type: {loss_cfg.TRANS_LOSS_TYPE}")
                loss_dict["loss_trans_xy"] *= loss_cfg.TRANS_LW
                loss_dict["loss_trans_z"] *= loss_cfg.TRANS_LW
            else:
                if loss_cfg.TRANS_LOSS_TYPE == "L1":
                    loss_dict["loss_trans_LPnP"] = nn.L1Loss(reduction="mean")(out_trans, gt_trans)
                elif loss_cfg.TRANS_LOSS_TYPE == "L2":
                    loss_dict["loss_trans_LPnP"] = L2Loss(reduction="mean")(out_trans, gt_trans)

                elif loss_cfg.TRANS_LOSS_TYPE == "MSE":
                    loss_dict["loss_trans_LPnP"] = nn.MSELoss(reduction="mean")(out_trans, gt_trans)
                else:
                    raise ValueError(f"Unknown trans loss type: {loss_cfg.TRANS_LOSS_TYPE}")
                loss_dict["loss_trans_LPnP"] *= loss_cfg.TRANS_LW

        # bind loss (R^T@t)
        if loss_cfg.get("BIND_LW", 0.0) > 0.0:
            pred_bind = torch.bmm(out_rot.permute(0, 2, 1), out_trans.view(-1, 3, 1)).view(-1, 3)
            gt_bind = torch.bmm(gt_rot.permute(0, 2, 1), gt_trans.view(-1, 3, 1)).view(-1, 3)
            if loss_cfg.BIND_LOSS_TYPE == "L1":
                loss_dict["loss_bind"] = nn.L1Loss(reduction="mean")(pred_bind, gt_bind)
            elif loss_cfg.BIND_LOSS_TYPE == "L2":
                loss_dict["loss_bind"] = L2Loss(reduction="mean")(pred_bind, gt_bind)
            elif loss_cfg.CENTROID_LOSS_TYPE == "MSE":
                loss_dict["loss_bind"] = nn.MSELoss(reduction="mean")(pred_bind, gt_bind)
            else:
                raise ValueError(f"Unknown bind loss (R^T@t) type: {loss_cfg.BIND_LOSS_TYPE}")
            loss_dict["loss_bind"] *= loss_cfg.BIND_LW

        if net_cfg.USE_MTL:
            for _k in loss_dict:
                _name = _k.replace("loss_", "log_var_")
                cur_log_var = getattr(self, _name)
                loss_dict[_k] = loss_dict[_k] * torch.exp(-cur_log_var) + torch.log(1 + torch.exp(cur_log_var))
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! GDRN_RHO LOSS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # import sys
        # sys.exit()  
        return loss_dict


def build_model_optimizer(cfg, is_test=False):
    net_cfg = cfg.MODEL.POSE_NET
    backbone_cfg = net_cfg.BACKBONE

    params_lr_list = []
    # backbone
    init_backbone_args = copy.deepcopy(backbone_cfg.INIT_CFG)
    backbone_type = init_backbone_args.pop("type")
    if "timm/" in backbone_type or "tv/" in backbone_type:
        init_backbone_args["model_name"] = backbone_type.split("/")[-1]

    backbone = BACKBONES[backbone_type](**init_backbone_args)
    if backbone_cfg.FREEZE:
        for param in backbone.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {"params": filter(lambda p: p.requires_grad, backbone.parameters()), "lr": float(cfg.SOLVER.BASE_LR)}
        )

    # neck --------------------------------
    neck, neck_params = get_neck(cfg)
    params_lr_list.extend(neck_params)

    # geo head -----------------------------------------------------
    geo_head, geo_head_params = get_geo_head(cfg)
    params_lr_list.extend(geo_head_params)

    # pnp net -----------------------------------------------
    pnp_net, pnp_net_params = get_pnp_net(cfg)
    params_lr_list.extend(pnp_net_params)
    # build model
    model = GDRN_Rho_flow(cfg, backbone, neck=neck, geo_head_net=geo_head, pnp_net=pnp_net)
    if net_cfg.USE_MTL:
        params_lr_list.append(
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [_param for _name, _param in model.named_parameters() if "log_var" in _name],
                ),
                "lr": float(cfg.SOLVER.BASE_LR),
            }
        )

    # get optimizer
    if is_test:
        optimizer = None
    else:
        optimizer = build_optimizer_with_params(cfg, params_lr_list)

    if cfg.MODEL.WEIGHTS == "":
        ## backbone initialization
        backbone_pretrained = backbone_cfg.get("PRETRAINED", "")
        if backbone_pretrained == "":
            logger.warning("Randomly initialize weights for backbone!")
        elif backbone_pretrained in ["timm", "internal"]:
            # skip if it has already been initialized by pretrained=True
            logger.info("Check if the backbone has been initialized with its own method!")
            if backbone_pretrained == "timm":
                if init_backbone_args.pretrained and init_backbone_args.in_chans != 3:
                    load_timm_pretrained(
                        model.backbone, in_chans=init_backbone_args.in_chans, adapt_input_mode="custom", strict=False
                    )
                    logger.warning("override input conv weight adaptation of timm")
        else:
            # initialize backbone with official weights
            tic = time.time()
            logger.info(f"load backbone weights from: {backbone_pretrained}")
            load_checkpoint(model.backbone, backbone_pretrained, strict=False, logger=logger)
            logger.info(f"load backbone weights took: {time.time() - tic}s")

    model.to(torch.device(cfg.MODEL.DEVICE))
    return model, optimizer
