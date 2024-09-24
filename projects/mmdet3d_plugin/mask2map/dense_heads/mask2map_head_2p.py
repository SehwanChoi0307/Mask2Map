import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.runner import force_fp32
from mmcv.cnn import Linear, bias_init_with_prob
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.core import multi_apply, reduce_mean, build_assigner
from mmcv.utils import TORCH_VERSION, digit_version


def denormalize_3d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    new_pts[..., 2:3] = pts[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    return new_pts


def normalize_3d_pts(pts, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    patch_z = pc_range[5] - pc_range[2]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    new_pts[..., 2:3] = pts[..., 2:3] - pc_range[2]
    factor = pts.new_tensor([patch_w, patch_h, patch_z])
    normalized_pts = new_pts / factor
    return normalized_pts


def normalize_2d_bbox(bboxes, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[..., 0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[..., 1:2] = cxcywh_bboxes[..., 1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes


def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


def denormalize_2d_bbox(bboxes, pc_range):
    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = bboxes[..., 0::2] * (pc_range[3] - pc_range[0]) + pc_range[0]
    bboxes[..., 1::2] = bboxes[..., 1::2] * (pc_range[4] - pc_range[1]) + pc_range[1]

    return bboxes


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    return new_pts


@HEADS.register_module()
class Mask2MapHead_2Phase(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(
        self,
        *args,
        with_box_refine=False,
        as_two_stage=False,
        transformer=None,
        bbox_coder=None,
        code_weights=None,
        bev_h=30,
        bev_w=30,
        num_vec_one2one=50,
        num_pts_per_vec=2,
        num_pts_per_gt_vec=2,
        dir_interval=1,
        aux_seg=dict(use_aux_seg=False, bev_seg=False, pv_seg=False, seg_classes=1, feat_down_sample=32),
        z_cfg=dict(pred_z_flag=False, gt_z_flag=False),
        loss_pts=dict(type="ChamferDistance", loss_src_weight=1.0, loss_dst_weight=1.0),
        loss_seg=dict(type="SimpleLoss", pos_weight=2.13, loss_weight=1.0),
        loss_pv_seg=dict(type="SimpleLoss", pos_weight=2.13, loss_weight=1.0),
        loss_dir=dict(type="PtsDirCosLoss", loss_weight=2.0),
        loss_segm_cls=None,
        loss_segm_mask=None,
        loss_segm_dice=None,
        dn_enabled=False,
        cls_join=True,
        tr_cls_join=False,
        dn_weight=1,
        pv_seg_dim=None,
        **kwargs,
    ):
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer["as_two_stage"] = self.as_two_stage
        if "code_size" in kwargs:
            self.code_size = 2 if not z_cfg["pred_z_flag"] else 3
        else:
            self.code_size = 2
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        num_vec = num_vec_one2one
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval
        self.aux_seg = aux_seg
        self.z_cfg = z_cfg
        self.pv_seg_dim = pv_seg_dim

        super().__init__(*args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False),
            requires_grad=False,
        )
        self.loss_pts = build_loss(loss_pts)
        self.loss_dir = build_loss(loss_dir)

        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.num_vec_one2one = num_vec_one2one

        self.loss_seg = build_loss(loss_seg)
        self.loss_pv_seg = build_loss(loss_pv_seg)

        self._init_layers()

        self.loss_segm_cls = build_loss(loss_segm_cls)
        self.loss_segm_mask = build_loss(loss_segm_mask)
        self.loss_segm_dice = build_loss(loss_segm_dice)

        self.dn_enabled = dn_enabled
        self.cls_join = cls_join
        self.tr_cls_join = tr_cls_join
        self.dn_weight = dn_weight
        if self.train_cfg is not None and isinstance(self.train_cfg.assigner_segm, dict):
            self.assigner_segm = build_assigner(self.train_cfg.assigner_segm)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])

        if self.aux_seg["use_aux_seg"]:
            if not (self.aux_seg["bev_seg"] or self.aux_seg["pv_seg"]):
                raise ValueError("aux_seg must have bev_seg or pv_seg")
            if self.aux_seg["bev_seg"]:
                self.seg_head = nn.Sequential(
                    nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False),
                    # nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.embed_dims, self.aux_seg["seg_classes"], kernel_size=1, padding=0),
                )
            if self.aux_seg["pv_seg"]:
                if self.pv_seg_dim is None:
                    self.pv_seg_dim = self.embed_dims
                self.pv_seg_head = nn.Sequential(
                    nn.Conv2d(self.pv_seg_dim, self.embed_dims, kernel_size=3, padding=1, bias=False),
                    # nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.embed_dims, self.aux_seg["seg_classes"], kernel_size=1, padding=0),
                )

        if not self.as_two_stage:
            self.bev_embedding = None
            self.query_embedding = None
            self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims * 2)
        self.positional_encoding = None

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=("mlvl_feats", "prev_bev"))
    def forward(self, mlvl_feats, lidar_feat, img_metas, prev_bev=None, only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        num_vec = self.num_vec_one2one

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        pts_query = self.pts_embedding.weight.unsqueeze(0)

        bev_queries = None
        bev_mask = None
        bev_pos = None

        # make attn mask
        """ attention mask to prevent information leakage
        """
        self_attn_mask = torch.zeros([num_vec, num_vec]).bool().to(mlvl_feats[0].device)
        self_attn_mask[self.num_vec_one2one :, 0 : self.num_vec_one2one] = True
        self_attn_mask[0 : self.num_vec_one2one, self.num_vec_one2one :] = True

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )["bev"]
        else:
            outputs = self.transformer(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                pts_query,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=(self.reg_branches if self.with_box_refine else None),  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
                self_attn_mask=self_attn_mask,
                num_vec=num_vec,
                num_pts_per_vec=self.num_pts_per_vec,
            )

        bev_embed, segm_mask_pred_list, segm_cls_scores_list, depth, hs, init_reference, inter_references, mask_dict = outputs
        hs = hs.permute(0, 2, 1, 3)
        dn_pad_size = mask_dict["pad_size"] if mask_dict is not None else 0

        outputs_segm_mask_pred_dn = []
        outputs_segm_cls_scores_dn = []

        outputs_segm_mask_pred_one2one = []
        outputs_segm_cls_scores_one2one = []

        outputs_segm_mask_pred_one2many = []
        outputs_segm_cls_scores_one2many = []

        for lvl in range(len(segm_mask_pred_list)):
            outputs_segm_mask_pred_dn.append(segm_mask_pred_list[lvl][:, :dn_pad_size])
            outputs_segm_cls_scores_dn.append(segm_cls_scores_list[lvl][:, :dn_pad_size])

            outputs_segm_mask_pred_one2one.append(segm_mask_pred_list[lvl][:, dn_pad_size : dn_pad_size + self.num_vec_one2one])
            outputs_segm_cls_scores_one2one.append(segm_cls_scores_list[lvl][:, dn_pad_size : dn_pad_size + self.num_vec_one2one])

            outputs_segm_mask_pred_one2many.append(segm_mask_pred_list[lvl][:, dn_pad_size + self.num_vec_one2one :])
            outputs_segm_cls_scores_one2many.append(segm_cls_scores_list[lvl][:, dn_pad_size + self.num_vec_one2one :])

        outputs_segm_mask_pred_dn = torch.stack(outputs_segm_mask_pred_dn)
        outputs_segm_cls_scores_dn = torch.stack(outputs_segm_cls_scores_dn)
        outputs_segm_mask_pred_one2one = torch.stack(outputs_segm_mask_pred_one2one)
        outputs_segm_cls_scores_one2one = torch.stack(outputs_segm_cls_scores_one2one)
        outputs_segm_mask_pred_one2many = torch.stack(outputs_segm_mask_pred_one2many)
        outputs_segm_cls_scores_one2many = torch.stack(outputs_segm_cls_scores_one2many)

        outputs_classes_dn = []
        outputs_coords_dn = []
        outputs_pts_coords_dn = []

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_pts_coords_one2one = []

        outputs_classes_one2many = []
        outputs_coords_one2many = []
        outputs_pts_coords_one2many = []

        num_vec = hs.shape[2] // self.num_pts_per_vec
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                # import pdb;pdb.set_trace()
                reference = init_reference[..., 0:2] if not self.z_cfg["gt_z_flag"] else init_reference[..., 0:3]
            else:
                reference = inter_references[lvl - 1][..., 0:2] if not self.z_cfg["gt_z_flag"] else inter_references[lvl - 1][..., 0:3]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl].view(bs, num_vec, self.num_pts_per_vec, -1).mean(2))
            tmp = self.reg_branches[lvl](hs[lvl])
            tmp = tmp[..., 0:2] if not self.z_cfg["gt_z_flag"] else tmp[..., 0:3]
            tmp += reference

            tmp = tmp.sigmoid()  # cx,cy,w,h

            outputs_coord, outputs_pts_coord = self.transform_box(tmp, num_vec=num_vec)

            outputs_classes_one2one.append(outputs_class[:, dn_pad_size : dn_pad_size + self.num_vec_one2one])
            outputs_coords_one2one.append(outputs_coord[:, dn_pad_size : dn_pad_size + self.num_vec_one2one])
            outputs_pts_coords_one2one.append(outputs_pts_coord[:, dn_pad_size : dn_pad_size + self.num_vec_one2one])

            outputs_classes_dn.append(outputs_class[:, :dn_pad_size])
            outputs_coords_dn.append(outputs_coord[:, :dn_pad_size])
            outputs_pts_coords_dn.append(outputs_pts_coord[:, :dn_pad_size])

            outputs_classes_one2many.append(outputs_class[:, dn_pad_size + self.num_vec_one2one :])
            outputs_coords_one2many.append(outputs_coord[:, dn_pad_size + self.num_vec_one2one :])
            outputs_pts_coords_one2many.append(outputs_pts_coord[:, dn_pad_size + self.num_vec_one2one :])

        outputs_classes_dn = torch.stack(outputs_classes_dn)
        outputs_coords_dn = torch.stack(outputs_coords_dn)
        outputs_pts_coords_dn = torch.stack(outputs_pts_coords_dn)

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        outputs_pts_coords_one2one = torch.stack(outputs_pts_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)
        outputs_pts_coords_one2many = torch.stack(outputs_pts_coords_one2many)

        if self.dn_enabled and self.training:
            mask_dict["output_known_lbs_bboxes"] = (
                outputs_segm_cls_scores_dn,
                outputs_segm_mask_pred_dn,
                outputs_classes_dn,
                outputs_coords_dn,
                outputs_pts_coords_dn,
            )

        outputs_seg = None
        outputs_pv_seg = None
        if self.aux_seg["use_aux_seg"]:
            seg_bev_embed = bev_embed.permute(1, 0, 2).reshape(bs, self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2).contiguous()
            if self.aux_seg["bev_seg"]:
                outputs_seg = self.seg_head(seg_bev_embed)
            bs, num_cam, embed_dims, feat_h, feat_w = mlvl_feats[-1].shape
            if self.aux_seg["pv_seg"]:
                outputs_pv_seg = self.pv_seg_head(mlvl_feats[-1].flatten(0, 1))
                outputs_pv_seg = outputs_pv_seg.view(bs, num_cam, -1, feat_h, feat_w)

        if self.tr_cls_join and self.training:
            outputs_classes_one2one = outputs_classes_one2one + outputs_segm_cls_scores_one2one[-1:][..., :3].contiguous()

        if self.cls_join and not self.training:
            outputs_classes_one2one = outputs_classes_one2one + outputs_segm_cls_scores_one2one[-1:][..., :3].contiguous()
        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes_one2one,
            "all_bbox_preds": outputs_coords_one2one,
            "all_pts_preds": outputs_pts_coords_one2one,
            "enc_bbox_preds": None,
            "enc_pts_preds": None,
            "depth": depth,
            "seg": outputs_seg,
            "segm_mask_preds": outputs_segm_mask_pred_one2one,
            "segm_cls_scores": outputs_segm_cls_scores_one2one,
            "pv_seg": outputs_pv_seg,
            "one2many_outs": dict(
                all_cls_scores=outputs_classes_one2many,
                all_bbox_preds=outputs_coords_one2many,
                all_pts_preds=outputs_pts_coords_one2many,
                enc_bbox_preds=None,
                enc_pts_preds=None,
                segm_mask_preds=outputs_segm_mask_pred_one2many,
                segm_cls_scores=outputs_segm_cls_scores_one2many,
                seg=None,
                pv_seg=None,
            ),
            "dn_mask_dict": mask_dict,
        }

        return outs

    def transform_box(self, pts, num_vec=50, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        if self.z_cfg["gt_z_flag"]:
            pts_reshape = pts.view(pts.shape[0], num_vec, self.num_pts_per_vec, 3)
        else:
            pts_reshape = pts.view(pts.shape[0], num_vec, self.num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        xmin = pts_x.min(dim=2, keepdim=True)[0]
        xmax = pts_x.max(dim=2, keepdim=True)[0]
        ymin = pts_y.min(dim=2, keepdim=True)[0]
        ymax = pts_y.max(dim=2, keepdim=True)[0]
        bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
        bbox = bbox_xyxy_to_cxcywh(bbox)
        return bbox, pts_reshape

    def _get_target_single(
        self,
        cls_score,
        bbox_pred,
        pts_pred,
        gt_labels,
        gt_bboxes,
        gt_shifts_pts,
        gt_bboxes_ignore=None,
    ):
        """ "Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # import pdb;pdb.set_trace()

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        # import pdb;pdb.set_trace()
        assign_result, order_index = self.assigner.assign(bbox_pred, cls_score, pts_pred, gt_bboxes, gt_labels, gt_shifts_pts, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # pts targets
        # num_query, num_order, num_points, num_coords
        if order_index is None:
            # import pdb;pdb.set_trace()
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0), pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds, assigned_shift, :, :]
        return labels, label_weights, bbox_targets, bbox_weights, pts_targets, pts_weights, pos_inds, neg_inds

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        pts_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_shifts_pts_list,
        gt_bboxes_ignore_list=None,
    ):
        assert gt_bboxes_ignore_list is None, "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            pts_preds_list,
            gt_labels_list,
            gt_bboxes_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def _get_target_single_pts(
        self,
        cls_score,
        bbox_pred,
        pts_pred,
        gt_labels,
        gt_bboxes,
        gt_shifts_pts,
        pos_inds,
        neg_inds,
        pos_assigned_gt_inds,
        gt_bboxes_ignore=None,
    ):
        # import pdb;pdb.set_trace()
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        cls_assigned = cls_score[pos_inds].unsqueeze(1)
        bbox_assigned = bbox_pred[pos_inds].unsqueeze(1)
        pts_assigned = pts_pred[pos_inds].unsqueeze(1)
        gt_labels_assigned = gt_labels[pos_assigned_gt_inds].unsqueeze(1)
        gt_bboxes_assigned = gt_bboxes[pos_assigned_gt_inds].unsqueeze(1)
        gt_shifts_pts_assigned = gt_shifts_pts[pos_assigned_gt_inds].unsqueeze(1)

        _, order_index = multi_apply(
            self.assigner.assign, bbox_assigned, cls_assigned, pts_assigned, gt_bboxes_assigned, gt_labels_assigned, gt_shifts_pts_assigned
        )

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels_assigned.squeeze(1)
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # pts targets
        # num_query, num_order, num_points, num_coords
        if order_index is None:
            assigned_shift = gt_labels[pos_assigned_gt_inds]
        else:
            assigned_shift = torch.tensor(order_index)
        pts_targets = pts_pred.new_zeros((pts_pred.size(0), pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = gt_bboxes_assigned.squeeze(1)
        pts_targets[pos_inds] = gt_shifts_pts[pos_assigned_gt_inds, assigned_shift, :, :]
        return labels, label_weights, bbox_targets, bbox_weights, pts_targets, pts_weights, pos_inds, neg_inds

    def _get_target_single_segm(self, cls_score, mask_pred, gt_labels, gt_masks):
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]
        # assign and sample
        assign_result = self.assigner_segm.assign(cls_score, mask_pred, gt_labels, gt_masks)
        sampling_result = self.sampler.sample(assign_result, mask_pred, gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((num_queries,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((num_queries,))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((num_queries,))
        mask_weights[pos_inds] = 1.0

        return labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds

    def get_targets_segm(self, cls_scores_list, mask_preds_list, gt_labels_list, gt_masks_list):
        (
            labels_list,
            label_weights_list,
            mask_targets_list,
            mask_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single_segm,
            cls_scores_list,
            mask_preds_list,
            gt_labels_list,
            gt_masks_list,
        )

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return labels_list, label_weights_list, mask_targets_list, mask_weights_list, num_total_pos, num_total_neg

    def loss_single_pts(
        self,
        cls_scores,
        bbox_preds,
        pts_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_shifts_pts_list,
        gt_bboxes_ignore_list=None,
    ):
        """ "Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list, bbox_preds_list, pts_preds_list, gt_bboxes_list, gt_labels_list, gt_shifts_pts_list, gt_bboxes_ignore_list
        )
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, pts_targets_list, pts_weights_list, num_total_pos, num_total_neg) = (
            cls_reg_targets
        )
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :4], normalized_bbox_targets[isnotnan, :4], bbox_weights[isnotnan, :4], avg_factor=num_total_pos
        )

        # regression pts CD loss

        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = (
            normalize_2d_pts(pts_targets, self.pc_range) if not self.z_cfg["gt_z_flag"] else normalize_3d_pts(pts_targets, self.pc_range)
        )

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0, 2, 1)
            pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec), mode="linear", align_corners=True)
            pts_preds = pts_preds.permute(0, 2, 1).contiguous()

        # import pdb;pdb.set_trace()
        loss_pts = self.loss_pts(
            pts_preds[isnotnan, :, :],
            normalized_pts_targets[isnotnan, :, :],
            pts_weights[isnotnan, :, :],
            avg_factor=num_total_pos,
        )
        dir_weights = pts_weights[:, : -self.dir_interval, 0]
        denormed_pts_preds = (
            denormalize_2d_pts(pts_preds, self.pc_range) if not self.z_cfg["gt_z_flag"] else denormalize_3d_pts(pts_preds, self.pc_range)
        )
        denormed_pts_preds_dir = denormed_pts_preds[:, self.dir_interval :, :] - denormed_pts_preds[:, : -self.dir_interval, :]
        pts_targets_dir = pts_targets[:, self.dir_interval :, :] - pts_targets[:, : -self.dir_interval, :]

        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan, :, :],
            pts_targets_dir[isnotnan, :, :],
            dir_weights[isnotnan, :],
            avg_factor=num_total_pos,
        )

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes[isnotnan, :4],
            bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos,
        )

        if digit_version(TORCH_VERSION) >= digit_version("1.8"):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    def prepare_for_dn_loss(self, mask_dict):

        (
            segm_cls_scores,
            segm_mask_pred,
            pts_cls_scores,
            pts_bbox_preds,
            pts_preds,
        ) = mask_dict["output_known_lbs_bboxes"]
        known_labels, known_masks = mask_dict["known_lbs_bboxes"]
        map_known_indice = mask_dict["map_known_indice"].long()
        known_indice = mask_dict["known_indice"].long()
        batch_idx = mask_dict["batch_idx"].long()
        bid = batch_idx[known_indice]
        num_tgt = known_indice.numel()

        if len(pts_cls_scores) > 0:
            pts_cls_scores = pts_cls_scores.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            pts_bbox_preds = pts_bbox_preds.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            pts_preds = pts_preds.permute(1, 2, 0, 3, 4)[(bid, map_known_indice)].permute(1, 0, 2, 3)
            segm_cls_scores = segm_cls_scores.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            segm_mask_pred = segm_mask_pred.permute(1, 2, 0, 3, 4)[(bid, map_known_indice)].permute(1, 0, 2, 3)

        return known_labels, known_masks, pts_cls_scores, pts_bbox_preds, pts_preds, segm_cls_scores, segm_mask_pred, num_tgt

    def dn_loss_single_pts(
        self,
        cls_scores,
        bbox_preds,
        pts_preds,
        known_bboxs,
        known_pts_list,
        known_shifts_pts,
        known_labels,
        num_total_pos=None,
    ):
        num_total_pos = cls_scores.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(num_total_pos, min=1.0).item()

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0

        cls_avg_factor = max(cls_avg_factor, 1)

        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)
        pts_weights = torch.ones_like(known_pts_list)

        num_pts_per_predline = pts_preds.size(1)
        normalized_gt_pts = (
            normalize_2d_pts(known_shifts_pts[: int(num_total_pos)], self.pc_range)
            if not self.z_cfg["gt_z_flag"]
            else normalize_3d_pts(known_shifts_pts[: int(num_total_pos)], self.pc_range)
        )
        _, num_orders, num_pts_per_gtline, num_coords = known_shifts_pts.shape
        num_gts, num_bboxes = known_bboxs.size(0), bbox_preds.size(0)

        if num_pts_per_predline != num_pts_per_gtline:
            pts_pred_interpolated = F.interpolate(pts_preds.permute(0, 2, 1), size=(num_pts_per_gtline), mode="linear", align_corners=True)
            pts_pred_interpolated = pts_pred_interpolated.permute(0, 2, 1).contiguous()
        else:
            pts_pred_interpolated = pts_preds
        # num_q, num_pts, 2 <-> num_gt, num_pts, 2
        pts_pred_flatten = pts_pred_interpolated.view(pts_pred_interpolated.shape[0], -1)

        normalized_gt_pts_flatten = normalized_gt_pts.flatten(2).view(num_gts * num_orders, -1)
        pts_cost_ordered = torch.cdist(pts_pred_flatten, normalized_gt_pts_flatten, p=1)

        pts_cost_ordered = pts_cost_ordered.view(num_bboxes, num_gts, num_orders)
        _, order_index = torch.min(pts_cost_ordered, 2)

        assigned_shift = order_index[torch.arange(len(order_index)).cuda(), torch.arange(len(order_index)).cuda()]
        pos_inds = torch.arange(len(known_shifts_pts), device=known_shifts_pts.device).long()

        # DETR
        pts_targets = known_shifts_pts[pos_inds, assigned_shift, :, :]

        loss_cls = self.loss_cls(cls_scores, known_labels, label_weights, avg_factor=cls_avg_factor)

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))

        normalized_bbox_targets = normalize_2d_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        # regression pts CD loss

        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = (
            normalize_2d_pts(pts_targets, self.pc_range) if not self.z_cfg["gt_z_flag"] else normalize_3d_pts(known_pts_list, self.pc_range)
        )

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        loss_pts = self.loss_pts(
            pts_preds[isnotnan, :, :],
            normalized_pts_targets[isnotnan, :, :],
            pts_weights[isnotnan, :, :],
            avg_factor=num_total_pos,
        )
        dir_weights = pts_weights[:, : -self.dir_interval, 0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:, self.dir_interval :, :] - denormed_pts_preds[:, : -self.dir_interval, :]
        pts_targets_dir = known_pts_list[:, self.dir_interval :, :] - known_pts_list[:, : -self.dir_interval, :]
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan, :, :],
            pts_targets_dir[isnotnan, :, :],
            dir_weights[isnotnan, :],
            avg_factor=num_total_pos,
        )

        loss_cls = self.dn_weight * torch.nan_to_num(loss_cls)
        loss_pts = self.dn_weight * torch.nan_to_num(loss_pts)
        loss_dir = self.dn_weight * torch.nan_to_num(loss_dir)

        # return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir
        return loss_cls, loss_pts, loss_dir

    def dn_loss_single_segm(
        self,
        segm_cls_scores,
        segm_mask_preds,
        known_labels,
        known_masks,
        num_total_pos=None,
    ):
        num_total_pos = segm_cls_scores.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(num_total_pos, min=1.0).item()

        segm_mask_shape = segm_mask_preds.shape[-2:]
        segm_mask_points_num = segm_mask_shape[0] * segm_mask_shape[1]

        segm_label_weights = torch.ones_like(known_labels)
        segm_class_weight = segm_cls_scores.new_tensor(self.loss_segm_cls.class_weight)
        loss_segm_cls = self.loss_segm_cls(
            segm_cls_scores,
            known_labels,
            segm_label_weights,
            avg_factor=segm_class_weight[known_labels].sum(),
        )

        if known_masks.shape[0] == 0:
            # zero match
            loss_dice = segm_mask_preds.sum()
            loss_mask = segm_mask_preds.sum()
            return loss_segm_cls, loss_mask, loss_dice

        # dice loss
        loss_segm_dice = self.loss_segm_dice(segm_mask_preds, known_masks, avg_factor=num_total_pos)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        segm_mask_preds = segm_mask_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        known_masks = known_masks.reshape(-1)
        loss_segm_mask = self.loss_segm_mask(
            segm_mask_preds,
            known_masks,
            avg_factor=num_total_pos * segm_mask_points_num,
        )

        loss_segm_dice = self.dn_weight * torch.nan_to_num(loss_segm_dice)
        loss_segm_cls = self.dn_weight * torch.nan_to_num(loss_segm_cls)
        loss_segm_mask = self.dn_weight * torch.nan_to_num(loss_segm_mask)

        return loss_segm_dice, loss_segm_cls, loss_segm_mask

    @force_fp32(apply_to=("preds_dicts"))
    def calc_dn_loss(
        self,
        loss_dict,
        preds_dicts,
        gt_bboxes_list,
        gt_pts_list,
        gt_shifts_pts_list,
        num_dec_layers,
    ):
        (known_labels, known_masks, pts_cls_scores, pts_bbox_preds, pts_preds, segm_cls_scores, segm_mask_pred, num_tgt) = self.prepare_for_dn_loss(
            preds_dicts["dn_mask_dict"]
        )

        dn_group_num = preds_dicts["dn_mask_dict"]["pad_size"] // preds_dicts["dn_mask_dict"]["dn_single_pad"]

        bboxes = torch.cat(gt_bboxes_list).clone()
        pts = torch.cat(gt_pts_list).clone()
        shifts_pts = torch.cat(gt_shifts_pts_list).clone()

        known_bboxes = bboxes.repeat(dn_group_num, 1)
        known_pts = pts.repeat(dn_group_num, 1, 1)
        known_shifts_pts = shifts_pts.repeat(dn_group_num, 1, 1, 1)

        all_known_bboxs_list = [known_bboxes for _ in range(num_dec_layers)]
        all_known_labels_list = [known_labels.long() for _ in range(num_dec_layers)]
        all_known_pts_list = [known_pts for _ in range(num_dec_layers)]
        all_num_tgts_list = [num_tgt for _ in range(num_dec_layers)]
        all_known_shifts_pts_list = [known_shifts_pts for _ in range(num_dec_layers)]

        all_known_masks = [known_masks for _ in range(num_dec_layers)]

        dn_loss_segm_dice, dn_loss_segm_cls, dn_loss_segm_mask = multi_apply(
            self.dn_loss_single_segm, segm_cls_scores, segm_mask_pred, all_known_labels_list, all_known_masks, all_num_tgts_list
        )

        all_known_bboxs_list = [known_bboxes for _ in range(len(pts_cls_scores))]
        all_known_labels_list = [known_labels.long() for _ in range(len(pts_cls_scores))]
        all_known_pts_list = [known_pts for _ in range(len(pts_cls_scores))]
        all_num_tgts_list = [num_tgt for _ in range(len(pts_cls_scores))]
        all_known_shifts_pts_list = [known_shifts_pts for _ in range(len(pts_cls_scores))]

        dn_loss_cls, dn_loss_pts, dn_loss_dir = multi_apply(
            self.dn_loss_single_pts,
            pts_cls_scores,
            pts_bbox_preds,
            pts_preds,
            all_known_bboxs_list,
            all_known_pts_list,
            all_known_shifts_pts_list,
            all_known_labels_list,
            all_num_tgts_list,
        )

        loss_dict["loss_cls_dn"] = dn_loss_cls[-1]
        # loss_dict["loss_bbox_dn"] = dn_loss_bbox[-1]
        # loss_dict["loss_iou_dn"] = dn_loss_iou[-1]
        loss_dict["loss_pts_dn"] = dn_loss_pts[-1]
        loss_dict["loss_dir_dn"] = dn_loss_dir[-1]
        loss_dict["loss_segm_cls_dn"] = dn_loss_segm_cls[-1]
        loss_dict["loss_segm_dice_dn"] = dn_loss_segm_dice[-1]
        loss_dict["loss_segm_mask_dn"] = dn_loss_segm_mask[-1]

        num_dec_layer = 0
        for loss_cls_i, loss_pts_i, loss_dir_i in zip(dn_loss_cls[:-1], dn_loss_pts[:-1], dn_loss_dir[:-1]):
            loss_dict[f"d{num_dec_layer}.loss_cls_dn"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_pts_dn"] = loss_pts_i
            loss_dict[f"d{num_dec_layer}.loss_dir_dn"] = loss_dir_i
            num_dec_layer += 1

        num_dec_layer = 0
        for loss_segm_cls_i, loss_segm_dice_i, loss_segm_mask_i in zip(
            dn_loss_segm_cls[:-1],
            dn_loss_segm_dice[:-1],
            dn_loss_segm_mask[:-1],
        ):
            loss_dict[f"d{num_dec_layer}.loss_segm_cls_dn"] = loss_segm_cls_i
            loss_dict[f"d{num_dec_layer}.loss_segm_dice_dn"] = loss_segm_dice_i
            loss_dict[f"d{num_dec_layer}.loss_segm_mask_dn"] = loss_segm_mask_i
            num_dec_layer += 1
        # loss_dict
        return loss_dict

    def loss_single_segm(
        self,
        segm_cls_scores=None,
        segm_mask_preds=None,
        gt_segm_mask_list=None,
        gt_segm_labels_list=None,
    ):
        """ "Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        segm_mask_shape = segm_mask_preds.shape[-2:]
        segm_mask_points_num = segm_mask_shape[0] * segm_mask_shape[1]
        segm_targets = self.get_targets_segm(segm_cls_scores, segm_mask_preds, gt_segm_labels_list, gt_segm_mask_list)
        (segm_labels_list, segm_label_weights_list, segm_mask_targets_list, segm_mask_weights_list, segm_num_total_pos, segm_num_total_neg) = (
            segm_targets
        )

        # shape (batch_size, num_queries)
        segm_labels = torch.stack(segm_labels_list, dim=0)
        # shape (batch_size, num_queries)
        segm_label_weights = torch.stack(segm_label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        segm_mask_targets = torch.cat(segm_mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        segm_mask_weights = torch.stack(segm_mask_weights_list, dim=0)
        segm_cls_scores = segm_cls_scores.flatten(0, 1)
        segm_labels = segm_labels.flatten(0, 1)
        segm_label_weights = segm_label_weights.flatten(0, 1)

        segm_class_weight = segm_cls_scores.new_tensor(self.loss_segm_cls.class_weight)
        loss_segm_cls = self.loss_segm_cls(
            segm_cls_scores,
            segm_labels,
            segm_label_weights,
            avg_factor=segm_class_weight[segm_labels].sum(),
        )

        num_total_masks = reduce_mean(segm_cls_scores.new_tensor([segm_num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        segm_mask_preds = segm_mask_preds[segm_mask_weights > 0]

        if segm_mask_targets.shape[0] == 0:
            # zero match
            loss_dice = segm_mask_preds.sum()
            loss_mask = segm_mask_preds.sum()
            return loss_segm_cls, loss_mask, loss_dice

        # dice loss
        loss_segm_dice = self.loss_segm_dice(segm_mask_preds, segm_mask_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        segm_mask_preds = segm_mask_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        segm_mask_targets = segm_mask_targets.reshape(-1)
        loss_segm_mask = self.loss_segm_mask(
            segm_mask_preds,
            segm_mask_targets,
            avg_factor=num_total_masks * segm_mask_points_num,
        )

        return loss_segm_dice, loss_segm_mask, loss_segm_cls

    def loss_single(
        self,
        cls_scores,
        bbox_preds,
        pts_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_shifts_pts_list,
        gt_bboxes_ignore_list=None,
        segm_cls_scores=None,
        segm_mask_preds=None,
        gt_segm_mask_list=None,
        gt_segm_labels_list=None,
    ):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            pts_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        segm_mask_shape = segm_mask_preds.shape[-2:]
        segm_mask_points_num = segm_mask_shape[0] * segm_mask_shape[1]

        segm_labels_list = [labels for labels in labels_list]
        segm_label_weights_list = [segm_label_weights for segm_label_weights in label_weights_list]
        segm_mask_weights_list = [bbox_weights.sum(1) > 0 for bbox_weights in bbox_weights_list]
        segm_num_total_pos = num_total_pos
        target_idxes = []
        for gt_bboxes, bbox_targets in zip(gt_bboxes_list, bbox_targets_list):
            target_idx = []
            for gt_box in gt_bboxes:
                target_idx.append(torch.nonzero(torch.all(bbox_targets[torch.all(bbox_targets != 0.0, dim=1)] == gt_box, dim=1)).squeeze())
            target_idxes.append(target_idx)
        segm_mask_targets_list = [gt_segm_mask[torch.stack(target_i)] for target_i, gt_segm_mask in zip(target_idxes, gt_segm_mask_list)]

        # shape (batch_size, num_queries)
        segm_labels = torch.stack(segm_labels_list, dim=0)
        # shape (batch_size, num_queries)
        segm_label_weights = torch.stack(segm_label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        segm_mask_targets = torch.cat(segm_mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        segm_mask_weights = torch.stack(segm_mask_weights_list, dim=0)
        segm_cls_scores = segm_cls_scores.flatten(0, 1)
        segm_labels = segm_labels.flatten(0, 1)
        segm_label_weights = segm_label_weights.flatten(0, 1)

        segm_class_weight = segm_cls_scores.new_tensor(self.loss_segm_cls.class_weight)
        loss_segm_cls = self.loss_segm_cls(segm_cls_scores, segm_labels, segm_label_weights, avg_factor=segm_class_weight[segm_labels].sum())

        num_total_masks = reduce_mean(segm_cls_scores.new_tensor([segm_num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        segm_mask_preds = segm_mask_preds[segm_mask_weights > 0]

        if segm_mask_targets.shape[0] == 0:
            # zero match
            loss_dice = segm_mask_preds.sum()
            loss_mask = segm_mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        # dice loss
        loss_segm_dice = self.loss_segm_dice(segm_mask_preds, segm_mask_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        segm_mask_preds = segm_mask_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        segm_mask_targets = segm_mask_targets.reshape(-1)
        loss_segm_mask = self.loss_segm_mask(
            segm_mask_preds,
            segm_mask_targets,
            avg_factor=num_total_masks * segm_mask_points_num,
        )

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :4],
            normalized_bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos,
        )

        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = (
            normalize_2d_pts(pts_targets, self.pc_range) if not self.z_cfg["gt_z_flag"] else normalize_3d_pts(pts_targets, self.pc_range)
        )

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0, 2, 1)
            pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec), mode="linear", align_corners=True)
            pts_preds = pts_preds.permute(0, 2, 1).contiguous()

        # import pdb;pdb.set_trace()
        loss_pts = self.loss_pts(
            pts_preds[isnotnan, :, :],
            normalized_pts_targets[isnotnan, :, :],
            pts_weights[isnotnan, :, :],
            avg_factor=num_total_pos,
        )
        dir_weights = pts_weights[:, : -self.dir_interval, 0]
        denormed_pts_preds = (
            denormalize_2d_pts(pts_preds, self.pc_range) if not self.z_cfg["gt_z_flag"] else denormalize_3d_pts(pts_preds, self.pc_range)
        )
        denormed_pts_preds_dir = denormed_pts_preds[:, self.dir_interval :, :] - denormed_pts_preds[:, : -self.dir_interval, :]
        pts_targets_dir = pts_targets[:, self.dir_interval :, :] - pts_targets[:, : -self.dir_interval, :]
        # dir_weights = pts_weights[:, indice,:-1,0]
        # import pdb;pdb.set_trace()
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan, :, :],
            pts_targets_dir[isnotnan, :, :],
            dir_weights[isnotnan, :],
            avg_factor=num_total_pos,
        )

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes[isnotnan, :4],
            bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos,
        )

        if digit_version(TORCH_VERSION) >= digit_version("1.8"):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
        return (
            loss_cls,
            loss_bbox,
            loss_iou,
            loss_pts,
            loss_dir,
            loss_segm_dice,
            loss_segm_mask,
            loss_segm_cls,
        )

    @force_fp32(apply_to=("preds_dicts"))
    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        gt_pts_list,
        gt_seg_mask,
        gt_pv_seg_mask,
        gt_shifts_pts_list,
        gt_segms_list,
        preds_dicts,
        gt_bboxes_ignore=None,
        img_metas=None,
        is_one2many=False,
    ):
        assert gt_bboxes_ignore is None, f"{self.__class__.__name__} only supports " f"for gt_bboxes_ignore setting to None."
        all_cls_scores = preds_dicts["all_cls_scores"]
        all_bbox_preds = preds_dicts["all_bbox_preds"]
        all_pts_preds = preds_dicts["all_pts_preds"]
        enc_bbox_preds = preds_dicts["enc_bbox_preds"]
        enc_pts_preds = preds_dicts["enc_pts_preds"]

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_pts_list = [gt_pts_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        segm_cls_scores = preds_dicts["segm_cls_scores"]
        segm_mask_preds = preds_dicts["segm_mask_preds"]

        all_gt_segms_mask_list = [gt_segms_list for _ in range(len(segm_mask_preds))]
        all_gt_segms_labels_list = [gt_labels_list for _ in range(len(segm_mask_preds))]

        losses_cls, losses_bbox, losses_iou, losses_pts, losses_dir = multi_apply(
            self.loss_single_pts,
            all_cls_scores,
            all_bbox_preds,
            all_pts_preds,
            all_gt_bboxes_list,
            all_gt_labels_list,
            all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list,
        )
        losses_segm_dice, losses_segm_mask, losses_segm_cls = multi_apply(
            self.loss_single_segm,
            segm_cls_scores,
            segm_mask_preds,
            all_gt_segms_mask_list,
            all_gt_segms_labels_list,
        )

        loss_dict = dict()
        if self.aux_seg["use_aux_seg"]:
            if self.aux_seg["bev_seg"]:
                if preds_dicts["seg"] is not None:
                    seg_output = preds_dicts["seg"]
                    num_imgs = seg_output.size(0)
                    seg_gt = torch.stack([gt_seg_mask[i] for i in range(num_imgs)], dim=0)
                    loss_seg = self.loss_seg(seg_output, seg_gt.float())
                    loss_dict["loss_seg"] = loss_seg
            if self.aux_seg["pv_seg"]:
                if preds_dicts["pv_seg"] is not None:
                    pv_seg_output = preds_dicts["pv_seg"]
                    num_imgs = pv_seg_output.size(0)
                    pv_seg_gt = torch.stack([gt_pv_seg_mask[i] for i in range(num_imgs)], dim=0)
                    loss_pv_seg = self.loss_pv_seg(pv_seg_output, pv_seg_gt.float())
                    loss_dict["loss_pv_seg"] = loss_pv_seg

        if "dn_mask_dict" in preds_dicts and preds_dicts["dn_mask_dict"] is not None and is_one2many == False:
            loss_dict = self.calc_dn_loss(
                loss_dict,
                preds_dicts,
                gt_bboxes_list,
                gt_pts_list,
                gt_shifts_pts_list,
                len(segm_mask_preds),
            )

        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]
        loss_dict["loss_iou"] = losses_iou[-1]
        loss_dict["loss_pts"] = losses_pts[-1]
        loss_dict["loss_dir"] = losses_dir[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_pts_i, loss_dir_i in zip(
            losses_cls[:-1],
            losses_bbox[:-1],
            losses_iou[:-1],
            losses_pts[:-1],
            losses_dir[:-1],
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            loss_dict[f"d{num_dec_layer}.loss_iou"] = loss_iou_i
            loss_dict[f"d{num_dec_layer}.loss_pts"] = loss_pts_i
            loss_dict[f"d{num_dec_layer}.loss_dir"] = loss_dir_i
            num_dec_layer += 1

        loss_dict["loss_segm_cls"] = losses_segm_cls[-1]
        loss_dict["loss_segm_dice"] = losses_segm_dice[-1]
        loss_dict["loss_segm_mask"] = losses_segm_mask[-1]
        num_segm_dec_layer = 0
        for loss_segm_cls, loss_segm_dice, loss_segm_mask in zip(losses_segm_cls[:-1], losses_segm_dice[:-1], losses_segm_mask[:-1]):
            loss_dict[f"d{num_segm_dec_layer}.loss_segm_cls"] = loss_segm_cls
            loss_dict[f"d{num_segm_dec_layer}.loss_segm_dice"] = loss_segm_dice
            loss_dict[f"d{num_segm_dec_layer}.loss_segm_mask"] = loss_segm_mask
            num_segm_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=("preds_dicts"))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # bboxes: xmin, ymin, xmax, ymax
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            # code_size = bboxes.shape[-1]
            # bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds["scores"]
            labels = preds["labels"]
            pts = preds["pts"]

            ret_list.append([bboxes, scores, labels, pts])

        return ret_list
