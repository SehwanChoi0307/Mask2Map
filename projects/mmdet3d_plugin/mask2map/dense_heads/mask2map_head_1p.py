import copy
import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmcv.runner import force_fp32
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmdet.core import multi_apply, reduce_mean, build_assigner


@HEADS.register_module()
class Mask2MapHead_1Phase(DETRHead):
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
        pc_range=None,
        num_cls_fcs=2,
        code_weights=None,
        bev_h=30,
        bev_w=30,
        num_vec_one2one=50,
        num_pts_per_vec=20,
        num_pts_per_gt_vec=20,
        dir_interval=1,
        aux_seg=dict(use_aux_seg=False, bev_seg=False, pv_seg=False, seg_classes=1, feat_down_sample=32),
        z_cfg=dict(pred_z_flag=False, gt_z_flag=False),
        loss_seg=dict(type="SimpleLoss", pos_weight=2.13, loss_weight=1.0),
        loss_pv_seg=dict(type="SimpleLoss", pos_weight=2.13, loss_weight=1.0),
        loss_segm_cls=None,
        loss_segm_mask=None,
        loss_segm_dice=None,
        dn_enabled=False,
        dn_weight=1,
        pv_seg_dim=None,
        **kwargs,
    ):
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.bev_encoder_type = transformer.encoder.type
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

        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

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
        self.code_weights = nn.Parameter(torch.tensor(self.code_weights, requires_grad=False), requires_grad=False)

        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_pts_per_vec = num_pts_per_vec
        self.num_vec_one2one = num_vec_one2one

        self.loss_seg = build_loss(loss_seg)
        self.loss_pv_seg = build_loss(loss_pv_seg)

        self._init_layers()

        self.loss_segm_cls = build_loss(loss_segm_cls)
        self.loss_segm_mask = build_loss(loss_segm_mask)
        self.loss_segm_dice = build_loss(loss_segm_dice)

        self.dn_enabled = dn_enabled
        self.dn_weight = dn_weight

        self.assigner_segm = build_assigner(self.train_cfg["assigner_segm"])

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        # cls_branch.append(Linear(self.embed_dims * 2, self.embed_dims))
        # cls_branch.append(nn.LayerNorm(self.embed_dims))
        # cls_branch.append(nn.ReLU(inplace=True))
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

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

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

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

        pts_embeds = self.pts_embedding.weight.unsqueeze(0)

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
                pts_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
                self_attn_mask=self_attn_mask,
                num_vec=num_vec,
                num_pts_per_vec=self.num_pts_per_vec,
            )

        bev_embed, segm_mask_pred_list, segm_cls_scores_list, depth, hs, init_reference, inter_references, mask_dict = outputs
        # hs = hs.permute(0, 2, 1, 3)
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

        if self.dn_enabled and self.training:
            mask_dict["output_known_lbs_bboxes"] = (outputs_segm_cls_scores_dn, outputs_segm_mask_pred_dn, None, None, None)

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

        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": None,
            "all_bbox_preds": None,
            "all_pts_preds": None,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
            "enc_pts_preds": None,
            "depth": depth,
            "seg": outputs_seg,
            "segm_mask_preds": outputs_segm_mask_pred_one2one,
            "segm_cls_scores": outputs_segm_cls_scores_one2one,
            "pv_seg": outputs_pv_seg,
            "one2many_outs": dict(
                all_cls_scores=None,
                all_bbox_preds=None,
                all_pts_preds=None,
                enc_cls_scores=None,
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
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single_segm,
            cls_scores_list,
            mask_preds_list,
            gt_labels_list,
            gt_masks_list,
        )

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return labels_list, label_weights_list, mask_targets_list, mask_weights_list, num_total_pos, num_total_neg

    def prepare_for_dn_loss(self, mask_dict):

        segm_cls_scores, segm_mask_pred, pts_cls_scores, pts_bbox_preds, pts_preds = mask_dict["output_known_lbs_bboxes"]
        known_labels, known_masks = mask_dict["known_lbs_bboxes"]
        map_known_indice = mask_dict["map_known_indice"].long()
        known_indice = mask_dict["known_indice"].long()
        batch_idx = mask_dict["batch_idx"].long()
        bid = batch_idx[known_indice]
        num_tgt = known_indice.numel()

        if len(segm_cls_scores) > 0:
            segm_cls_scores = segm_cls_scores.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            segm_mask_pred = segm_mask_pred.permute(1, 2, 0, 3, 4)[(bid, map_known_indice)].permute(1, 0, 2, 3)

        return known_labels, known_masks, pts_cls_scores, pts_bbox_preds, pts_preds, segm_cls_scores, segm_mask_pred, num_tgt

    def dn_loss_single(
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
        (
            known_labels,
            known_masks,
            pts_cls_scores,
            pts_bbox_preds,
            pts_preds,
            segm_cls_scores,
            segm_mask_pred,
            num_tgt,
        ) = self.prepare_for_dn_loss(preds_dicts["dn_mask_dict"])

        dn_group_num = preds_dicts["dn_mask_dict"]["pad_size"] // preds_dicts["dn_mask_dict"]["dn_single_pad"]

        all_known_labels_list = [known_labels.long() for _ in range(num_dec_layers)]
        all_num_tgts_list = [num_tgt for _ in range(num_dec_layers)]
        all_known_masks = [known_masks for _ in range(num_dec_layers)]

        dn_loss_segm_dice, dn_loss_segm_cls, dn_loss_segm_mask = multi_apply(
            self.dn_loss_single,
            segm_cls_scores,
            segm_mask_pred,
            all_known_labels_list,
            all_known_masks,
            all_num_tgts_list,
        )

        loss_dict["loss_segm_cls_dn"] = dn_loss_segm_cls[-1]
        loss_dict["loss_segm_dice_dn"] = dn_loss_segm_dice[-1]
        loss_dict["loss_segm_mask_dn"] = dn_loss_segm_mask[-1]

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
        segm_targets = self.get_targets_segm(
            segm_cls_scores,
            segm_mask_preds,
            gt_segm_labels_list,
            gt_segm_mask_list,
        )
        segm_labels_list, segm_label_weights_list, segm_mask_targets_list, segm_mask_weights_list, segm_num_total_pos, _ = segm_targets

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

        enc_cls_scores = preds_dicts["enc_cls_scores"]
        enc_bbox_preds = preds_dicts["enc_bbox_preds"]
        enc_pts_preds = preds_dicts["enc_pts_preds"]

        device = gt_labels_list[0].device

        segm_cls_scores = preds_dicts["segm_cls_scores"]
        segm_mask_preds = preds_dicts["segm_mask_preds"]

        all_gt_segms_mask_list = [gt_segms_list for _ in range(len(segm_mask_preds))]
        all_gt_segms_labels_list = [gt_labels_list for _ in range(len(segm_mask_preds))]

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
            loss_dict = self.calc_dn_loss(loss_dict, preds_dicts, gt_bboxes_list, gt_pts_list, gt_shifts_pts_list, len(segm_mask_preds))

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
