import cv2
import torch
import numpy as np
import torch.nn as nn
from shapely import affinity
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmdet3d.models import builder
from .encoder import SimpleBEVEncoder
from shapely.geometry import LineString
from .builder import build_fuser, FUSERS
from mmcv.runner.base_module import BaseModule
from mmdet.datasets.pipelines import to_tensor
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet.models.utils.positional_encoding import SinePositionalEncoding
from projects.mmdet3d_plugin.bevformer.modules.decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention import TemporalSelfAttention
from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import MSDeformableAttention3D


def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    return new_pts


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 2)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))


@TRANSFORMER.register_module()
class Mask2Map_Transformer_2Phase_CP(BaseModule):

    def __init__(
        self,
        num_feature_levels=4,
        num_cams=6,
        z_cfg=dict(
            pred_z_flag=False,
            gt_z_flag=False,
        ),
        two_stage_num_proposals=300,
        fuser=None,
        encoder=None,
        decoder=None,
        embed_dims=256,
        rotate_prev_bev=True,
        modality="vision",
        feat_down_sample_indice=-1,
        bev_encoder=None,
        bev_neck=None,
        dropout=0.1,
        thr=0.5,
        sam=20,
        num_transformer_feat_level=3,
        num_vec_one2one=50,
        segm_decoder=None,
        num_classes=3,
        dn_enabled=False,
        dn_group_num=5,
        dn_noise_scale=0.0,
        thresh_of_mask_for_pos=0.5,
        dn_label_noise_ratio=0.2,
        feat_down_dim=None,
        mask_noise_scale=0.2,
        mask_pred_detach=False,
        gt_mask_for_point_sampling=None,
        pts2mask_noise_scale=0.2,
        patch_size=[60.0, 30.0],
        bev_h=200,
        bev_w=100,
        num_max_survival=4,
        **kwargs,
    ):
        super(Mask2Map_Transformer_2Phase_CP, self).__init__(**kwargs)
        if modality == "fusion":
            self.fuser = build_fuser(fuser)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.sam = sam
        self.thr = thr
        self.num_feature_levels = num_feature_levels

        self.ref_pc_range = [0.0, 0.0, -10.0, bev_w, bev_h, 10.0]
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.rotate_prev_bev = rotate_prev_bev
        self.dropout = dropout
        self.two_stage_num_proposals = two_stage_num_proposals
        self.z_cfg = z_cfg

        if bev_encoder is None:
            self.bev_encoder = SimpleBEVEncoder(self.embed_dims, self.embed_dims)
        else:
            self.bev_encoder = builder.build_backbone(bev_encoder)
        if bev_neck is not None:
            self.bev_neck = builder.build_neck(bev_neck)
        else:
            self.bev_neck = None
        self.num_vec_one2one = num_vec_one2one
        self.num_classes = num_classes
        self.num_transformer_feat_level = num_transformer_feat_level
        self.instance_query_feat = nn.Embedding(self.num_vec_one2one, self.embed_dims)

        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, self.embed_dims)
        self.cls_embed = nn.Linear(self.embed_dims, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.decoder_positional_encoding = SinePositionalEncoding(num_feats=self.embed_dims // 2, normalize=True)

        self.num_heads = segm_decoder.layer_cfg.cross_attn_cfg.num_heads
        self.num_segm_decoder_layers = segm_decoder.num_layers
        self.segm_decoder = build_transformer_layer_sequence(segm_decoder)
        self.decoder_embed_dims = self.segm_decoder.embed_dims

        self.dn_enabled = dn_enabled
        self.dn_group_num = dn_group_num
        self.dn_noise_scale = dn_noise_scale
        self.pts2mask_noise_scale = pts2mask_noise_scale
        self.patch_size = patch_size
        self.thresh_of_mask_for_pos = thresh_of_mask_for_pos
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.label_enc = nn.Embedding(num_classes, self.embed_dims)

        self.feat_down_dim = feat_down_dim
        if self.feat_down_dim is not None:
            self.mlvl_down_layer = nn.ModuleList()
            for i in range(1):
                self.mlvl_down_layer.append(nn.Linear(self.feat_down_dim[0], self.feat_down_dim[1]))
        self.mask_noise_scale = mask_noise_scale
        self.mask_pred_detach = mask_pred_detach
        self.gt_mask_for_point_sampling = gt_mask_for_point_sampling
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.canvas_size = bev_h, bev_w
        self.num_max_survival = num_max_survival

        self.init_layers()
        self.feat_down_sample_indice = feat_down_sample_indice

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.reference_points = nn.Linear(self.embed_dims, 2) if not self.z_cfg["gt_z_flag"] else nn.Linear(self.embed_dims, 3)

        self.quert_dir = nn.Sequential(nn.Linear(self.embed_dims, self.embed_dims))

        self.lin_q = nn.Linear(self.embed_dims * 2, self.embed_dims * 2)
        self.lin_k_p = nn.Sequential(
            nn.Linear(2, self.embed_dims * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
        )
        self.lin_v_p = nn.Sequential(
            nn.Linear(2, self.embed_dims * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
        )

        self.lin_k = nn.Linear(self.embed_dims, self.embed_dims * 2)
        self.lin_v = nn.Linear(self.embed_dims, self.embed_dims * 2)

        self.img_coord_embed = nn.Linear(2, self.embed_dims)

        self.lin_k_pf_cat = nn.Sequential(
            nn.Linear(self.embed_dims * 4, self.embed_dims * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
        )

        self.lin_v_pf_cat = nn.Sequential(
            nn.Linear(self.embed_dims * 4, self.embed_dims * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
        )

        self.lin_self = nn.Linear(self.embed_dims * 2, self.embed_dims * 2)
        self.lin_ih = nn.Linear(self.embed_dims * 2, self.embed_dims * 2)
        self.lin_hh = nn.Linear(self.embed_dims * 2, self.embed_dims * 2)
        self.out_proj = nn.Linear(self.embed_dims * 2, self.embed_dims * 2)

        self.norm1 = nn.LayerNorm(self.embed_dims * 2)
        self.norm2 = nn.LayerNorm(self.embed_dims * 2)

        self.mlp2 = nn.Sequential(
            nn.Linear(self.embed_dims * 2, self.embed_dims * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims * 4, self.embed_dims * 2),
        )

        self.m_attn = nn.MultiheadAttention(self.embed_dims * 2, 8)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        xavier_init(self.reference_points, distribution="uniform", bias=0.0)

        for p in self.segm_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def lss_bev_encode(self, mlvl_feats, prev_bev=None, **kwargs):

        images = mlvl_feats[self.feat_down_sample_indice]
        img_metas = kwargs["img_metas"]
        encoder_outputdict = self.encoder(images, img_metas)
        bev_embed = encoder_outputdict["bev"]
        depth = encoder_outputdict["depth"]
        bs, c, _, _ = bev_embed.shape
        bev_embed = bev_embed.view(bs, c, -1).permute(0, 2, 1).contiguous()
        ret_dict = dict(bev=bev_embed, depth=depth)
        return ret_dict

    def get_bev_features(
        self,
        mlvl_feats,
        lidar_feat,
        bev_queries,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        prev_bev=None,
        **kwargs,
    ):
        """
        obtain bev features.
        """

        ret_dict = self.lss_bev_encode(mlvl_feats, prev_bev=prev_bev, **kwargs)
        bev_embed = ret_dict["bev"]
        depth = ret_dict["depth"]

        if lidar_feat is not None:
            bs = mlvl_feats[0].size(0)
            bev_embed = bev_embed.view(bs, bev_h, bev_w, -1).permute(0, 3, 1, 2).contiguous()
            lidar_feat = lidar_feat.permute(0, 1, 3, 2).contiguous()  # B C H W
            lidar_feat = nn.functional.interpolate(lidar_feat, size=(bev_h, bev_w), mode="bicubic", align_corners=False)
            fused_bev = self.fuser([bev_embed, lidar_feat])
            fused_bev = fused_bev.flatten(2).permute(0, 2, 1).contiguous()
            bev_embed = fused_bev
        ret_dict = dict(bev=bev_embed, depth=depth)
        return ret_dict

    def prepare_for_dn_input_pts2mask(self, batch_size, mask_feat, instance_query_feat, size_lists, img_metas):
        bs, f_dim, h, w = mask_feat.shape
        device = instance_query_feat.device
        num_vec = instance_query_feat.shape[1]
        targets = [
            {
                "gt_pts_list": m["gt_bboxes_3d"].data.fixed_num_sampled_points_condi.cuda(),
                "gt_masks_list": m["gt_bboxes_3d"].data.instance_segments_condi.cuda(),
                "labels": m["gt_bboxes_3d"].data.gt_labels.cuda().long(),
                "gt_bboxes_list": m["gt_bboxes_3d"].data.bbox_condi.cuda(),
            }
            for m in img_metas
        ]
        known = [torch.ones_like(t["labels"], device=device) for t in targets]
        known_num = [sum(k) for k in known]
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets]).clone()
        bboxes = torch.cat([t["gt_bboxes_list"] for t in targets]).clone()
        masks = [
            F.interpolate(targets[i]["gt_masks_list"].unsqueeze(1).float(), size=mask_feat.shape[-2:], mode="bilinear") for i in range(len(targets))
        ]

        gt_pts_list = torch.cat([t["gt_pts_list"] for t in targets]).clone()

        dn_single_pad = int(max(known_num))
        dn_pad_size = int(dn_single_pad * self.dn_group_num)
        batch_idx = torch.cat([torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)
        known_indice = known_indice.repeat(self.dn_group_num).view(-1)

        known_labels = labels.repeat(self.dn_group_num).view(-1)
        known_bid = batch_idx.repeat(self.dn_group_num).view(-1)
        known_bboxes = bboxes.repeat(self.dn_group_num, 1)
        known_masks = torch.cat(masks, dim=0).squeeze(1).repeat(self.dn_group_num, 1, 1)
        known_gt_pts_list = gt_pts_list.repeat(self.dn_group_num, 1, 1)

        yx_scale = known_gt_pts_list[..., 1].max() / known_gt_pts_list[..., 0].max()
        pts_distance = torch.sqrt(((known_gt_pts_list[:, 0] - known_gt_pts_list[:, 1]) ** 2).sum(-1))
        rand_prob = torch.rand(known_gt_pts_list.shape).cuda()
        diff = (rand_prob * pts_distance[:, None, None]) * self.pts2mask_noise_scale * torch.tensor((1, yx_scale)).cuda()[None]
        noise_known_gt_pts_list = known_gt_pts_list + diff

        segm_list = []
        scale_y = self.canvas_size[0] / self.patch_size[0]
        scale_x = self.canvas_size[1] / self.patch_size[1]
        trans_x = self.canvas_size[1] / 2
        trans_y = self.canvas_size[0] / 2
        for pts in noise_known_gt_pts_list:
            instance_segm = np.zeros(self.canvas_size, dtype=np.uint8)
            line_ego = affinity.scale(LineString(pts), scale_x, scale_y, origin=(0, 0))
            line_ego = affinity.affine_transform(line_ego, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            coords = np.array(list(line_ego.coords), dtype=np.int32)[:, :2]
            coords = coords.reshape((-1, 2))
            assert len(coords) >= 2
            cv2.polylines(instance_segm, np.int32([coords]), False, color=1, thickness=3)
            segm_list.append(instance_segm)

        segm_tensor = to_tensor(segm_list).cuda()
        masks_for_attn = (F.interpolate(segm_tensor.float().unsqueeze(1), size=size_lists[-1], mode="nearest") <= 1e-8).squeeze(1)
        padding_mask_3level = []
        for i in range(len(size_lists)):
            padding_mask = torch.ones([bs, dn_pad_size, size_lists[i][0] * size_lists[i][1]]).cuda().bool()
            padding_mask_3level.append(padding_mask)

        masks_3level = []
        known_boxes_expand = torch.stack(
            (
                (known_bboxes[..., 3] + known_bboxes[..., 1]) / 2,
                (known_bboxes[..., 2] + known_bboxes[..., 0]) / 2,
                (known_bboxes[..., 3] - known_bboxes[..., 1]),
                (known_bboxes[..., 2] - known_bboxes[..., 0]),
            ),
            dim=1,
        )
        diff = torch.zeros_like(known_boxes_expand)
        diff[..., :2] = torch.abs(known_boxes_expand[..., :2]) / 4 * self.mask_noise_scale
        diff[..., 2:] = known_boxes_expand[..., 2:] / 2 * self.mask_noise_scale
        delta_masks = torch.mul((torch.rand_like(known_boxes_expand) * 2 - 1.0), diff)
        is_scale = torch.rand_like(known_boxes_expand[..., 0])
        new_masks = []
        scale_noise = torch.rand_like(known_boxes_expand[..., 0]).to(known_boxes_expand) * self.mask_noise_scale * 1.5

        scale_size = (torch.tensor(size_lists[-1]).float().to(known_boxes_expand)[None] * (1 + scale_noise)[:, None]).long() + 1
        delta_center = (torch.tensor(size_lists[-1])[None].to(known_boxes_expand) - scale_size).to(known_boxes_expand) * (
            known_boxes_expand[..., :2] / torch.tensor(size_lists[-1]).to(known_boxes_expand)[None]
        )
        scale_size = scale_size.tolist()

        for mask, delta_mask, sc, noise_scale, dc in zip(masks_for_attn, delta_masks, is_scale, scale_size, delta_center):
            mask_scale = F.interpolate(mask[None][None].float(), noise_scale, mode="nearest")[0][0]
            x_, y_ = torch.where(mask_scale < 0.5)
            x_ += dc[0].long()
            y_ += dc[1].long()
            delta_x = delta_mask[0]
            delta_y = delta_mask[1]
            x_ = x_ + delta_x
            y_ = y_ + delta_y
            x_ = x_.clamp(min=0, max=size_lists[-1][-2] - 1)
            y_ = y_.clamp(min=0, max=size_lists[-1][-1] - 1)
            mask = torch.ones_like(mask, dtype=torch.bool)
            mask[x_.long(), y_.long()] = False
            new_masks.append(mask)

        new_masks = torch.stack(new_masks)
        noise_mask = new_masks.flatten(1)
        masks_3level.append(noise_mask)
        noise_mask = (F.interpolate(new_masks.unsqueeze(1).float(), size=size_lists[-2], mode="nearest") > 0.5).flatten(1)
        masks_3level.append(noise_mask)
        noise_mask = (F.interpolate(new_masks.unsqueeze(1).float(), size=size_lists[0], mode="nearest") > 0.5).flatten(1)
        masks_3level.append(noise_mask)

        dn_instance_query_feat = torch.zeros([dn_pad_size, instance_query_feat.shape[-1]], device=device).expand(batch_size, -1, -1)
        instance_query_feat = torch.cat([dn_instance_query_feat, instance_query_feat], dim=1)

        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + dn_single_pad * i for i in range(self.dn_group_num)]).long()

        if len(known_bid):
            known_labels_expand = known_labels.clone()
            if self.dn_label_noise_ratio > 0:
                prob = torch.rand_like(known_labels_expand.float())
                chosen_indice = prob < self.dn_label_noise_ratio
                new_label = torch.randint_like(known_labels_expand[chosen_indice], 0, self.num_classes)  # randomly put a new one here
                # gt_labels_expand.scatter_(0, chosen_indice, new_label)
                known_labels_expand[chosen_indice] = new_label
            known_features = self.label_enc(known_labels_expand)
            instance_query_feat[known_bid.long(), map_known_indice] = known_features

        total_size = dn_pad_size + num_vec
        attn_mask = torch.ones([total_size, total_size], device=device) < 0

        # original
        # match query cannot see the reconstruct
        attn_mask[dn_pad_size:, :dn_pad_size] = True
        for i in range(self.dn_group_num):
            if i == 0:
                attn_mask[dn_single_pad * i : dn_single_pad * (i + 1), dn_single_pad * (i + 1) : dn_pad_size] = True
            if i == self.dn_group_num - 1:
                attn_mask[dn_single_pad * i : dn_single_pad * (i + 1), : dn_single_pad * i] = True
            else:
                attn_mask[dn_single_pad * i : dn_single_pad * (i + 1), dn_single_pad * (i + 1) : dn_pad_size] = True
                attn_mask[dn_single_pad * i : dn_single_pad * (i + 1), : dn_single_pad * i] = True

        for i, padding_mask in enumerate(padding_mask_3level):
            padding_mask_3level[i][(known_bid, map_known_indice)] = masks_3level[2 - i]
            padding_mask_3level[i] = padding_mask_3level[i].unsqueeze(1).repeat([1, self.num_heads, 1, 1])

        mask_dict = {
            "known_indice": torch.as_tensor(known_indice).long(),
            "batch_idx": torch.as_tensor(batch_idx).long(),
            "map_known_indice": torch.as_tensor(map_known_indice, device=device).long(),
            "known_lbs_bboxes": (known_labels, known_masks),
            "pad_size": dn_pad_size,
            "dn_single_pad": dn_single_pad,
            "known_num": known_num,
        }
        return instance_query_feat, padding_mask_3level, attn_mask, mask_dict, known_bid, map_known_indice, dn_pad_size, known_masks

    def _forward_head(self, decoder_out, mask_feature, attn_mask_target_size):

        decoder_out = self.segm_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feature)
        attn_mask = F.interpolate(mask_pred, attn_mask_target_size, mode="bilinear", align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask

    def PositionalQueryGenerator(
        self,
        mask_aware_query_feat,
        pts_query,
        mask_pred,
        known_masks=None,
        known_bid=None,
        map_known_indice=None,
    ):
        bs = pts_query.shape[0]
        pts_query_pos, pts_query_feat = torch.split(pts_query, self.embed_dims, dim=-1)
        pts_query_pos = pts_query_pos.unsqueeze(0).expand(bs, -1, -1, -1)
        pts_query_feat = pts_query_feat.unsqueeze(0).expand(bs, -1, -1, -1)

        mask_aware_query_feat_pia = self.quert_dir(mask_aware_query_feat.detach())

        bs, nq, h, w = mask_pred.shape
        instance_pos_embed_mask = torch.zeros(bs, h, w).bool()
        instance_pos_embed_mask = instance_pos_embed_mask.to(mask_pred.device)
        pos_embed_map = self.decoder_positional_encoding(instance_pos_embed_mask)
        pos_embed_map = pos_embed_map.flatten(2)
        pos_embed_map = pos_embed_map.permute(0, 2, 1)
        mask_prob = mask_pred.clone().sigmoid()
        mask_prob = mask_prob.flatten(2).permute(0, 2, 1)
        if known_masks is not None:
            mask_prob = mask_prob.clone()
            mask_prob = mask_prob.permute(0, 2, 1).contiguous()
            mask_prob[known_bid, map_known_indice] = known_masks.view(known_masks.shape[0], -1)
            mask_prob = mask_prob.permute(0, 2, 1).contiguous()
        mask_bool = mask_prob > self.thresh_of_mask_for_pos
        query_pos_embeds = []
        for b_id in range(bs):
            query_pos_embed_list = []
            for q_id in range(nq):
                if mask_bool[b_id, :, q_id].sum() != 0:
                    query_pos_embed_list.append(
                        (pos_embed_map[b_id][mask_bool[b_id, :, q_id]] * mask_prob[b_id, :, q_id][mask_bool[b_id, :, q_id], None]).mean(0)
                    )
                else:
                    query_pos_embed_list.append(pos_embed_map[b_id].mean(0))

            query_pos_embeds.append(torch.stack(query_pos_embed_list))
        mask_aware_query_pos = torch.stack(query_pos_embeds)
        mask_aware_query_pos = torch.nan_to_num(mask_aware_query_pos, 0)

        mask_aware_query_feat_pia = mask_aware_query_feat_pia.unsqueeze(2) + pts_query_feat
        mask_aware_query_pos = mask_aware_query_pos.unsqueeze(2) + pts_query_pos

        return mask_aware_query_feat_pia, mask_aware_query_pos

    def GeometricFeatureExtractor(self, mask_pred_list, mask_aware_query_feat, mask_aware_query_pos, bev_embed):
        mask_aware_query_feat = mask_aware_query_feat.flatten(1, 2)
        mask_aware_query_pos = mask_aware_query_pos.flatten(1, 2)
        mask_pred = torch.stack(mask_pred_list, 1)

        seg_mask_pred, _ = mask_pred.clone().sigmoid().max(1)
        seg_mask_pred[seg_mask_pred < self.thr] = 0
        bs, q, H, W = seg_mask_pred.shape

        mask_pred_max, _ = seg_mask_pred.max(1)
        mask_pred_max = mask_pred_max.reshape(
            bs, H // self.num_max_survival, self.num_max_survival, W // self.num_max_survival, self.num_max_survival
        )
        mask_pred_max = mask_pred_max.permute(0, 1, 3, 2, 4)
        heat_map = mask_pred_max.reshape(bs, H // self.num_max_survival, W // self.num_max_survival, -1)

        _, max_inds = heat_map.max(dim=-1, keepdim=True)
        pad_heat_map = torch.zeros_like(heat_map)
        pad_heat_map = pad_heat_map.scatter_(-1, max_inds, heat_map.gather(-1, max_inds))

        pad_heat_map = pad_heat_map.reshape(bs, H // self.num_max_survival, W // self.num_max_survival, self.num_max_survival, self.num_max_survival)
        pad_heat_map = pad_heat_map.permute(0, 1, 3, 2, 4)
        pad_heat_map = pad_heat_map.reshape(bs, H, W)

        p_nodes = pad_heat_map.nonzero()
        bool_mat = torch.zeros_like(pad_heat_map)
        bool_mat[p_nodes[:, 0], p_nodes[:, 1], p_nodes[:, 2]] = 1
        bool_mat = bool_mat.bool()
        seg_mask_pred = seg_mask_pred * (bool_mat.unsqueeze(1))

        cnt_seg_mask_pred = (seg_mask_pred > 0).reshape(bs, q, -1).sum(-1)
        al_max_q = cnt_seg_mask_pred.max()
        sam_mask = cnt_seg_mask_pred > self.sam

        temp_al_result = []
        for bs_i in range(len(seg_mask_pred)):
            bs_mask_pred = seg_mask_pred[bs_i]
            nonzero_indices = bs_mask_pred.nonzero()

            result = torch.full((q, al_max_q, 2), -1, dtype=torch.long, device=bs_mask_pred.device)
            for idx, count in enumerate(cnt_seg_mask_pred[bs_i]):
                if count > self.sam:
                    result[idx, :count] = nonzero_indices[nonzero_indices[:, 0] == idx, 1:]
                    result[idx, count:] = nonzero_indices[nonzero_indices[:, 0] == idx, 1:][0]
            temp_al_result.append(result.unsqueeze(0))

        temp_al_result = torch.cat(temp_al_result, 0)
        al_result = torch.full((bs, q, self.sam, 2), -1, dtype=torch.long, device=seg_mask_pred.device)
        al_result_bool = torch.full((bs, q, self.sam), 1, dtype=torch.bool, device=seg_mask_pred.device)
        if al_max_q > self.sam:
            fps_point_ind = farthest_point_sample(temp_al_result[sam_mask].float(), self.sam)
            indices = torch.arange(fps_point_ind.shape[0])
            indices = indices.unsqueeze(1).expand(-1, self.sam)
            al_result[sam_mask] = temp_al_result[sam_mask][indices, fps_point_ind]
            al_result_bool[~sam_mask] = 0
        else:
            al_result_bool = torch.full((bs, q, self.sam), 0, dtype=torch.bool, device=seg_mask_pred.device)

        bs_pointset = al_result.clone()
        bs_pointset[bs_pointset < 0] = 0
        al_result = normalize_2d_pts(al_result[:, :, :, [1, 0]], self.ref_pc_range)
        mask_aware_query = torch.cat([mask_aware_query_feat, mask_aware_query_pos], -1)
        mask_aware_query_norm = self.norm1(mask_aware_query)
        query = self.lin_q(mask_aware_query_norm)
        query = query.reshape(bs, q, -1, self.embed_dims * 2)
        query = query.reshape(bs * q, -1, self.embed_dims * 2)
        query = query.permute(1, 0, 2)

        seg_bev_embed = bev_embed.view(bs, self.bev_h, self.bev_w, -1)
        seg_bev_embed = seg_bev_embed.permute(0, 3, 1, 2).contiguous()

        key_p = self.lin_k_p(al_result.float())
        value_p = self.lin_v_p(al_result.float())

        H, W = seg_bev_embed.shape[2:]
        Ws = torch.linspace(-1.0, 1.0, W)
        Hs = torch.linspace(-1.0, 1.0, H)
        image_coords = torch.stack(torch.meshgrid(Hs, Ws), dim=-1)
        image_coords = image_coords.to(seg_bev_embed.device)
        image_coord_embeddings = self.img_coord_embed(image_coords)
        seg_bev_embed += image_coord_embeddings[None].permute(0, 3, 1, 2)

        bs_pointset_feats = []
        for bs_i in range(bs):
            bs_pointset_feats.append(seg_bev_embed[bs_i, :, bs_pointset[bs_i, :, :, 0].long(), bs_pointset[bs_i, :, :, 1].long()].unsqueeze(0))
        bs_pointset_feats = torch.cat(bs_pointset_feats, 0)
        bs_pointset_feats = bs_pointset_feats.permute(0, 2, 3, 1)

        key_f = self.lin_k(bs_pointset_feats)
        value_f = self.lin_v(bs_pointset_feats)

        key = self.lin_k_pf_cat(torch.cat([key_f, key_p], -1))  # _cat
        value = self.lin_v_pf_cat(torch.cat([value_f, value_p], -1))

        key[~al_result_bool] = 0
        value[~al_result_bool] = 0

        key = key.reshape(bs * q, -1, self.embed_dims * 2).permute(1, 0, 2)
        value = value.reshape(bs * q, -1, self.embed_dims * 2).permute(1, 0, 2)

        return query, key, value, mask_aware_query, mask_aware_query_norm

    def IMPNet(self, mlvl_feats, lidar_feat, bev_queries, bev_h, bev_w, grid_length, bev_pos, prev_bev, **kwargs):
        ouput_dic = self.get_bev_features(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs,
        )  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bev_embed = ouput_dic["bev"]
        depth = ouput_dic["depth"]
        bs = mlvl_feats[0].size(0)

        num_vecs = self.num_vec_one2one

        bev_embed_single = bev_embed.view(bs, self.bev_h, self.bev_w, bev_embed.shape[-1]).permute(0, 3, 1, 2).contiguous()
        bev_embed_ms = self.bev_encoder(bev_embed_single)
        bev_embed_ms = self.bev_neck(bev_embed_ms)
        mask_feat = bev_embed_ms[0].to(torch.float32)

        ms_memorys = bev_embed_ms[:0:-1]

        decoder_inputs = []
        decoder_pos_encodings = []
        size_list = []
        for i in range(self.num_transformer_feat_level):
            size_list.append(ms_memorys[i].shape[-2:])
            decoder_input = ms_memorys[i]
            decoder_input = decoder_input.flatten(2)
            decoder_input = decoder_input.permute(0, 2, 1)  # [bs, hw, c]
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            mask = decoder_input.new_zeros((bs,) + ms_memorys[i].shape[-2:], dtype=torch.bool)
            decoder_pos_encoding = self.decoder_positional_encoding(mask).flatten(2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_pos_encodings.append(decoder_pos_encoding)

        instance_query_feat = self.instance_query_feat.weight[0:num_vecs].unsqueeze(0).expand(bs, -1, -1)
        instance_query_pos = None

        if self.dn_enabled and self.training:
            (instance_query_feat, padding_mask_3level, self_attn_mask, mask_dict, known_bid, map_known_indice, dn_pad_size, known_masks) = (
                self.prepare_for_dn_input_pts2mask(bs, mask_feat, instance_query_feat, size_list, kwargs["img_metas"])
            )
        else:
            mask_dict, known_masks, known_bid, map_known_indice = (None, None, None, None)
        dn_information = [mask_dict, known_masks, known_bid, map_known_indice]
        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(instance_query_feat, mask_feat, ms_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        if self.dn_enabled == False or self.training == False:
            self_attn_mask = torch.zeros([num_vecs, num_vecs]).bool().to(mlvl_feats[0].device)
            self_attn_mask[self.num_vec_one2one :, 0 : self.num_vec_one2one] = True
            self_attn_mask[0 : self.num_vec_one2one, self.num_vec_one2one :] = True
            self_attn_mask = self_attn_mask.unsqueeze(0).expand(bs, -1, -1)
            self_attn_mask = self_attn_mask.unsqueeze(1)
            self_attn_mask = self_attn_mask.repeat((1, self.num_heads, 1, 1))
            self_attn_mask = self_attn_mask.flatten(0, 1)

        else:
            attn_mask = attn_mask.view([bs, self.num_heads, -1, attn_mask.shape[-1]])
            attn_mask[:, :, :-num_vecs] = padding_mask_3level[0]
            attn_mask = attn_mask.flatten(0, 1)
            self_attn_mask = self_attn_mask.clone()
            self_attn_mask[: dn_pad_size + self.num_vec_one2one, dn_pad_size + self.num_vec_one2one :] = True
            self_attn_mask[dn_pad_size + self.num_vec_one2one :, : dn_pad_size + self.num_vec_one2one] = True
            kwargs["num_vec"] = self_attn_mask.shape[0]
            kwargs["self_attn_mask"] = self_attn_mask.clone()

        for i in range(self.num_segm_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.segm_decoder.layers[i]
            instance_query_feat = layer(
                query=instance_query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=instance_query_pos,
                key_pos=decoder_pos_encodings[level_idx],
                cross_attn_mask=attn_mask,
                self_attn_mask=self_attn_mask,
                query_key_padding_mask=None,
                key_padding_mask=None,
            )
            cls_pred, mask_pred, attn_mask = self._forward_head(
                instance_query_feat, mask_feat, ms_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:]
            )

            if self.dn_enabled and i != self.num_segm_decoder_layers - 1 and self.training:
                padding_mask = padding_mask_3level[(i + 1) % self.num_segm_decoder_layers]
                attn_mask = attn_mask.view([bs, self.num_heads, -1, attn_mask.shape[-1]])
                attn_mask[:, :, :-num_vecs] = padding_mask
                attn_mask = attn_mask.flatten(0, 1)

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return mask_pred_list, cls_pred_list, instance_query_feat, bev_embed, bev_embed_ms, depth, dn_information, kwargs

    def MaskGuidedMapDecoder(self, query, key, value, bev_embed_ms, mask_aware_query, mask_aware_query_norm, reg_branches, cls_branches, **kwargs):
        bs = mask_aware_query.shape[0]
        q = query.shape[1]//bs
        attn_output, _ = self.m_attn(query, key, value)
        attn_output = attn_output.permute(1, 0, 2)
        attn_output = attn_output.reshape(bs, q, -1, self.embed_dims * 2)
        attn_output = attn_output.reshape(bs, -1, self.embed_dims * 2)

        gate = torch.sigmoid(self.lin_ih(attn_output) + self.lin_hh(mask_aware_query_norm))
        query = mask_aware_query + self.out_proj(attn_output + gate * (self.lin_self(mask_aware_query_norm) - attn_output))
        query = query + self.mlp2(self.norm2(query))

        query_pos, query_feat = torch.split(query, self.embed_dims, dim=-1)

        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query_feat = query_feat.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed_ms_flatten = []
        spatial_flatten = []
        for lvl in range(len(bev_embed_ms)):
            B, C, H, W = bev_embed_ms[lvl].shape
            bev_embed_ms_flatten.append(bev_embed_ms[lvl].permute(0, 2, 3, 1).reshape(B, -1, C).permute(1, 0, 2))
            spatial_flatten.append((H, W))
        bev_embed_ms_flatten = torch.cat(bev_embed_ms_flatten, dim=0)
        spatial_flatten = torch.as_tensor(spatial_flatten, dtype=torch.long, device=query_feat.device)
        level_start_index = torch.cat((spatial_flatten.new_zeros((1,)), spatial_flatten.prod(1).cumsum(0)[:-1]))
        inter_states, inter_references = self.decoder(
            query=query_feat,
            key=None,
            value=bev_embed_ms_flatten,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=spatial_flatten,
            level_start_index=level_start_index,
            mlvl_feats=None,
            feat_flatten=None,
            feat_spatial_shapes=None,
            feat_level_start_index=None,
            **kwargs,
        )
        return inter_states, inter_references, init_reference_out

    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(
        self,
        mlvl_feats,
        lidar_feat,
        bev_queries,
        pts_query,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        reg_branches=None,
        cls_branches=None,
        prev_bev=None,
        **kwargs,
    ):

        (mask_pred_list, cls_pred_list, mask_aware_query_feat, bev_embed, bev_embed_ms, depth, dn_information, kwargs) = self.IMPNet(
            mlvl_feats, lidar_feat, bev_queries, bev_h, bev_w, grid_length=grid_length, bev_pos=bev_pos, prev_bev=prev_bev, **kwargs
        )

        # MMPNet
        mask_dict, known_masks, known_bid, map_known_indice = dn_information

        mask_aware_query_feat, mask_aware_query_pos = self.PositionalQueryGenerator(
            mask_aware_query_feat,
            pts_query,
            mask_pred_list[-1],
            known_masks=known_masks,
            known_bid=known_bid,
            map_known_indice=map_known_indice,
        )

        query, key, value, mask_aware_query, mask_aware_query_norm = self.GeometricFeatureExtractor(
            mask_pred_list,
            mask_aware_query_feat,
            mask_aware_query_pos,
            bev_embed,
        )

        inter_states, inter_references_out, init_reference_out = self.MaskGuidedMapDecoder(
            query,
            key,
            value,
            bev_embed_ms,
            mask_aware_query,
            mask_aware_query_norm,
            reg_branches,
            cls_branches,
            **kwargs,
        )

        return bev_embed, mask_pred_list, cls_pred_list, depth, inter_states, init_reference_out, inter_references_out, mask_dict
