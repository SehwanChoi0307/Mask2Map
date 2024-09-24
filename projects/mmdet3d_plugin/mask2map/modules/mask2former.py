import torch
import numpy as np
import torch.nn as nn
from mmdet.models import NECKS
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from torch.nn.modules.utils import _pair
from mmcv.ops import MultiScaleDeformableAttention
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmdet.models.utils.positional_encoding import SinePositionalEncoding
from mmcv.cnn import Conv2d, ConvModule, normal_init, caffe2_xavier_init, xavier_init

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


class DetrTransformerEncoderLayer(BaseModule):
    """Implements encoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(
        self,
        self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
        ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, num_fcs=2, ffn_drop=0.0, act_cfg=dict(type="ReLU", inplace=True)),
        norm_cfg=dict(type="LN"),
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        if "batch_first" not in self.self_attn_cfg:
            self.self_attn_cfg["batch_first"] = True
        else:
            assert (
                self.self_attn_cfg["batch_first"] is True
            ), "First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag."

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [build_norm_layer(self.norm_cfg, self.embed_dims)[1] for _ in range(2)]
        self.norms = ModuleList(norms_list)

    def forward(self, query, query_pos, key_padding_mask, **kwargs):
        """Forward function of an encoder layer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `query`.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor. has shape (bs, num_queries).
        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        query = self.self_attn(
            query=query, key=query, value=query, query_pos=query_pos, key_pos=query_pos, key_padding_mask=key_padding_mask, **kwargs
        )
        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.norms[1](query)

        return query


class DetrTransformerEncoder(BaseModule):
    """Encoder of DETR.

    Args:
        num_layers (int): Number of encoder layers.
        layer_cfg (:obj:`ConfigDict` or dict): the config of each encoder
            layer. All the layers will share the same config.
        num_cp (int): Number of checkpointing blocks in encoder layer.
            Default to -1.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self, num_layers, layer_cfg, num_cp=-1, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.layer_cfg = layer_cfg
        self.num_cp = num_cp
        assert self.num_cp <= self.num_layers
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([DetrTransformerEncoderLayer(**self.layer_cfg) for _ in range(self.num_layers)])

        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    "If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale."
                )
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query, query_pos, key_padding_mask, **kwargs):
        """Forward function of encoder.

        Args:
            query (Tensor): Input queries of encoder, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional embeddings of the queries, has
                shape (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).

        Returns:
            Tensor: Has shape (bs, num_queries, dim) if `batch_first` is
            `True`, otherwise (num_queries, bs, dim).
        """
        for layer in self.layers:
            query = layer(query, query_pos, key_padding_mask, **kwargs)
        return query


class DeformableDetrTransformerEncoderLayer(DetrTransformerEncoderLayer):
    """Encoder layer of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        self.self_attn = MultiScaleDeformableAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [build_norm_layer(self.norm_cfg, self.embed_dims)[1] for _ in range(2)]
        self.norms = ModuleList(norms_list)


class DeformableDetrTransformerEncoder(DetrTransformerEncoder):
    """Transformer encoder of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([DeformableDetrTransformerEncoderLayer(**self.layer_cfg) for _ in range(self.num_layers)])

        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    "If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale."
                )
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query, query_pos, key_padding_mask, spatial_shapes, level_start_index, valid_ratios, **kwargs):
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            Tensor: Output queries of Transformer encoder, which is also
            called 'encoder output embeddings' or 'memory', has shape
            (bs, num_queries, dim)
        """
        reference_points = self.get_encoder_reference_points(spatial_shapes, valid_ratios, device=query.device)
        for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
                **kwargs,
            )
        return query

    @staticmethod
    def get_encoder_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in encoder.

        Args:
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            device (obj:`device` or str): The device acquired by the
                `reference_points`.

        Returns:
            Tensor: Reference points used in decoder, has shape (bs, length,
            num_levels, 2).
        """

        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        # [bs, sum(hw), num_level, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points


class Mask2FormerTransformerEncoder(DeformableDetrTransformerEncoder):
    """Encoder in PixelDecoder of Mask2Former."""

    def forward(self, query, query_pos, key_padding_mask, spatial_shapes, level_start_index, valid_ratios, reference_points, **kwargs):
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim). If not None, it will be added to the
                `query` before forward function. Defaults to None.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 2) with the last dimension arranged
                as (cx, cy).

        Returns:
            Tensor: Output queries of Transformer encoder, which is also
            called 'encoder output embeddings' or 'memory', has shape
            (bs, num_queries, dim)
        """
        for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
                **kwargs,
            )
        return query


class MlvlPointGenerator:
    """Standard points generator for multi-level (Mlvl) feature maps in 2D
    points-based detectors.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        offset (float): The offset of points, the value is normalized with
            corresponding stride. Defaults to 0.5.
    """

    def __init__(self, strides, offset=0.5):
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self) -> int:
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    @property
    def num_base_priors(self):
        """list[int]: The number of priors (points) at a point
        on the feature grid"""
        return [1 for _ in range(len(self.strides))]

    def _meshgrid(self, x, y, row_major=True):
        yy, xx = torch.meshgrid(y, x)
        if row_major:
            # warning .flatten() would cause error in ONNX exporting
            # have to use reshape here
            return xx.reshape(-1), yy.reshape(-1)

        else:
            return yy.reshape(-1), xx.reshape(-1)

    def grid_priors(self, featmap_sizes, dtype=torch.float32, device="cuda", with_stride=False):
        """Generate grid points of multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            dtype (:obj:`dtype`): Dtype of priors. Defaults to torch.float32.
            device (str | torch.device): The device where the anchors will be
                put on.
            with_stride (bool): Whether to concatenate the stride to
                the last dimension of points.

        Return:
            list[torch.Tensor]: Points of  multiple feature levels.
            The sizes of each tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """

        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(featmap_sizes[i], level_idx=i, dtype=dtype, device=device, with_stride=with_stride)
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(self, featmap_size, level_idx, dtype=torch.float32, device="cuda", with_stride=False):
        """Generate grid Points of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps, arrange as
                (h, w).
            level_idx (int): The index of corresponding feature map level.
            dtype (:obj:`dtype`): Dtype of priors. Defaults to torch.float32.
            device (str | torch.device): The device the tensor will be put on.
                Defaults to 'cuda'.
            with_stride (bool): Concatenate the stride to the last dimension
                of points.

        Return:
            Tensor: Points of single feature levels.
            The shape of tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w, device=device) + self.offset) * stride_w
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_x = shift_x.to(dtype)

        shift_y = (torch.arange(0, feat_h, device=device) + self.offset) * stride_h
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_y = shift_y.to(dtype)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            # use `shape[0]` instead of `len(shift_xx)` for ONNX export
            stride_w = shift_xx.new_full((shift_xx.shape[0],), stride_w).to(dtype)
            stride_h = shift_xx.new_full((shift_yy.shape[0],), stride_h).to(dtype)
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h], dim=-1)
        all_points = shifts.to(device)
        return all_points

    def valid_flags(self, featmap_sizes, pad_shape, device="cuda"):
        """Generate valid flags of points of multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            pad_shape (tuple(int)): The padded shape of the image,
                arrange as (h, w).
            device (str | torch.device): The device where the anchors will be
                put on.

        Return:
            list(torch.Tensor): Valid flags of points of multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            point_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / point_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w), device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self, featmap_size, valid_size, device="cuda"):
        """Generate the valid flags of points of a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps, arrange as
                as (h, w).
            valid_size (tuple[int]): The valid size of the feature maps.
                The size arrange as as (h, w).
            device (str | torch.device): The device where the flags will be
            put on. Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each points in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid

    def sparse_priors(self, prior_idxs, featmap_size, level_idx, dtype=torch.float32, device="cuda"):
        """Generate sparse points according to the ``prior_idxs``.

        Args:
            prior_idxs (Tensor): The index of corresponding anchors
                in the feature map.
            featmap_size (tuple[int]): feature map size arrange as (w, h).
            level_idx (int): The level index of corresponding feature
                map.
            dtype (obj:`torch.dtype`): Date type of points. Defaults to
                ``torch.float32``.
            device (str | torch.device): The device where the points is
                located.
        Returns:
            Tensor: Anchor with shape (N, 2), N should be equal to
            the length of ``prior_idxs``. And last dimension
            2 represent (coord_x, coord_y).
        """
        height, width = featmap_size
        x = (prior_idxs % width + self.offset) * self.strides[level_idx][0]
        y = ((prior_idxs // width) % height + self.offset) * self.strides[level_idx][1]
        prioris = torch.stack([x, y], 1).to(dtype)
        prioris = prioris.to(device)
        return prioris


@NECKS.register_module()
class MSDeformAttnPixelDecoder(BaseModule):
    """Pixel decoder with multi-scale deformable attention.

    Args:
        in_channels (list[int] | tuple[int]): Number of channels in the
            input feature maps.
        strides (list[int] | tuple[int]): Output strides of feature from
            backbone.
        feat_channels (int): Number of channels for feature.
        out_channels (int): Number of channels for output.
        num_outs (int): Number of output scales.
        norm_cfg (:obj:`ConfigDict` or dict): Config for normalization.
            Defaults to dict(type='GN', num_groups=32).
        act_cfg (:obj:`ConfigDict` or dict): Config for activation.
            Defaults to dict(type='ReLU').
        encoder (:obj:`ConfigDict` or dict): Config for transformer
            encoder. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer encoder position encoding. Defaults to
            dict(num_feats=128, normalize=True).
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(
        self,
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_outs=3,
        norm_cfg=dict(type="GN", num_groups=32),
        act_cfg=dict(type="ReLU"),
        encoder=None,
        positional_encoding=dict(num_feats=128, normalize=True),
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.strides = strides
        self.num_input_levels = len(in_channels)
        self.num_encoder_levels = encoder.layer_cfg.self_attn_cfg.num_levels
        assert self.num_encoder_levels >= 1, "num_levels in attn_cfgs must be at least one"
        input_conv_list = []
        # from top to down (low to high resolution)
        for i in range(self.num_input_levels - 1, self.num_input_levels - self.num_encoder_levels - 1, -1):
            input_conv = ConvModule(in_channels[i], feat_channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=None, bias=True)
            input_conv_list.append(input_conv)
        self.input_convs = ModuleList(input_conv_list)

        self.encoder = Mask2FormerTransformerEncoder(**encoder)
        self.postional_encoding = SinePositionalEncoding(**positional_encoding)
        # high resolution to low resolution
        self.level_encoding = nn.Embedding(self.num_encoder_levels, feat_channels)

        # fpn-like structure
        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = norm_cfg is None
        # from top to down (low to high resolution)
        # fpn for the rest features that didn't pass in encoder
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1, -1):
            lateral_conv = ConvModule(in_channels[i], feat_channels, kernel_size=1, bias=self.use_bias, norm_cfg=norm_cfg, act_cfg=None)
            output_conv = ConvModule(
                feat_channels, feat_channels, kernel_size=3, stride=1, padding=1, bias=self.use_bias, norm_cfg=norm_cfg, act_cfg=act_cfg
            )
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self.mask_feature = Conv2d(feat_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.num_outs = num_outs
        self.point_generator = MlvlPointGenerator(strides)

    def init_weights(self) -> None:
        """Initialize weights."""
        for i in range(0, self.num_encoder_levels):
            xavier_init(self.input_convs[i].conv, gain=1, bias=0, distribution="uniform")

        for i in range(0, self.num_input_levels - self.num_encoder_levels):
            caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
            caffe2_xavier_init(self.output_convs[i].conv, bias=0)

        caffe2_xavier_init(self.mask_feature, bias=0)

        normal_init(self.level_encoding, mean=0, std=1)
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        # init_weights defined in MultiScaleDeformableAttention
        for m in self.encoder.layers.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()

    def forward(self, feats):
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c, h, w).

        Returns:
            tuple: A tuple containing the following:

                - mask_feature (Tensor): shape (batch_size, c, h, w).
                - multi_scale_features (list[Tensor]): Multi scale \
                        features, each in shape (batch_size, c, h, w).
        """
        # generate padding mask for each level, for each image
        batch_size = feats[0].shape[0]
        encoder_input_list = []
        padding_mask_list = []
        level_positional_encoding_list = []
        spatial_shapes = []
        reference_points_list = []
        for i in range(self.num_encoder_levels):
            level_idx = self.num_input_levels - i - 1
            feat = feats[level_idx]
            feat_projected = self.input_convs[i](feat)
            feat_hw = torch._shape_as_tensor(feat)[2:].to(feat.device)

            # no padding
            padding_mask_resized = feat.new_zeros((batch_size,) + feat.shape[-2:], dtype=torch.bool)
            pos_embed = self.postional_encoding(padding_mask_resized)
            level_embed = self.level_encoding.weight[i]
            level_pos_embed = level_embed.view(1, -1, 1, 1) + pos_embed
            # (h_i * w_i, 2)
            reference_points = self.point_generator.single_level_grid_priors(feat.shape[-2:], level_idx, device=feat.device)
            # normalize
            feat_wh = feat_hw.unsqueeze(0).flip(dims=[0, 1])
            factor = feat_wh * self.strides[level_idx]
            reference_points = reference_points / factor

            # shape (batch_size, c, h_i, w_i) -> (h_i * w_i, batch_size, c)
            feat_projected = feat_projected.flatten(2).permute(0, 2, 1)
            level_pos_embed = level_pos_embed.flatten(2).permute(0, 2, 1)
            padding_mask_resized = padding_mask_resized.flatten(1)

            encoder_input_list.append(feat_projected)
            padding_mask_list.append(padding_mask_resized)
            level_positional_encoding_list.append(level_pos_embed)
            spatial_shapes.append(feat_hw)
            reference_points_list.append(reference_points)
        # shape (batch_size, total_num_queries),
        # total_num_queries=sum([., h_i * w_i,.])
        padding_masks = torch.cat(padding_mask_list, dim=1)
        # shape (total_num_queries, batch_size, c)
        encoder_inputs = torch.cat(encoder_input_list, dim=1)
        level_positional_encodings = torch.cat(level_positional_encoding_list, dim=1)
        # shape (num_encoder_levels, 2), from low
        # resolution to high resolution
        num_queries_per_level = [e[0] * e[1] for e in spatial_shapes]
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        # shape (0, h_0*w_0, h_0*w_0+h_1*w_1, ...)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = torch.cat(reference_points_list, dim=0)
        reference_points = reference_points[None, :, None].repeat(batch_size, 1, self.num_encoder_levels, 1)
        valid_radios = reference_points.new_ones((batch_size, self.num_encoder_levels, 2))
        # shape (num_total_queries, batch_size, c)
        memory = self.encoder(
            query=encoder_inputs,
            query_pos=level_positional_encodings,
            key_padding_mask=padding_masks,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_radios,
        )
        # (batch_size, c, num_total_queries)
        memory = memory.permute(0, 2, 1)

        # from low resolution to high resolution
        outs = torch.split(memory, num_queries_per_level, dim=-1)
        outs = [x.reshape(batch_size, -1, spatial_shapes[i][0], spatial_shapes[i][1]) for i, x in enumerate(outs)]

        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1, -1):
            x = feats[i]
            cur_feat = self.lateral_convs[i](x)
            y = cur_feat + F.interpolate(outs[-1], size=cur_feat.shape[-2:], mode="bilinear", align_corners=False)
            y = self.output_convs[i](y)
            outs.append(y)
        multi_scale_features = outs[: self.num_outs]

        mask_feature = self.mask_feature(outs[-1])
        multi_scale_features[-1] = mask_feature
        # return mask_feature, multi_scale_features
        return multi_scale_features[::-1]


class DetrTransformerDecoder(BaseModule):

    def __init__(self, num_layers, layer_cfg, post_norm_cfg=dict(type="LN"), return_intermediate=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.layer_cfg = layer_cfg
        self.num_layers = num_layers
        self.post_norm_cfg = post_norm_cfg
        self.return_intermediate = return_intermediate
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([DetrTransformerDecoderLayer(**self.layer_cfg) for _ in range(self.num_layers)])
        self.embed_dims = self.layers[0].embed_dims
        self.post_norm = build_norm_layer(self.post_norm_cfg, self.embed_dims)[1]

    def forward(self, query, key, value, query_pos, key_pos, key_padding_mask, **kwargs):
        """Forward function of decoder
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor): The input key, has shape (bs, num_keys, dim).
            value (Tensor): The input value with the same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`, with the
                same shape as `query`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`.
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).

        Returns:
            Tensor: The forwarded results will have shape
            (num_decoder_layers, bs, num_queries, dim) if
            `return_intermediate` is `True` else (1, bs, num_queries, dim).
        """
        intermediate = []
        for layer in self.layers:
            query = layer(query, key=key, value=value, query_pos=query_pos, key_pos=key_pos, key_padding_mask=key_padding_mask, **kwargs)
            if self.return_intermediate:
                intermediate.append(self.post_norm(query))
        query = self.post_norm(query)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return query.unsqueeze(0)


class DetrTransformerDecoderLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(
        self,
        self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0, batch_first=True),
        cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0, batch_first=True),
        ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, num_fcs=2, ffn_drop=0.0, act_cfg=dict(type="ReLU", inplace=True)),
        norm_cfg=dict(type="LN"),
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        self.cross_attn_cfg = cross_attn_cfg
        if "batch_first" not in self.self_attn_cfg:
            self.self_attn_cfg["batch_first"] = True
        else:
            assert (
                self.self_attn_cfg["batch_first"] is True
            ), "First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag."

        if "batch_first" not in self.cross_attn_cfg:
            self.cross_attn_cfg["batch_first"] = True
        else:
            assert (
                self.cross_attn_cfg["batch_first"] is True
            ), "First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag."

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiheadAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [build_norm_layer(self.norm_cfg, self.embed_dims)[1] for _ in range(3)]
        self.norms = ModuleList(norms_list)

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        self_attn_mask=None,
        cross_attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):

        query = self.self_attn(query=query, key=query, value=query, query_pos=query_pos, key_pos=query_pos, attn_mask=self_attn_mask, **kwargs)
        query = self.norms[0](query)
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Mask2FormerTransformerDecoder(DetrTransformerDecoder):
    """Decoder of Mask2Former."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([Mask2FormerTransformerDecoderLayer(**self.layer_cfg) for _ in range(self.num_layers)])
        self.embed_dims = self.layers[0].embed_dims
        self.post_norm = build_norm_layer(self.post_norm_cfg, self.embed_dims)[1]


class Mask2FormerTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Implements decoder layer in Mask2Former transformer."""

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        self_attn_mask=None,
        cross_attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):

        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )
        query = self.norms[0](query)
        query = self.self_attn(query=query, key=query, value=query, query_pos=query_pos, key_pos=query_pos, attn_mask=self_attn_mask, **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query
