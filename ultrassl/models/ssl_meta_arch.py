"""
Simplified SSL meta-architecture for DINOv2 ultrasound pretraining.

Same teacher-student self-distillation logic as dinov2/train/ssl_meta_arch.py
but without FSDP or xformers dependencies:
- Standard nn.Module (wrap with DDP externally if needed)
- torch.amp.GradScaler instead of ShardedGradScaler
- Sequential head processing instead of xformers BlockDiagonalMask
- Direct parameter iteration for EMA instead of get_fsdp_modules
"""

import copy
import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.layers import DINOHead

from .backbone import build_backbone

logger = logging.getLogger("ultrassl")


class UltraSSLMetaArch(nn.Module):
    """Teacher-student SSL architecture for DINOv2 ultrasound pretraining.

    Implements:
    - DINO CLS token self-distillation loss
    - iBOT masked patch prediction loss
    - KoLeo diversity regularizer
    - EMA teacher update

    No FSDP/xformers dependencies.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # --- Build backbones ---
        student_backbone, embed_dim = build_backbone(
            model_name=cfg.model.arch,
            patch_size=cfg.model.patch_size,
            pretrained=cfg.model.pretrained,
            img_size=cfg.crops.global_crops_size,
            drop_path_rate=cfg.model.drop_path_rate,
        )

        # Teacher: same arch, no drop path, no pretrained (will be copied from student)
        teacher_backbone, _ = build_backbone(
            model_name=cfg.model.arch,
            patch_size=cfg.model.patch_size,
            pretrained="",  # Will copy from student
            img_size=cfg.crops.global_crops_size,
            drop_path_rate=0.0,
        )

        self.embed_dim = embed_dim

        # --- Build heads ---
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.get("separate_head", False)

        student_model_dict = {"backbone": student_backbone}
        teacher_model_dict = {"backbone": teacher_backbone}

        dino_head_fn = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck_dim,
            nlayers=cfg.dino.head_nlayers,
        )

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head_fn()
            teacher_model_dict["dino_head"] = dino_head_fn()

        if self.do_dino:
            self.dino_loss_weight = cfg.dino.loss_weight
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                self.koleo_loss = KoLeoLoss()

        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            self.ibot_out_dim = cfg.dino.head_n_prototypes  # shared head
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)

            if self.ibot_separate_head:
                ibot_head_fn = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.get("head_hidden_dim", cfg.dino.head_hidden_dim),
                    bottleneck_dim=cfg.ibot.get("head_bottleneck_dim", cfg.dino.head_bottleneck_dim),
                    nlayers=cfg.ibot.get("head_nlayers", cfg.dino.head_nlayers),
                )
                student_model_dict["ibot_head"] = ibot_head_fn()
                teacher_model_dict["ibot_head"] = ibot_head_fn()

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # Copy student weights to teacher and freeze teacher
        self._copy_student_to_teacher()
        for p in self.teacher.parameters():
            p.requires_grad = False

        logger.info(f"UltraSSLMetaArch built: {cfg.model.arch}, embed_dim={embed_dim}")
        logger.info(f"  DINO: {self.do_dino}, iBOT: {self.do_ibot}, KoLeo: {self.do_koleo}")

    def _copy_student_to_teacher(self):
        """Copy student parameters to teacher."""
        for k in self.student.keys():
            self.teacher[k].load_state_dict(self.student[k].state_dict())

    @torch.no_grad()
    def update_teacher(self, m):
        """EMA update: teacher = m * teacher + (1 - m) * student."""
        student_params = []
        teacher_params = []
        for k in self.student.keys():
            for sp, tp in zip(self.student[k].parameters(), self.teacher[k].parameters()):
                student_params.append(sp.data)
                teacher_params.append(tp.data)
        torch._foreach_mul_(teacher_params, m)
        torch._foreach_add_(teacher_params, student_params, alpha=1 - m)

    def forward_backward(self, images, teacher_temp):
        """Full forward-backward pass for one iteration.

        Args:
            images: Dict from collate_data_and_cast with keys:
                collated_global_crops, collated_local_crops, collated_masks,
                mask_indices_list, n_masked_patches, upperbound, masks_weight
            teacher_temp: Current teacher temperature.

        Returns:
            loss_dict with individual loss components.
        """
        n_global_crops = 2
        n_local_crops = self.cfg.crops.local_crops_number

        # Device-agnostic: move tensors to wherever the model parameters live
        device = next(self.student.parameters()).device
        global_crops = images["collated_global_crops"].to(device, non_blocking=True)
        local_crops = images["collated_local_crops"].to(device, non_blocking=True)
        masks = images["collated_masks"].to(device, non_blocking=True)
        mask_indices_list = images["mask_indices_list"].to(device, non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].to(device, non_blocking=True)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        # ============================================================
        # TEACHER FORWARD (no grad)
        # ============================================================
        with torch.no_grad():
            teacher_out = self.teacher.backbone(global_crops, is_training=True)
            teacher_cls_tokens = teacher_out["x_norm_clstoken"]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops)
            # Swap: A matched to B (cross-view consistency)
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))

            ibot_teacher_patch_tokens = teacher_out["x_norm_patchtokens"]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if self.do_ibot and not self.ibot_separate_head:
                # Process CLS + masked patches through shared head
                buffer = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                buffer[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer[n_cls_tokens:n_cls_tokens + n_masked_patches],
                )
                tokens_after_head = self.teacher.dino_head(buffer)
                teacher_cls_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_after_head = tokens_after_head[n_cls_tokens:n_cls_tokens + n_masked_patches]
            elif self.do_ibot and self.ibot_separate_head:
                buffer = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer[:n_masked_patches],
                )
                teacher_cls_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_patch_after_head = self.teacher.ibot_head(buffer)[:n_masked_patches]
            else:
                teacher_cls_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_ibot_softmaxed = None

            # Center and sharpen teacher outputs
            teacher_dino_softmaxed = self.dino_loss.softmax_center_teacher(
                teacher_cls_after_head, teacher_temp=teacher_temp
            ).view(n_global_crops, -1, *teacher_cls_after_head.shape[1:])
            self.dino_loss.update_center(teacher_cls_after_head)

            if self.do_ibot:
                masked_teacher_patch_after_head_unsq = masked_teacher_patch_after_head.unsqueeze(0)
                masked_teacher_ibot_softmaxed = self.ibot_patch_loss.softmax_center_teacher(
                    masked_teacher_patch_after_head_unsq[:, :n_masked_patches],
                    teacher_temp=teacher_temp,
                )
                masked_teacher_ibot_softmaxed = masked_teacher_ibot_softmaxed.squeeze(0)
                self.ibot_patch_loss.update_center(
                    masked_teacher_patch_after_head_unsq[:n_masked_patches]
                )

        # ============================================================
        # STUDENT FORWARD (with grad)
        # ============================================================
        # Process global crops (with masks for iBOT)
        student_global_out = self.student.backbone(global_crops, masks=masks, is_training=True)
        student_global_cls = student_global_out["x_norm_clstoken"]
        student_global_patches = student_global_out["x_norm_patchtokens"]

        # Process local crops (no masks)
        student_local_out = self.student.backbone(local_crops, is_training=True)
        student_local_cls = student_local_out["x_norm_clstoken"]

        # --- Apply heads ---
        # Process CLS tokens through dino_head
        student_local_cls_after_head = self.student.dino_head(student_local_cls)
        student_global_cls_after_head = self.student.dino_head(student_global_cls)

        # Process masked patches through head
        if self.do_ibot:
            _dim = student_global_cls.shape[-1]
            buffer_student = student_global_patches.new_zeros(upperbound, _dim)
            buffer_student[:n_masked_patches].copy_(
                torch.index_select(
                    student_global_patches.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                )
            )
            if not self.ibot_separate_head:
                student_masked_patches_after_head = self.student.dino_head(buffer_student)[:n_masked_patches]
            else:
                student_masked_patches_after_head = self.student.ibot_head(buffer_student)[:n_masked_patches]

        # ============================================================
        # LOSS COMPUTATION
        # ============================================================
        loss_dict = {}
        loss_accumulator = 0.0

        # DINO loss on local crops
        if n_local_crops > 0 and self.do_dino:
            dino_local_loss = self.dino_loss(
                student_output_list=student_local_cls_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            loss_dict["dino_local_crops_loss"] = dino_local_loss
            loss_accumulator = loss_accumulator + self.dino_loss_weight * dino_local_loss

        # DINO loss on global crops (scale by 2 since processed together)
        loss_scales = 2
        if self.do_dino:
            dino_global_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_after_head],
                    teacher_out_softmaxed_centered_list=[teacher_dino_softmaxed.flatten(0, 1)],
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )
            loss_dict["dino_global_crops_loss"] = dino_global_loss
            loss_accumulator = loss_accumulator + self.dino_loss_weight * dino_global_loss

            # KoLeo regularizer
            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_global_cls.chunk(2)
                )
                loss_accumulator = loss_accumulator + koleo_loss
                loss_dict["koleo_loss"] = koleo_loss / loss_scales

        # iBOT masked patch loss
        if self.do_ibot:
            ibot_loss_scale = 1.0 / n_global_crops
            ibot_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_masked_patches_after_head,
                    masked_teacher_ibot_softmaxed,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )
            loss_dict["ibot_loss"] = ibot_loss / 2
            loss_accumulator = loss_accumulator + self.ibot_loss_weight * ibot_loss

        return loss_accumulator, loss_dict

    def get_params_groups(self):
        """Get parameter groups with layerwise LR decay for the student."""
        # Separate backbone, head, and last-layer parameters
        backbone_params = []
        head_params = []
        last_layer_params = []

        # Backbone with layerwise decay
        depth = len(list(self.student.backbone.blocks))
        lr_decay = self.cfg.optim.layerwise_decay
        patch_embed_lr_mult = self.cfg.optim.patch_embed_lr_mult

        # Patch embed: reduced LR
        for name, param in self.student.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if "patch_embed" in name:
                backbone_params.append({
                    "params": [param],
                    "lr_multiplier": patch_embed_lr_mult,
                    "wd_multiplier": 1.0,
                    "is_last_layer": False,
                })
            elif "blocks." in name:
                # Extract block index for layerwise decay
                block_idx = int(name.split("blocks.")[1].split(".")[0])
                layer_mult = lr_decay ** (depth - 1 - block_idx)
                wd_mult = 1.0 if ("weight" in name and param.ndim >= 2) else 0.0
                backbone_params.append({
                    "params": [param],
                    "lr_multiplier": layer_mult,
                    "wd_multiplier": wd_mult,
                    "is_last_layer": False,
                })
            else:
                wd_mult = 1.0 if ("weight" in name and param.ndim >= 2) else 0.0
                backbone_params.append({
                    "params": [param],
                    "lr_multiplier": 1.0,
                    "wd_multiplier": wd_mult,
                    "is_last_layer": False,
                })

        # Heads: standard LR
        for head_name in ["dino_head", "ibot_head"]:
            if head_name not in self.student:
                continue
            for name, param in self.student[head_name].named_parameters():
                if not param.requires_grad:
                    continue
                is_last = "last_layer" in name
                wd_mult = 1.0 if ("weight" in name and param.ndim >= 2) else 0.0
                head_params.append({
                    "params": [param],
                    "lr_multiplier": 1.0,
                    "wd_multiplier": wd_mult,
                    "is_last_layer": is_last,
                })

        return backbone_params + head_params

    def train(self, mode=True):
        super().train(mode)
        # Teacher always in eval mode
        self.teacher.eval()
        return self
