import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.image import one_hot_mask
from ...utils.math import generate_permute_matrix
from ..layers.basic import seq_to_2d


class AOTEngine(nn.Module):
    def __init__(self, aot_model, gpu_id=0, long_term_mem_gap=9999, short_term_mem_skip=1, max_len_long_term=9999):
        super().__init__()

        self.cfg = aot_model.cfg
        self.align_corners = aot_model.cfg.MODEL_ALIGN_CORNERS
        self.AOT = aot_model

        self.max_obj_num = aot_model.max_obj_num
        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip
        self.max_len_long_term = max_len_long_term

        self.restart_engine()

    def encode_one_img_mask(self, img=None, mask=None, frame_step=-1):
        if frame_step == -1:
            frame_step = self.frame_step

        if self.enable_offline_enc:
            curr_enc_embs = self.offline_enc_embs[frame_step]
        elif img is None:
            curr_enc_embs = None
        else:
            curr_enc_embs = self.AOT.encode_image(img)

        if mask is not None:
            curr_one_hot_mask = one_hot_mask(mask, self.max_obj_num)
        elif self.enable_offline_enc:
            curr_one_hot_mask = self.offline_one_hot_masks[frame_step]
        else:
            curr_one_hot_mask = None

        return curr_enc_embs, curr_one_hot_mask

    def offline_encoder(self, all_frames, all_masks=None):
        self.enable_offline_enc = True
        self.offline_frames = all_frames.size(0) // self.batch_size

        # extract backbone features
        self.offline_enc_embs = self.split_frames(self.AOT.encode_image(all_frames), self.batch_size)
        self.total_offline_frame_num = len(self.offline_enc_embs)

        if all_masks is not None:
            # extract mask embeddings
            offline_one_hot_masks = one_hot_mask(all_masks, self.max_obj_num)
            self.offline_masks = list(torch.split(all_masks, self.batch_size, dim=0))
            self.offline_one_hot_masks = list(torch.split(offline_one_hot_masks, self.batch_size, dim=0))

        if self.input_size_2d is None:
            self.update_size(all_frames.size()[2:], self.offline_enc_embs[0][-1].size()[2:])

    def assign_identity(self, one_hot_mask):
        if self.enable_id_shuffle:
            one_hot_mask = torch.einsum("bohw,bot->bthw", one_hot_mask, self.id_shuffle_matrix)

        id_emb = self.AOT.get_id_emb(one_hot_mask).view(self.batch_size, -1, self.enc_hw).permute(2, 0, 1)

        if self.training and self.freeze_id:
            id_emb = id_emb.detach()

        return id_emb

    def split_frames(self, xs, chunk_size):
        new_xs = []
        for x in xs:
            all_x = list(torch.split(x, chunk_size, dim=0))
            new_xs.append(all_x)
        return list(zip(*new_xs, strict=False))

    def add_reference_frame(self, img=None, mask=None, frame_step=-1, obj_nums=None, img_embs=None):
        if self.obj_nums is None and obj_nums is None:
            print("No objects for reference frame!")
            exit()
        elif obj_nums is not None:
            self.obj_nums = obj_nums

        if frame_step == -1:
            frame_step = self.frame_step

        if img_embs is None:
            curr_enc_embs, curr_one_hot_mask = self.encode_one_img_mask(img, mask, frame_step)
        else:
            _, curr_one_hot_mask = self.encode_one_img_mask(None, mask, frame_step)
            curr_enc_embs = img_embs

        if curr_enc_embs is None:
            print("No image for reference frame!")
            exit()

        if curr_one_hot_mask is None:
            print("No mask for reference frame!")
            exit()

        if self.input_size_2d is None:
            self.update_size(img.size()[2:], curr_enc_embs[-1].size()[2:])

        self.curr_enc_embs = curr_enc_embs
        self.curr_one_hot_mask = curr_one_hot_mask

        if self.pos_emb is None:
            self.pos_emb = (
                self.AOT.get_pos_emb(curr_enc_embs[-1])
                .expand(self.batch_size, -1, -1, -1)
                .view(self.batch_size, -1, self.enc_hw)
                .permute(2, 0, 1)
            )

        curr_id_emb = self.assign_identity(curr_one_hot_mask)
        self.curr_id_embs = curr_id_emb

        # self matching and propagation
        self.curr_lstt_output = self.AOT.LSTT_forward(
            curr_enc_embs, None, None, curr_id_emb, pos_emb=self.pos_emb, size_2d=self.enc_size_2d
        )

        lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories = self.curr_lstt_output

        if self.long_term_memories is None:
            self.long_term_memories = lstt_long_memories
        else:
            self.update_long_term_memory(lstt_long_memories)

        self.last_mem_step = self.frame_step

        self.short_term_memories_list = [lstt_short_memories]
        self.short_term_memories = lstt_short_memories

    def set_prev_frame(self, img=None, mask=None, frame_step=1):
        self.frame_step = frame_step
        curr_enc_embs, curr_one_hot_mask = self.encode_one_img_mask(img, mask, frame_step)

        if curr_enc_embs is None:
            print("No image for previous frame!")
            exit()

        if curr_one_hot_mask is None:
            print("No mask for previous frame!")
            exit()

        self.curr_enc_embs = curr_enc_embs
        self.curr_one_hot_mask = curr_one_hot_mask

        curr_id_emb = self.assign_identity(curr_one_hot_mask)
        self.curr_id_embs = curr_id_emb

        # self matching and propagation
        self.curr_lstt_output = self.AOT.LSTT_forward(
            curr_enc_embs, None, None, curr_id_emb, pos_emb=self.pos_emb, size_2d=self.enc_size_2d
        )

        lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories = self.curr_lstt_output

        if self.long_term_memories is None:
            self.long_term_memories = lstt_long_memories
        else:
            self.update_long_term_memory(lstt_long_memories)
        self.last_mem_step = frame_step

        self.short_term_memories_list = [lstt_short_memories]
        self.short_term_memories = lstt_short_memories

    def update_long_term_memory(self, new_long_term_memories):
        TOKEN_NUM = new_long_term_memories[0][0].shape[0]
        if self.long_term_memories is None:
            self.long_term_memories = new_long_term_memories
        updated_long_term_memories = []
        for new_long_term_memory, last_long_term_memory in zip(
            new_long_term_memories, self.long_term_memories, strict=False
        ):
            updated_e = []
            for new_e, last_e in zip(new_long_term_memory, last_long_term_memory, strict=False):
                if new_e is None or last_e is None:
                    updated_e.append(None)
                else:
                    if last_e.shape[0] >= self.max_len_long_term * TOKEN_NUM:
                        last_e = last_e[: (self.max_len_long_term - 1) * TOKEN_NUM]
                    updated_e.append(torch.cat([new_e, last_e], dim=0))
            updated_long_term_memories.append(updated_e)
        self.long_term_memories = updated_long_term_memories

    def update_short_term_memory(self, curr_mask, curr_id_emb=None, skip_long_term_update=False):
        if curr_id_emb is None:
            if len(curr_mask.size()) == 3 or curr_mask.size()[0] == 1:
                curr_one_hot_mask = one_hot_mask(curr_mask, self.max_obj_num)
            else:
                curr_one_hot_mask = curr_mask
            curr_id_emb = self.assign_identity(curr_one_hot_mask)

        lstt_curr_memories = self.curr_lstt_output[1]
        lstt_curr_memories_2d = []
        for layer_idx in range(len(lstt_curr_memories)):
            curr_k, curr_v = lstt_curr_memories[layer_idx][0], lstt_curr_memories[layer_idx][1]
            curr_k, curr_v = self.AOT.LSTT.layers[layer_idx].fuse_key_value_id(curr_k, curr_v, curr_id_emb)
            lstt_curr_memories[layer_idx][0], lstt_curr_memories[layer_idx][1] = curr_k, curr_v
            lstt_curr_memories_2d.append(
                [
                    seq_to_2d(lstt_curr_memories[layer_idx][0], self.enc_size_2d),
                    seq_to_2d(lstt_curr_memories[layer_idx][1], self.enc_size_2d),
                ]
            )

        self.short_term_memories_list.append(lstt_curr_memories_2d)
        self.short_term_memories_list = self.short_term_memories_list[-self.short_term_mem_skip :]
        self.short_term_memories = self.short_term_memories_list[0]

        if self.frame_step - self.last_mem_step >= self.long_term_mem_gap:
            # skip the update of long-term memory or not
            if not skip_long_term_update:
                self.update_long_term_memory(lstt_curr_memories)
            self.last_mem_step = self.frame_step

    def match_propogate_one_frame(self, img=None, img_embs=None):
        self.frame_step += 1
        if img_embs is None:
            curr_enc_embs, _ = self.encode_one_img_mask(img, None, self.frame_step)
        else:
            curr_enc_embs = img_embs
        self.curr_enc_embs = curr_enc_embs

        self.curr_lstt_output = self.AOT.LSTT_forward(
            curr_enc_embs,
            self.long_term_memories,
            self.short_term_memories,
            None,
            pos_emb=self.pos_emb,
            size_2d=self.enc_size_2d,
        )

    def decode_current_logits(self, output_size=None):
        curr_enc_embs = self.curr_enc_embs
        curr_lstt_embs = self.curr_lstt_output[0]

        pred_id_logits = self.AOT.decode_id_logits(curr_lstt_embs, curr_enc_embs)

        if self.enable_id_shuffle:  # reverse shuffle
            pred_id_logits = torch.einsum("bohw,bto->bthw", pred_id_logits, self.id_shuffle_matrix)

        # remove unused identities
        for batch_idx, obj_num in enumerate(self.obj_nums):
            pred_id_logits[batch_idx, (obj_num + 1) :] = -1e10 if pred_id_logits.dtype == torch.float32 else -1e4

        self.pred_id_logits = pred_id_logits

        if output_size is not None:
            pred_id_logits = F.interpolate(
                pred_id_logits, size=output_size, mode="bilinear", align_corners=self.align_corners
            )

        return pred_id_logits

    def predict_current_mask(self, output_size=None, return_prob=False):
        if output_size is None:
            output_size = self.input_size_2d

        pred_id_logits = F.interpolate(
            self.pred_id_logits, size=output_size, mode="bilinear", align_corners=self.align_corners
        )
        pred_mask = torch.argmax(pred_id_logits, dim=1)

        if not return_prob:
            return pred_mask
        else:
            pred_prob = torch.softmax(pred_id_logits, dim=1)
            return pred_mask, pred_prob

    def restart_engine(self, batch_size=1, enable_id_shuffle=False):
        self.batch_size = batch_size
        self.frame_step = 0
        self.last_mem_step = -1
        self.enable_id_shuffle = enable_id_shuffle
        self.freeze_id = False

        self.obj_nums = None
        self.pos_emb = None
        self.enc_size_2d = None
        self.enc_hw = None
        self.input_size_2d = None

        self.long_term_memories = None
        self.short_term_memories_list = []
        self.short_term_memories = None

        self.enable_offline_enc = False
        self.offline_enc_embs = None
        self.offline_one_hot_masks = None
        self.offline_frames = -1
        self.total_offline_frame_num = 0

        self.curr_enc_embs = None
        self.curr_memories = None
        self.curr_id_embs = None

        if enable_id_shuffle:
            self.id_shuffle_matrix = generate_permute_matrix(self.max_obj_num + 1, batch_size, gpu_id=self.gpu_id)
        else:
            self.id_shuffle_matrix = None

    def update_size(self, input_size, enc_size):
        self.input_size_2d = input_size
        self.enc_size_2d = enc_size
        self.enc_hw = self.enc_size_2d[0] * self.enc_size_2d[1]


class AOTInferEngine(nn.Module):
    def __init__(
        self,
        aot_model,
        gpu_id=0,
        long_term_mem_gap=9999,
        short_term_mem_skip=1,
        max_aot_obj_num=None,
        max_len_long_term=9999,
    ):
        super().__init__()

        self.cfg = aot_model.cfg
        self.AOT = aot_model

        if max_aot_obj_num is None or max_aot_obj_num > aot_model.max_obj_num:
            self.max_aot_obj_num = aot_model.max_obj_num
        else:
            self.max_aot_obj_num = max_aot_obj_num

        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip
        self.max_len_long_term = max_len_long_term
        self.aot_engines = []

        self.restart_engine()

    def restart_engine(self):
        del self.aot_engines
        self.aot_engines = []
        self.obj_nums = None

    def separate_mask(self, mask, obj_nums):
        if mask is None:
            return [None] * len(self.aot_engines)
        if len(self.aot_engines) == 1:
            return [mask], [obj_nums]

        separated_obj_nums = [self.max_aot_obj_num for _ in range(len(self.aot_engines))]
        if obj_nums % self.max_aot_obj_num > 0:
            separated_obj_nums[-1] = obj_nums % self.max_aot_obj_num

        if len(mask.size()) == 3 or mask.size()[0] == 1:
            separated_masks = []
            for idx in range(len(self.aot_engines)):
                start_id = idx * self.max_aot_obj_num + 1
                end_id = (idx + 1) * self.max_aot_obj_num
                fg_mask = ((mask >= start_id) & (mask <= end_id)).float()
                separated_mask = (fg_mask * mask - start_id + 1) * fg_mask
                separated_masks.append(separated_mask)
            return separated_masks, separated_obj_nums
        else:
            prob = mask
            separated_probs = []
            for idx in range(len(self.aot_engines)):
                start_id = idx * self.max_aot_obj_num + 1
                end_id = (idx + 1) * self.max_aot_obj_num
                fg_prob = prob[start_id : (end_id + 1)]
                bg_prob = 1.0 - torch.sum(fg_prob, dim=1, keepdim=True)
                separated_probs.append(torch.cat([bg_prob, fg_prob], dim=1))
            return separated_probs, separated_obj_nums

    def min_logit_aggregation(self, all_logits):
        if len(all_logits) == 1:
            return all_logits[0]

        fg_logits = []
        bg_logits = []

        for logit in all_logits:
            bg_logits.append(logit[:, 0:1])
            fg_logits.append(logit[:, 1 : 1 + self.max_aot_obj_num])

        bg_logit, _ = torch.min(torch.cat(bg_logits, dim=1), dim=1, keepdim=True)
        merged_logit = torch.cat([bg_logit] + fg_logits, dim=1)  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.

        return merged_logit

    def soft_logit_aggregation(self, all_logits):
        if len(all_logits) == 1:
            return all_logits[0]

        fg_probs = []
        bg_probs = []

        for logit in all_logits:
            prob = torch.softmax(logit, dim=1)
            bg_probs.append(prob[:, 0:1])
            fg_probs.append(prob[:, 1 : 1 + self.max_aot_obj_num])

        bg_prob = torch.prod(torch.cat(bg_probs, dim=1), dim=1, keepdim=True)
        merged_prob = torch.cat([bg_prob] + fg_probs, dim=1).clamp(1e-5, 1 - 1e-5)  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
        merged_logit = torch.logit(merged_prob)

        return merged_logit

    def add_reference_frame(self, img, mask, obj_nums, frame_step=-1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        self.obj_nums = obj_nums
        aot_num = max(np.ceil(obj_nums / self.max_aot_obj_num), 1)
        while aot_num > len(self.aot_engines):
            new_engine = AOTEngine(
                self.AOT,
                self.gpu_id,
                self.long_term_mem_gap,
                self.short_term_mem_skip,
                self.max_len_long_term,
            )
            new_engine.eval()
            self.aot_engines.append(new_engine)

        separated_masks, separated_obj_nums = self.separate_mask(mask, obj_nums)
        img_embs = None
        for aot_engine, separated_mask, separated_obj_num in zip(
            self.aot_engines, separated_masks, separated_obj_nums, strict=False
        ):
            aot_engine.add_reference_frame(
                img, separated_mask, obj_nums=[separated_obj_num], frame_step=frame_step, img_embs=img_embs
            )

            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

        self.update_size()

    def match_propogate_one_frame(self, img=None):
        img_embs = None
        for aot_engine in self.aot_engines:
            aot_engine.match_propogate_one_frame(img, img_embs=img_embs)
            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

    def decode_current_logits(self, output_size=None):
        all_logits = []
        for aot_engine in self.aot_engines:
            all_logits.append(aot_engine.decode_current_logits(output_size))
        pred_id_logits = self.soft_logit_aggregation(all_logits)
        return pred_id_logits

    def update_memory(self, curr_mask, skip_long_term_update=False):
        _curr_mask = F.interpolate(curr_mask, self.input_size_2d)
        separated_masks, _ = self.separate_mask(_curr_mask, self.obj_nums)
        for aot_engine, separated_mask in zip(self.aot_engines, separated_masks, strict=False):
            aot_engine.update_short_term_memory(separated_mask, skip_long_term_update=skip_long_term_update)

    def update_size(self):
        self.input_size_2d = self.aot_engines[0].input_size_2d
        self.enc_size_2d = self.aot_engines[0].enc_size_2d
        self.enc_hw = self.aot_engines[0].enc_hw
