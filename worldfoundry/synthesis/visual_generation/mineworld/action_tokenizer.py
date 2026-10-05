import attr
import collections
import numpy as np
from typing import Dict, Union

from utils import print0


# https://github.com/openai/Video-Pre-Training/blob/main/lib/actions.py#L8 with some modifications
class Buttons:
    # 14 in total without hotbar and camera
    ATTACK = "attack"
    BACK = "back"
    FORWARD = "forward"
    JUMP = "jump"
    LEFT = "left"
    RIGHT = "right"
    SNEAK = "sneak"
    SPRINT = "sprint"
    USE = "use"
    DROP = "drop"
    INVENTORY = "inventory"
    # added by Yang
    ESC = "ESC"
    SWAPHANDS = "swapHands"
    PICKITEM = "pickItem"

    ALL = [
        USE,
        ATTACK,

        
        FORWARD,
        BACK,
        LEFT,
        RIGHT,

        JUMP,
        SNEAK,
        SPRINT,
        
        DROP,
        SWAPHANDS,
        PICKITEM,

        INVENTORY,
        ESC,
    ] + [f"hotbar.{i}" for i in range(1, 10)]


class QuantizationScheme:
    LINEAR = "linear"
    MU_LAW = "mu_law"


# https://github.com/openai/Video-Pre-Training/blob/main/lib/actions.py#L49
@attr.s(auto_attribs=True)
class CameraQuantizer:
    """
    A camera quantizer that discretizes and undiscretizes a continuous camera input with y (pitch) and x (yaw) components.

    Parameters:
    - camera_binsize: The size of the bins used for quantization. In case of mu-law quantization, it corresponds to the average binsize.
    - camera_maxval: The maximum value of the camera action.
    - quantization_scheme: The quantization scheme to use. Currently, two quantization schemes are supported:
    - Linear quantization (default): Camera actions are split uniformly into discrete bins
    - Mu-law quantization: Transforms the camera action using mu-law encoding (https://en.wikipedia.org/wiki/%CE%9C-law_algorithm)
    followed by the same quantization scheme used by the linear scheme.
    - mu: Mu is the parameter that defines the curvature of the mu-law encoding. Higher values of
    mu will result in a sharper transition near zero. Below are some reference values listed
    for choosing mu given a constant maxval and a desired max_precision value.
    maxval = 10 | max_precision = 0.5  | μ ≈ 2.93826
    maxval = 10 | max_precision = 0.4  | μ ≈ 4.80939
    maxval = 10 | max_precision = 0.25 | μ ≈ 11.4887
    maxval = 20 | max_precision = 0.5  | μ ≈ 2.7
    maxval = 20 | max_precision = 0.4  | μ ≈ 4.39768
    maxval = 20 | max_precision = 0.25 | μ ≈ 10.3194
    maxval = 40 | max_precision = 0.5  | μ ≈ 2.60780
    maxval = 40 | max_precision = 0.4  | μ ≈ 4.21554
    maxval = 40 | max_precision = 0.25 | μ ≈ 9.81152
    """

    camera_maxval: int
    camera_binsize: int
    quantization_scheme: str = attr.ib(
        default=QuantizationScheme.LINEAR,
        validator=attr.validators.in_([QuantizationScheme.LINEAR, QuantizationScheme.MU_LAW]),
    )
    mu: float = attr.ib(default=5)

    def discretize(self, xy):
        xy = np.clip(xy, -self.camera_maxval, self.camera_maxval)

        if self.quantization_scheme == QuantizationScheme.MU_LAW:
            xy = xy / self.camera_maxval
            v_encode = np.sign(xy) * (np.log(1.0 + self.mu * np.abs(xy)) / np.log(1.0 + self.mu))
            v_encode *= self.camera_maxval
            xy = v_encode

        # Quantize using linear scheme
        return np.round((xy + self.camera_maxval) / self.camera_binsize).astype(np.int64)

    def undiscretize(self, xy):
        xy = xy * self.camera_binsize - self.camera_maxval

        if self.quantization_scheme == QuantizationScheme.MU_LAW:
            xy = xy / self.camera_maxval
            v_decode = np.sign(xy) * (1.0 / self.mu) * ((1.0 + self.mu) ** np.abs(xy) - 1.0)
            v_decode *= self.camera_maxval
            xy = v_decode
        return xy


class MinecraftActionTokenizer:
    """Convert MineWorld action dictionaries into model action token ids."""

    def __init__(self,
                 action_length: int = 11,  # including bos and eos
                 camera_binsize: int = 9,  # 2 in vpt
                 camera_maxval: int = 90,  # 10 in vpt
                 camera_mu: float = 11.4887,  # 10 in vpt
                 quantization_scheme: str = "mu_law",
    ):
        self.action_length = action_length
        self.camera_quantizer = CameraQuantizer(
            camera_binsize=camera_binsize,
            camera_maxval=camera_maxval,
            mu=camera_mu,
            quantization_scheme=quantization_scheme,
        )

    def make_action_vocab(self,
                          num_cam_bins: int = 21,
                          action_vocab_offset: int = 0,
                          verbose: bool = False):
        action_vocab = collections.OrderedDict()
        # 14 actions and hotbar.1-9
        for i, action in enumerate(Buttons.ALL):
            action_vocab[action] = i
        # camera 0 
        for i in range(num_cam_bins):
            action_vocab[f"cam_0_{i}"] = len(Buttons.ALL) + i
        # camera 1
        for i in range(num_cam_bins):
            action_vocab[f"cam_1_{i}"] = len(Buttons.ALL) + num_cam_bins + i
        # bos, null, eos
        action_vocab["<act_bos>"] = len(Buttons.ALL) + 2 * num_cam_bins
        action_vocab["<null_act>"] = len(Buttons.ALL) + 2 * num_cam_bins + 1
        action_vocab["<act_eos>"] = len(Buttons.ALL) + 2 * num_cam_bins + 2

        if action_vocab_offset > 0:
            action_vocab = {k: v + action_vocab_offset for k, v in action_vocab.items()}

        if verbose:
            print0(f"[bold yellow][MineWorldActionTokenizer][/bold yellow] Action Vocab: {action_vocab}")

        self.action_vocab = action_vocab

    def _handle_conflict_action_index(self,
                                action_dict: Dict[str, Union[int, np.ndarray]],
                                key1: str,
                                key2: str,
                                null_key: str,
                                verbose: bool = False):
        if action_dict[key1] == 1 and action_dict[key2] == 1:
            if verbose:
                print0(f"[bold yellow][MineWorldActionTokenizer][/bold yellow] {key1} and {key2} are both pressed")
            return self.action_vocab[null_key]
        elif action_dict[key1] == 1:
            return self.action_vocab[key1]
        elif action_dict[key2] == 1:
            return self.action_vocab[key2]
        else:
            return self.action_vocab[null_key]

    def get_action_index_from_actiondict(self,
                                         action_dict: Dict[str, Union[int, np.ndarray]],
                                         action_vocab_offset: int = 0,
                                         verbose: bool = False):

        if not hasattr(self, "action_vocab"):
            self.make_action_vocab(action_vocab_offset=action_vocab_offset, verbose=verbose)

        # action_list = [boa, camy, camx, hotbar, fore_back, left_right, sprint_sneak, use_attack, jump, drop_pick, eoa]
        # 11 actions
        action_list = [self.action_vocab["<null_act>"]] * self.action_length
        # 0 & 10
        action_list[0] = self.action_vocab["<act_bos>"]
        action_list[-1] = self.action_vocab["<act_eos>"]

        camera_action = action_dict["camera"]
        assert len(camera_action) == 2, f"MineWorld camera action length is not 2: {camera_action}"
        # camera_action should be numpy array
        if not isinstance(camera_action, np.ndarray):
            camera_action = np.array(camera_action)
        camera_action = self.camera_quantizer.discretize(camera_action)
        # 1 & 2
        action_list[1] = self.action_vocab[f"cam_0_{camera_action[0]}"]
        action_list[2] = self.action_vocab[f"cam_1_{camera_action[1]}"]

        # 3
        for i in range(1, 10):
            if f"hotbar.{i}" in action_dict and action_dict[f"hotbar.{i}"] == 1:
                action_list[3] = self.action_vocab[f"hotbar.{i}"]
                break

        # 4 forward/back
        action_list[4] = self._handle_conflict_action_index(action_dict, "forward", "back", "<null_act>", verbose=verbose)
        # 5 left/right
        action_list[5] = self._handle_conflict_action_index(action_dict, "left", "right", "<null_act>", verbose=verbose)
        # 6 sprint/sneak
        action_list[6] = self._handle_conflict_action_index(action_dict, "sprint", "sneak", "<null_act>", verbose=verbose)
        # 7 use/attack
        action_list[7] = self._handle_conflict_action_index(action_dict, "use", "attack", "<null_act>", verbose=verbose)
        # 8 jump
        action_list[8] = self.action_vocab["jump"] if action_dict["jump"] == 1 else self.action_vocab["<null_act>"]
        # 9 drop/pick
        action_list[9] = self._handle_conflict_action_index(action_dict, "drop", "pickItem", "<null_act>", verbose=verbose)

        if verbose:
            print0(f"[bold yellow][MineWorldActionTokenizer][/bold yellow] Action List: {action_list}")

        return action_list
