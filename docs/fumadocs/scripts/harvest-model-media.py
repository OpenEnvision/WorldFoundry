#!/usr/bin/env python3
"""Collect paper titles and teaser images for model homepages.

Reads the generated recipe JSON, pulls arXiv metadata, and downloads Hugging
Face paper thumbnails when they exist. Writes ``lib/model-paper-media.json``
and patches recipe/index JSON in place so the pages pick up paper figures
without a full catalog regenerate.

Never attach catalog / Studio demo clips (``kind: video``, ``/demos/*.mp4``)
to model homepages. Home, Studio gallery, and welcome still use those files
on disk — this harvest just must not emit them into model-page media.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from model_paper_figures import figure_records

DOCS_ROOT = Path(__file__).resolve().parents[1]
RECIPES = DOCS_ROOT / "lib" / "model-recipes-data.json"
INDEX = DOCS_ROOT / "lib" / "model-recipes-index.json"
OUT = DOCS_ROOT / "lib" / "model-paper-media.json"
PUBLIC = DOCS_ROOT / "public"
THUMB_DIR = PUBLIC / "models"

USER_AGENT = "WorldFoundryDocs/1.0 (model homepage media harvest)"
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf|html)|huggingface\.co/papers|hf\.co/papers)/(\d{4}\.\d{4,5})(?:v\d+)?",
    re.I,
)
GITHUB_RE = re.compile(r"https?://(?:www\.)?github\.com/([^/]+)/([^/?#]+)", re.I)

# Official paper teaser / overview figures. Download only these author-published
# assets; never a PDF page-1 render (that stays paper.png beside the title).
CURATED_FIGURES: dict[str, dict[str, str]] = {
    "a1": {
        "overview": "https://arxiv.org/html/2604.05672v3/framework.png",
    },
    "abot-m0": {
        "teaser": "https://arxiv.org/html/2607.00678v2/teaser.png",
        "overview": "https://arxiv.org/html/2607.00678v2/model.png",
    },
    "abot-world-0-5b-lf": {
        "teaser": "https://arxiv.org/html/2607.19191v1/Figs/abot-world-0.png",
        "overview": "https://arxiv.org/html/2607.19191v1/abot_pipeline_new_ppt_editable.png",
    },
    "ac3d": {
        "teaser": "https://raw.githubusercontent.com/snap-research/ac3d/main/assets/teaser.png",
        "overview": "https://arxiv.org/html/2411.18673v4/architecture.png",
    },
    "adaworld": {
        "teaser": "https://arxiv.org/html/2503.18938v4/figs/teaser-min.jpg",
        "overview": "https://arxiv.org/html/2503.18938v4/wm_compressed.png",
    },
    "ahawam": {
        "teaser": "https://arxiv.org/html/2606.09811v1/teaser.png",
        "overview": "https://arxiv.org/html/2606.09811v1/model_arch.png",
    },
    "alayaworld": {
        "teaser": "https://arxiv.org/html/2607.06291v1/fig1.png",
    },
    "allegro": {
        "teaser": "https://arxiv.org/html/2410.15458v1/teaser.png",
    },
    "animatediff": {
        "overview": "https://raw.githubusercontent.com/guoyww/AnimateDiff/main/__assets__/figs/adapter_explain.png",
    },
    "astra": {
        "overview": "https://arxiv.org/html/2512.08931v3/diagram-v4.png",
    },
    "being-h05": {
        "teaser": "https://arxiv.org/html/2601.12993/x1.png",
        "overview": "https://arxiv.org/html/2601.12993/x5.png",
    },
    "bernini": {
        "teaser": "https://arxiv.org/html/2605.22344v2/figure0_0521.png",
        "overview": "https://arxiv.org/html/2605.22344v2/framework.png",
    },
    "cameractrl": {
        "teaser": "https://arxiv.org/html/2404.02101v2/teaser.png",
        "overview": "https://arxiv.org/html/2404.02101v2/fig2.png",
    },
    "causal-forcing": {
        "teaser": "https://arxiv.org/html/2602.02214v5/Fig2.png",
        "overview": "https://arxiv.org/html/2602.02214v5/qualitive.png",
    },
    "causal-rcm": {
        "teaser": "https://arxiv.org/html/2510.08431v3/t2v-crop.png",
        "overview": "https://arxiv.org/html/2510.08431v3/pipeline-crop.png",
    },
    "cogact": {
        "overview": "https://arxiv.org/html/2411.19650v1/method-V3.png",
    },
    "cogvideox": {
        "teaser": "https://raw.githubusercontent.com/THUDM/CogVideo/CogVideo/assets/intro-image.png",
    },
    "cosmos-predict-2": {
        "overview": "https://raw.githubusercontent.com/nvidia-cosmos/cosmos-predict2/main/assets/cosmos-predict-diagram.png",
    },
    "cosmos-predict-2.5": {
        "teaser": "https://arxiv.org/html/2511.00062/x1.png",
        "overview": "https://arxiv.org/html/2511.00062/x2.png",
    },
    "cosmos3": {
        "teaser": "https://arxiv.org/html/2606.02800/x1.png",
        "overview": "https://arxiv.org/html/2606.02800/x2.png",
    },
    "cut3r": {
        "overview": "https://arxiv.org/html/2501.12387v1/v6.png",
    },
    "dap": {
        "teaser": "https://raw.githubusercontent.com/Insta360-Research-Team/DAP/main/assets/depth_teaser2_00.png",
        "overview": "https://arxiv.org/html/2512.16913/x3.png",
    },
    "depth-anything-v1": {
        "teaser": "https://raw.githubusercontent.com/LiheYoung/Depth-Anything/main/assets/teaser.png",
    },
    "depth-anything-v2-prior": {
        "teaser": "https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/assets/teaser.png",
    },
    "depth-anything-v3-prior": {
        "overview": "https://arxiv.org/html/2511.10647v1/figs/pdfs/pipeline.png",
        "teaser": "https://arxiv.org/html/2511.10647v1/x1.png",
    },
    "dexora-1b": {
        "teaser": "https://arxiv.org/html/2605.18722v1/teaser.png",
    },
    "diamond": {
        "overview": "https://arxiv.org/html/2405.12399v2/architectures_02.png",
    },
    "diffusion-policy": {
        "teaser": "https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/media/teaser.png",
    },
    "dino-wm": {
        "teaser": "https://arxiv.org/html/2411.04983v2/figures/intro.png",
        "overview": "https://arxiv.org/html/2411.04983v2/arch.png",
    },
    "dm0": {
        "overview": "https://arxiv.org/html/2510.23511v1/teaser.png",
    },
    "dreamdojo": {
        "overview": "https://arxiv.org/html/2602.06949v1/overview_compressed.png",
    },
    "dreamx-world-5b": {
        "teaser": "https://arxiv.org/html/2606.16993v1/figures/dreamx-world_teaser_fig.jpg",
        "overview": "https://arxiv.org/html/2606.16993v1/pipeline_overall.png",
    },
    "dreamzero": {
        "teaser": "https://arxiv.org/html/2602.15922v1/dreamzero-header-v2.png",
        "overview": "https://arxiv.org/html/2602.15922v1/dreamzero_model.png",
    },
    "dualcamctrl": {
        "overview": "https://arxiv.org/html/2511.23127v2/dualcam_framework.png",
    },
    "dust3r": {
        "teaser": "https://raw.githubusercontent.com/naver/dust3r/main/assets/dust3r.jpg",
    },
    "dvlt": {
        "teaser": "https://arxiv.org/html/2605.30215v2/teaser_figure.png",
        "overview": "https://arxiv.org/html/2605.30215v2/method_overview_v3.png",
    },
    "dynamicrafter": {
        "overview": "https://arxiv.org/html/2310.12190v2/overview3.png",
    },
    "easyanimate": {
        "teaser": "https://arxiv.org/html/2405.18991/x1.png",
        "overview": "https://arxiv.org/html/2405.18991/x2.png",
    },
    "echo-infinity": {
        "teaser": "https://arxiv.org/html/2606.04527v1/1-teaser.png",
    },
    "echo-memory-context-k1": {
        "teaser": "https://arxiv.org/html/2606.09803v1/figure_1_abs_framework.png",
    },
    "eventvla": {
        "overview": "https://arxiv.org/html/2606.20092v2/Overview.png",
    },
    "evoke": {
        "teaser": "https://arxiv.org/html/2608.13546v2/fig_l_hourscale_demo.png",
    },
    "fantasyworld": {
        "teaser": "https://arxiv.org/html/2509.21657v2/FantasyWorld-Overview_v2-reduced.png",
        "overview": "https://raw.githubusercontent.com/Fantasy-AMAP/fantasy-world/main/assets/overview.png",
    },
    "fastvideo-causal-wan2.2": {
        "teaser": "https://arxiv.org/html/2606.03159v2/teaser.png",
    },
    "fastwam": {
        "teaser": "https://arxiv.org/html/2603.16666v2/teaser_main_new.png",
        "overview": "https://arxiv.org/html/2603.16666v2/model_arch.png",
    },
    "flashworld": {
        "teaser": "https://arxiv.org/html/2510.13678v1/teaser.png",
        "overview": "https://arxiv.org/html/2510.13678v1/method.png",
    },
    "framepack": {
        "teaser": "https://arxiv.org/html/2504.12626v3/sampling_v2.png",
    },
    "galaxea-vla": {
        "teaser": "https://arxiv.org/html/2608.11739v1/tokenizer_v2.png",
    },
    "gamma-world": {
        "teaser": "https://arxiv.org/html/2605.28816v1/teaser.png",
        "overview": "https://arxiv.org/html/2605.28816v1/multiagent_method.png",
    },
    "gen3c": {
        "teaser": "https://arxiv.org/html/2503.03751v1/motivation_img.png",
        "overview": "https://arxiv.org/html/2503.03751v1/method_compatible.png",
    },
    "genie-envisioner": {
        "teaser": "https://arxiv.org/html/2508.05635v3/Banner.png",
        "overview": "https://raw.githubusercontent.com/AgibotTech/Genie-Envisioner/master/figs/overview.png",
    },
    "geocalib-prior": {
        "teaser": "https://arxiv.org/html/2409.06704v2/teaser_v1_compressed.png",
        "overview": "https://arxiv.org/html/2409.06704v2/architecture_v2_compressed.png",
    },
    "giga-brain-0": {
        "overview": "https://arxiv.org/html/2608.15875v1/gigabrain07_teaser_compressed.png",
    },
    "giga-world-0": {
        "overview": "https://arxiv.org/html/2511.19861v2/Dreamer_.png",
    },
    "giga-world-policy-0.5": {
        "overview": "https://arxiv.org/html/2607.13960v3/framework.png",
    },
    "go1": {
        "teaser": "https://arxiv.org/html/2503.06669v4/data_collection_pipeline_v2.png",
    },
    "gr00t": {
        "teaser": "https://raw.githubusercontent.com/NVIDIA/Isaac-GR00T/main/media/header_compress.png",
        "overview": "https://raw.githubusercontent.com/NVIDIA/Isaac-GR00T/main/media/model-architecture.png",
    },
    "helios": {
        "teaser": "https://arxiv.org/html/2603.04379/x1.png",
        "overview": "https://arxiv.org/html/2603.04379/x2.png",
    },
    "hma": {
        "overview": "https://arxiv.org/html/2502.04296v1/framework_figure_xinlei2.png",
    },
    "hunyuan-game-craft": {
        "overview": "https://arxiv.org/html/2506.17201v1/gamecraft-new.png",
    },
    "hunyuanvideo-1.5": {
        "overview": "https://raw.githubusercontent.com/Tencent-Hunyuan/HunyuanVideo-1.5/refs/heads/main/assets/hy_video_1_5_dit.png",
    },
    "hunyuanworld-1": {
        "teaser": "https://arxiv.org/html/2507.21809v2/teaser.png",
        "overview": "https://arxiv.org/html/2507.21809v2/method.png",
    },
    "hunyuanworld-mirror": {
        "overview": "https://arxiv.org/html/2510.10726v2/Figs/pipeline.png",
    },
    "hunyuanworld-voyager": {
        "teaser": "https://arxiv.org/html/2405.07719v5/lb.png",
    },
    "hy-embodied-vla": {
        "teaser": "https://arxiv.org/html/2606.14409v2/teaser_1.png",
        "overview": "https://arxiv.org/html/2606.14409v2/pipeline.png",
    },
    "hy-world-2.0": {
        "teaser": "https://arxiv.org/html/2604.14268v1/pics/teaser0.jpeg",
    },
    "hy-worldplay": {
        "teaser": "https://arxiv.org/html/2602.09022v1/model1.png",
    },
    "hydra": {
        "overview": "https://arxiv.org/html/2603.25716v2/pipeline.png",
    },
    "i2vgen-xl": {
        "overview": "https://arxiv.org/html/2311.04145v1/Fig2_framework.png",
    },
    "infinite-vggt": {
        "overview": "https://arxiv.org/html/2507.11539v2/framework.png",
    },
    "infinite-world": {
        "overview": "https://arxiv.org/html/2602.02393v2/main_figure_cropped.png",
    },
    "inspatio-world": {
        "overview": "https://arxiv.org/html/2604.07209v2/teaser.png",
    },
    "internvla-a1": {
        "teaser": "https://arxiv.org/html/2607.04988v1/teaser.png",
        "overview": "https://arxiv.org/html/2607.04988v1/Model-A1.5.png",
    },
    "irasim": {
        "overview": "https://arxiv.org/html/2406.14540/x2.png",
    },
    "kairos-sensenova": {
        "teaser": "https://arxiv.org/html/2606.16533v3/figures/motivation_v2.png",
        "overview": "https://arxiv.org/html/2606.16533v3/figures/framework_new_v1.png",
    },
    "lagernvs": {
        "overview": "https://arxiv.org/html/2603.20176v3/method_v5.png",
    },
    "lapa": {
        "teaser": "https://arxiv.org/html/2410.11758v2/Figure1.png",
        "overview": "https://arxiv.org/html/2410.11758v2/latent_action_model.png",
    },
    "last-r1": {
        "overview": "https://arxiv.org/html/2604.28192v3/teaser.png",
    },
    "lda-1b": {
        "teaser": "https://arxiv.org/html/2505.03233v3/datagen-v2.png",
        "overview": "https://arxiv.org/html/2505.03233v3/pipeline.png",
    },
    "leworldmodel": {
        "teaser": "https://arxiv.org/html/2603.19312v3/lewm.png",
    },
    "lingbot-map": {
        "overview": "https://arxiv.org/html/2604.14141v2/Network.png",
    },
    "lingbot-va": {
        "teaser": "https://arxiv.org/html/2601.21998v2/teaser_v3.png",
        "overview": "https://arxiv.org/html/2601.21998v2/framework2.png",
    },
    "lingbot-video": {
        "teaser": "https://arxiv.org/html/2607.07675v1/teaser_final.png",
        "overview": "https://arxiv.org/html/2607.07675v1/architecture.png",
    },
    "lingbot-vla": {
        "overview": "https://arxiv.org/html/2508.02317/x2.png",
    },
    "lingbot-vla-v2": {
        "overview": "https://arxiv.org/html/2607.06403v1/framework.png",
    },
    "lingbot-world": {
        "teaser": "https://arxiv.org/html/2601.20540v1/figures/teaser.png",
        "overview": "https://arxiv.org/html/2601.20540v1/overview.png",
    },
    "lingbot-world-v2": {
        "teaser": "https://arxiv.org/html/2607.07534v1/teaser.png",
        "overview": "https://arxiv.org/html/2607.07534v1/data_engine.png",
    },
    "loger": {
        "teaser": "https://arxiv.org/html/2603.03269v2/figure2v2.png",
        "overview": "https://arxiv.org/html/2603.03269v2/data_ablation.png",
    },
    "longcat-video": {
        "teaser": "https://arxiv.org/html/2510.22200v2/teaser-4-v2.png",
        "overview": "https://arxiv.org/html/2510.22200v2/overview_training.png",
    },
    "longvie-2": {
        "overview": "https://arxiv.org/html/2512.13604v1/framework.png",
    },
    "ltx-2.x": {
        "overview": "https://arxiv.org/html/2601.03233/2601.03233v1/assets/figures/fig-1-overview-v2.png",
    },
    "ltx-video": {
        "teaser": "https://arxiv.org/html/2501.00103/x1.png",
        "overview": "https://arxiv.org/html/2501.00103/x4.png",
    },
    "magi-1": {
        "teaser": "https://arxiv.org/html/2505.13211v1/algorithm_v4.png",
        "overview": "https://arxiv.org/html/2505.13211v1/vae_new.png",
    },
    "matrix-game-2": {
        "overview": "https://arxiv.org/html/2508.13009v4/asset/overall_architecture.png",
    },
    "matrix-game-3": {
        "teaser": "https://arxiv.org/html/2604.08995v2/teaser.png",
        "overview": "https://arxiv.org/html/2604.08995v2/mg3_overview.png",
    },
    "mem-0": {
        "teaser": "https://arxiv.org/html/2603.01229v3/benchmark.png",
    },
    "mineworld": {
        "overview": "https://arxiv.org/html/2504.08388v1/archv2-crop.png",
    },
    "minwm-hy-action2v": {
        "teaser": "https://arxiv.org/html/2605.30263v1/paper_pipeline.png",
    },
    "mmaudio": {
        "teaser": "https://arxiv.org/html/2412.15322v2/teaser-print-crop.png",
        "overview": "https://arxiv.org/html/2412.15322v2/vis-crop-compressed.png",
    },
    "mme-vla": {
        "overview": "https://arxiv.org/html/2603.04639v3/model_design.png",
    },
    "modelscope-t2v": {
        "teaser": "https://arxiv.org/html/2308.06571/x1.png",
        "overview": "https://arxiv.org/html/2308.06571/x2.png",
    },
    "molmoact2": {
        "teaser": "https://arxiv.org/html/2605.02881v2/MAF11.png",
    },
    "molmobot": {
        "teaser": "https://arxiv.org/html/2603.16861v2/MolmoBotTeaser_11.png",
    },
    "monst3r": {
        "teaser": "https://raw.githubusercontent.com/Junyi42/monst3r/main/assets/fig1_teaser.png",
    },
    "mosaicmem": {
        "teaser": "https://arxiv.org/html/2603.17117v1/figs/mem_comparison_V7.jpg",
        "overview": "https://arxiv.org/html/2603.17117v1/figs/method_v4.jpg",
    },
    "motionbricks": {
        "teaser": "https://arxiv.org/html/2604.24833v1/teaser_motion_bricks.png",
    },
    "motionctrl": {
        "teaser": "https://arxiv.org/html/2312.03641v2/teaser.png",
        "overview": "https://arxiv.org/html/2312.03641v2/framework_v2.png",
    },
    "moverse": {
        "overview": "https://arxiv.org/html/2606.13376v2/pipeline_overview.png",
    },
    "mvdiffusion": {
        "teaser": "https://arxiv.org/html/2307.01097v7/teaser_with_mesh_short_compress.png",
    },
    "neoverse": {
        "overview": "https://arxiv.org/html/2601.00393v2/framework.png",
    },
    "oasis-500m": {
        "overview": "https://raw.githubusercontent.com/etched-ai/open-oasis/master/media/arch.png",
    },
    "octo": {
        "teaser": "https://raw.githubusercontent.com/octo-models/octo/main/docs/assets/teaser.jpg",
    },
    "omniforcing": {
        "overview": "https://arxiv.org/html/2603.11647v2/teaser_2.png",
    },
    "open-dreamer": {
        "teaser": "https://arxiv.org/html/2509.24527v1/imag.png",
    },
    "open-magvit2": {
        "overview": "https://arxiv.org/html/2409.04410v3/framework.png",
    },
    "open-sora": {
        "teaser": "https://arxiv.org/html/2503.09642/x1.png",
        "overview": "https://arxiv.org/html/2503.09642/x2.png",
    },
    "open-sora-plan": {
        "overview": "https://arxiv.org/html/2412.00131v1/overview.png",
    },
    "openvla": {
        "teaser": "https://arxiv.org/html/2406.09246/x1.png",
        "overview": "https://arxiv.org/html/2406.09246/x2.png",
    },
    "openvla-oft": {
        "teaser": "https://arxiv.org/html/2502.19645v2/fig/figure_1_openvla_aloha.001.jpeg",
    },
    "pi0": {
        "teaser": "https://arxiv.org/html/2410.24164/x1.png",
        "overview": "https://arxiv.org/html/2410.24164/x2.png",
    },
    "pi05": {
        "teaser": "https://arxiv.org/html/2504.16054/x1.png",
        "overview": "https://arxiv.org/html/2504.16054/x3.png",
    },
    "pi3": {
        "teaser": "https://arxiv.org/html/2507.13347v3/teaser.png",
        "overview": "https://arxiv.org/html/2507.13347v3/pipeline.png",
    },
    "pixelsplat": {
        "teaser": "https://arxiv.org/html/2312.12337v4/teaser_bigger_text.png",
        "overview": "https://arxiv.org/html/2312.12337v4/point_clouds_fig.png",
    },
    "pointworld": {
        "overview": "https://arxiv.org/html/2601.03782v1/method.png",
    },
    "pusa-vidgen": {
        "teaser": "https://arxiv.org/html/2410.03160v1/figures/Teaser.png",
        "overview": "https://arxiv.org/html/2410.03160v1/figures/Pipeline.png",
    },
    "real-time-chunking": {
        "teaser": "https://arxiv.org/html/2506.07339v2/candle_frame1.jpg",
    },
    "recammaster": {
        "overview": "https://arxiv.org/html/2503.11647v2/fig_pipe.png",
    },
    "rolling-forcing": {
        "teaser": "https://arxiv.org/html/2509.25161v1/teaser.png",
        "overview": "https://arxiv.org/html/2509.25161v1/method.png",
    },
    "rt-1": {
        "teaser": "https://arxiv.org/html/2212.06817v2/figures/rt1_teaser_tasks.png",
    },
    "sama-14b": {
        "overview": "https://arxiv.org/html/2603.19228v1/pipeline_newest.png",
    },
    "sana": {
        "teaser": "https://arxiv.org/html/2410.10629v3/teaser.png",
        "overview": "https://arxiv.org/html/2410.10629v3/model.png",
    },
    "sana-wm": {
        "teaser": "https://arxiv.org/html/2605.15178v1/teaser.png",
        "overview": "https://arxiv.org/html/2605.15178v1/pipeline_overview.png",
    },
    "scope": {
        "teaser": "https://arxiv.org/html/2605.23345v2/teaser.png",
        "overview": "https://arxiv.org/html/2605.23345v2/method.png",
    },
    "self-forcing": {
        "overview": "https://arxiv.org/html/2506.08009v2/overview.png",
    },
    "shotstream": {
        "teaser": "https://arxiv.org/html/2603.25746v1/overflow.png",
        "overview": "https://arxiv.org/html/2603.25746v1/method_teacher.png",
    },
    "show-o": {
        "overview": "https://arxiv.org/html/2408.12528v7/method_comparisons.png",
    },
    "simworld": {
        "teaser": "https://arxiv.org/html/2512.01078v2/figure1.png",
        "overview": "https://arxiv.org/html/2512.01078v2/whitepaper-figure2.png",
    },
    "skyreels-v2": {
        "teaser": "https://arxiv.org/html/2504.13074v3/demo_fig1_v2.png",
        "overview": "https://arxiv.org/html/2504.13074v3/mainpipeline_v2.png",
    },
    "skyreels-v3": {
        "teaser": "https://arxiv.org/html/2601.17323v2/mo2v1.png",
    },
    "solaris": {
        "teaser": "https://arxiv.org/html/2602.22208v2/teaser.png",
        "overview": "https://arxiv.org/html/2602.22208v2/arch.png",
    },
    "solarwm": {
        "overview": "https://arxiv.org/html/2609.02886v1/pipeline_2.png",
    },
    "spatia": {
        "overview": "https://arxiv.org/html/2512.15716v1/overview.png",
    },
    "splatt3r": {
        "overview": "https://arxiv.org/html/2408.13912v2/methodology.png",
    },
    "stable-video-infinity": {
        "teaser": "https://arxiv.org/html/2510.09212v1/Figure/intro.png",
        "overview": "https://arxiv.org/html/2510.09212v1/Figure/overall.png",
    },
    "stable-virtual-camera": {
        "overview": "https://arxiv.org/html/2503.14489v2/overview.png",
    },
    "starvla": {
        "teaser": "https://arxiv.org/html/2604.05014v1/vla-form.png",
    },
    "starwm": {
        "teaser": "https://arxiv.org/html/2602.14857v2/online_casestudy.png",
        "overview": "https://arxiv.org/html/2602.14857v2/starwm-agent.png",
    },
    "step-video-t2v": {
        "overview": "https://arxiv.org/html/2502.10248v3/figure/model_architecture.png",
    },
    "t2v_turbo_t2v": {
        "teaser": "https://arxiv.org/html/2405.18750/x1.png",
        "overview": "https://arxiv.org/html/2405.18750/x2.png",
    },
    "tesseract": {
        "overview": "https://arxiv.org/html/2504.20995v1/arch-new.png",
    },
    "tinyvla": {
        "overview": "https://arxiv.org/html/2402.03766v1/MobileVLMv2arch.png",
    },
    "track-anything-prior": {
        "overview": "https://arxiv.org/html/2305.06558v1/overview.png",
    },
    "uni3c": {
        "overview": "https://arxiv.org/html/2504.14899v2/pipeline.png",
    },
    "unidepth-v2-prior": {
        "teaser": "https://arxiv.org/html/2502.20110v2/teaser1.png",
        "overview": "https://arxiv.org/html/2502.20110v2/overview3.png",
    },
    "unik3d-prior": {
        "teaser": "https://arxiv.org/html/2503.16591v1/teaser_cr.png",
        "overview": "https://arxiv.org/html/2503.16591v1/overview_cr.png",
    },
    "uwm": {
        "overview": "https://arxiv.org/html/2504.02792v3/teaser.png",
    },
    "vchitect-2-t2v": {
        "teaser": "https://arxiv.org/html/2501.08453v1/teaser.png",
        "overview": "https://arxiv.org/html/2501.08453v1/model_overview.png",
    },
    "versecrafter": {
        "teaser": "https://arxiv.org/html/2601.05138v2/teaser2.png",
        "overview": "https://arxiv.org/html/2601.05138v2/framework4.png",
    },
    "vggt-omega": {
        "overview": "https://arxiv.org/html/2605.15195v1/architecture_v8.png",
    },
    "vggt-world": {
        "teaser": "https://arxiv.org/html/2603.12655v1/Figure/ECCV_26_VGGTWorld.png",
        "overview": "https://arxiv.org/html/2603.12655v1/fig_tartanair_3.png",
    },
    "vid2world": {
        "teaser": "https://arxiv.org/html/2505.14357v3/v2w_overview.png",
        "overview": "https://arxiv.org/html/2505.14357v3/pipeline.png",
    },
    "video-depth-anything-prior": {
        "overview": "https://arxiv.org/html/2501.12375v3/overview_head_fix.png",
    },
    "vlanext": {
        "teaser": "https://arxiv.org/html/2602.18532v3/performance_first_glance.png",
        "overview": "https://arxiv.org/html/2602.18532v3/framework.png",
    },
    "vmem": {
        "teaser": "https://arxiv.org/html/2506.18903v3/vmem_method.png",
        "overview": "https://arxiv.org/html/2506.18903v3/ood_demo.png",
    },
    "wan2.1": {
        "teaser": "https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/assets/t2v_res.jpg",
        "overview": "https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/assets/video_dit_arch.jpg",
    },
    "wan2.1-vace": {
        "teaser": "https://raw.githubusercontent.com/ali-vilab/VACE/main/assets/materials/teaser.jpg",
    },
    "wan2.2": {
        "overview": "https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/assets/moe_arch.png",
    },
    "wildworld": {
        "teaser": "https://arxiv.org/html/2603.23497v1/teaser.png",
        "overview": "https://arxiv.org/html/2603.23497v1/framework-arxiv.png",
    },
    "wonderjourney": {
        "overview": "https://arxiv.org/html/2312.03884v2/overview.png",
    },
    "wonderworld": {
        "overview": "https://arxiv.org/html/2406.09394v4/overview.png",
    },
    "worldcam": {
        "teaser": "https://arxiv.org/html/2603.16871v1/teaser.png",
    },
    "worldfm": {
        "teaser": "https://arxiv.org/html/2603.11911v3/files/20260309-153804_final_0309.png",
        "overview": "https://arxiv.org/html/2603.11911v3/arch.png",
    },
    "worldgen": {
        "overview": "https://arxiv.org/html/2511.16825v1/pipeline_w_prompt_anno_group_edit_white.png",
    },
    "worldgrow": {
        "overview": "https://arxiv.org/html/2510.21682v1/pipeline_v2-resized.png",
    },
    "worldmem": {
        "teaser": "https://arxiv.org/html/2504.12369v3/teaser.png",
    },
    "wow": {
        "teaser": "https://arxiv.org/html/2509.22642v2/teaser.png",
        "overview": "https://arxiv.org/html/2509.22642v2/figs/Brain_and_Mind_Model.png",
    },
    "x-wam": {
        "teaser": "https://arxiv.org/html/2604.26694v2/teaser_new.png",
        "overview": "https://arxiv.org/html/2604.26694v2/framework.png",
    },
    "xiaomi-robotics-0": {
        "teaser": "https://arxiv.org/html/2602.12684v2/fig1.png",
    },
    "xiaomi-robotics-1": {
        "teaser": "https://arxiv.org/html/2607.15330v2/teaser.png",
        "overview": "https://arxiv.org/html/2607.15330v2/model.png",
    },
    "xvla": {
        "teaser": "https://arxiv.org/html/2510.10274v1/intro_small.png",
        "overview": "https://arxiv.org/html/2510.10274v1/archi.png",
    },
    "yume": {
        "teaser": "https://arxiv.org/html/2512.22096v1/dataset.png",
    },
    "ati-wan21-14b": {
        "teaser": "https://arxiv.org/html/2505.22944v3/figures/example1.jpg",
        "overview": "https://arxiv.org/html/2505.22944v3/figures/Pipeline.jpg",
    },
    "egowm": {
        "teaser": "https://arxiv.org/html/2601.15284v2/teaser.png",
        "overview": "https://arxiv.org/html/2601.15284v2/method.png",
    },
    "emu3.5": {
        "overview": "https://raw.githubusercontent.com/baaivision/Emu3.5/main/assets/arch.png",
    },
    "h-rdt": {
        "overview": "https://arxiv.org/html/2507.23523v2/figure2.png",
    },
    "joyai-echo-wm": {
        "teaser": "https://raw.githubusercontent.com/jd-opensource/JoyAI-Echo/main/assets/teaser.png",
    },
    "liveworld": {
        "teaser": "https://arxiv.org/html/2603.07145v2/imgs/teaser.jpg",
        "overview": "https://arxiv.org/html/2603.07145v2/main.png",
    },
    "lyra": {
        "teaser": "https://arxiv.org/html/2604.13036v1/teaser_v3.png",
        "overview": "https://arxiv.org/html/2604.13036v1/method_v3.png",
    },
    "magicworld": {
        "overview": "https://arxiv.org/html/2511.18886v2/Fig_Model.png",
    },
    "matrix-game-3.5-first-person": {
        "teaser": "https://matrix-game-v3-5.github.io/static/imgs/mg35/fig4_memory.jpg",
        "overview": "https://matrix-game-v3-5.github.io/static/imgs/mg35/fig3_sequence.jpg",
    },
    "metric3d-prior": {
        "teaser": "https://arxiv.org/html/2404.15506v4/page2.png",
        "overview": "https://arxiv.org/html/2404.15506v4/ours.png",
    },
    "nwm": {
        "teaser": "https://arxiv.org/html/2412.03572v2/figure_ood_v2.png",
        "overview": "https://arxiv.org/html/2412.03572v2/CDiTv6.png",
    },
    "pandora": {
        "overview": "https://arxiv.org/html/2406.09455v1/architecture_v2.png",
    },
    "prior-depth-anything": {
        "teaser": "https://arxiv.org/html/2505.10565v1/motivation.png",
        "overview": "https://arxiv.org/html/2505.10565v1/model.png",
    },
    "spatial-forcing": {
        "teaser": "https://arxiv.org/html/2510.12276v2/fig_teaser.png",
        "overview": "https://arxiv.org/html/2510.12276v2/fig_compare.png",
    },
    "unianimate-dit": {
        "teaser": "https://arxiv.org/html/2504.11289v1/figures.png",
        "overview": "https://arxiv.org/html/2504.11289v1/Network.png",
    },
    "vggt": {
        "teaser": "https://arxiv.org/html/2503.11651v1/comparison_vggt_dust3r.png",
        "overview": "https://arxiv.org/html/2503.11651v1/architecture_v4.png",
    },
    "wilddet3d": {
        "teaser": "https://arxiv.org/html/2604.08626v2/teaser_5_flat.png",
        "overview": "https://arxiv.org/html/2604.08626v2/model_arch_new.png",
    },
}

AR5IV_FALLBACK = {
    "being-h05": {
        "teaser": "https://ar5iv.labs.arxiv.org/html/2601.12993/x1.png",
        "overview": "https://ar5iv.labs.arxiv.org/html/2601.12993/x5.png",
    },
    "cosmos-predict-2.5": {
        "teaser": "https://ar5iv.labs.arxiv.org/html/2511.00062/x1.png",
        "overview": "https://ar5iv.labs.arxiv.org/html/2511.00062/x2.png",
    },
    "cosmos3": {
        "teaser": "https://ar5iv.labs.arxiv.org/html/2606.02800/x1.png",
        "overview": "https://ar5iv.labs.arxiv.org/html/2606.02800/x2.png",
    },
    "dap": {
        "overview": "https://ar5iv.labs.arxiv.org/html/2512.16913/x3.png",
    },
    "easyanimate": {
        "teaser": "https://ar5iv.labs.arxiv.org/html/2405.18991/x1.png",
        "overview": "https://ar5iv.labs.arxiv.org/html/2405.18991/x2.png",
    },
    "helios": {
        "teaser": "https://ar5iv.labs.arxiv.org/html/2603.04379/x1.png",
        "overview": "https://ar5iv.labs.arxiv.org/html/2603.04379/x2.png",
    },
    "ltx-video": {
        "teaser": "https://ar5iv.labs.arxiv.org/html/2501.00103/x1.png",
        "overview": "https://ar5iv.labs.arxiv.org/html/2501.00103/x4.png",
    },
    "modelscope-t2v": {
        "teaser": "https://ar5iv.labs.arxiv.org/html/2308.06571/x1.png",
        "overview": "https://ar5iv.labs.arxiv.org/html/2308.06571/x2.png",
    },
    "open-sora": {
        "teaser": "https://ar5iv.labs.arxiv.org/html/2503.09642/x1.png",
        "overview": "https://ar5iv.labs.arxiv.org/html/2503.09642/x2.png",
    },
    "openvla": {
        "teaser": "https://ar5iv.labs.arxiv.org/html/2406.09246/x1.png",
        "overview": "https://ar5iv.labs.arxiv.org/html/2406.09246/x2.png",
    },
    "pi0": {
        "teaser": "https://ar5iv.labs.arxiv.org/html/2410.24164/x1.png",
        "overview": "https://ar5iv.labs.arxiv.org/html/2410.24164/x2.png",
    },
    "pi05": {
        "teaser": "https://ar5iv.labs.arxiv.org/html/2504.16054/x1.png",
        "overview": "https://ar5iv.labs.arxiv.org/html/2504.16054/x3.png",
    },
}

# In-tree paper / architecture stills only. Demo mp4s stay on disk for home /
# Studio / CDN — they must not be harvested onto model homepages.
LOCAL_MEDIA: dict[str, list[dict[str, str]]] = {
    "hunyuanvideo": [
        {
            "kind": "image",
            "src": "/models/hunyuanvideo/overall.png",
            "caption": "HunyuanVideo overall architecture — Causal 3D VAE, LLM text encoder, and latent diffusion",
            "captionZh": "HunyuanVideo 整体架构：因果 3D VAE、大语言模型文本编码与潜空间扩散",
        },
        {
            "kind": "image",
            "src": "/models/hunyuanvideo/backbone.png",
            "caption": "Dual-stream to single-stream Transformer backbone for unified image and video generation",
            "captionZh": "双流到单流 Transformer 骨干，用于统一的图视频生成",
        },
        {
            "kind": "image",
            "src": "/models/hunyuanvideo/text_encoder.png",
            "caption": "MLLM text encoder with a bidirectional token refiner",
            "captionZh": "带双向 token 优化器的 MLLM 文本编码器",
        },
        {
            "kind": "image",
            "src": "/models/hunyuanvideo/3dvae.png",
            "caption": "CausalConv3D VAE — temporal 4×, spatial 8×, 16-channel latent",
            "captionZh": "CausalConv3D VAE：时间 4 倍、空间 8 倍压缩，16 通道潜空间",
        },
        {
            "kind": "image",
            "src": "/models/hunyuanvideo/i2v_backbone.png",
            "caption": "HunyuanVideo-I2V architecture — token replace of the reference image into the video generator",
            "captionZh": "HunyuanVideo-I2V 架构：用 token replace 将参考图写入视频生成过程",
        },
    ],
}


MAX_FIGURE_BYTES = 750_000
MAX_FIGURE_WIDTH = 1600


def looks_like_pdf_page(path: Path) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False
    with Image.open(path) as image:
        width, height = image.size
    if width < 200 or height < 200:
        return True
    if height > width * 1.2 and height >= 1200:
        return True
    return False


def compress_figure(src: Path, dest: Path) -> Path | None:
    if not src.is_file() or src.stat().st_size < 800:
        return None
    if looks_like_pdf_page(src):
        print(f"  skip pdf-page-like {src.name}", flush=True)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.stat().st_size <= MAX_FIGURE_BYTES and src.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        return dest
    cwebp = shutil.which("cwebp")
    webp_dest = dest.with_suffix(".webp")
    if cwebp:
        cmd = [cwebp, "-quiet", "-q", "78", "-m", "6", "-resize", str(MAX_FIGURE_WIDTH), "0", str(src), "-o", str(webp_dest)]
        if subprocess.run(cmd, check=False, capture_output=True).returncode == 0 and webp_dest.is_file():
            if dest.exists() and dest != webp_dest:
                dest.unlink()
            return webp_dest
    sips = shutil.which("sips")
    jpeg_dest = dest.with_suffix(".jpg")
    if sips:
        cmd = [
            sips,
            "-s",
            "format",
            "jpeg",
            "-s",
            "formatOptions",
            "82",
            "--resampleWidth",
            str(MAX_FIGURE_WIDTH),
            str(src),
            "--out",
            str(jpeg_dest),
        ]
        if subprocess.run(cmd, check=False, capture_output=True).returncode == 0 and jpeg_dest.is_file():
            if dest.exists() and dest != jpeg_dest:
                dest.unlink()
            return jpeg_dest
    shutil.copy2(src, dest)
    return dest


def download_curated_figures() -> dict[str, list[str]]:
    saved: dict[str, list[str]] = {}
    for model_id, roles in CURATED_FIGURES.items():
        dest_dir = THUMB_DIR / model_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        for role, url in roles.items():
            existing = list(dest_dir.glob(f"{role}.*"))
            if any(path.is_file() and path.stat().st_size > 800 for path in existing):
                saved.setdefault(model_id, []).append(f"{role}:exists")
                continue
            raw = request(url, timeout=40)
            if (not raw or len(raw) < 4000) and model_id in AR5IV_FALLBACK and role in AR5IV_FALLBACK[model_id]:
                time.sleep(0.3)
                raw = request(AR5IV_FALLBACK[model_id][role], timeout=40)
            if not raw or len(raw) < 4000:
                print(f"  miss {model_id}/{role}", flush=True)
                continue
            suffix = ".png"
            if raw[:3] == b"\xff\xd8\xff":
                suffix = ".jpg"
            elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
                suffix = ".webp"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(raw)
                tmp = Path(handle.name)
            try:
                if looks_like_pdf_page(tmp):
                    print(f"  skip pdf-page-like {model_id}/{role}", flush=True)
                    continue
                dest = dest_dir / f"{role}{suffix}"
                written = compress_figure(tmp, dest)
                if written:
                    saved.setdefault(model_id, []).append(f"{role}:{written.name}:{written.stat().st_size}")
                    print(f"  wrote {model_id}/{written.name} ({written.stat().st_size} bytes)", flush=True)
            finally:
                tmp.unlink(missing_ok=True)
            time.sleep(0.2)
    return saved


def is_demo_video_figure(item: dict[str, str]) -> bool:
    kind = item.get("kind") or "image"
    src = str(item.get("src") or "").strip().lower()
    if kind == "video":
        return True
    return src.endswith((".mp4", ".webm", ".mov", ".m4v")) or src.startswith("/demos/")


def without_demo_videos(figures: list[dict[str, str]]) -> list[dict[str, str]]:
    return [item for item in figures if not is_demo_video_figure(item)]


def local_figure_exists(src: str) -> bool:
    if not src.startswith("/"):
        return False
    path = PUBLIC / src.lstrip("/")
    if not path.is_file():
        return False
    # Demo clips stay on disk for home / Studio / CDN; model homes must not list them.
    if src.startswith("/demos/"):
        return False
    return path.stat().st_size > 800


def merge_official_figures(figures: list[dict[str, str]], model_id: str) -> list[dict[str, str]]:
    official = figure_records(model_id)
    if not official:
        return without_demo_videos(
            [
                item
                for item in figures
                if local_figure_exists(str(item.get("src") or ""))
                and not str(item.get("src") or "").lower().endswith("/paper.png")
            ]
        )
    kept = [
        item
        for item in figures
        if str(item.get("src") or "") not in {row["src"] for row in official}
        and not str(item.get("src") or "").lower().endswith("/paper.png")
        and local_figure_exists(str(item.get("src") or ""))
    ]
    return without_demo_videos(official + kept)


def request(url: str, timeout: int = 20) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status >= 400:
                return None
            return response.read()
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def arxiv_id_from(url: str) -> str | None:
    match = ARXIV_ID_RE.search(url or "")
    return match.group(1) if match else None


def fetch_arxiv(ids: list[str]) -> dict[str, dict[str, object]]:
    papers: dict[str, dict[str, object]] = {}
    for start in range(0, len(ids), 16):
        batch = ids[start : start + 16]
        url = "http://export.arxiv.org/api/query?id_list=" + ",".join(batch) + "&max_results=16"
        raw = request(url, timeout=30)
        if not raw:
            time.sleep(1.2)
            continue
        root = ET.fromstring(raw)
        for entry in root.findall("a:entry", ARXIV_NS):
            entry_id = (entry.findtext("a:id", default="", namespaces=ARXIV_NS) or "")
            match = re.search(r"(\d{4}\.\d{4,5})", entry_id)
            if not match:
                continue
            title = re.sub(r"\s+", " ", (entry.findtext("a:title", default="", namespaces=ARXIV_NS) or "")).strip()
            abstract = re.sub(r"\s+", " ", (entry.findtext("a:summary", default="", namespaces=ARXIV_NS) or "")).strip()
            published = entry.findtext("a:published", default="", namespaces=ARXIV_NS) or ""
            comment = entry.findtext("arxiv:comment", default="", namespaces=ARXIV_NS) or ""
            venue = None
            venue_match = re.search(
                r"\b(CVPR|ICCV|ECCV|NeurIPS|ICML|ICLR|AAAI|IJCAI|SIGGRAPH|ACL|EMNLP|IROS|ICRA|CoRL|RSS|3DV|WACV)[^.;,]{0,24}",
                comment,
                re.I,
            )
            if venue_match:
                venue = venue_match.group(0).strip(" ;,")
            papers[match.group(1)] = {
                "title": title,
                "year": int(published[:4]) if published[:4].isdigit() else None,
                "venue": venue,
                "arxivId": match.group(1),
                "abstract": abstract[:900] if abstract else None,
            }
        time.sleep(0.8)
    return papers


def download_thumb(arxiv_id: str, model_id: str) -> str | None:
    url = f"https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/{arxiv_id}.png"
    raw = request(url, timeout=15)
    if not raw or len(raw) < 4000:
        return None
    dest_dir = THUMB_DIR / model_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "paper.png"
    dest.write_bytes(raw)
    return f"/models/{model_id}/paper.png"


def teaser_src(figures: list[dict[str, str]]) -> str | None:
    if not figures:
        return None
    first = figures[0]
    return first.get("poster") or (first.get("src") if first.get("kind") != "video" else None)


def apply_payload(media: dict[str, dict[str, object]]) -> None:
    recipes = json.loads(RECIPES.read_text(encoding="utf-8"))
    for recipe in recipes["recipes"]:
        extra = media.get(recipe["id"]) or {}
        docs = recipe.setdefault("docs", {})
        if extra.get("paper"):
            docs["paper"] = extra["paper"]
        else:
            docs.setdefault("paper", None)
        if extra.get("figures"):
            docs["figures"] = without_demo_videos(list(extra["figures"]))
        else:
            docs.setdefault("figures", [])
    RECIPES.write_text(json.dumps(recipes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    index = json.loads(INDEX.read_text(encoding="utf-8"))
    for recipe in index["recipes"]:
        extra = media.get(recipe["id"]) or {}
        recipe["teaser"] = teaser_src(extra.get("figures") or [])
    INDEX.write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-figures",
        action="store_true",
        help="Download curated official teaser/overview figures only; do not refetch arXiv metadata",
    )
    args = parser.parse_args()
    if args.paper_figures:
        print("downloading curated official paper figures", flush=True)
        saved = download_curated_figures()
        if OUT.exists():
            media = json.loads(OUT.read_text(encoding="utf-8"))
        else:
            media = {}
        for model_id in sorted({*CURATED_FIGURES, *saved}):
            extra = media.get(model_id) if isinstance(media.get(model_id), dict) else {}
            figures = merge_official_figures(list(extra.get("figures") or []), model_id)
            extra = dict(extra)
            extra["figures"] = figures
            if figures or extra.get("paper"):
                media[model_id] = extra
        for model_id, extra in list(media.items()):
            if not isinstance(extra, dict):
                continue
            extra["figures"] = without_demo_videos(list(extra.get("figures") or []))
            media[model_id] = extra
        OUT.write_text(json.dumps(media, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"curated figures for {len(saved)} models -> {OUT}", flush=True)
        return

    recipes = json.loads(RECIPES.read_text(encoding="utf-8"))["recipes"]
    arxiv_for: dict[str, str] = {}
    for recipe in recipes:
        for source in recipe.get("sources") or []:
            found = arxiv_id_from(source.get("url") or "")
            if found:
                arxiv_for[recipe["id"]] = found
                break

    unique_ids = sorted(set(arxiv_for.values()))
    print(f"arxiv ids: {len(unique_ids)} across {len(arxiv_for)} models", flush=True)
    papers = fetch_arxiv(unique_ids)
    print(f"arxiv metadata: {len(papers)}", flush=True)

    media: dict[str, dict[str, object]] = {}
    downloaded = 0
    for recipe in recipes:
        model_id = recipe["id"]
        entry: dict[str, object] = {}
        arxiv_id = arxiv_for.get(model_id)
        if arxiv_id and arxiv_id in papers:
            entry["paper"] = papers[arxiv_id]
        figures = without_demo_videos([dict(item) for item in LOCAL_MEDIA.get(model_id, [])])
        if arxiv_id and not (THUMB_DIR / model_id / "paper.png").is_file():
            thumb = download_thumb(arxiv_id, model_id)
            if thumb:
                downloaded += 1
                time.sleep(0.15)
        figures = merge_official_figures(figures, model_id)
        if figures:
            entry["figures"] = figures
        if entry:
            media[model_id] = entry

    OUT.write_text(json.dumps(media, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    apply_payload(media)
    print(f"wrote {len(media)} media records, {downloaded} paper thumbs -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
