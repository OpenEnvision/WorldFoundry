"""Module for base_models -> three_dimensions -> general_3d -> mvdiffusion -> mvdiffusion_runtime -> demo.py functionality."""


import argparse
import os

import cv2
import numpy as np
import torch
import yaml
from src.checkpoint_utils import resolve_mvdiffusion_checkpoint
from src.pano_generator import PanoGenerator
from src.pano_outpaint_generator import PanoOutpaintGenerator

from worldfoundry.core.io.paths import package_data_path

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from datetime import datetime

from generate_video_tool.pano_video_generation import generate_video
from PIL import Image

try:
    from exiftool import ExifToolHelper
except Exception:
    ExifToolHelper = None

torch.manual_seed(0)


def _checkpoint_path(filename):
    return str(resolve_mvdiffusion_checkpoint("MVDiffusion", "weights", filename))

def load_mvdiffusion_state_dict(model, checkpoint_path):
    """Load mvdiffusion state dict.

    Args:
        model: The model.
        checkpoint_path: The checkpoint path.
    """
    state_dict = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=True,
        mmap=True,
    )['state_dict']
    key_map = {
        '.query.': '.to_q.',
        '.key.': '.to_k.',
        '.value.': '.to_v.',
        '.proj_attn.': '.to_out.0.',
    }
    remapped = {}
    for key, value in state_dict.items():
        if key == 'text_encoder.text_model.embeddings.position_ids':
            continue
        new_key = key
        if key.startswith('vae.'):
            for old, new in key_map.items():
                new_key = new_key.replace(old, new)
        remapped[new_key] = value
    model.load_state_dict(remapped, strict=True)

def get_K_R(FOV, THETA, PHI, height, width):
    """Get k r.

    Args:
        FOV: The fov.
        THETA: The theta.
        PHI: The phi.
        height: The height.
        width: The width.
    """
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R

def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--text', type=str, default='This kitchen is a charming blend of rustic and modern, featuring a large reclaimed wood island with marble countertop, a sink surrounded by cabinets. To the left of the island, a stainless-steel refrigerator stands tall. To the right of the sink, built-in wooden cabinets painted in a muted.')
    parser.add_argument(
        '--image_path', type=str, default=None, help='image condition outpainting')
    parser.add_argument('--gen_video',
                    action='store_true', help='generate video')
    parser.add_argument('--text_path',
                    type=str, help='text path allow to specify 8 texts')
    parser.add_argument(
        '--steps', type=int, default=None,
        help='override the configured diffusion timestep count')

    return parser.parse_args()

def resize_and_center_crop(img, size):
    """Resize and center crop.

    Args:
        img: The img.
        size: The size.
    """
    H, W, _ = img.shape
    if H==W:
        img = cv2.resize(img, (size, size))
    elif H > W:
        current_size = int(size*H/W)
        img = cv2.resize(img, (size, current_size))
        # center crop to square
        margin_l=(current_size-size)//2
        margin_r=current_size-margin_l-size
        img=img[margin_l:-margin_r, :]
    else:
        current_size=int(size*W/H)
        img = cv2.resize(img, (current_size, size))
        margin_l=(current_size-size)//2
        margin_r=current_size-margin_l-size
        img=img[:, margin_l:-margin_r]
    return img

args = parse_args()
if args.image_path is None:
    config_file = package_data_path('models', 'runtime', 'configs', 'mvdiffusion', 'pano_generation.yaml')
    config = yaml.load(open(config_file, 'rb'), Loader=yaml.SafeLoader)
    if args.steps is not None:
        config['model']['diff_timestep'] = max(int(args.steps), 1)
    model = PanoGenerator(config)
    load_mvdiffusion_state_dict(model, _checkpoint_path('pano.ckpt'))
    #saved_ckpt = torch.load('weights/pano.ckpt', map_location='cpu')
    #model.load_state_dict(saved_ckpt, strict=False)
    model=model.cuda()
    img=None
else:

    config_file = package_data_path('models', 'runtime', 'configs', 'mvdiffusion', 'pano_generation_outpaint.yaml')
    config = yaml.load(open(config_file, 'rb'), Loader=yaml.SafeLoader)
    if args.steps is not None:
        config['model']['diff_timestep'] = max(int(args.steps), 1)
    model = PanoOutpaintGenerator(config)
    load_mvdiffusion_state_dict(model, _checkpoint_path('pano_outpaint.ckpt'))
    #saved_ckpt = torch.load('weights/pano_outpaint.ckpt', map_location='cpu')
    #model.load_state_dict(saved_ckpt, strict=False)
    model=model.cuda()

    img=cv2.imread(args.image_path)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=resize_and_center_crop(img, config['dataset']['resolution'])
    img=img/127.5-1       
    img=torch.tensor(img).cuda()
    
    # Read Stable Diffusion prompts from PNG metadata when exiftool is available.
    # Headless inference should still work in minimal conda environments.
    if ExifToolHelper is not None:
        try:
            with ExifToolHelper() as et:
                EXIF_dict = et.get_metadata(args.image_path)
                print(f'Metadata: {EXIF_dict}')
                if EXIF_dict and 'PNG:Parameters' in EXIF_dict[0]:
                    PNGparameters = EXIF_dict[0]['PNG:Parameters']
                    parsed_parameters = PNGparameters.split("\n")
                    positive_prompt = parsed_parameters[0]
                    negative_prompt = parsed_parameters[1] if len(parsed_parameters) > 1 else ""
                    print(f'Positive promts: ' +  positive_prompt)
                    print(f'Negative promts: ' +  negative_prompt)
                    args.text = positive_prompt
        except Exception as exc:
            print(f'Warning: could not read PNG metadata with exiftool: {exc}')
    else:
        print('Warning: exiftool is not installed; using --text prompt for MVDiffusion outpainting.')
    
resolution=config['dataset']['resolution']
Rs=[]
Ks=[]
for i in range(8):
    degree = (45*i) % 360
    K, R = get_K_R(90, degree, 0,
                    resolution, resolution)

    Rs.append(R)
    Ks.append(K)

images=torch.zeros((1,8,resolution,resolution, 3)).cuda()
if img is not None:
    images[0,0]=img


if args.text_path is not None:
    prompt=[]
    with open(args.text_path, 'r') as f:
        for i, line in enumerate(f):
            prompt.append(line.strip())
    if len(prompt)<8:
        raise ValueError('text file should contain 8 lines for each camera view')
    args.text=prompt[0]
else:
    prompt=[args.text]*8
K=torch.tensor(Ks).cuda()[None]
R=torch.tensor(Rs).cuda()[None]

batch= {
        'images': images,
        'prompt': prompt,
        'R': R,
        'K': K
    }
images_pred=model.inference(batch)
#res_dir=os.path.join('outputs/',args.text[:20])
res_dir=os.path.join('outputs/',f'results'+datetime.now().strftime('--%Y%m%d-%H%M%S'))
print('saved to the folder: {}'.format(res_dir))
os.makedirs(res_dir, exist_ok=True)
with open(os.path.join(res_dir, 'prompt.txt'), 'w') as f:
    f.write(args.text)    
image_paths=[]
for i in range(8):
    im = Image.fromarray(images_pred[0,i])
    image_path=os.path.join(res_dir, '{}.png'.format(i))
    image_paths.append(image_path)
    im.save(image_path)
generate_video(image_paths, res_dir, args.gen_video)
