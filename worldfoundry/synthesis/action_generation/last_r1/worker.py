"""Isolated LaST-R1 step using the pinned official Qwen3-VL implementation."""

import json
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


if len(sys.argv) != 2:
    raise SystemExit('usage: python -m worldfoundry.synthesis.action_generation.last_r1.worker REQUEST_JSON')
request = json.loads(Path(sys.argv[1]).read_text())
root = Path(request['run_dir']).resolve()
root.mkdir(parents=True, exist_ok=True)
source = Path(request['source_root']).resolve()
checkpoint = Path(request['checkpoint_path']).resolve()
tokenizer_spec = importlib.util.spec_from_file_location('last_r1_action_tokenizer', source / 'verl/workers/actor/action_tokenizer.py')
tokenizer_module = importlib.util.module_from_spec(tokenizer_spec)
tokenizer_spec.loader.exec_module(tokenizer_module)
ActionTokenizer = tokenizer_module.ActionTokenizer
status = {'status': 'starting', 'started_at': time.time(), 'checkpoint': str(checkpoint), 'source': str(source), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}


def write(name, payload):
    (root / name).write_text(json.dumps(payload, indent=2, default=str) + '\n')


write('status.json', status)
try:
    import transformers

    assert Path(transformers.__file__).resolve().is_relative_to(source), transformers.__file__
    assert checkpoint.joinpath('config.json').is_file()
    assert checkpoint.joinpath('model.safetensors.index.json').is_file()
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    tokenizer = processor.tokenizer
    action_zero = tokenizer.vocab['<action_0>']
    latent_start = tokenizer.vocab['<latent_start>']
    latent_end = tokenizer.vocab['<latent_end>']
    latent_pad = tokenizer.vocab['<latent_pad>']
    assert [tokenizer.vocab[f'<action_{index}>'] for index in range(256)] == list(range(action_zero, action_zero + 256))
    action_tokenizer = ActionTokenizer(tokenizer, need_to_sub=3)
    assert action_tokenizer.my_vocab_size == action_zero + 256

    picture = Image.open(request['image_path']).convert('RGB')
    if request.get('center_crop', True):
        width, height = picture.size
        crop_width, crop_height = int(width * 0.9), int(height * 0.9)
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        picture = picture.crop((left, top, left + crop_width, top + crop_height)).resize((width, height))
    instruction = str(request['instruction'])
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': picture}, {'type': 'text', 'text': instruction}]}]
    prepared = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
    ids_prefix = prepared.input_ids[0]
    prompt_len = max(len(ids_prefix), 100)
    G, max_groups, action_chunks, action_width = 2, 4, 8, 7
    L, action_count = G * max_groups, action_chunks * action_width
    body_len = max_groups * (G + 1)
    tail = [latent_start] + ([latent_pad] * G + [latent_end]) * max_groups + [action_zero] * action_count
    ids = torch.full((1, prompt_len + len(tail)), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    ids[0, prompt_len - len(ids_prefix):prompt_len] = ids_prefix
    attention[0, prompt_len - len(ids_prefix):prompt_len] = 1
    ids[0, prompt_len:] = torch.tensor(tail)
    attention[0, prompt_len:] = 1
    pixels = prepared.pixel_values.to(dtype=torch.bfloat16).unsqueeze(0)
    grid = prepared.image_grid_thw[0].unsqueeze(0)
    write('input.json', {'image_path': str(request['image_path']), 'instruction': instruction, 'prompt_tokens': len(ids_prefix), 'padded_prompt_tokens': prompt_len, 'pixel_shape': list(pixels.shape), 'image_grid_thw': grid.tolist(), 'action_zero_id': action_zero, 'latent_start_id': latent_start, 'latent_end_id': latent_end, 'latent_pad_id': latent_pad, 'checkpoint_tensor_count': len(json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map'])})

    status['status'] = 'loading_model'
    write('status.json', status)
    model, loading = Qwen3VLForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.bfloat16, local_files_only=True, device_map='cuda:0', output_loading_info=True)
    model.eval()
    write('loading.json', {key: value for key, value in loading.items() if key in {'missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'}})
    assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading.get('mismatched_keys') and not loading['error_msgs'], loading
    status['status'] = 'predicting'
    write('status.json', status)

    seed = int(request.get('seed', 42))
    temperature = float(request.get('temperature', 1.6))
    if not 0 < temperature <= 10:
        raise ValueError('temperature must be in (0, 10]')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda:0')
    ids, attention, pixels, grid = ids.to(device), attention.to(device), pixels.to(device), grid.to(device)
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        prefix_end = ids.shape[1] - body_len - action_count
        prefix_attention = attention[:, :prefix_end]
        output = model.model(input_ids=ids[:, :prefix_end], pixel_values=pixels, image_grid_thw=grid, attention_mask=prefix_attention, return_dict=True, action_length=0, use_cache=True, attn_mode='causal')
        cache = output.past_key_values
        cache_length = cache.get_seq_length()
        hidden = output.last_hidden_state[:, -1, :]
        D = hidden.shape[-1]
        end_embed = model.language_model.get_input_embeddings()(torch.tensor([[latent_end]], device=device)).to(hidden.dtype)
        current_attention = prefix_attention.clone()
        end_logits = []
        end_hiddens = []
        for _group in range(max_groups):
            for _latent in range(G):
                current_attention = torch.cat([current_attention, torch.ones((1, 1), device=device, dtype=current_attention.dtype)], dim=1)
                output = model.model(inputs_embeds=hidden.unsqueeze(1), pixel_values=None, image_grid_thw=None, attention_mask=current_attention, cache_position=torch.arange(cache_length, cache_length + 1, device=device), past_key_values=cache, return_dict=True, action_length=0, use_cache=True, attn_mode='causal')
                cache = output.past_key_values
                cache_length += 1
                hidden = output.last_hidden_state[:, -1, :]
            end_logits.append(model.lm_head(hidden.unsqueeze(1)).squeeze(1))
            current_attention = torch.cat([current_attention, torch.ones((1, 1), device=device, dtype=current_attention.dtype)], dim=1)
            output = model.model(inputs_embeds=end_embed, pixel_values=None, image_grid_thw=None, attention_mask=current_attention, cache_position=torch.arange(cache_length, cache_length + 1, device=device), past_key_values=cache, return_dict=True, action_length=0, use_cache=True, attn_mode='causal')
            cache = output.past_key_values
            cache_length += 1
            hidden = output.last_hidden_state[:, -1, :]
            end_hiddens.append(hidden)
        logits = torch.stack(end_logits, dim=1)
        assert torch.isfinite(logits).all(), 'nonfinite latent group logits'
        group_logprobs = F.log_softmax(logits, dim=-1)[:, :, latent_end]
        # Official validation uses stochastic latent-end sampling at temperature 1.6.
        group_probs = torch.softmax(logits[:, :, latent_end] / temperature, dim=-1)
        chosen_group = int(torch.multinomial(group_probs, 1).item())
        chosen_length = G * (chosen_group + 1)
        body_mask = (torch.arange(body_len, device=device).unsqueeze(0) < (chosen_group + 1) * (G + 1)).long()
        final_attention = torch.cat([prefix_attention, body_mask, torch.ones((1, action_count), device=device, dtype=attention.dtype)], dim=1)
        zero_action = torch.zeros((1, action_count, D), device=device, dtype=hidden.dtype)
        action_output = model.model(inputs_embeds=zero_action, pixel_values=None, image_grid_thw=None, attention_mask=final_attention, cache_position=torch.arange(prefix_end + L + max_groups, prefix_end + L + max_groups + action_count, device=device), past_key_values=cache, return_dict=True, use_cache=False, action_length=action_count, attn_mode='causal')
        action_hidden = action_output.last_hidden_state[:, -action_count:, :]
        value = model.value_head(torch.stack(end_hiddens, dim=1)[:, chosen_group, :]).squeeze(-1)
        action_logits = model.lm_head(action_hidden)[..., action_zero:action_zero + 256]
        assert torch.isfinite(action_logits).all(), 'nonfinite action logits'
        assert torch.isfinite(value).all(), 'nonfinite value'
        probabilities = torch.softmax(action_logits / temperature, dim=-1)
        action_bins = torch.multinomial(probabilities.reshape(-1, 256), 1).reshape(1, action_count)
        action_ids = (action_bins + action_zero).cpu().numpy()

    normalized = action_tokenizer.decode_token_ids_to_actions(action_ids).reshape(1, action_chunks, action_width)
    normalized[..., 6] = (normalized[..., 6] >= 0.5).astype(int)
    stats = json.loads((checkpoint / 'statistics.json').read_text())['libero']['action']
    mask = np.asarray(stats['mask'])
    low, high = np.asarray(stats['q01']), np.asarray(stats['q99'])
    actions = np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)
    assert actions.shape == (1, 8, 7) and np.isfinite(actions).all(), actions.shape
    result = {'status': 'success', 'model_id': 'last-r1', 'checkpoint': str(checkpoint), 'official_source': str(source), 'official_revision': 'ba4e939632427e066f734bf9f7abfa11cc4911f6', 'input_kind': 'image_language_offline', 'instruction': instruction, 'action_shape': list(actions.shape), 'actions': actions[0].tolist(), 'normalized_actions': normalized[0].tolist(), 'action_token_ids': action_ids[0].tolist(), 'chosen_latent_group': chosen_group, 'chosen_latent_length': chosen_length, 'latent_end_log_prob': float(group_logprobs[0, chosen_group].float().item()), 'value': float(value[0].float().item()), 'all_finite': True, 'seed': seed, 'temperature': temperature, 'max_cuda_memory_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    write('result.json', result)
    status.update(status='success', elapsed_seconds=round(time.time() - status['started_at'], 2))
except BaseException as exc:
    status.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(), elapsed_seconds=round(time.time() - status['started_at'], 2))
    traceback.print_exc()
finally:
    write('status.json', status)
    print(status, flush=True)
if status['status'] != 'success':
    raise SystemExit(1)
