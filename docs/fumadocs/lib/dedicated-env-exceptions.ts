export type DedicatedEnvGroup = 'visual' | 'embodied' | 'prepare';

export type DedicatedEnvEntry = {
  id: string;
  models: string[];
  env: string;
  setupModel: string;
  group: DedicatedEnvGroup;
  notes: { en: string; zh: string };
};

export const dedicatedEnvExceptions: DedicatedEnvEntry[] = [
  {
    id: 'ac3d',
    models: ['ac3d'],
    env: 'ac3d',
    setupModel: 'ac3d',
    group: 'visual',
    notes: {
      en: 'Official runtime pins Python 3.10, PyTorch 2.4, and CUDA 12.4. Runtime code is in tree under worldfoundry/synthesis/visual_generation/ac3d/ac3d_runtime; only checkpoints/assets are external.',
      zh: '官方 runtime 固定 Python 3.10、PyTorch 2.4 和 CUDA 12.4。推理代码在仓内 worldfoundry/synthesis/visual_generation/ac3d/ac3d_runtime；外部只需要 checkpoint/asset。',
    },
  },
  {
    id: 'cosmos3',
    models: ['cosmos3'],
    env: 'worldfoundry-cosmos3-cu128',
    setupModel: 'cosmos3',
    group: 'visual',
    notes: {
      en: 'Cosmos3 uses the native diffusion runner, checkpoint-shaped transformer, Wan VAE38, AVAE decoder, and core runtime policies. Guardrails remain an optional orchestration layer; Diffusers is not an inference backend.',
      zh: 'Cosmos3 使用原生 diffusion runner、checkpoint-shaped transformer、Wan VAE38、AVAE decoder 与 core runtime policy。Guardrail 保持为可选 orchestration layer；Diffusers 不是 inference backend。',
    },
  },
  {
    id: 'gen3c',
    models: ['gen3c'],
    env: 'gen3c',
    setupModel: 'gen3c',
    group: 'visual',
    notes: {
      en: 'Inference-only Cosmos Predict1 stack with Python 3.10, CUDA 12.4, torch 2.6, Transformer Engine, and MoGe. Setup applies the Transformer Engine link patch from released packages; it does not clone or build the training-only Apex repository.',
      zh: '仅推理的 Cosmos Predict1 栈需要 Python 3.10、CUDA 12.4、torch 2.6、Transformer Engine 和 MoGe。setup 只对发布包做 Transformer Engine include/link patch，不再 clone 或构建训练专用的 Apex 仓库。',
    },
  },
  {
    id: 'lyra-1',
    models: ['lyra-1'],
    env: 'lyra',
    setupModel: 'lyra-1',
    group: 'visual',
    notes: {
      en: 'Uses the same inference-only Cosmos Predict1 / Transformer Engine family as GEN3C.',
      zh: '复用 GEN3C 同类的仅推理 Cosmos Predict1 / Transformer Engine 依赖。',
    },
  },
  {
    id: 'magi-1',
    models: ['magi-1'],
    env: 'magi-official-cu124',
    setupModel: 'magi-1',
    group: 'visual',
    notes: {
      en: 'Official stack is Python 3.10.12 with PyTorch 2.4 / CUDA 12.4. Keep flash-attn and flashinfer in this env because their wheels are ABI-sensitive.',
      zh: '官方栈是 Python 3.10.12、PyTorch 2.4 / CUDA 12.4。flash-attn 和 flashinfer wheel 对 ABI 敏感，需要留在这个环境。',
    },
  },
  {
    id: 'warp-as-history',
    models: ['warp-as-history'],
    env: 'warp-as-history',
    setupModel: 'warp-as-history',
    group: 'visual',
    notes: {
      en: 'Pins Python 3.10, torch 2.5.0+cu121, diffusers 0.36, transformers 4.51, and peft 0.17.',
      zh: '固定 Python 3.10、torch 2.5.0+cu121、diffusers 0.36、transformers 4.51 和 peft 0.17。',
    },
  },
  {
    id: 'hunyuanvideo',
    models: ['hunyuanvideo-t2v', 'hunyuanvideo-i2v', 'hunyuanvideo-1.5-t2v', 'hunyuanvideo-1.5-i2v'],
    env: 'worldfoundry-unified-cu128',
    setupModel: 'hunyuanvideo-t2v',
    group: 'visual',
    notes: {
      en: 'All four IDs use the shared native diffusion runner and role-based Hunyuan components. The former Diffusers/xFuser CLI environment is no longer an inference backend.',
      zh: '四个 ID 全部使用共享 native diffusion runner 与按角色组织的 Hunyuan component；旧 Diffusers/xFuser CLI 环境不再作为 inference backend。',
    },
  },
  {
    id: 'kairos-sensenova',
    models: ['kairos-sensenova'],
    env: 'kairos-sensenova',
    setupModel: 'kairos-sensenova',
    group: 'visual',
    notes: {
      en: 'Uses the official torch/CUDA 12.6-class stack and native attention packages such as SageAttention. Use this env when unified CUDA does not provide the Kairos attention stack.',
      zh: '使用官方 torch/CUDA 12.6 级别依赖和 SageAttention 等 native attention 包；统一 CUDA 环境没有 Kairos attention 栈时用这个环境。',
    },
  },
  {
    id: 'lingbot-world',
    models: ['lingbot-world', 'lingbot-world-act'],
    env: 'lingbot-world',
    setupModel: 'lingbot-world',
    group: 'visual',
    notes: {
      en: 'Base-Cam and the independently cataloged public Base-Act checkpoint share the in-tree runtime. Parity requires exact torch and flash-attn pins; the unified env produced invalid near-black act2cam output in upstream-style tests.',
      zh: 'Base-Cam 与独立 catalog 的公开 Base-Act checkpoint 复用仓内 runtime。parity 需要精确 torch 和 flash-attn pins；统一环境在上游式测试里生成过近黑色视频。',
    },
  },
  {
    id: 'wonderworld',
    models: ['wonderworld'],
    env: 'wonderworld',
    setupModel: 'wonderworld',
    group: 'visual',
    notes: {
      en: 'Inference imports PyTorch3D CUDA extensions. Build or install PyTorch3D wheels compatible with this env before running the official path.',
      zh: '推理路径会 import PyTorch3D CUDA 扩展；运行 official path 前需要安装或构建与该环境匹配的 PyTorch3D wheel。',
    },
  },
  {
    id: 'wonderjourney',
    models: ['wonderjourney'],
    env: 'wonderworld',
    setupModel: 'wonderjourney',
    group: 'visual',
    notes: {
      en: 'Shares the WonderWorld PyTorch3D extension environment and uses in-tree WonderJourney runtime code.',
      zh: '复用 WonderWorld 的 PyTorch3D 扩展环境，WonderJourney runtime 代码已在仓内。',
    },
  },
  {
    id: 'hunyuan-game-craft',
    models: ['hunyuan-game-craft'],
    env: 'HYGameCraft',
    setupModel: 'hunyuan-game-craft',
    group: 'visual',
    notes: {
      en: 'Official inference relies on HYGameCraft plus torchrun sequence parallelism. The dedicated env includes the CUDA 12.4 toolkit and builds the pinned flash-attn 2.6.3 PyPI release without cloning its Git repository. Set cuda_visible_devices / torchrun_nproc_per_node explicitly on shared GPU machines.',
      zh: '官方推理依赖 HYGameCraft 和 torchrun sequence parallelism。专用环境包含 CUDA 12.4 toolkit，并从 PyPI 发布包构建固定版本 flash-attn 2.6.3，不 clone 它的 Git 仓库；共享 GPU 机器上要显式设置 cuda_visible_devices / torchrun_nproc_per_node。',
    },
  },
  {
    id: 'octo',
    models: ['octo'],
    env: 'worldfoundry-octo-jax',
    setupModel: 'octo',
    group: 'embodied',
    notes: {
      en: 'JAX CUDA 11.8, Flax 0.7.5, TensorFlow, dlimp, and Octo tokenizer assets are isolated from the PyTorch CUDA env.',
      zh: 'JAX CUDA 11.8、Flax 0.7.5、TensorFlow、dlimp 和 Octo tokenizer 资产需要与 PyTorch CUDA 环境隔离。',
    },
  },
  {
    id: 'lapa',
    models: ['lapa'],
    env: 'worldfoundry-lapa-jax-cu118',
    setupModel: 'lapa',
    group: 'embodied',
    notes: {
      en: 'Uses the official Python 3.10, JAX 0.4.25 CUDA 11/cuDNN 8.6, Flax 0.8.2, Tux 0.0.2, and Transformers 4.40 inference stack. It stays isolated because Transformers 5 removed the Flax generation modules required by LAPA.',
      zh: '使用官方 Python 3.10、JAX 0.4.25 CUDA 11/cuDNN 8.6、Flax 0.8.2、Tux 0.0.2 和 Transformers 4.40 推理栈。Transformers 5 已移除 LAPA 依赖的 Flax generation 模块，因此必须隔离。',
    },
  },
  {
    id: 'openvla-oft',
    models: ['openvla-oft'],
    env: 'worldfoundry-vla-openvla-compat-cu121',
    setupModel: 'openvla-oft',
    group: 'embodied',
    notes: {
      en: 'Prismatic/OpenVLA-compatible stack with older transformers/timm/tokenizers. Use this env for LIBERO/ALOHA OpenVLA-OFT inference.',
      zh: 'Prismatic/OpenVLA 兼容栈，使用较旧 transformers/timm/tokenizers；LIBERO/ALOHA OpenVLA-OFT 推理走这个环境。',
    },
  },
  {
    id: 'cogact',
    models: ['cogact'],
    env: 'worldfoundry-vla-openvla-compat-cu121',
    setupModel: 'cogact',
    group: 'embodied',
    notes: {
      en: 'Shares the OpenVLA compatibility env and reuses a staged OpenVLA Llama tokenizer for inference-only loading. A token is only needed when no compatible local tokenizer or complete Llama-2 mirror exists.',
      zh: '和 OpenVLA-OFT 复用兼容环境，inference-only 加载会复用已落盘的 OpenVLA Llama tokenizer；只有本地既没有兼容 tokenizer、也没有完整 Llama-2 mirror 时才需要 token。',
    },
  },
  {
    id: 'lingbot-va',
    models: ['lingbot-va'],
    env: 'lingbot-va',
    setupModel: 'lingbot-va',
    group: 'prepare',
    notes: {
      en: 'Prepare-only profile for the official Python 3.10, torch 2.9, CUDA 12.6, flash-attn stack; stage the LingBot-VA base/posttrain checkpoints before rollout.',
      zh: 'prepare-only profile，对齐官方 Python 3.10、torch 2.9、CUDA 12.6、flash-attn 栈；rollout 前先 stage LingBot-VA base/posttrain checkpoints。',
    },
  },
  {
    id: 'being-h05',
    models: ['being-h05'],
    env: 'beingh',
    setupModel: 'being-h05',
    group: 'prepare',
    notes: {
      en: 'Prepare-only profile. Stage all Being-H0.5 checkpoints; full get_action parity needs the upstream policy stack and more headroom than a single 80GB GPU profile.',
      zh: 'prepare-only profile。需要 stage 全部 Being-H0.5 checkpoints；完整 get_action parity 需要上游 policy 栈，并且单张 80GB GPU profile 余量不够。',
    },
  },
  {
    id: 'dreamdojo',
    models: ['dreamdojo'],
    env: 'dreamdojo',
    setupModel: 'dreamdojo',
    group: 'prepare',
    notes: {
      en: 'Prepare-only profile for the official torchcodec/CUDA 12.8 setup on H100-class hosts. Stage the GR-1 eval dataset before enabling real command templates.',
      zh: 'prepare-only profile，记录官方面向 H100 级主机的 torchcodec/CUDA 12.8 设置；启用真实命令模板前需要先准备 GR-1 eval dataset。',
    },
  },
];

export function dedicatedEnvSetupCommand(setupModel: string) {
  return `bash scripts/inference/prepare_model_infer.sh ${setupModel} --download`;
}
