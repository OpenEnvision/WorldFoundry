type Locale = 'en' | 'zh';

type Localized = Record<Locale, string>;

type EnvVar = {
  name: string | readonly string[];
  fallback: Localized;
  effect: Localized;
};

type EnvVarGroup = {
  id: string;
  title: Localized;
  items: readonly EnvVar[];
};

const groups: readonly EnvVarGroup[] = [
  {
    id: 'playback',
    title: { en: 'Playback & generation', zh: '播放与生成' },
    items: [
      {
        name: 'WORLDFOUNDRY_REALTIME_FPS',
        fallback: { en: '16', zh: '16' },
        effect: {
          en: 'WebRTC video pacing and input-resampler rate.',
          zh: 'WebRTC 播放节奏与输入重采样频率。',
        },
      },
      {
        name: 'WORLDFOUNDRY_REALTIME_CHUNK_FRAMES',
        fallback: { en: '9', zh: '9' },
        effect: {
          en: 'Target frames per generated chunk; known model temporal contracts may override it.',
          zh: '每个生成 chunk 的目标帧数；有严格时序约束的模型会覆盖该值。',
        },
      },
      {
        name: 'WORLDFOUNDRY_REALTIME_INFERENCE_STEPS',
        fallback: {
          en: 'model default; LingBot Fast 4',
          zh: '模型默认；LingBot Fast 为 4',
        },
        effect: {
          en: 'Explicit latency/quality override. Non-distilled models are not silently reduced to four steps.',
          zh: '显式的延迟/质量覆盖；非蒸馏模型不会被静默降到 4 步。',
        },
      },
      {
        name: 'WORLDFOUNDRY_REALTIME_WARMUP_CHUNKS',
        fallback: { en: '1', zh: '1' },
        effect: {
          en: 'Background stream warmup; set 0 to disable or raise it for compile/autotune experiments.',
          zh: '后台 stream warmup；设为 0 可关闭，编译/autotune 实验可调高。',
        },
      },
      {
        name: 'WORLDFOUNDRY_REALTIME_PREWARM_TIMEOUT_SECONDS',
        fallback: { en: 'unset', zh: '未设置' },
        effect: {
          en: 'Optional deadline for configure plus warmup chunks. The server reports an overrun after the current non-cancellable model call returns; 0 disables it.',
          zh: 'configure 与 warmup chunks 的可选 deadline；当前不可取消的模型调用返回后才报告超时，设为 0 可关闭。',
        },
      },
    ],
  },
  {
    id: 'queue',
    title: { en: 'Queue & presentation', zh: '队列与呈现' },
    items: [
      {
        name: 'WORLDFOUNDRY_REALTIME_FRAME_QUEUE',
        fallback: { en: 'max(chunk, 8)', zh: 'max(chunk, 8)' },
        effect: {
          en: 'One steady-state chunk of bounded playback buffering.',
          zh: '单个 steady-state chunk 的有界播放队列。',
        },
      },
      {
        name: 'WORLDFOUNDRY_REALTIME_FRAME_QUEUE_POLICY',
        fallback: { en: 'latest-interactive', zh: 'latest-interactive' },
        effect: {
          en: 'Use latest-interactive to bound latency under congestion or ordered-quality to preserve every frame. Count-exact prompt/segment sessions always enforce ordered-quality.',
          zh: 'latest-interactive 在拥塞时限制延迟，ordered-quality 保留每一帧；严格计数的 prompt/segment 会话始终强制使用 ordered-quality。',
        },
      },
      {
        name: 'WORLDFOUNDRY_REALTIME_PRESENTATION_MODE',
        fallback: { en: 'hold-last', zh: 'hold-last' },
        effect: {
          en: 'WebRTC A/B switch. hold-last preserves the compatible constant clock; real-frames gives pacing to the chunk presentation manager and sends no repeated frames through its two-frame mailbox.',
          zh: 'WebRTC A/B 开关。hold-last 保留兼容的恒定时钟；real-frames 将 pacing 交给 chunk presentation manager，并通过两帧 mailbox 只发送真实帧。',
        },
      },
      {
        name: 'WORLDFOUNDRY_REALTIME_PRESENTATION_MAX_FPS',
        fallback: { en: '2 × model FPS', zh: '模型 FPS × 2' },
        effect: {
          en: 'Maximum active-chunk drain rate while a pending chunk is waiting in real-frames mode.',
          zh: 'real-frames 模式存在 pending chunk 时，active chunk 的最大 drain 速率。',
        },
      },
      {
        name: 'WORLDFOUNDRY_REALTIME_SOCKET_PACING_MODE',
        fallback: { en: 'legacy-dual', zh: 'legacy-dual' },
        effect: {
          en: 'WebSocket A/B switch. client-only disables server-side frame sleeps and leaves the existing browser presentation clock as the sole pacing owner.',
          zh: 'WebSocket A/B 开关。client-only 关闭服务端逐帧 sleep，仅保留浏览器现有的 presentation clock。',
        },
      },
      {
        name: 'WORLDFOUNDRY_REALTIME_BACKPRESSURE_MS',
        fallback: { en: 'half a chunk', zh: '半个 chunk' },
        effect: {
          en: 'Preserve every frame during normal playback; evict stale frames only after this congestion budget expires.',
          zh: '正常播放时保留全部帧；只有拥塞超过预算才丢弃旧帧。',
        },
      },
    ],
  },
  {
    id: 'rtx',
    title: { en: 'RTX post-process', zh: 'RTX 后处理' },
    items: [
      {
        name: 'WORLDFOUNDRY_REALTIME_POSTPROCESS_PRESET',
        fallback: { en: 'identity', zh: 'identity' },
        effect: {
          en: 'Select identity, rtx-super-resolution, rtx-super-resolution-ultra, or rtx-deblur-ultra. RTX choices are explicit opt-ins and fail before serving if their runtime is unavailable.',
          zh: '可选 identity、rtx-super-resolution、rtx-super-resolution-ultra 或 rtx-deblur-ultra。RTX 必须显式启用；运行条件不满足时会在启动服务前失败。',
        },
      },
      {
        name: 'WORLDFOUNDRY_REALTIME_RTX_SCALE',
        fallback: {
          en: 'preset default (2, or 1 for deblur)',
          zh: 'preset 默认值（VSR 为 2，deblur 为 1）',
        },
        effect: {
          en: 'Spatial VSR scale. NVIDIA VFX supports up to 4x; same-resolution denoise/deblur modes require 1.',
          zh: 'VSR 空间放大倍数；NVIDIA VFX 最高支持 4x，同分辨率去噪/去模糊模式必须为 1。',
        },
      },
      {
        name: [
          'WORLDFOUNDRY_REALTIME_RTX_OUTPUT_WIDTH',
          'WORLDFOUNDRY_REALTIME_RTX_OUTPUT_HEIGHT',
        ],
        fallback: { en: 'unset', zh: '未设置' },
        effect: {
          en: 'Optional explicit RTX output dimensions. Set both or neither.',
          zh: '可选的 RTX 显式输出尺寸；必须同时设置或同时不设。',
        },
      },
      {
        name: 'WORLDFOUNDRY_REALTIME_RTX_QUALITY',
        fallback: { en: 'preset default', zh: 'preset 默认值' },
        effect: {
          en: 'Override the NVIDIA VFX quality enum, for example HIGH, ULTRA, DEBLUR_ULTRA, or HIGHBITRATE_HIGH.',
          zh: '覆盖 NVIDIA VFX quality 枚举，例如 HIGH、ULTRA、DEBLUR_ULTRA 或 HIGHBITRATE_HIGH。',
        },
      },
      {
        name: 'WORLDFOUNDRY_REALTIME_RTX_DEVICE',
        fallback: {
          en: 'Studio device index, otherwise 0',
          zh: 'Studio device 中的 index，否则为 0',
        },
        effect: {
          en: 'CUDA device index owned by the VFX effect.',
          zh: 'VFX effect 使用的 CUDA device index。',
        },
      },
      {
        name: 'WORLDFOUNDRY_REALTIME_RTX_NON_BLOCKING',
        fallback: { en: 'false', zh: 'false' },
        effect: {
          en: 'Request asynchronous VFX execution. WorldFoundry still synchronizes before cloning the borrowed DLPack output, so this does not weaken memory safety.',
          zh: '请求异步 VFX 执行。WorldFoundry 在 clone 借用的 DLPack 输出前仍会同步，不会降低内存安全性。',
        },
      },
      {
        name: 'WORLDFOUNDRY_REALTIME_RTX_USE_CURRENT_STREAM',
        fallback: { en: 'true', zh: 'true' },
        effect: {
          en: 'Pass the current PyTorch CUDA stream pointer to NVIDIA VFX; disable only when diagnosing stream interop.',
          zh: '把当前 PyTorch CUDA stream pointer 传给 NVIDIA VFX；只应在排查 stream interop 时关闭。',
        },
      },
    ],
  },
  {
    id: 'perf',
    title: { en: 'Profiling', zh: '性能采样' },
    items: [
      {
        name: 'WORLDFOUNDRY_REALTIME_PERF_LOG_INTERVAL_CHUNKS',
        fallback: { en: '5', zh: '5' },
        effect: {
          en: 'Emit a structured timing summary with stage p50/p90 values every N chunks; set 0 to disable periodic summaries.',
          zh: '每 N 个 chunk 输出一次含各阶段 p50/p90 的结构化 timing 摘要；设为 0 可关闭周期摘要。',
        },
      },
      {
        name: 'WORLDFOUNDRY_REALTIME_PERF_WARMUP_CHUNKS',
        fallback: { en: '0', zh: '0' },
        effect: {
          en: 'Mark and exclude the first N session chunks from timing summaries. Raw JSONL events retain the warmup marker.',
          zh: '将会话开头 N 个 chunk 标记为 warmup 并从摘要中排除；原始 JSONL 仍保留 warmup 标记。',
        },
      },
      {
        name: 'WORLDFOUNDRY_REALTIME_PERF_JSONL',
        fallback: { en: 'unset', zh: '未设置' },
        effect: {
          en: 'Optional path for one canonical realtime.chunk_timing JSONL event per chunk. Use only while profiling to avoid hot-path file I/O.',
          zh: '可选的 JSONL 路径，每个 chunk 写一条规范化 realtime.chunk_timing 事件；仅建议 profiling 时启用，避免热路径文件 I/O。',
        },
      },
    ],
  },
  {
    id: 'session',
    title: { en: 'Session & network', zh: '会话与网络' },
    items: [
      {
        name: 'WORLDFOUNDRY_REALTIME_LIVENESS_TIMEOUT_SECONDS',
        fallback: { en: '30', zh: '30' },
        effect: {
          en: 'Reclaim a session whose browser heartbeats disappear without killing a slow in-flight chunk.',
          zh: '浏览器 heartbeat 消失后自动回收会话，同时避免误杀较慢的在途 chunk。',
        },
      },
      {
        name: 'WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS',
        fallback: { en: '5', zh: '5' },
        effect: {
          en: 'Total deadline for cooperative async teardown of preload, session workers, transports, and runtime reset. It bounds the shutdown handler but does not force-kill a running model thread.',
          zh: 'preload、会话 worker、传输层与 runtime reset 共享的协作式异步 teardown 总 deadline；它限制 shutdown handler 的等待时间，但不会强杀仍在运行的模型线程。',
        },
      },
      {
        name: 'WORLDFOUNDRY_REALTIME_ICE_SERVERS_JSON',
        fallback: { en: 'unset', zh: '未设置' },
        effect: {
          en: 'Optional browser-compatible STUN/TURN iceServers JSON for hosts that are not directly reachable.',
          zh: '浏览器无法直连 GPU 主机时使用的 STUN/TURN iceServers JSON。',
        },
      },
    ],
  },
];

const copy = {
  en: { fallback: 'Default' },
  zh: { fallback: '默认' },
} as const;

function namesOf(name: EnvVar['name']) {
  return typeof name === 'string' ? [name] : [...name];
}

export function StudioRealtimeEnvVars({ locale = 'en' }: { locale?: Locale }) {
  const t = copy[locale];

  return (
    <div className="pi-kv-catalog is-vars not-prose">
      {groups.map((group) => (
        <section className="pi-kv-group" key={group.id}>
          <h3>{group.title[locale]}</h3>
          <ul>
            {group.items.map((item) => (
              <li className="pi-kv-row" key={namesOf(item.name).join('/')}>
                <div className="pi-kv-head">
                  <div className="pi-kv-names">
                    {namesOf(item.name).map((name) => (
                      <code key={name}>{name}</code>
                    ))}
                  </div>
                  <span className="pi-kv-default">
                    <span>{t.fallback}</span>
                    {item.fallback[locale]}
                  </span>
                </div>
                <p>{item.effect[locale]}</p>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
