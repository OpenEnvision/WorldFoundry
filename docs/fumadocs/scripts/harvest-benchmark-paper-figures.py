#!/usr/bin/env python3
"""Download curated official benchmark paper figures and write the docs manifest.

Only author-published teaser / method figures. Never a PDF page-1 cover.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmark_paper_figures import (
    PUBLIC_BENCHMARKS,
    caption_pair,
    overview_src,
    results_src,
    teaser_src,
)

DOCS_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = DOCS_ROOT / "lib" / "benchmark-paper-figures.json"

USER_AGENT = "WorldFoundryDocs/1.0 (benchmark paper figure harvest)"
MAX_FIGURE_BYTES = 400_000
MAX_FIGURE_WIDTH = 1600

# Official HTML / project figures. Keys are hub benchmark ids.
CURATED_FIGURES: dict[str, dict[str, str]] = {
    "ai2thor": {
        "teaser": "https://arxiv.org/html/1712.05474v4/figures/thor-cover-2.jpg",
    },
    "behavior1k": {
        "teaser": "https://arxiv.org/html/2403.09227v1/pull-new.png",
    },
    "calvin": {
        "teaser": "https://arxiv.org/html/2112.03227v4/coverfigure.png",
    },
    "larybench": {
        "teaser": "https://arxiv.org/html/2604.11689v1/lary.png",
        "overview": "https://arxiv.org/html/2604.11689v1/framework.png",
    },
    "libero": {
        "teaser": "https://arxiv.org/html/2306.03310v2/libero_fig1.png",
        "overview": "https://arxiv.org/html/2306.03310v2/figures/libero_fig2.png",
    },
    "libero-mem": {
        "teaser": "https://arxiv.org/html/2511.11478v3/Fig-memory-in-nonMarkov.png",
    },
    "libero-plus": {
        "overview": "https://arxiv.org/html/2510.13626v3/imgs/benchmark_component.png",
    },
    "libero-pro": {
        "overview": "https://arxiv.org/html/2510.03827v2/main_figure.png",
    },
    "likephys": {
        "teaser": "https://arxiv.org/html/2510.11512v3/jianhao_teaser_iclr2026.png",
        "overview": "https://arxiv.org/html/2510.11512v3/VDMPhysEvalMain.png",
    },
    "maniskill": {
        "teaser": "https://arxiv.org/html/2107.14483v5/assets/teaser.png",
    },
    "maniskill2": {
        "teaser": "https://arxiv.org/html/2302.04659v1/figures/teaser.png",
    },
    "metaworld": {
        "teaser": "https://arxiv.org/html/1910.10897v2/figure_1_metaworld.png",
        "results": "https://arxiv.org/html/1910.10897v2/figure_3_metaworld.png",
    },
    "phygenbench": {
        "teaser": "https://arxiv.org/html/2410.05363v1/crop_demo_compressed.png",
        "overview": "https://arxiv.org/html/2410.05363v1/crop_overview.png",
    },
    "physics-iq": {
        "teaser": "https://arxiv.org/html/2501.09038v3/category_samples.png",
        "overview": "https://arxiv.org/html/2501.09038v3/method.png",
    },
    "physics-iq-verified": {
        "overview": "https://arxiv.org/html/2606.18943v1/figure_1_phys_iq.png",
    },
    "rlbench": {
        "teaser": "https://arxiv.org/html/1909.12271v1/task_grid.png",
    },
    "robocasa": {
        "teaser": "https://arxiv.org/html/2406.02523v1/robots_in_kitchen.png",
        "overview": "https://arxiv.org/html/2406.02523v1/task_gen_fig_v3_comp.png",
    },
    "robotwin": {
        "teaser": "https://arxiv.org/html/2506.18088v2/teaser.png",
        "overview": "https://arxiv.org/html/2506.18088v2/main.png",
    },
    "simpler-env": {
        "teaser": "https://arxiv.org/html/2405.05941v1/teaser_v13.png",
        "overview": "https://arxiv.org/html/2405.05941v1/fig2.png",
    },
    "t2v-compbench": {
        "teaser": "https://arxiv.org/html/2407.14505v2/teaser.png",
        "overview": "https://arxiv.org/html/2407.14505v2/prompt_generation_process.png",
    },
    "vbench": {
        "teaser": "https://arxiv.org/html/2311.17982v1/fig_paper_teaser.png",
        "results": "https://arxiv.org/html/2311.17982v1/fig_paper_radar_combine.png",
    },
    "4dworldbench": {
        "teaser": "https://arxiv.org/html/2511.19836v1/framework_all.png",
    },
    "aigcbench": {
        "teaser": "https://arxiv.org/html/2401.01651v3/I2VFramework.png",
        "overview": "https://arxiv.org/html/2401.01651v3/generation_pipeline.png",
    },
    "apple-pi": {
        "teaser": "https://arxiv.org/html/2607.16401v1/teaser.png",
    },
    "bridgedata-v2": {
        "teaser": "https://arxiv.org/html/2308.12952v3/teaser.png",
        "overview": "https://arxiv.org/html/2308.12952v3/tasks.png",
    },
    "camerabench": {
        "teaser": "https://arxiv.org/html/2504.15376v2/images/teaser1.jpg",
        "overview": "https://arxiv.org/html/2504.15376v2/images/pipeline.jpg",
    },
    "chronomagic-bench": {
        "teaser": "https://arxiv.org/html/2406.18522v2/categories_example.png",
    },
    "devil-dynamics": {
        "overview": "https://arxiv.org/html/2407.01094v1/workflow.png",
    },
    "evalcrafter": {
        "teaser": "https://arxiv.org/html/2310.11440v3/teaser_img.png",
        "overview": "https://arxiv.org/html/2310.11440v3/pipeline.png",
    },
    "ewmbench": {
        "teaser": "https://arxiv.org/html/2505.09694v2/comparsion_general_video_embodied_video.png",
        "overview": "https://arxiv.org/html/2505.09694v2/pipeline.png",
    },
    "fetv": {
        "teaser": "https://arxiv.org/html/2311.01813v3/categorization.png",
        "results": "https://arxiv.org/html/2311.01813v3/leaderboard.png",
    },
    "genai-bench": {
        "teaser": "https://arxiv.org/html/2406.04485v4/gen_ai_arena.png",
    },
    "ipv-bench": {
        "teaser": "https://arxiv.org/html/2503.14378v1/teaser_example_print.png",
        "overview": "https://arxiv.org/html/2503.14378v1/main_fig.png",
    },
    "iworld-bench": {
        "teaser": "https://arxiv.org/html/2605.03941v2/overview.png",
        "overview": "https://arxiv.org/html/2605.03941v2/datapipline.png",
    },
    "kinetix": {
        "teaser": "https://arxiv.org/html/2410.23208v2/images/fig1_v6.png",
        "results": "https://arxiv.org/html/2410.23208v2/plot_levels_l.png",
    },
    "memobench": {
        "teaser": "https://arxiv.org/html/2606.27537v6/teaser.png",
        "overview": "https://arxiv.org/html/2606.27537v6/data_curation_v4.png",
    },
    "mikasa": {
        "teaser": "https://arxiv.org/html/2502.10550v3/visual-abstract-scheme_v2.png",
        "overview": "https://arxiv.org/html/2502.10550v3/envs-demo.png",
        "results": "https://arxiv.org/html/2502.10550v3/spider_chart_v2.png",
    },
    "mind": {
        "teaser": "https://arxiv.org/html/2602.08025v2/first_figure.png",
        "overview": "https://arxiv.org/html/2602.08025v2/overview.png",
    },
    "molmospaces": {
        "teaser": "https://arxiv.org/html/2603.16861v2/MolmoBotTeaser_11.png",
    },
    "pawbench": {
        "teaser": "https://arxiv.org/html/2608.27345v3/one_plausible_future.png",
        "overview": "https://arxiv.org/html/2608.27345v3/paweval_overview.png",
    },
    "phyeduvideo": {
        "teaser": "https://arxiv.org/html/2601.00943v1/intro-fig.png",
    },
    "phyfps-bench-gen": {
        "teaser": "https://arxiv.org/html/2603.14375v2/Chronometric_Hallucination_V2.png",
    },
    "phyground": {
        "teaser": "https://arxiv.org/html/2605.10806v1/teaser.png",
    },
    "physical-ai-bench": {
        "teaser": "https://arxiv.org/html/2512.01989v1/physical-ai-bench-teaser-20250928.png",
    },
    "physvidbench": {
        "teaser": "https://arxiv.org/html/2507.15824v1/PhysVidBench-Teaser.png",
        "overview": "https://arxiv.org/html/2507.15824v1/PhysVidBench-Benchmark-v2.png",
    },
    "rbench": {
        "teaser": "https://arxiv.org/html/2601.15282v1/Fig1_teaser.png",
        "overview": "https://arxiv.org/html/2601.15282v1/Fig2_datapipeline.png",
    },
    "robocerebra": {
        "teaser": "https://arxiv.org/html/2506.06677v2/vla-fig1.png",
        "overview": "https://arxiv.org/html/2506.06677v2/vla-fig2-big.png",
    },
    "robomme": {
        "teaser": "https://arxiv.org/html/2603.04639v3/robomme_bench.png",
    },
    "sana-wm-bench": {
        "teaser": "https://arxiv.org/html/2605.15178v1/benchmark_initial_frame_grid.png",
    },
    "stevo-bench": {
        "teaser": "https://arxiv.org/html/2603.13215v1/teaser_v1.png",
        "overview": "https://arxiv.org/html/2603.13215v1/benchmark_figure.png",
    },
    "t2v-safety-bench": {
        "teaser": "https://arxiv.org/html/2407.05965v3/crop_fig3_1.png",
    },
    "t2vworldbench": {
        "teaser": "https://arxiv.org/html/2507.18107v1/benchmark_example.png",
    },
    "vbench-2.0": {
        "teaser": "https://arxiv.org/html/2503.21755v2/fig_paper_teaser.png",
        "overview": "https://arxiv.org/html/2503.21755v2/anomaly_framework.png",
    },
    "vbench-plus-plus": {
        "teaser": "https://arxiv.org/html/2411.13503v1/fig_extention_teaser.png",
    },
    "video-bench": {
        "overview": "https://arxiv.org/html/2504.04907v2/chain_of_query_v2.png",
    },
    "videophy": {
        "teaser": "https://arxiv.org/html/2406.03520v2/VideoPhysics.png",
    },
    "videophy2": {
        "teaser": "https://arxiv.org/html/2503.06800v1/videophy.png",
    },
    "videoscore": {
        "teaser": "https://arxiv.org/html/2406.15252v3/teaser.png",
    },
    "videoverse": {
        "teaser": "https://arxiv.org/html/2510.08398v4/eval-dimensions_0301.png",
    },
    "visual-chronometer": {
        "teaser": "https://arxiv.org/html/2603.14375v2/Teaser.png",
    },
    "vlabench": {
        "teaser": "https://arxiv.org/html/2412.18194v1/Figure1_overview.png",
        "overview": "https://arxiv.org/html/2412.18194v1/Figure3_task_example.png",
    },
    "vmbench": {
        "teaser": "https://arxiv.org/html/2503.10076v2/intro_v6.png",
        "overview": "https://arxiv.org/html/2503.10076v2/eval_pipeline_v3.png",
    },
    "wbench": {
        "teaser": "https://arxiv.org/html/2605.25874v1/teaser.png",
    },
    "world-in-world": {
        "teaser": "https://arxiv.org/html/2510.18135v2/teaser_figure.png",
        "overview": "https://arxiv.org/html/2510.18135v2/eval_framework.png",
    },
    "worldarena": {
        "teaser": "https://arxiv.org/html/2602.08971v2/video_metric_part2.png",
        "overview": "https://arxiv.org/html/2602.08971v2/embtask.png",
    },
    "worldbench": {
        "teaser": "https://arxiv.org/html/2601.21282v2/overview.png",
        "overview": "https://arxiv.org/html/2601.21282v2/perfphys_overview.png",
    },
    "worldmodelbench": {
        "teaser": "https://arxiv.org/html/2502.20694v1/VILA-EWM-Figure1-new.png",
    },
    "worldolympiad": {
        "teaser": "https://arxiv.org/html/2606.11129v2/overview.png",
        "overview": "https://arxiv.org/html/2606.11129v2/dataOverview.png",
    },
    "worldreasonbench": {
        "teaser": "https://arxiv.org/html/2605.10434v1/bench_overview_2.png",
        "overview": "https://arxiv.org/html/2605.10434v1/data_pipe.png",
    },
    "worldscore": {
        "teaser": "https://arxiv.org/html/2504.00983v2/comparison_vbench.png",
        "overview": "https://arxiv.org/html/2504.00983v2/framework.png",
    },
    "wrbench": {
        "teaser": "https://arxiv.org/html/2606.20545v1/teaser3.png",
        "overview": "https://arxiv.org/html/2606.20545v1/Overview.png",
    },
}

AR5IV_FALLBACK: dict[str, dict[str, str]] = {
    bench: {role: url.replace("https://arxiv.org/html/", "https://ar5iv.labs.arxiv.org/html/") for role, url in roles.items()}
    for bench, roles in CURATED_FIGURES.items()
}


def request(url: str, timeout: int = 40) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status >= 400:
                return None
            return response.read()
    except urllib.error.HTTPError as exc:
        print(f"  http {exc.code} {url}", flush=True)
        return None
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        print(f"  err {exc} {url}", flush=True)
        return None


def looks_like_pdf_page(path: Path, role: str = "") -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False
    with Image.open(path) as image:
        width, height = image.size
    # HTML extracts can be short wide teasers. Only reject tall PDF page-1 renders.
    # Official results plots (curves, radar) are often tall — do not treat those as covers.
    if width < 80 or height < 80:
        return True
    if role != "results" and height > width * 1.25 and height >= 1200:
        return True
    return False


def compress_figure(src: Path, dest: Path) -> Path | None:
    if not src.is_file() or src.stat().st_size < 800:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    cwebp = shutil.which("cwebp")
    webp_dest = dest.with_suffix(".webp")
    if cwebp:
        quality = "78" if src.stat().st_size > MAX_FIGURE_BYTES else "82"
        cmd = [
            cwebp,
            "-quiet",
            "-q",
            quality,
            "-m",
            "6",
            "-resize",
            str(MAX_FIGURE_WIDTH),
            "0",
            str(src),
            "-o",
            str(webp_dest),
        ]
        if subprocess.run(cmd, check=False, capture_output=True).returncode == 0 and webp_dest.is_file():
            if dest.exists() and dest != webp_dest:
                dest.unlink()
            return webp_dest
    sips = shutil.which("sips")
    jpeg_dest = dest.with_suffix(".jpg")
    if sips:
        cmd = [
            "sips",
            "-s",
            "format",
            "jpeg",
            "-s",
            "formatOptions",
            "80",
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


def existing_role(dest_dir: Path, role: str) -> Path | None:
    for path in dest_dir.glob(f"{role}.*"):
        if path.is_file() and path.stat().st_size > 800:
            return path
    return None


def harvest_one(bench_id: str, role: str, url: str) -> tuple[str, str, str]:
    dest_dir = PUBLIC_BENCHMARKS / bench_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = existing_role(dest_dir, role)
    if existing:
        return bench_id, f"{role}:exists:{existing.name}", "exists"
    raw = request(url, timeout=40)
    if (not raw or len(raw) < 4000) and bench_id in AR5IV_FALLBACK and role in AR5IV_FALLBACK[bench_id]:
        time.sleep(0.2)
        raw = request(AR5IV_FALLBACK[bench_id][role], timeout=40)
    if not raw or len(raw) < 4000:
        print(f"  miss {bench_id}/{role}", flush=True)
        return bench_id, f"{role}:miss", "miss"
    suffix = ".png"
    if raw[:3] == b"\xff\xd8\xff":
        suffix = ".jpg"
    elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        suffix = ".webp"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(raw)
        tmp = Path(handle.name)
    try:
        if looks_like_pdf_page(tmp, role):
            print(f"  skip pdf-page-like {bench_id}/{role}", flush=True)
            return bench_id, f"{role}:pdf-page", "skip"
        dest = dest_dir / f"{role}{suffix}"
        written = compress_figure(tmp, dest)
        if written:
            note = f"{role}:{written.name}:{written.stat().st_size}"
            print(f"  wrote {bench_id}/{written.name} ({written.stat().st_size} bytes)", flush=True)
            return bench_id, note, "wrote"
    finally:
        tmp.unlink(missing_ok=True)
    return bench_id, f"{role}:fail", "fail"


def download_curated_figures() -> dict[str, list[str]]:
    saved: dict[str, list[str]] = {}
    jobs = [(bench_id, role, url) for bench_id, roles in CURATED_FIGURES.items() for role, url in roles.items()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(harvest_one, bench_id, role, url) for bench_id, role, url in jobs]
        for fut in as_completed(futs):
            bench_id, note, _status = fut.result()
            saved.setdefault(bench_id, []).append(note)
    return saved


def write_manifest() -> dict[str, dict[str, str]]:
    previous: dict[str, dict[str, str]] = {}
    if OUT_PATH.is_file():
        try:
            previous = json.loads(OUT_PATH.read_text(encoding="utf-8")).get("figures", {})
        except json.JSONDecodeError:
            previous = {}
    registry: dict[str, dict[str, str]] = {}
    ids = sorted(
        {
            *CURATED_FIGURES,
            *{p.name for p in PUBLIC_BENCHMARKS.iterdir() if p.is_dir()},
        }
        if PUBLIC_BENCHMARKS.is_dir()
        else set(CURATED_FIGURES)
    )
    for bench_id in ids:
        teaser = teaser_src(bench_id)
        overview = overview_src(bench_id)
        results = results_src(bench_id)
        if not teaser and not overview and not results:
            continue
        entry: dict[str, str] = {"benchmarkId": bench_id}
        if teaser:
            en, zh = caption_pair(bench_id, "teaser")
            entry["teaserSrc"] = teaser
            entry["teaserAlt"] = en
            entry["teaserAltZh"] = zh
            entry["teaserCaption"] = en
            entry["teaserCaptionZh"] = zh
        if overview:
            en, zh = caption_pair(bench_id, "overview")
            entry["src"] = overview
            entry["alt"] = en
            entry["altZh"] = zh
            entry["caption"] = en
            entry["captionZh"] = zh
        if results:
            en, zh = caption_pair(bench_id, "results")
            entry["resultsSrc"] = results
            entry["resultsAlt"] = en
            entry["resultsAltZh"] = zh
            entry["resultsCaption"] = en
            entry["resultsCaptionZh"] = zh
        registry[bench_id] = entry
    for bench_id, entry in previous.items():
        if bench_id not in registry and (teaser_src(bench_id) or overview_src(bench_id) or results_src(bench_id)):
            registry[bench_id] = entry
    payload = {"version": 1, "figures": dict(sorted(registry.items()))}
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return registry


def main() -> int:
    print("downloading curated official benchmark paper figures", flush=True)
    saved = download_curated_figures()
    registry = write_manifest()
    with_teaser = sum(1 for item in registry.values() if item.get("teaserSrc"))
    with_overview = sum(1 for item in registry.values() if item.get("src"))
    with_results = sum(1 for item in registry.values() if item.get("resultsSrc"))
    print(
        f"wrote {OUT_PATH.name}: {len(registry)} benches, "
        f"{with_teaser} teasers, {with_overview} overview figures, "
        f"{with_results} results figures; "
        f"downloaded/kept {len(saved)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
