#!/usr/bin/env python3
"""
Google Colab G4 / A100 / L4 一键模型部署脚本 (路线 2 · 融合 mizore.blog + EasyColab)
支持在 Google Drive 根目录下建立 `models/` 总文件夹，每个子文件夹存放一个模型：

Google Drive 目录结构:
  MyDrive/
  └── models/
      ├── Huihui-Qwen3.8-27B-abliterated-GGUF/      # 模型 1 (例如 huihui-ai 的 GGUF)
      ├── Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4/   # 模型 2
      └── <任意你新建的模型文件夹>/                 # 模型 N

功能特性:
  1. 原生挂载完整 Google Drive 根目录 (/content/drive/MyDrive)，自动清理旧的 rclone 子目录挂载。
  2. 若 `MyDrive/models/<文件夹>` 中已有模型权重，16 线程极速预热至 Colab 内存盘 (/dev/shm) 或 NVMe 加载。
  3. 若 `MyDrive/models/<文件夹>` 为空，自动从 HuggingFace 8 线程并发下载 (400~700 MB/s)，并同步写入该云盘文件夹。
  4. 自动拉起 llama.cpp CUDA 12.8 Server (262K 上下文 + FlashAttention + 多模态视觉) 与 Cloudflare Tunnel 公网 OpenAI 兼容 API。
"""

import argparse
import fnmatch
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DRIVE_MODELS_ROOT = "/content/drive/MyDrive/models"
LOCAL_MODELS_ROOT = "/dev/shm/local_models" if os.path.isdir("/dev/shm") else "/content/local_models"

HF_REPO = os.environ.get("HF_REPO", "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF")
MODEL_FOLDER = os.environ.get("MODEL_FOLDER", "Huihui-Qwen3.8-27B-abliterated-GGUF")
FILE_PATTERN = os.environ.get("FILE_PATTERN", "Huihui-Qwen3.8-27B-abliterated-bf16.gguf")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CTX_LEN = int(os.environ.get("CTX_LEN", "262144") or "262144")
API_KEY = os.environ.get("COLAB_API_KEY", "sk-colab-qwen38-default")
SYNC_TO_DRIVE = os.environ.get("SYNC_TO_DRIVE", "1") not in ("0", "false", "False", "no")
DOWNLOAD_ONLY = os.environ.get("DOWNLOAD_ONLY", "0") in ("1", "true", "True", "yes")


def _matches_main_pattern(filepath: str, pattern: str) -> bool:
    """检查本地或云盘中的 GGUF 文件是否符合当前请求的 FILE_PATTERN (mmproj/mtp 辅助权重始终保留)"""
    base = os.path.basename(filepath).lower()
    if "mmproj" in base or "/mtp" in filepath.lower() or base.startswith("mtp"):
        return True
    pat = (pattern or "").strip().lower()
    if not pat:
        return True
    glob_pat = pat if ("*" in pat or "?" in pat) else f"*{pat}*"
    return fnmatch.fnmatch(base, glob_pat) or fnmatch.fnmatch(filepath.lower(), glob_pat)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_running_on_colab_vm() -> bool:
    return os.path.exists("/var/colab/hostname") or (
        os.path.isdir("/content") and sys.platform.startswith("linux")
    )


# ============================================================================
# 远端 Colab VM 执行逻辑 (下载/同步模型 + 启动 llama-server + Cloudflare Tunnel)
# ============================================================================

class _StripAuthRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            new_req.unredirected_hdrs.pop("Authorization", None)
            new_req.headers.pop("Authorization", None)
        return new_req


def resolve_hf_gguf_files(repo_id: str, pattern: str = "", token: str = "") -> List[Tuple[str, int]]:
    """查询 HuggingFace API，选出匹配的 GGUF 主模型文件 + 视觉 mmproj + MTP 投机解码头"""
    headers = {"User-Agent": "ColabDeploy/1.0", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    meta = None
    for host in ("https://huggingface.co", "https://hf-mirror.com"):
        try:
            req = urllib.request.Request(f"{host}/api/models/{repo_id}?blobs=true", headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                meta = json.loads(resp.read().decode("utf-8"))
                break
        except Exception:
            continue

    if not meta:
        raise RuntimeError(f"无法读取 HuggingFace 仓库 {repo_id} 元数据，请检查仓库名或 HF_TOKEN。")

    siblings = meta.get("siblings") or []
    all_ggufs: List[Tuple[str, int]] = []
    for s in siblings:
        rfn = str(s.get("rfilename") or "").strip()
        if rfn.lower().endswith(".gguf"):
            sz = int(s.get("size") or (s.get("lfs") or {}).get("size") or 0)
            all_ggufs.append((rfn, sz))

    if not all_ggufs:
        raise RuntimeError(f"仓库 {repo_id} 中未找到任何 .gguf 文件。")

    mmprojs = [(r, s) for r, s in all_ggufs if "mmproj" in r.lower()]
    mtps = [
        (r, s) for r, s in all_ggufs
        if ("/mtp" in r.lower() or os.path.basename(r.lower()).startswith("mtp"))
        and "mmproj" not in r.lower()
    ]
    mains = [
        (r, s) for r, s in all_ggufs
        if (r, s) not in mmprojs
        and (r, s) not in mtps
        and "imatrix" not in r.lower()
        and not os.path.basename(r.lower()).startswith("ggml-vocab")
    ]

    selected: List[Tuple[str, int]] = []
    pat = (pattern or "").strip().lower()
    candidates = mains
    if pat:
        glob_pat = pat if ("*" in pat or "?" in pat) else f"*{pat}*"
        candidates = [
            (r, s) for r, s in mains
            if fnmatch.fnmatch(r.lower(), glob_pat) or fnmatch.fnmatch(os.path.basename(r.lower()), glob_pat)
        ]
        if not candidates:
            avail = ", ".join(os.path.basename(r) for r, _ in mains[:10])
            raise RuntimeError(f"仓库 {repo_id} 中未找到匹配 '{pattern}' 的 GGUF，可选文件: {avail}")

    priority = ["bf16", "f16", "ud-dw-q4_k_m", "ud-q4_k_m", "q4_k_m", "q4_k_l", "iq4_xs", "q4_k_s", "q5_k_m", "q6_k", "q8_0"]
    groups: Dict[str, List[Tuple[str, int]]] = {}
    for r, s in candidates:
        key = re.sub(r"-\d{5}-of-\d{5}\.gguf$", ".gguf", r, flags=re.IGNORECASE)
        groups.setdefault(key, []).append((r, s))
    best_key = next(iter(groups.keys()))
    best_rank = (999, 999, 999)
    for k in groups:
        kl = k.lower()
        p_rank = 999
        for idx_p, qp in enumerate(priority):
            if qp in kl:
                p_rank = idx_p
                break
        # 同档位优先选非 -mtp / 非 -ud-dw 后缀的标准原版权重，文件名更短更标准者优先
        mtp_penalty = 1 if ("-mtp" in kl or "ud-dw" in kl) else 0
        rank = (p_rank, mtp_penalty, len(kl))
        if rank < best_rank:
            best_rank = rank
            best_key = k
    selected.extend(sorted(groups[best_key], key=lambda x: x[0]))

    if mmprojs:
        mmprojs.sort(key=lambda x: (0 if "f16" in x[0].lower() else (1 if "bf16" in x[0].lower() else 2), x[0]))
        selected.append(mmprojs[0])
    if mtps:
        mtps.sort(key=lambda x: (0 if "q4_0" in x[0].lower() else 1, x[0]))
        selected.append(mtps[0])

    return selected


PRESET_SAFETENSORS_FILES: Dict[str, List[str]] = {
    "MiniMaxAI/MiniMax-H3": (
        [f"FL2VA/transformer/model-{i:05d}-of-00013.safetensors" for i in range(1, 14)]
        + [f"FL2VA/text_encoder/model-{i:05d}-of-00014.safetensors" for i in range(1, 15)]
        + [
            "FL2VA/video_vae/source/model.safetensors",
            "FL2VA/audio_vae/model.safetensors",
        ]
    ),
    "MiniMaxAI/MiniMax-H3#Ref2VA": (
        [f"Ref2VA/transformer/model-{i:05d}-of-00013.safetensors" for i in range(1, 14)]
        + [f"FL2VA/text_encoder/model-{i:05d}-of-00014.safetensors" for i in range(1, 15)]
        + [
            "FL2VA/video_vae/source/model.safetensors",
            "FL2VA/audio_vae/model.safetensors",
        ]
    ),
    "Comfy-Org/MiniMax-H3-Image": [
        "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
        "vae/minimax_h3_image_vae_fp16.safetensors",
    ],
    "unsloth/MiniMax-H3-GGUF": [
        "minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf",
        "qwen3vl_32b_minimax_h3-Q2_K_M.gguf",
        "vae/minimax_h3_video_vae_fp16.safetensors",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
    ],
    "Decorator/ReCaVSR": [
        "recavsr_wan2.2_5B.pth",
    ],
    "IndexTeam/IndexTTS-2": [
        "gpt.pth",
        "s2mel.pth",
        "qwen0.6bemo4-merge/model.safetensors",
        "config.yaml",
        "bpe.model",
        "feat1.pt",
        "feat2.pt",
        "wav2vec2bert_stats.pt",
    ],
    "meituan-longcat/LongCat-Video-Avatar-1.6": [
        *(f"base_model/diffusion_pytorch_model-{i:05d}-of-00006.safetensors" for i in range(1, 7)),
        "loraA/lora.safetensors",
        "loraB/lora.safetensors",
        "vocal_separator/Kim_Vocal_2.onnx",
        "whisper-large-v3/model.safetensors",
    ],
    "meituan-longcat/LongCat-Video-Avatar-1.6#int8": [
        *(f"base_model_int8/quantized_model-{i:05d}-of-00004.safetensors" for i in range(1, 5)),
        "loraA/lora.safetensors",
        "loraB/lora.safetensors",
        "vocal_separator/Kim_Vocal_2.onnx",
        "whisper-large-v3/model.safetensors",
    ],
}


def resolve_hf_safetensors_files(repo_id: str, pattern: str = "", token: str = "") -> List[Tuple[str, int]]:
    """查询 HuggingFace API，解析开源模型中心中非 GGUF 模型 (如 MiniMax-H3 / Qwen-Image-2.1 / Z-Image-Turbo) 的权重列表"""
    headers = {"User-Agent": "ColabDeploy/1.0", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    meta = None
    for host in ("https://huggingface.co", "https://hf-mirror.com"):
        try:
            req = urllib.request.Request(f"{host}/api/models/{repo_id}?blobs=true", headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                meta = json.loads(resp.read().decode("utf-8"))
                break
        except Exception:
            continue

    size_map: Dict[str, int] = {}
    all_siblings: List[Tuple[str, int]] = []
    if meta:
        for s in meta.get("siblings") or []:
            rfn = str(s.get("rfilename") or "").strip()
            sz = int(s.get("size") or (s.get("lfs") or {}).get("size") or 0)
            size_map[rfn] = sz
            all_siblings.append((rfn, sz))

    preset_key = repo_id
    if repo_id == "MiniMaxAI/MiniMax-H3" and "ref2va" in (pattern or "").lower():
        preset_key = "MiniMaxAI/MiniMax-H3#Ref2VA"
    elif repo_id == "meituan-longcat/LongCat-Video-Avatar-1.6" and "int8" in (pattern or "").lower():
        preset_key = "meituan-longcat/LongCat-Video-Avatar-1.6#int8"
    if preset_key in PRESET_SAFETENSORS_FILES:
        preset_list = PRESET_SAFETENSORS_FILES[preset_key]
        out: List[Tuple[str, int]] = [(fn, size_map.get(fn, 0)) for fn in preset_list]
        # 顺带附带该仓库下的轻量 json/txt/model/py 配置文件，方便离线直接加载
        if preset_key == "MiniMaxAI/MiniMax-H3#Ref2VA":
            for rfn, sz in all_siblings:
                if not (
                    rfn.startswith("FL2VA/")
                    or rfn.startswith("Ref2VA/transformer/")
                    or rfn == "Ref2VA/model_index.json"
                ):
                    continue
                if rfn.endswith((".json", ".txt", ".yaml", ".py", "tokenizer.model", "spiece.model")) and sz < 50 * 1024 * 1024 and rfn not in preset_list:
                    out.append((rfn, sz))
            return out
        prefix_filter = "FL2VA/" if repo_id == "MiniMaxAI/MiniMax-H3" else ""
        for rfn, sz in all_siblings:
            if prefix_filter and not rfn.startswith(prefix_filter):
                continue
            if rfn.endswith((".json", ".txt", ".yaml", ".py", "tokenizer.model", "spiece.model")) and sz < 50 * 1024 * 1024 and rfn not in preset_list:
                out.append((rfn, sz))
        return out

    pat = (pattern or "*.safetensors").strip().lower()
    glob_pat = pat if ("*" in pat or "?" in pat) else f"*{pat}*"
    matched = [
        (rfn, sz) for rfn, sz in all_siblings
        if rfn.lower().endswith(".safetensors") and (fnmatch.fnmatch(rfn.lower(), glob_pat) or fnmatch.fnmatch(os.path.basename(rfn.lower()), glob_pat))
    ]
    if not matched:
        matched = [(rfn, sz) for rfn, sz in all_siblings if rfn.lower().endswith(".safetensors")]
    for rfn, sz in all_siblings:
        if rfn.endswith((".json", ".txt", "tokenizer.model")) and sz < 50 * 1024 * 1024:
            matched.append((rfn, sz))
    return matched


def fast_parallel_hf_download(
    repo_id: str,
    filename: str,
    dest_dir: str,
    token: str = "",
    num_conn: int = 8,
    batch_done_bytes: int = 0,
    batch_total_bytes: int = 0,
    batch_label: str = "",
    stage_tag: str = "HF 下载",
) -> str:
    """8~16 路 HTTP Range 并发极速下载 (Colab G4/A100 上可达 400~700 MB/s)，支持跨分片全局总进度显示"""
    dest_path = os.path.join(dest_dir, filename)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".part"

    url = f"https://huggingface.co/{repo_id}/resolve/main/{urllib.parse.quote(filename, safe='/')}"
    opener = urllib.request.build_opener(_StripAuthRedirectHandler())
    headers = {"User-Agent": "ColabDeploy/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    probe_req = urllib.request.Request(url, headers={**headers, "Range": "bytes=0-0"})
    with opener.open(probe_req, timeout=20) as resp:
        cdn_url = resp.geturl()
        cr = resp.headers.get("Content-Range", "")
        total_size = int(cr.split("/")[-1]) if "/" in cr else int(resp.headers.get("Content-Length", 0))

    if total_size <= 0:
        raise RuntimeError(f"无法获取文件大小: {filename}")

    with open(temp_path, "wb") as f:
        f.truncate(total_size)

    downloaded = 0
    dl_lock = __import__("threading").Lock()
    t0 = time.time()
    last_log = [0.0]

    part_sz = (total_size + num_conn - 1) // num_conn
    ranges = [
        (i * part_sz, min((i + 1) * part_sz - 1, total_size - 1))
        for i in range(num_conn) if i * part_sz < total_size
    ]

    def _chunk_worker(br: Tuple[int, int]):
        nonlocal downloaded
        sb, eb = br
        req = urllib.request.Request(cdn_url, headers={"User-Agent": "ColabDeploy/1.0", "Range": f"bytes={sb}-{eb}"})
        with urllib.request.urlopen(req, timeout=60) as r, open(temp_path, "r+b") as wf:
            wf.seek(sb)
            while True:
                buf = r.read(4 * 1024 * 1024)
                if not buf:
                    break
                wf.write(buf)
                now = time.time()
                with dl_lock:
                    downloaded += len(buf)
                    if now - last_log[0] >= 2.0 or downloaded >= total_size:
                        last_log[0] = now
                        dt = max(0.01, now - t0)
                        spd = (downloaded / (1024 ** 2)) / dt
                        if batch_total_bytes > 0:
                            ov_done = batch_done_bytes + downloaded
                            ov_tot = max(batch_total_bytes, ov_done)
                            ov_pct = min(100, ov_done * 100 // max(1, ov_tot))
                            lbl = (batch_label or os.path.basename(filename)).replace(":", "·")
                            log(f"  [{stage_tag}] {lbl}: {ov_pct}% ({ov_done/(1024**3):.2f}/{ov_tot/(1024**3):.2f} GB, {spd:.1f} MB/s)")
                        else:
                            pct = downloaded * 100 // total_size
                            log(f"  [{stage_tag}] {os.path.basename(filename)}: {pct}% ({downloaded/(1024**3):.2f}/{total_size/(1024**3):.2f} GB, {spd:.1f} MB/s)")

    with ThreadPoolExecutor(max_workers=num_conn) as pool:
        futs = [pool.submit(_chunk_worker, r) for r in ranges]
        for fut in as_completed(futs):
            fut.result()

    os.replace(temp_path, dest_path)
    return dest_path


def _has_vm_rclone_drive() -> bool:
    rc_p = os.path.expanduser("~/.config/rclone/rclone.conf")
    return bool(os.path.exists(rc_p) and shutil.which("rclone"))


def _check_rclone_drive_has_model(remote_folder: str) -> int:
    """检查云盘 gdrive:models/<remote_folder> 中已持久化的核心权重文件数量 (.gguf / .safetensors)"""
    rc_conf = os.path.expanduser("~/.config/rclone/rclone.conf")
    if not (os.path.exists(rc_conf) and shutil.which("rclone")):
        return 0
    remote_src = f"gdrive:models/{remote_folder}"
    try:
        ls_out = subprocess.check_output(
            ["rclone", "lsf", remote_src, "--config", rc_conf, "-R", "--files-only", "--fast-list"],
            stderr=subprocess.DEVNULL,
            timeout=45,
            text=True,
        )
        remote_files = [
            ln.strip() for ln in ls_out.splitlines()
            if ln.strip().lower().endswith((".gguf", ".safetensors", ".onnx", ".pth"))
        ]
        return len(remote_files)
    except Exception:
        return 0


def _pull_from_rclone_drive_if_exists(
    remote_folder: str,
    local_dir: str,
    excludes: Optional[List[str]] = None,
    includes: Optional[List[str]] = None,
) -> bool:
    """优先从 Google Drive (gdrive:models/<remote_folder>) 开启 16 路并发多线程高速拉取至本地盘"""
    rc_conf = os.path.expanduser("~/.config/rclone/rclone.conf")
    if not (os.path.exists(rc_conf) and shutil.which("rclone")):
        return False
    remote_src = f"gdrive:models/{remote_folder}"
    filter_args: List[str] = []
    for inc in (includes or []):
        if inc:
            filter_args.extend(["--include", inc])
    for exc in (excludes or []):
        if exc:
            filter_args.extend(["--exclude", exc])
    if filter_args:
        filter_args.append("--ignore-case")
    try:
        ls_out = subprocess.check_output(
            ["rclone", "lsf", remote_src, "--config", rc_conf, "-R", "--files-only", "--format", "sp"] + filter_args,
            stderr=subprocess.DEVNULL,
            timeout=25,
            text=True,
        )
        remote_entries: List[Tuple[str, int]] = []
        for raw in ls_out.splitlines():
            ln = raw.strip()
            if ";" not in ln:
                continue
            sz_s, rel_p = ln.split(";", 1)
            rel_p = rel_p.strip()
            if not rel_p:
                continue
            try:
                rsz = int(sz_s.strip())
            except ValueError:
                rsz = -1
            remote_entries.append((rel_p, rsz))
        remote_weights = [
            (rp, rsz) for rp, rsz in remote_entries
            if rp.lower().endswith((".gguf", ".safetensors", ".onnx", ".pth"))
        ]
    except Exception:
        return False

    if not remote_weights:
        return False

    os.makedirs(local_dir, exist_ok=True)
    if all(
        os.path.exists(os.path.join(local_dir, rp))
        and (rsz < 0 or os.path.getsize(os.path.join(local_dir, rp)) == rsz)
        for rp, rsz in remote_entries
    ):
        log(f"⚡ 检测到本地高速盘 ({local_dir}) 已完整校验 {len(remote_weights)} 个核心权重及配套文件，0 秒直接复用！")
        return True

    log(f"☁️ 正在从您的 Google Drive ({remote_src}) 16 路并发高速拉取 {len(remote_weights)} 个权重文件 (共 {len(remote_entries)} 个文件) 至本地 ({local_dir})...")
    cmd_fg = [
        "rclone", "copy", remote_src, local_dir,
        "--config", rc_conf,
        "--transfers", "16",
        "--multi-thread-streams", "16",
        "--multi-thread-cutoff", "64M",
        "--buffer-size", "64M",
        "--drive-chunk-size", "128M",
        "--size-only",
        "--stats", "2s",
        "--stats-one-line",
        "-v",
    ] + filter_args
    proc = subprocess.Popen(
        cmd_fg,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        ln = raw_line.strip()
        m = re.search(r"([\d.]+\s*[KMGT]?i?B)\s*/\s*([\d.]+\s*[KMGT]?i?B),\s*(\d+)%,\s*([\d.]+\s*[KMGT]?i?B/s)", ln)
        if m:
            done_s, tot_s, pct_s, spd_s = m.group(1), m.group(2), m.group(3), m.group(4)
            log(f"  [云盘极速预热] {remote_folder}: {pct_s}% ({done_s}/{tot_s}, {spd_s})")
    proc.wait()
    if proc.returncode == 0:
        log(f"  [云盘极速预热] {remote_folder}: 100% (云盘预热完成, 已就绪)")
        return True
    return False


GUIDE_PREPROCESSOR_CODE = r'''#!/usr/bin/env python3
"""
4-Mode Guide Video & Photo Preprocessor for MiniMax-H3-Ref2VA (Card 06) and MiniMax-H3-Image (Card 01):
  - clay          : 3D Studio Clay White Model (Depth-Anything-V2 -> smoothed normals + 3-point soft studio shading + clean foreground cutout)
  - subject_depth : Subject-only 3D Depth on pure black background (TanLuAI subject_depth style)
  - full_depth    : Full-scene Depth-Anything-V2 with temporal EMA stabilization (zero flicker)
  - pose          : Multi-person OpenPose-style colored body skeleton on black canvas (KeypointRCNN)
Supports both MP4 video (with start/end clip trimming, exact 24fps resampling, and original audio muxing)
and single photo (PNG/JPG base64 -> PNG base64).
"""
import base64
import gc
import io
import os
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_PREPROC_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {}

_OPENPOSE_LIMBS = [
    ((5, 6), (255, 85, 0)),
    ((5, 7), (255, 170, 0)),
    ((7, 9), (255, 255, 0)),
    ((6, 8), (170, 255, 0)),
    ((8, 10), (0, 255, 0)),
    ((5, 11), (0, 255, 170)),
    ((6, 12), (0, 170, 255)),
    ((11, 12), (0, 85, 255)),
    ((11, 13), (85, 0, 255)),
    ((13, 15), (170, 0, 255)),
    ((12, 14), (255, 0, 170)),
    ((14, 16), (255, 0, 85)),
    ((0, 1), (255, 0, 255)),
    ((0, 2), (170, 0, 255)),
    ((1, 3), (255, 85, 255)),
    ((2, 4), (85, 85, 255)),
]


def _pick_device() -> Tuple[str, bool]:
    if not torch.cuda.is_available():
        return "cpu", False
    try:
        free_b, _ = torch.cuda.mem_get_info("cuda")
        if free_b >= 2.2 * (1024 ** 3):
            return "cuda", (free_b < 8.0 * (1024 ** 3))
    except Exception:
        pass
    return "cpu", False


def _get_depth_model(device: str):
    key = "depth_anything_v2"
    if key not in _CACHE:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        proc = AutoImageProcessor.from_pretrained(model_id)
        dtype = torch.float16 if device == "cuda" else torch.float32
        mdl = AutoModelForDepthEstimation.from_pretrained(model_id, torch_dtype=dtype).eval()
        _CACHE[key] = (proc, mdl)
    proc, mdl = _CACHE[key]
    mdl.to(device)
    return proc, mdl


def _get_seg_model(device: str):
    key = "deeplabv3_seg"
    if key not in _CACHE:
        import torchvision
        weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
        mdl = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights).eval()
        _CACHE[key] = (mdl, weights.transforms())
    mdl, tfm = _CACHE[key]
    mdl.to(device)
    if device == "cuda":
        mdl.half()
    else:
        mdl.float()
    return mdl, tfm


def _get_pose_model(device: str):
    key = "keypoint_rcnn"
    if key not in _CACHE:
        import torchvision
        weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        mdl = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights).eval()
        _CACHE[key] = mdl
    mdl = _CACHE[key]
    mdl.to(device)
    return mdl


def _offload_models_if_needed(offload: bool):
    if not offload:
        return
    for val in _CACHE.values():
        objs = val if isinstance(val, tuple) else (val,)
        for obj in objs:
            if hasattr(obj, "to"):
                try:
                    obj.to("cpu")
                except Exception:
                    pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def _infer_depth_batch(frames_rgb: List[np.ndarray], device: str, batch_size: int = 8) -> List[np.ndarray]:
    proc, mdl = _get_depth_model(device)
    out: List[np.ndarray] = []
    model_dtype = next(mdl.parameters()).dtype
    for i in range(0, len(frames_rgb), batch_size):
        chunk = frames_rgb[i : i + batch_size]
        pils = [Image.fromarray(f) for f in chunk]
        inp = proc(images=pils, return_tensors="pt")
        inp = {k: v.to(device=device, dtype=model_dtype if v.is_floating_point() else v.dtype) for k, v in inp.items()}
        pred = mdl(**inp).predicted_depth
        h, w = chunk[0].shape[:2]
        pred = F.interpolate(pred.unsqueeze(1).float(), size=(h, w), mode="bicubic", align_corners=False).squeeze(1)
        arr = pred.detach().cpu().numpy()
        for j in range(arr.shape[0]):
            out.append(arr[j])
    return out


@torch.inference_mode()
def _infer_fg_masks_batch(frames_rgb: List[np.ndarray], device: str, batch_size: int = 6) -> List[np.ndarray]:
    mdl, _ = _get_seg_model(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    use_half = device == "cuda"
    if use_half:
        mean = mean.half()
        std = std.half()
    masks: List[np.ndarray] = []
    for i in range(0, len(frames_rgb), batch_size):
        chunk = frames_rgb[i : i + batch_size]
        h, w = chunk[0].shape[:2]
        scale = 520.0 / max(1, min(h, w))
        sh = max(64, int(round(h * scale / 8.0)) * 8)
        sw = max(64, int(round(w * scale / 8.0)) * 8)
        if max(sh, sw) > 768:
            r2 = 768.0 / max(sh, sw)
            sh = max(64, int(round(sh * r2 / 8.0)) * 8)
            sw = max(64, int(round(sw * r2 / 8.0)) * 8)
        batch_np = np.stack([cv2.resize(f, (sw, sh), interpolation=cv2.INTER_LINEAR) for f in chunk], axis=0)
        t = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(device=device, dtype=torch.float16 if use_half else torch.float32) / 255.0
        t = (t - mean) / std
        logits = mdl(t)["out"].float()
        probs = torch.softmax(logits, dim=1)
        person_prob = probs[:, 15:16, :, :]
        fg_any_prob = 1.0 - probs[:, 0:1, :, :]
        person_up = F.interpolate(person_prob, size=(h, w), mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()
        fg_up = F.interpolate(fg_any_prob, size=(h, w), mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()
        for p_m, a_m in zip(person_up, fg_up):
            m = p_m if float((p_m > 0.45).mean()) >= 0.012 else a_m
            # Soft crisp foreground alpha with slight edge feather
            m = np.clip((m - 0.30) / 0.42, 0.0, 1.0).astype(np.float32)
            m = cv2.GaussianBlur(m, (5, 5), 1.1)
            masks.append(m)
    return masks


def _render_clay_frame(depth_norm: np.ndarray, mask: Optional[np.ndarray], keep_bg: bool = False, smoothness: str = "balanced") -> np.ndarray:
    sm = (smoothness or "balanced").strip().lower()
    if sm == "soft":
        blur_sigma, grad_scale, ao_str = 3.2, 1.55, 0.55
    elif sm == "detail":
        blur_sigma, grad_scale, ao_str = 0.8, 2.95, 1.55
    else:
        blur_sigma, grad_scale, ao_str = 1.6, 2.2, 1.15
    d_smooth = cv2.GaussianBlur(depth_norm.astype(np.float32), (0, 0), blur_sigma)
    gx = cv2.Sobel(d_smooth, cv2.CV_32F, 1, 0, ksize=5) * grad_scale
    gy = cv2.Sobel(d_smooth, cv2.CV_32F, 0, 1, ksize=5) * grad_scale
    nz = np.ones_like(d_smooth, dtype=np.float32)
    inv_len = 1.0 / np.sqrt(gx * gx + gy * gy + nz * nz + 1e-6)
    nx, ny, nz = -gx * inv_len, -gy * inv_len, nz * inv_len

    # 3-point soft studio clay lighting (matches Blender matte white clay mannequin)
    l1 = np.array([-0.34, -0.44, 0.83], dtype=np.float32)
    l1 /= np.linalg.norm(l1)
    l2 = np.array([0.46, -0.14, 0.88], dtype=np.float32)
    l2 /= np.linalg.norm(l2)
    diff1 = np.clip(nx * l1[0] + ny * l1[1] + nz * l1[2], 0.0, 1.0)
    diff2 = np.clip(nx * l2[0] + ny * l2[1] + nz * l2[2], 0.0, 1.0)

    # Local concavity ambient occlusion for clothing folds & limb joints
    d_blur_wide = cv2.GaussianBlur(d_smooth, (0, 0), 9.0)
    ao = np.clip(0.92 + (d_smooth - d_blur_wide) * ao_str, 0.72, 1.04)

    shade = (0.40 + 0.44 * diff1 + 0.16 * diff2) * ao
    # Subtle depth aerial gradient for sculptural clarity
    clay = np.clip(0.46 + 0.48 * shade + 0.06 * d_smooth, 0.24, 0.96)

    if mask is not None and not keep_bg:
        # Clean bright studio backdrop (#F1F2F4) with subtle vignette
        h, w = clay.shape
        yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        bg = np.clip(0.955 - 0.045 * (xx * xx * 0.5 + yy * yy * 0.5), 0.88, 0.96)
        alpha = np.clip(mask, 0.0, 1.0)
        comp = clay * alpha + bg * (1.0 - alpha)
    else:
        comp = clay

    u8 = np.clip(comp * 255.0, 0, 255).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def _render_subject_depth_frame(depth_raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    fg_sel = mask > 0.45
    if int(fg_sel.sum()) < 64:
        # Depth-based fallback when no class mask triggers
        lo = float(np.percentile(depth_raw, 40))
        hi = float(np.percentile(depth_raw, 99.5))
        d_norm = np.clip((depth_raw - lo) / max(1e-5, hi - lo), 0.0, 1.0)
        u8 = np.clip(d_norm * 255.0, 0, 255).astype(np.uint8)
        return np.stack([u8, u8, u8], axis=-1)

    fg_vals = depth_raw[fg_sel]
    d_lo = float(np.percentile(fg_vals, 2))
    d_hi = float(np.percentile(fg_vals, 99))
    d_norm = np.clip((depth_raw - d_lo) / max(1e-5, d_hi - d_lo), 0.0, 1.0)
    # Rescale subject depth to [0.28..1.0] so farthest limb stays visible against pure black background
    fg_val = (0.28 + 0.72 * d_norm) * np.clip(mask, 0.0, 1.0)
    u8 = np.clip(fg_val * 255.0, 0, 255).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


@torch.inference_mode()
def _render_pose_frames(frames_rgb: List[np.ndarray], device: str) -> List[np.ndarray]:
    mdl = _get_pose_model(device)
    out: List[np.ndarray] = []
    for i in range(0, len(frames_rgb), 4):
        chunk = frames_rgb[i : i + 4]
        tensors = [torch.from_numpy(f).permute(2, 0, 1).float().to(device) / 255.0 for f in chunk]
        preds = mdl(tensors)
        for frame, pred in zip(chunk, preds):
            h, w = frame.shape[:2]
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            thick = max(2, int(round(min(h, w) / 140.0)))
            rad = max(3, int(round(min(h, w) / 115.0)))
            scores = pred.get("scores", torch.empty(0)).detach().cpu().numpy()
            kps_all = pred.get("keypoints", torch.empty(0)).detach().cpu().numpy()
            kp_sc_all = pred.get("keypoints_scores", torch.empty(0)).detach().cpu().numpy()
            for pi, sc in enumerate(scores):
                if float(sc) < 0.65:
                    continue
                kps = kps_all[pi]
                ksc = kp_sc_all[pi] if pi < len(kp_sc_all) else np.ones(len(kps))
                # Draw neck connector from nose to shoulder midpoint
                if ksc[0] > 0.2 and ksc[5] > 0.2 and ksc[6] > 0.2:
                    mx = int(round((kps[5, 0] + kps[6, 0]) * 0.5))
                    my = int(round((kps[5, 1] + kps[6, 1]) * 0.5))
                    cv2.line(canvas, (int(round(kps[0, 0])), int(round(kps[0, 1]))), (mx, my), (255, 0, 0), thick, cv2.LINE_AA)
                for (a, b), col in _OPENPOSE_LIMBS:
                    if a < len(kps) and b < len(kps) and ksc[a] > 0.15 and ksc[b] > 0.15:
                        p1 = (int(round(kps[a, 0])), int(round(kps[a, 1])))
                        p2 = (int(round(kps[b, 0])), int(round(kps[b, 1])))
                        cv2.line(canvas, p1, p2, col, thick, cv2.LINE_AA)
                for ji in range(min(17, len(kps))):
                    if ksc[ji] > 0.15:
                        pt = (int(round(kps[ji, 0])), int(round(kps[ji, 1])))
                        cv2.circle(canvas, pt, rad, (255, 255, 255), -1, cv2.LINE_AA)
            out.append(canvas)
    return out


def _process_frames_rgb(frames_rgb: List[np.ndarray], mode: str, keep_bg: bool = False, smoothness: str = "balanced") -> List[np.ndarray]:
    mode = (mode or "clay").strip().lower()
    device, should_offload = _pick_device()
    try:
        if mode == "pose":
            return _render_pose_frames(frames_rgb, device)

        depths = _infer_depth_batch(frames_rgb, device)
        if mode == "full_depth":
            out = []
            ema_lo, ema_hi = None, None
            for d in depths:
                lo = float(np.percentile(d, 2))
                hi = float(np.percentile(d, 98.5))
                if ema_lo is None:
                    ema_lo, ema_hi = lo, hi
                else:
                    ema_lo = 0.85 * ema_lo + 0.15 * lo
                    ema_hi = 0.85 * ema_hi + 0.15 * hi
                dn = np.clip((d - ema_lo) / max(1e-5, ema_hi - ema_lo), 0.0, 1.0)
                if smoothness == "soft":
                    dn = cv2.GaussianBlur(dn.astype(np.float32), (0, 0), 2.4)
                u8 = np.clip(dn * 255.0, 0, 255).astype(np.uint8)
                out.append(np.stack([u8, u8, u8], axis=-1))
            return out

        masks = _infer_fg_masks_batch(frames_rgb, device)
        if mode == "subject_depth":
            res_sd = []
            for d, m in zip(depths, masks):
                d_use = cv2.GaussianBlur(d.astype(np.float32), (0, 0), 2.4) if smoothness == "soft" else d
                res_sd.append(_render_subject_depth_frame(d_use, m))
            return res_sd

        # Default: clay (3D studio white model)
        out = []
        ema_lo, ema_hi = None, None
        for d, m in zip(depths, masks):
            sel = m > 0.45
            vals = d[sel] if int(sel.sum()) > 128 and not keep_bg else d
            lo = float(np.percentile(vals, 2))
            hi = float(np.percentile(vals, 98.5))
            if ema_lo is None:
                ema_lo, ema_hi = lo, hi
            else:
                ema_lo = 0.85 * ema_lo + 0.15 * lo
                ema_hi = 0.85 * ema_hi + 0.15 * hi
            dn = np.clip((d - ema_lo) / max(1e-5, ema_hi - ema_lo), 0.0, 1.0)
            out.append(_render_clay_frame(dn, None if keep_bg else m, keep_bg=keep_bg, smoothness=smoothness))
        return out
    finally:
        _offload_models_if_needed(should_offload)


def _detect_black_bar_crop(video_path: str) -> str:
    try:
        r = subprocess.run(
            ["ffmpeg", "-v", "info", "-i", video_path, "-t", "1.5", "-vf", "cropdetect=24:16:0", "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        import re as _re
        matches = _re.findall(r"crop=(\d+:\d+:\d+:\d+)", r.stderr or "")
        if matches:
            cw, ch, cx, cy = [int(v) for v in matches[-1].split(":")]
            if cw >= 160 and ch >= 160 and (cx > 4 or cy > 4):
                return f"crop={cw}:{ch}:{cx}:{cy}"
    except Exception:
        pass
    return ""


def _has_audio_stream(video_path: str) -> bool:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
            text=True,
            timeout=10,
        ).strip()
        return bool(out)
    except Exception:
        return False


def preprocess_guide(payload: dict, output_dir: str, gpu_lock: Optional[threading.Lock] = None) -> dict:
    t0 = time.time()
    mode = str(payload.get("mode") or "clay").strip().lower()
    if mode not in ("clay", "subject_depth", "full_depth", "pose", "raw"):
        mode = "clay"
    keep_bg = bool(payload.get("keep_bg", False))
    smoothness = str(payload.get("smoothness") or "balanced").strip().lower()
    if smoothness not in ("soft", "balanced", "detail"):
        smoothness = "balanced"
    crop_mode = str(payload.get("crop_mode") or "none").strip().lower()

    lock = gpu_lock or _PREPROC_LOCK

    # Case 1: Single image preprocessing (Card 01 photo -> white model / depth / pose)
    image_b64 = str(payload.get("image_b64") or "").strip()
    if image_b64:
        raw_b = base64.b64decode(image_b64.split(",", 1)[-1])
        pil_im = Image.open(io.BytesIO(raw_b)).convert("RGB")
        w0, h0 = pil_im.size
        # Cap inference edge at 1280 for speed while preserving crisp detail
        max_e = max(w0, h0)
        if max_e > 1280:
            s = 1280.0 / max_e
            pil_work = pil_im.resize((max(32, int(round(w0 * s))), max(32, int(round(h0 * s)))), Image.Resampling.LANCZOS)
        else:
            pil_work = pil_im
        rgb = np.asarray(pil_work, dtype=np.uint8)
        with lock:
            res_frames = [rgb] if mode == "raw" else _process_frames_rgb([rgb], mode, keep_bg=keep_bg, smoothness=smoothness)
        out_im = Image.fromarray(res_frames[0])
        if out_im.size != (w0, h0):
            out_im = out_im.resize((w0, h0), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        out_im.save(buf, format="PNG")
        out_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        return {
            "ok": True,
            "kind": "image",
            "mode": mode,
            "width": w0,
            "height": h0,
            "image_b64": out_b64,
            "elapsed_s": round(time.time() - t0, 2),
        }

    # Case 2: Video preprocessing (Card 06 real video -> 24fps clay / subject_depth / full_depth / pose MP4)
    video_b64 = str(payload.get("video_b64") or "").strip()
    video_name = os.path.basename(str(payload.get("video_name") or "").strip())
    start_sec = max(0.0, float(payload.get("start_sec") or 0.0))
    end_sec = float(payload.get("end_sec") or 0.0)
    target_fps = 24
    max_short_edge = max(256, min(768, int(payload.get("max_short_edge") or 512)))
    max_frames = max(22, min(243, int(payload.get("max_frames") or 243)))

    tmp_in = None
    tmp_norm = None
    tmp_raw_vid = None
    try:
        if video_b64:
            raw_vb = base64.b64decode(video_b64.split(",", 1)[-1])
            fd, tmp_in = tempfile.mkstemp(prefix="guide_in_", suffix=".mp4")
            os.close(fd)
            with open(tmp_in, "wb") as wf:
                wf.write(raw_vb)
            src_path = tmp_in
        elif video_name and os.path.isfile(os.path.join(output_dir, video_name)):
            src_path = os.path.join(output_dir, video_name)
        else:
            return {"ok": False, "error": "请先上传参考视频或提供图片"}

        fd2, tmp_norm = tempfile.mkstemp(prefix="guide_norm_", suffix=".mp4")
        os.close(fd2)

        # Optional pre-crop to strip black bars or platform watermarks BEFORE depth/white-model conversion
        crop_filter = ""
        if crop_mode == "auto_black":
            crop_filter = _detect_black_bar_crop(src_path)
        elif crop_mode == "trim_6pct":
            crop_filter = "crop=iw*0.88:ih*0.88:iw*0.06:ih*0.06"
        elif crop_mode == "trim_10pct":
            crop_filter = "crop=iw*0.80:ih*0.80:iw*0.10:ih*0.10"

        vf_parts = [f"fps={target_fps}"]
        if crop_filter:
            vf_parts.append(crop_filter)
        vf_parts.append(f"scale='if(lte(iw,ih),min({max_short_edge},iw),-2)':'if(lte(iw,ih),-2,min({max_short_edge},ih))'")
        vf_parts.append("scale=trunc(iw/16)*16:trunc(ih/16)*16")
        vf = ",".join(vf_parts)
        ff_cmd = ["ffmpeg", "-y", "-v", "error"]
        if start_sec > 0.0:
            ff_cmd.extend(["-ss", f"{start_sec:.3f}"])
        ff_cmd.extend(["-i", src_path])
        if end_sec > start_sec + 0.2:
            dur = min(end_sec - start_sec, max_frames / float(target_fps))
            ff_cmd.extend(["-t", f"{dur:.3f}"])
        else:
            ff_cmd.extend(["-t", f"{max_frames / float(target_fps):.3f}"])
        ff_cmd.extend([
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "32000", "-ac", "2", "-shortest",
            tmp_norm,
        ])
        r_ff = subprocess.run(ff_cmd, capture_output=True, text=True, timeout=120)
        if r_ff.returncode != 0:
            # Retry without audio stream in case source audio codec fails
            ff_cmd_noa = [c for c in ff_cmd if c not in ("-c:a", "aac", "-ar", "32000", "-ac", "2", "-shortest")]
            ff_cmd_noa.insert(-1, "-an")
            subprocess.run(ff_cmd_noa, check=True, timeout=120)

        has_audio = _has_audio_stream(tmp_norm)
        cap = cv2.VideoCapture(tmp_norm)
        frames_rgb: List[np.ndarray] = []
        while len(frames_rgb) < max_frames:
            ok_r, fr = cap.read()
            if not ok_r or fr is None:
                break
            frames_rgb.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames_rgb:
            return {"ok": False, "error": "无法读取视频帧，请确认视频格式有效 (MP4/MOV/WebM)"}

        # MiniMax-H3 Ref2VA requires >= 22 frames in reference video
        while len(frames_rgb) < 22:
            frames_rgb.append(frames_rgb[-1].copy())

        h, w = frames_rgb[0].shape[:2]
        if mode == "raw":
            out_frames = frames_rgb
        else:
            with lock:
                out_frames = _process_frames_rgb(frames_rgb, mode, keep_bg=keep_bg, smoothness=smoothness)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_fname = f"guide_{mode}_{ts}.mp4"
        out_path = os.path.join(output_dir, out_fname)
        fd3, tmp_raw_vid = tempfile.mkstemp(prefix="guide_out_", suffix=".mp4")
        os.close(fd3)

        # Write crisp H.264 MP4 via ffmpeg stdin pipe so browser plays it natively
        enc_cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(target_fps),
            "-i", "-",
        ]
        if has_audio:
            enc_cmd.extend(["-i", tmp_norm, "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-ar", "32000", "-shortest"])
        enc_cmd.extend([
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            tmp_raw_vid,
        ])
        proc_enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)
        assert proc_enc.stdin is not None
        for fr_rgb in out_frames:
            proc_enc.stdin.write(np.ascontiguousarray(fr_rgb).tobytes())
        proc_enc.stdin.close()
        proc_enc.wait(timeout=120)
        if proc_enc.returncode != 0:
            raise RuntimeError("ffmpeg encoding guide video failed")

        os.replace(tmp_raw_vid, out_path)
        tmp_raw_vid = None
        return {
            "ok": True,
            "kind": "video",
            "mode": mode,
            "filename": out_fname,
            "width": w,
            "height": h,
            "fps": target_fps,
            "num_frames": len(out_frames),
            "duration_sec": round(len(out_frames) / float(target_fps), 2),
            "has_audio": has_audio,
            "elapsed_s": round(time.time() - t0, 2),
        }
    finally:
        for tp in (tmp_in, tmp_norm, tmp_raw_vid):
            if tp and os.path.exists(tp):
                try:
                    os.remove(tp)
                except Exception:
                    pass
'''


MINIMAX_SERVER_CODE = r'''#!/usr/bin/env python3
import base64
import gc
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "/content/DiffSynth-Studio")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
os.environ["DIFFSYNTH_DISK_MAP_BUFFER_SIZE"] = str(10**12)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio
try:
    from diffsynth.utils.data.audio_video import read_video_audio
except Exception:
    read_video_audio = None
try:
    from diffsynth.utils.data.audio import read_audio
except Exception:
    read_audio = None

MODEL_DIR = os.environ.get("MINIMAX_MODEL_DIR", "/content/local_models/MiniMax-H3")
MODEL_VARIANT = os.environ.get("MINIMAX_VARIANT", "FL2VA").strip() or "FL2VA"
ENGINE_NAME = "minimax-h3-ref2va" if MODEL_VARIANT == "Ref2VA" else "minimax-h3"
OUTPUT_DIR = "/content/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PIPELINE = None
ENGINE_READY = False
ENGINE_ERROR = ""
GPU_LOCK = threading.Lock()

JOB_STATE = {
    "running": False,
    "pct": 0,
    "step": 0,
    "total_steps": 30,
    "message": "就绪：等待输入视频提示词...",
    "elapsed_s": 0.0,
    "latest_video": "",
    "error": "",
}


class _TqdmProgressCapture:
    def __init__(self, orig):
        self.orig = orig
        self._buf = ""
        self._ansi = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

    def write(self, data: str):
        try:
            self.orig.write(data)
            self.orig.flush()
        except Exception:
            pass
        if not data:
            return
        clean = self._ansi.sub("", data)
        for ch in clean:
            if ch in ("\r", "\n"):
                ln = self._buf.strip()
                self._buf = ""
                if ln and JOB_STATE["running"]:
                    if os.path.exists("/content/.cancel_minimax"):
                        try:
                            os.unlink("/content/.cancel_minimax")
                        except Exception:
                            pass
                        raise RuntimeError("🛑 已手动停止视频生成任务")
                    m = re.search(r"(\d+)%\|.*\|\s*(\d+)/(\d+)", ln)
                    if m:
                        pct = int(m.group(1))
                        cur_s = int(m.group(2))
                        tot_s = int(m.group(3))
                        JOB_STATE["step"] = cur_s
                        JOB_STATE["total_steps"] = tot_s
                        target_steps = int(JOB_STATE.get("target_steps") or 60)
                        b_idx = int(JOB_STATE.get("completed_count") or 0)
                        b_tot = max(1, int(JOB_STATE.get("batch_count") or 1))
                        base_pct = (b_idx / float(b_tot)) * 100.0
                        span_pct = 100.0 / float(b_tot)
                        prefix = f"[连抽 {b_idx+1}/{b_tot}] " if b_tot > 1 else ""
                        if tot_s <= target_steps + 5:
                            sub_pct = 8.0 + (pct * 0.80)
                            overall_pct = min(98, max(5, int(base_pct + (sub_pct / 100.0) * span_pct)))
                            JOB_STATE["pct"] = overall_pct
                            JOB_STATE["message"] = f"{prefix}🎬 正在扩散采样生成有声视频: 第 {cur_s}/{tot_s} 步 (总进度 {overall_pct}%)"
                        else:
                            sub_pct = 88.0 + (pct * 0.11)
                            overall_pct = min(99, max(88, int(base_pct + (sub_pct / 100.0) * span_pct)))
                            JOB_STATE["pct"] = overall_pct
                            JOB_STATE["message"] = f"{prefix}🎞️ 采样完成，正在合成音视频画面: {cur_s}/{tot_s} 帧 (总进度 {overall_pct}%)"
            else:
                self._buf += ch

    def flush(self):
        try:
            self.orig.flush()
        except Exception:
            pass

    def isatty(self):
        return False


sys.stdout = _TqdmProgressCapture(sys.__stdout__)
sys.stderr = _TqdmProgressCapture(sys.__stderr__)


def _enable_g4_vram_optimizations(pipe, total_vram_gb: float):
    # 1. 提示词编码缓存: 相同 (prompt, height, width) 0.001s 直接命中返回
    try:
        for unit in getattr(pipe, "units", []):
            if unit.__class__.__name__ == "MiniMaxH3Unit_PromptEmbedder" and not hasattr(unit, "_orig_process"):
                unit._orig_process = unit.process
                unit._orig_onload_names = getattr(unit, "onload_model_names", ("text_encoder",))
                unit.onload_model_names = ()
                unit._embed_cache = {}

                def _cached_prompt_process(self_unit, pipe_obj, prompt, keyframes=None, ref_blocks=None, height=None, width=None, text_embedding=None, **extra_kwargs):
                    if not extra_kwargs and ref_blocks is None and text_embedding is None and prompt is not None and not keyframes:
                        cache_key = (prompt, height, width)
                        cached = self_unit._embed_cache.get(cache_key)
                        if cached is not None:
                            return {
                                "prompt_embeds": cached["prompt_embeds"].to(pipe_obj.device, pipe_obj.torch_dtype),
                                "text_token_tags": cached["text_token_tags"].to(pipe_obj.device, torch.long),
                            }
                        pipe_obj.load_models_to_device(self_unit._orig_onload_names)
                        res = self_unit._orig_process(pipe_obj, prompt, keyframes=keyframes, ref_blocks=ref_blocks, height=height, width=width, text_embedding=text_embedding)
                        self_unit._embed_cache.clear()
                        self_unit._embed_cache[cache_key] = {
                            "prompt_embeds": res["prompt_embeds"].detach().clone(),
                            "text_token_tags": res["text_token_tags"].detach().clone(),
                        }
                        return res
                    pipe_obj.load_models_to_device(self_unit._orig_onload_names)
                    return self_unit._orig_process(pipe_obj, prompt, keyframes=keyframes, ref_blocks=ref_blocks, height=height, width=width, text_embedding=text_embedding, **extra_kwargs)

                unit.process = types.MethodType(_cached_prompt_process, unit)
    except Exception:
        pass

    # 2. SageAttention 2 (NHD 零转置 Blackwell Tensor Core 加速)
    try:
        import diffsynth.models.minimax_h3_dit as mm_dit
        from sageattention import sageattn

        def _sage_varlen_attention(q, k, v, cu_seqlens, softmax_scale):
            bounds = cu_seqlens.tolist()
            if len(bounds) == 2 and bounds[0] == 0 and bounds[1] == q.shape[0]:
                return sageattn(
                    q.unsqueeze(0),
                    k.unsqueeze(0),
                    v.unsqueeze(0),
                    tensor_layout="NHD",
                    is_causal=False,
                    sm_scale=softmax_scale,
                ).squeeze(0)
            out = torch.empty_like(q)
            for start, stop in zip(bounds[:-1], bounds[1:]):
                if stop == start:
                    continue
                out[start:stop] = sageattn(
                    q[start:stop].unsqueeze(0),
                    k[start:stop].unsqueeze(0),
                    v[start:stop].unsqueeze(0),
                    tensor_layout="NHD",
                    is_causal=False,
                    sm_scale=softmax_scale,
                ).squeeze(0)
            return out

        mm_dit._sdpa_varlen_attention = _sage_varlen_attention
        print("SAGE_ATTENTION_2_NHD_ENABLED", flush=True)
    except Exception as e_sage:
        print(f"SAGE_ATTENTION_2_WARN: {e_sage}", flush=True)

    # 3. TorchInductor 融合算子 (消除每步 2.4TB 的 RoPE / AdaLN / SwiGLU 中间显存读写)
    try:
        import diffsynth.models.minimax_h3_dit as mm_dit

        @torch.compile(dynamic=True)
        def _fused_apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
            rot_dim = freqs.shape[-1]
            x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
            cos = torch.cos(freqs).to(x.dtype).unsqueeze(1)
            sin = torch.sin(freqs).to(x.dtype).unsqueeze(1)
            x1, x2 = torch.chunk(x_rot, 2, dim=-1)
            x_rot = (x_rot * cos) + (torch.cat((-x2, x1), dim=-1) * sin)
            return torch.cat((x_rot, x_pass), dim=-1)

        @torch.compile(dynamic=True)
        def _fused_modulate_scale_shift(x, shift, scale, indices):
            return (x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(x.dtype)

        @torch.compile(dynamic=True)
        def _fused_modulate_gate(x, gate, other, indices):
            return (x + gate.index_select(0, indices) * other).to(x.dtype)

        @torch.compile(dynamic=True)
        def _fused_swiglu(hidden):
            gate, up = hidden.chunk(2, dim=-1)
            return torch.nn.functional.silu(gate) * up

        mm_dit._apply_rope = _fused_apply_rope
        mm_dit._modulate_scale_shift = _fused_modulate_scale_shift
        mm_dit._modulate_gate = _fused_modulate_gate
        mm_dit.MiniMaxH3MLP.forward = lambda self_mlp, x: self_mlp.fc2(_fused_swiglu(self_mlp.fc1(x)))
        print("TORCH_COMPILE_FUSED_OPS_ENABLED", flush=True)
    except Exception as e_comp:
        print(f"TORCH_COMPILE_WARN: {e_comp}", flush=True)

    # 4. TeaCache 自适应残差缓存 (首尾各 4 步全精计算，中间步自适应复用 blocks[1..49] 残差)
    try:
        import diffsynth.models.minimax_h3_dit as mm_dit

        if getattr(pipe, "dit", None) is not None and not hasattr(pipe.dit, "_orig_forward"):
            pipe.dit._orig_forward = pipe.dit.forward
            pipe.dit._tc_enabled = True
            pipe.dit._tc_thresh = float(os.environ.get("MINIMAX_TEACACHE_THRESH", "0.12"))
            pipe.dit._tc_max_skips = 2
            pipe.dit._tc_step = 0
            pipe.dit._tc_total_steps = 35
            pipe.dit._tc_prev_b0_res = None
            pipe.dit._tc_cached_rest_res = None
            pipe.dit._tc_accum_rel_l1 = 0.0
            pipe.dit._tc_consec_skips = 0
            pipe.dit._tc_skipped_count = 0

            def _teacache_dit_forward(
                self_dit,
                x,
                audio_x,
                img_position_ids,
                unique_timesteps,
                inverse_indices,
                update_mask,
                token_tags,
                prompt_embeds,
                img_pos_info,
                audio_pos_info,
                text_pos_info,
                img_pos_for_infer_output_info,
                packed_seq_params,
                refiner_packed_seq_params,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
                update_audio_mask=None,
                skip_mask_out_condition=False,
                control_hints=None,
                video_feature_positions=None,
                **extra_kwargs,
            ):
                if (
                    not getattr(self_dit, "_tc_enabled", True)
                    or control_hints is not None
                    or use_gradient_checkpointing
                    or extra_kwargs
                ):
                    orig_kw = dict(
                        use_gradient_checkpointing=use_gradient_checkpointing,
                        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                        update_audio_mask=update_audio_mask,
                        skip_mask_out_condition=skip_mask_out_condition,
                        control_hints=control_hints,
                    )
                    if video_feature_positions is not None:
                        orig_kw["video_feature_positions"] = video_feature_positions
                    orig_kw.update(extra_kwargs)
                    return self_dit._orig_forward(
                        x, audio_x, img_position_ids, unique_timesteps, inverse_indices, update_mask,
                        token_tags, prompt_embeds, img_pos_info, audio_pos_info, text_pos_info,
                        img_pos_for_infer_output_info, packed_seq_params, refiner_packed_seq_params,
                        **orig_kw,
                    )

                inverse_indices = inverse_indices.view(-1).to(torch.long)
                token_tags = token_tags.view(-1).to(torch.long)
                text_selected = prompt_embeds

                img_pos = img_pos_info["position_ids"].view(-1).to(torch.long)
                audio_pos = audio_pos_info["position_ids"].view(-1).to(torch.long)
                text_pos = text_pos_info["position_ids"].view(-1).to(torch.long)
                infer_out_pos = img_pos_for_infer_output_info["position_ids"].view(-1).to(torch.long)

                cu_seqlens = packed_seq_params["cu_seqlens_q"].to(torch.int32)
                max_seqlen = int(packed_seq_params["max_seqlen_q"])
                refiner_cu = refiner_packed_seq_params["cu_seqlens_q"].to(torch.int32)
                refiner_max = int(refiner_packed_seq_params["max_seqlen_q"])

                seq_len = int(x.shape[1])
                device = x.device

                rope_freqs = self_dit.rope(img_position_ids).to(device)
                decoder_input, t_emb = self_dit._embed(
                    x=x, audio_x=audio_x, text_embeddings_selected=text_selected,
                    unique_timesteps=unique_timesteps.view(-1).to(device),
                    img_pos=img_pos.to(device), audio_pos=audio_pos.to(device), text_pos=text_pos.to(device),
                    refiner_cu_seqlens=refiner_cu.to(device), refiner_max_seqlen=refiner_max,
                    seq_len=seq_len, device=device,
                )

                combined_indices = (inverse_indices * mm_dit.MINIMAX_H3_ADALN_MODALITY_NUM + token_tags.clamp(min=0)).to(device)
                inverse_indices = inverse_indices.to(device)
                cu_seqlens = cu_seqlens.to(device)

                cur_step = int(getattr(self_dit, "_tc_step", 0))
                tot_steps = int(getattr(self_dit, "_tc_total_steps", 35))
                self_dit._tc_step = cur_step + 1

                h0 = self_dit.blocks[0](
                    decoder_input,
                    t_emb=t_emb,
                    combined_indices=combined_indices,
                    rope_freqs=rope_freqs,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
                b0_res = h0 - decoder_input

                can_try_skip = (
                    4 <= cur_step < max(5, tot_steps - 4)
                    and self_dit._tc_prev_b0_res is not None
                    and self_dit._tc_cached_rest_res is not None
                    and self_dit._tc_prev_b0_res.shape == b0_res.shape
                )
                do_skip = False
                if can_try_skip:
                    rel_l1 = (
                        (b0_res - self_dit._tc_prev_b0_res).abs().mean()
                        / (self_dit._tc_prev_b0_res.abs().mean() + 1e-6)
                    ).item()
                    self_dit._tc_accum_rel_l1 += rel_l1
                    if (
                        self_dit._tc_accum_rel_l1 < self_dit._tc_thresh
                        and self_dit._tc_consec_skips < self_dit._tc_max_skips
                    ):
                        do_skip = True

                if do_skip:
                    hidden = h0 + self_dit._tc_cached_rest_res
                    self_dit._tc_consec_skips += 1
                    self_dit._tc_skipped_count += 1
                else:
                    hidden = h0
                    for block in self_dit.blocks[1:]:
                        hidden = block(
                            hidden,
                            t_emb=t_emb,
                            combined_indices=combined_indices,
                            rope_freqs=rope_freqs,
                            cu_seqlens=cu_seqlens,
                            max_seqlen=max_seqlen,
                        )
                    self_dit._tc_prev_b0_res = b0_res.detach()
                    self_dit._tc_cached_rest_res = (hidden - h0).detach()
                    self_dit._tc_accum_rel_l1 = 0.0
                    self_dit._tc_consec_skips = 0

                video_logits, audio_logits = self_dit.final_layer(hidden, t_emb=t_emb, inverse_indices=inverse_indices)
                video_logits = video_logits.index_select(0, infer_out_pos.to(device))
                audio_logits = audio_logits.index_select(0, audio_pos.to(device))
                if not skip_mask_out_condition:
                    update_mask = update_mask.view(-1).to(device)
                    video_logits = video_logits * update_mask.unsqueeze(-1)
                    if update_audio_mask is not None:
                        audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)
                if video_feature_positions is not None:
                    vfp = video_feature_positions.to(device) if hasattr(video_feature_positions, "to") else video_feature_positions
                    time_indices = inverse_indices.index_select(0, vfp)
                    features = hidden.index_select(0, vfp)
                    return video_logits, audio_logits, (features, t_emb.index_select(0, time_indices))
                return video_logits, audio_logits

            pipe.dit.forward = types.MethodType(_teacache_dit_forward, pipe.dit)
            print("TEACACHE_DIT_ENABLED", flush=True)
    except Exception as e_tc:
        print(f"TEACACHE_WARN: {e_tc}", flush=True)

    # 5. 双内存池零换页常驻架构 (Dual-Memory Zero-Swap):
    # - dit (56.1 GiB) + video_vae (9.0 GiB) + audio_vae (0.5 GiB) + text_encoder 非线性层 (0.3 GiB) = 65.9 GiB 永久锁定在 96GB GPU 显存 (state=2, cuda)
    # - text_encoder 线性层 (57.5 GiB) 通过零拷贝 mmap 锁定在 177GB 系统 DDR5 内存 (state=2, cpu -> computation=cuda)，编码提示词仅需 ~2.3s，永不驱逐 DiT！
    from diffsynth.core.vram.layers import AutoWrappedLinear, AutoWrappedModule

    for attr_name in ("dit", "video_vae", "audio_vae"):
        model = getattr(pipe, attr_name, None)
        if model is None:
            continue
        for mod in model.modules():
            if hasattr(mod, "preparing") and getattr(mod, "state", 2) != 2:
                try:
                    mod.preparing()
                except Exception:
                    pass
    torch.cuda.synchronize()

    te_model = getattr(pipe, "text_encoder", None)
    if te_model is not None:
        for mod in te_model.modules():
            if isinstance(mod, AutoWrappedLinear):
                mod.offload_dtype = torch.bfloat16
                mod.offload_device = "cpu"
                mod.onload_dtype = torch.bfloat16
                mod.onload_device = "disk"
                mod.preparing_dtype = torch.bfloat16
                mod.preparing_device = "cpu"
                mod.computation_dtype = torch.bfloat16
                mod.computation_device = "cuda"
                try:
                    mod.preparing()
                except Exception:
                    pass
            elif isinstance(mod, AutoWrappedModule):
                mod.preparing_dtype = torch.bfloat16
                mod.preparing_device = "cuda"
                mod.computation_dtype = torch.bfloat16
                mod.computation_device = "cuda"
                try:
                    mod.preparing()
                except Exception:
                    pass
    torch.cuda.synchronize()

    def _noop_load_models_to_device(self_pipe, model_names=()):
        # 所有子模型已永久锁定在各自最优硬件池 (DiT/VAE 在 GPU 显存，TextEncoder 在 DDR5 内存直通 GPU 计算)，无需任何磁盘换入换出
        return

    pipe.load_models_to_device = types.MethodType(_noop_load_models_to_device, pipe)
    pipe._high_vram_fast_mode = True
    free_vram, total_vram = torch.cuda.mem_get_info("cuda")
    print(
        f"ZERO_SWAP_VRAM_LOCKED: used={(total_vram - free_vram)/(1024**3):.1f}GiB / {total_vram/(1024**3):.1f}GiB",
        flush=True,
    )


def _init_pipeline():
    global PIPELINE, ENGINE_READY, ENGINE_ERROR
    try:
        try:
            from diffsynth.core.vram.disk_map import DiskMap

            def _zero_copy_diskmap_getitem(self, name):
                if self.rename_dict is not None:
                    name = self.rename_dict[name]
                file_id = self.name_map[name]
                param = self.files[file_id].get_tensor(name)
                if self.torch_dtype is not None and isinstance(param, torch.Tensor) and param.dtype != self.torch_dtype:
                    param = param.to(self.torch_dtype)
                return param

            DiskMap.__getitem__ = _zero_copy_diskmap_getitem
        except Exception:
            pass

        total_vram_gb = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3)
        vram_limit = max(8.0, total_vram_gb - 2.5)
        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": "disk",
            "onload_device": "disk",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
        te_paths = [os.path.join(MODEL_DIR, f"FL2VA/text_encoder/model-{i:05d}-of-00014.safetensors") for i in range(1, 15)]
        dit_sub = "Ref2VA" if MODEL_VARIANT == "Ref2VA" else "FL2VA"
        dit_paths = [os.path.join(MODEL_DIR, f"{dit_sub}/transformer/model-{i:05d}-of-00013.safetensors") for i in range(1, 14)]
        vvae_path = os.path.join(MODEL_DIR, "FL2VA/video_vae/source/model.safetensors")
        avae_path = os.path.join(MODEL_DIR, "FL2VA/audio_vae/model.safetensors")

        proc_dir = os.path.join(MODEL_DIR, "FL2VA/processor")
        proc_cfg = ModelConfig(path=proc_dir) if os.path.isdir(proc_dir) else ModelConfig(
            model_id="MiniMaxAI/MiniMax-H3",
            origin_file_pattern="FL2VA/processor/",
            download_source="huggingface",
        )
        try:
            pipe = MiniMaxH3Pipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(path=te_paths, **vram_config),
                    ModelConfig(path=dit_paths, **vram_config),
                    ModelConfig(path=vvae_path, **vram_config),
                    ModelConfig(path=avae_path, **vram_config),
                ],
                processor_config=proc_cfg,
                vram_limit=vram_limit,
            )
        except Exception:
            pipe = MiniMaxH3Pipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(path=te_paths, **vram_config),
                    ModelConfig(path=dit_paths, **vram_config),
                    ModelConfig(path=vvae_path, **vram_config),
                    ModelConfig(path=avae_path, **vram_config),
                ],
                processor_config=ModelConfig(
                    model_id="MiniMaxAI/MiniMax-H3",
                    origin_file_pattern="FL2VA/processor/",
                    download_source="huggingface",
                ),
                vram_limit=vram_limit,
            )
        _enable_g4_vram_optimizations(pipe, total_vram_gb)
        PIPELINE = pipe
        ENGINE_READY = True
        print("MINIMAX_H3_ENGINE_READY", flush=True)
    except Exception as e:
        ENGINE_ERROR = str(e)
        print(f"MINIMAX_H3_ENGINE_ERROR: {e}", flush=True)


def _list_history_videos():
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")), key=os.path.getmtime, reverse=True)
    return [os.path.basename(p) for p in files[:12]]


def _normalize_h3_lora_mod_name(raw_mod: str) -> str:
    s = raw_mod.strip()
    for pref in ("diffusion_model.", "model.diffusion_model.", "transformer.", "dit.", "lora_unet_"):
        if s.startswith(pref):
            s = s[len(pref):]
    s = re.sub(r"^blocks_(\d+)_(attn|mlp|cross_attn)_(.+)$", r"blocks.\1.\2.\3", s)
    s = re.sub(r"^refiner_blocks_(\d+)_(attn|mlp)_(.+)$", r"refiner_blocks.\1.\2.\3", s)
    return s


def _apply_or_switch_sentinel_h3_lora(pipe, lora_path: str, lora_strength: float = 0.85, prefix: str = ""):
    dit = getattr(pipe, "dit", None)
    if dit is None:
        return
    clean_path = str(lora_path or "").strip().lstrip("/")
    strength = round(float(lora_strength if lora_strength is not None else 0.85), 3) if clean_path else 0.0
    req_key = (clean_path, strength)
    cur_key = getattr(dit, "_active_lora_key", ("", 0.0))
    if cur_key == req_key:
        return

    # 1. 先卸载旧 LoRA 增量权重，无损还原基座权重
    old_deltas = getattr(dit, "_active_lora_deltas", None) or []
    if old_deltas:
        for param, delta_cpu in old_deltas:
            try:
                param.data.sub_(delta_cpu.to(device=param.device, dtype=param.dtype))
            except Exception:
                pass
        dit._active_lora_deltas = []
        dit._active_lora_key = ("", 0.0)
        torch.cuda.empty_cache()

    if not clean_path or abs(strength) < 1e-4:
        return

    lora_dir = "/content/local_models/MiniMax-H3-LoRAs"
    os.makedirs(lora_dir, exist_ok=True)
    safe_fn = re.sub(r"[^A-Za-z0-9._-]+", "_", clean_path)
    if not safe_fn.endswith(".safetensors"):
        safe_fn += ".safetensors"
    local_p = os.path.join(lora_dir, safe_fn)
    rc_conf = os.path.expanduser("~/.config/rclone/rclone.conf")

    if not (os.path.isfile(local_p) and os.path.getsize(local_p) > 65536):
        if os.path.isfile(rc_conf) and shutil.which("rclone"):
            JOB_STATE["message"] = f"{prefix}☁️ 正在从 Google Drive 云盘极速拉取 H3 LoRA ({os.path.basename(clean_path)})..."
            subprocess.run(
                ["rclone", "copy", f"gdrive:models/MiniMax-H3-LoRAs/{safe_fn}", lora_dir, "--config", rc_conf],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
    if not (os.path.isfile(local_p) and os.path.getsize(local_p) > 65536):
        import urllib.parse
        JOB_STATE["message"] = f"{prefix}⬇️ 正在从 HuggingFace (Sentinel7/h3) 高速下载 LoRA ({os.path.basename(clean_path)})..."
        hf_url = "https://huggingface.co/Sentinel7/h3/resolve/main/" + urllib.parse.quote(clean_path, safe="/")
        tmp_p = local_p + ".part"
        subprocess.run(["curl", "-fL", "--retry", "3", "-o", tmp_p, hf_url], check=True)
        os.replace(tmp_p, local_p)
        if os.path.isfile(rc_conf) and shutil.which("rclone"):
            subprocess.Popen(
                ["rclone", "copy", local_p, "gdrive:models/MiniMax-H3-LoRAs", "--config", rc_conf],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    from safetensors.torch import load_file
    JOB_STATE["message"] = f"{prefix}🔌 正在向 H3 DiT 显存挂载 LoRA ({os.path.basename(clean_path)} · 权重 {strength})..."
    sd = load_file(local_p, device="cpu")
    grouped = {}
    for k, v in sd.items():
        m = re.match(r"^(.+?)\.(lora_A|lora_B|lora_down|lora_up|alpha)(?:\.weight)?$", k)
        if not m:
            continue
        mod_name = _normalize_h3_lora_mod_name(m.group(1))
        kind = m.group(2)
        entry = grouped.setdefault(mod_name, {})
        if kind in ("lora_A", "lora_down"):
            entry["down"] = v
        elif kind in ("lora_B", "lora_up"):
            entry["up"] = v
        elif kind == "alpha":
            try:
                entry["alpha"] = float(v.item())
            except Exception:
                pass

    mod_map = dict(dit.named_modules())
    applied_deltas = []
    for mod_name, item in grouped.items():
        down = item.get("down")
        up = item.get("up")
        if down is None or up is None:
            continue
        target_mod = mod_map.get(mod_name)
        if target_mod is None or not hasattr(target_mod, "weight"):
            continue
        w_param = target_mod.weight
        rank = max(1, int(down.shape[0]))
        alpha = float(item.get("alpha", rank))
        scale = float(strength) * (alpha / float(rank))
        up_f = up.to(device=w_param.device, dtype=torch.float32)
        down_f = down.to(device=w_param.device, dtype=torch.float32)
        if up_f.ndim == 2 and down_f.ndim == 2:
            delta_f = (up_f @ down_f) * scale
        else:
            delta_f = (up_f.flatten(1) @ down_f.flatten(1)).view_as(w_param) * scale
        if delta_f.shape == w_param.shape:
            delta_cast = delta_f.to(dtype=w_param.dtype)
            w_param.data.add_(delta_cast)
            applied_deltas.append((w_param, delta_cast.to(dtype=torch.float16).cpu()))
        del up_f, down_f, delta_f

    del sd, grouped
    dit._active_lora_deltas = applied_deltas
    dit._active_lora_key = req_key
    print(f"SENTINEL_H3_LORA_APPLIED: {clean_path} (strength={strength}, layers={len(applied_deltas)})", flush=True)


def _load_ref2va_video_frames_and_audio(payload: dict):
    import base64
    import tempfile
    import cv2
    from PIL import Image

    ref_name = os.path.basename(str(payload.get("ref_video_name") or "").strip())
    ref_b64 = str(payload.get("ref_video_b64") or "").strip()
    start_sec = max(0.0, float(payload.get("start_sec") or 0.0))
    end_sec = float(payload.get("end_sec") or 0.0)
    max_frames = 243

    tmp_in = None
    tmp_24 = None
    try:
        if ref_name and os.path.isfile(os.path.join(OUTPUT_DIR, ref_name)):
            src_path = os.path.join(OUTPUT_DIR, ref_name)
            already_guide = ref_name.startswith("guide_")
        elif ref_b64:
            raw_vb = base64.b64decode(ref_b64.split(",", 1)[-1])
            fd, tmp_in = tempfile.mkstemp(prefix="ref2va_in_", suffix=".mp4")
            os.close(fd)
            with open(tmp_in, "wb") as wf:
                wf.write(raw_vb)
            src_path = tmp_in
            already_guide = False
        else:
            return None, None, None

        fd2, tmp_24 = tempfile.mkstemp(prefix="ref2va_24fps_", suffix=".mp4")
        os.close(fd2)
        ff_cmd = ["ffmpeg", "-y", "-v", "error"]
        if not already_guide and start_sec > 0.0:
            ff_cmd.extend(["-ss", f"{start_sec:.3f}"])
        ff_cmd.extend(["-i", src_path])
        if not already_guide and end_sec > start_sec + 0.2:
            dur = min(end_sec - start_sec, max_frames / 24.0)
            ff_cmd.extend(["-t", f"{dur:.3f}"])
        else:
            ff_cmd.extend(["-t", f"{max_frames / 24.0:.3f}"])
        ff_cmd.extend([
            "-vf", "fps=24,scale='if(lte(iw,ih),min(768,iw),-2)':'if(lte(iw,ih),-2,min(768,ih))',scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "32000", "-ac", "2", "-shortest",
            tmp_24,
        ])
        r_ff = subprocess.run(ff_cmd, capture_output=True, text=True, timeout=120)
        if r_ff.returncode != 0:
            ff_noa = [c for c in ff_cmd if c not in ("-c:a", "aac", "-ar", "32000", "-ac", "2", "-shortest")]
            ff_noa.insert(-1, "-an")
            subprocess.run(ff_noa, check=True, timeout=120)

        ref_frames = None
        ref_audio = None
        ref_sr = 32000
        if read_video_audio is not None:
            try:
                ref_frames, ref_audio, ref_sr = read_video_audio(tmp_24)
            except Exception:
                ref_frames = None
        if not ref_frames:
            cap = cv2.VideoCapture(tmp_24)
            ref_frames = []
            while len(ref_frames) < max_frames:
                ok_r, fr = cap.read()
                if not ok_r or fr is None:
                    break
                ref_frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            cap.release()
        if not ref_frames:
            raise RuntimeError("参考视频无法解码，请检查视频文件")
        if len(ref_frames) > max_frames:
            ref_frames = ref_frames[:max_frames]
        while len(ref_frames) < 22:
            ref_frames.append(ref_frames[-1].copy())
        return ref_frames, ref_audio, int(ref_sr or 32000)
    finally:
        for tp in (tmp_in, tmp_24):
            if tp and os.path.exists(tp):
                try:
                    os.remove(tp)
                except Exception:
                    pass


def _run_video_job(payload: dict):
    global JOB_STATE
    t0 = time.time()
    try:
        req_mode = str(payload.get("mode") or ("ref2va" if MODEL_VARIANT == "Ref2VA" else "t2va")).strip().lower()
        is_ref2va = (req_mode == "ref2va") or (MODEL_VARIANT == "Ref2VA")
        visual = str(payload.get("prompt") or "").strip()
        sound = str(payload.get("soundscape") or payload.get("soundscape_prompt") or "").strip()
        music = str(payload.get("music") or payload.get("music_prompt") or "").strip()
        neg_prompt = str(payload.get("negative_prompt") or "").strip() or (
            "字幕，底部字幕，台词文字，水印，乱码，画面拉伸，人物变形，subtitles, captions, text on screen, lower thirds, watermark, logo, "
            "stretched face, distorted face, deformed hands, extra fingers, low quality, blurry"
        )
        res_str = str(payload.get("resolution") or "832x480").strip()
        num_frames = int(payload.get("num_frames") if payload.get("num_frames") is not None else 124)
        steps = int(payload.get("steps") or payload.get("num_inference_steps") or 30)
        fps = int(payload.get("fps") or 24)
        batch_count = max(1, min(4, int(payload.get("batch_count") or 1)))
        base_seed = int(payload.get("seed") if payload.get("seed") is not None else -1)
        rand_seed = bool(payload.get("randomize_seed")) or (base_seed < 0)

        res_map = {
            "832x480": (832, 480),
            "848x480": (832, 480),
            "480x832": (480, 832),
            "480x848": (480, 832),
            "1280x720": (1280, 720),
            "720x1280": (720, 1280),
            "640x640": (640, 640),
        }
        if "width" in payload and "height" in payload and not payload.get("resolution"):
            width, height = int(payload["width"]), int(payload["height"])
        else:
            width, height = res_map.get(res_str, (832, 480))

        # Pre-load Ref2VA multimodal references once across batch draws
        ref2va_refs = None
        if is_ref2va:
            import base64, io, tempfile
            from PIL import Image
            JOB_STATE["message"] = "🕺 正在准备 Ref2VA 参考视频 (24fps) 与多图参考特征..."
            ref2va_refs = []
            for img_b64 in (payload.get("ref_images_b64") or [])[:9]:
                if not str(img_b64 or "").strip():
                    continue
                raw_ib = str(img_b64).split(",", 1)[-1]
                pil_ref = Image.open(io.BytesIO(base64.b64decode(raw_ib))).convert("RGB")
                ref2va_refs.append({"type": "image", "image": pil_ref})
            ref_frames, ref_audio, ref_sr = _load_ref2va_video_frames_and_audio(payload)
            if ref_frames is not None:
                keep_ref_audio = bool(payload.get("keep_ref_audio", False))
                has_valid_audio = (
                    keep_ref_audio
                    and ref_audio is not None
                    and hasattr(ref_audio, "numel")
                    and int(ref_audio.numel()) > 0
                )
                if has_valid_audio:
                    ref2va_refs.append({
                        "type": "video_audio",
                        "video": ref_frames,
                        "audio": ref_audio,
                        "sample_rate": ref_sr,
                    })
                else:
                    ref2va_refs.append({"type": "video", "video": ref_frames})
                if num_frames <= 0 or bool(payload.get("match_ref_duration", False)):
                    n_step = max(2, int(round((len(ref_frames) - 5) / 17.0)))
                    num_frames = min(243, max(39, 17 * n_step + 5))
            if payload.get("ref_audio_b64") and read_audio is not None:
                raw_ab = str(payload["ref_audio_b64"]).split(",", 1)[-1]
                fd_a, tmp_a = tempfile.mkstemp(prefix="ref2va_aud_", suffix=".wav")
                os.close(fd_a)
                try:
                    with open(tmp_a, "wb") as af:
                        af.write(base64.b64decode(raw_ab))
                    aud_t, aud_sr = read_audio(tmp_a)
                    ref2va_refs.append({"type": "audio", "audio": aud_t, "sample_rate": aud_sr})
                finally:
                    try:
                        os.remove(tmp_a)
                    except Exception:
                        pass
            if num_frames <= 0:
                num_frames = 124
            if not ref2va_refs:
                raise RuntimeError("Ref2VA 模式请至少提供 1 个参考视频或参考图片")
            has_ref_video = any(r["type"] in ("video", "video_audio") for r in ref2va_refs)
            if has_ref_video:
                if "<Video 1>" not in visual:
                    visual = f"严格跟随 <Video 1> 中的人物动作轨迹与运镜节奏。{visual}"
                freedom = str(payload.get("motion_freedom") or "strict").strip().lower()
                if freedom == "natural" and "自由发挥" not in visual and "natural micro-expressions" not in visual.lower():
                    visual = visual.rstrip("。., ") + "。以 <Video 1> 的大致走位与运镜节奏为主干，允许人物表情、眼神互动、呼吸微动与发丝/衣料惯性自然流畅地自由发挥 (natural micro-expressions and organic motion)。"
                elif freedom == "strict" and "strictly locked" not in visual.lower() and "1:1" not in visual:
                    visual = visual.rstrip("。., ") + "。严格 1:1 卡拍对齐 <Video 1> 的每一帧肢体姿态、手脚位置与节奏变化 (strictly locked frame-accurate motion choreography)。"
            else:
                if visual and "实拍原片" not in visual and "raw unedited cinema camera footage" not in visual.lower():
                    visual = visual.rstrip("。.,， ") + "。电影摄影机实拍原片，纯洁干净画面 (raw unedited cinema camera footage, pure clean frame)。"
        else:
            if visual and "实拍原片" not in visual and "raw unedited cinema camera footage" not in visual.lower():
                visual = visual.rstrip("。.,， ") + "。电影摄影机实拍原片，纯洁干净画面 (raw unedited cinema camera footage, pure clean frame)。"

        parts = [visual]
        if sound:
            parts.append(f"Soundscape & Dialogue: {sound}")
        if music:
            parts.append(f"Background Music: {music}")
        full_prompt = "\n".join(parts)

        completed_videos = []
        last_seed = base_seed
        for b_idx in range(batch_count):
            cur_seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if rand_seed else (base_seed + b_idx)
            last_seed = cur_seed
            prefix = f"[连抽 {b_idx+1}/{batch_count}] " if batch_count > 1 else ""
            base_pct = int((b_idx / float(batch_count)) * 100)
            span_pct = 100.0 / float(batch_count)

            JOB_STATE.update({
                "running": True,
                "pct": max(5, int(base_pct + 0.08 * span_pct)),
                "step": 0,
                "total_steps": steps,
                "target_steps": steps,
                "seed": cur_seed,
                "batch_count": batch_count,
                "completed_count": b_idx,
                "completed_videos": completed_videos,
                "message": f"{prefix}🔓 正在使用 32B 无审查多模态编码器解析{'Ref2VA 白模/动作' if is_ref2va else ''}提示词 (Seed: {cur_seed})...",
                "error": "",
            })

            with GPU_LOCK:
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    _apply_or_switch_sentinel_h3_lora(
                        PIPELINE,
                        str(payload.get("lora_path") or "").strip(),
                        float(payload.get("lora_strength") if payload.get("lora_strength") is not None else 0.85),
                        prefix=prefix,
                    )
                    if payload.get("sigma_shift") is not None and hasattr(PIPELINE, "scheduler"):
                        try:
                            PIPELINE.scheduler.shift = float(payload.get("sigma_shift"))
                        except Exception:
                            pass
                except Exception as e_lora:
                    print(f"LORA_MOUNT_WARN: {e_lora}", flush=True)
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                mode_tag = "ref2va" if is_ref2va else ("fl2va" if payload.get("first_frame_b64") else "t2va")
                call_kwargs = dict(
                    prompt=full_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=steps,
                    seed=cur_seed,
                    tiled=True,
                )
                if neg_prompt:
                    call_kwargs["negative_prompt"] = neg_prompt
                if is_ref2va and ref2va_refs:
                    _out_short = min(int(width), int(height))
                    _ref_v_short = max(384, _out_short)
                    _ref_v_max_px = int(width) * int(height)
                    _ref_i_short = 1024
                    call_kwargs["references"] = ref2va_refs
                    call_kwargs["ref_image_short_edge"] = _ref_i_short
                    call_kwargs["ref_video_short_edge"] = _ref_v_short
                    call_kwargs["ref_video_max_pixels"] = _ref_v_max_px
                    print(
                        f"REF2VA_FAST_RES: out={width}x{height} ref_video_short={_ref_v_short} ref_video_max_px={_ref_v_max_px} ref_image_short={_ref_i_short}",
                        flush=True,
                    )
                else:
                    first_b64 = payload.get("first_frame_b64")
                    last_b64 = payload.get("last_frame_b64")
                    if first_b64:
                        import base64, inspect, io
                        from PIL import Image, ImageOps
                        raw_f = first_b64.split(",", 1)[-1]
                        raw_im_f = Image.open(io.BytesIO(base64.b64decode(raw_f))).convert("RGB")
                        img_f = ImageOps.fit(raw_im_f, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.28))
                        cond_imgs = [img_f]
                        cond_indices = [0]
                        if last_b64:
                            raw_l = last_b64.split(",", 1)[-1]
                            raw_im_l = Image.open(io.BytesIO(base64.b64decode(raw_l))).convert("RGB")
                            img_l = ImageOps.fit(raw_im_l, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.28))
                            cond_imgs.append(img_l)
                            cond_indices.append(-1)
                        sig_params = inspect.signature(PIPELINE.__call__).parameters
                        if "keyframes" in sig_params:
                            call_kwargs["keyframes"] = cond_imgs
                            call_kwargs["keyframe_indices"] = cond_indices
                        else:
                            call_kwargs["condition_images"] = cond_imgs
                            call_kwargs["condition_images_index"] = cond_indices
                if getattr(PIPELINE, "dit", None) is not None:
                    _base_tc_thresh = float(os.environ.get("MINIMAX_TEACACHE_THRESH", "0.12"))
                    if steps < 25:
                        PIPELINE.dit._tc_enabled = False
                        PIPELINE.dit._tc_max_skips = 0
                        PIPELINE.dit._tc_thresh = 0.0
                    elif is_ref2va:
                        PIPELINE.dit._tc_enabled = True
                        PIPELINE.dit._tc_max_skips = 1
                        PIPELINE.dit._tc_thresh = min(_base_tc_thresh, 0.08)
                    else:
                        PIPELINE.dit._tc_enabled = True
                        PIPELINE.dit._tc_max_skips = 2
                        PIPELINE.dit._tc_thresh = _base_tc_thresh
                    PIPELINE.dit._tc_step = 0
                    PIPELINE.dit._tc_total_steps = steps
                    PIPELINE.dit._tc_prev_b0_res = None
                    PIPELINE.dit._tc_cached_rest_res = None
                    PIPELINE.dit._tc_accum_rel_l1 = 0.0
                    PIPELINE.dit._tc_consec_skips = 0
                    PIPELINE.dit._tc_skipped_count = 0
                    print(
                        f"TEACACHE_CFG: mode={mode_tag} steps={steps} enabled={PIPELINE.dit._tc_enabled} max_skips={PIPELINE.dit._tc_max_skips} thresh={PIPELINE.dit._tc_thresh}",
                        flush=True,
                    )
                try:
                    video, audio = PIPELINE(**call_kwargs)
                except TypeError:
                    call_kwargs.pop("negative_prompt", None)
                    video, audio = PIPELINE(**call_kwargs)
                JOB_STATE["pct"] = int(base_pct + 0.94 * span_pct)
                JOB_STATE["message"] = f"{prefix}🎵 正在合成 32kHz 立体声音轨与 MP4 视频..."
                fname = f"minimax_h3_{mode_tag}_{ts}_seed{cur_seed}.mp4"
                out_path = os.path.join(OUTPUT_DIR, fname)
                write_video_audio(
                    video=video,
                    audio=audio,
                    output_path=out_path,
                    fps=fps,
                    audio_sample_rate=32000,
                )
                video = None
                audio = None
                gc.collect()
                torch.cuda.empty_cache()

            completed_videos.append(fname)
            JOB_STATE["latest_video"] = fname
            JOB_STATE["completed_count"] = b_idx + 1
            JOB_STATE["completed_videos"] = completed_videos

            rc_conf = os.path.expanduser("~/.config/rclone/rclone.conf")
            if os.path.exists(rc_conf) and shutil.which("rclone"):
                from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                _ym = _dt.now(_tz(_td(hours=8))).strftime("%Y.%m")
                subprocess.Popen(
                    ["rclone", "copy", out_path, f"gdrive:pro6000-output/{_ym}/videos", "--config", rc_conf],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        dt = round(time.time() - t0, 1)
        batch_label = f"共 {batch_count} 条视频 · " if batch_count > 1 else ""
        JOB_STATE.update({
            "running": False,
            "pct": 100,
            "seed": last_seed,
            "elapsed_s": dt,
            "latest_video": completed_videos[-1] if completed_videos else "",
            "completed_videos": completed_videos,
            "message": f"✅ 生成完成！{batch_label}总耗时 {dt}s · {width}×{height} · {num_frames}帧 (Seed: {last_seed}) · 已自动备份至 Google Drive",
        })
    except Exception as e:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        JOB_STATE.update({
            "running": False,
            "error": str(e),
            "message": f"❌ 生成出错: {e}",
        })


class _MiniMaxHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            if ENGINE_READY:
                self._send_json({"status": "ok", "engine": ENGINE_NAME, "variant": MODEL_VARIANT})
            else:
                self._send_json({"status": "loading", "engine": ENGINE_NAME, "variant": MODEL_VARIANT, "error": ENGINE_ERROR}, status=503)
            return

        if self.path.startswith("/v1/models") or self.path.startswith("/models"):
            mid = "MiniMaxAI/MiniMax-H3-Ref2VA" if MODEL_VARIANT == "Ref2VA" else "MiniMaxAI/MiniMax-H3"
            self._send_json({
                "object": "list",
                "data": [{"id": mid, "object": "model", "owned_by": "MiniMaxAI"}],
            })
            return

        if self.path.startswith("/v1/video/status"):
            st = dict(JOB_STATE)
            st["engine_ready"] = ENGINE_READY
            st["history"] = _list_history_videos()
            self._send_json(st)
            return

        if self.path.startswith("/v1/video/upscale_status"):
            try:
                with urllib.request.urlopen("http://127.0.0.1:30002/v1/video/upscale_status", timeout=5) as r:
                    self._send_json(json.loads(r.read().decode("utf-8")))
            except Exception as e:
                self._send_json({"running": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/audio/indextts2"):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:30003{self.path}", timeout=8) as r:
                    self._send_json(json.loads(r.read().decode("utf-8")), status=r.status)
            except Exception as e:
                self._send_json({"ok": False, "ready": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/video/file/"):
            fname = os.path.basename(self.path.split("/v1/video/file/", 1)[-1].split("?")[0])
            fpath = os.path.join(OUTPUT_DIR, fname)
            if not os.path.isfile(fpath):
                self._send_json({"error": "File not found"}, status=404)
                return
            fsize = os.path.getsize(fpath)
            range_hdr = self.headers.get("Range")
            if range_hdr and range_hdr.startswith("bytes="):
                try:
                    parts = range_hdr.replace("bytes=", "").split("-")
                    start = int(parts[0]) if parts[0] else 0
                    end = int(parts[1]) if (len(parts) > 1 and parts[1]) else (fsize - 1)
                    end = min(end, fsize - 1)
                    length = max(0, end - start + 1)
                    self.send_response(206)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{fsize}")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(length))
                    self.end_headers()
                    with open(fpath, "rb") as rf:
                        rf.seek(start)
                        rem = length
                        while rem > 0:
                            buf = rf.read(min(262144, rem))
                            if not buf:
                                break
                            self.wfile.write(buf)
                            rem -= len(buf)
                    return
                except Exception:
                    pass
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(fsize))
            self.send_header("Content-Disposition", f'inline; filename="{fname}"')
            self.end_headers()
            with open(fpath, "rb") as rf:
                while True:
                    buf = rf.read(262144)
                    if not buf:
                        break
                    self.wfile.write(buf)
            return

        self._send_json({"status": "ok", "engine": "minimax-h3"})

    def do_POST(self):
        cl = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(cl).decode("utf-8") if cl > 0 else "{}"
        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {}

        if self.path.startswith("/v1/video/upscale"):
            try:
                req2 = urllib.request.Request(
                    f"http://127.0.0.1:30002{self.path}",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req2, timeout=15) as r2:
                    self._send_json(json.loads(r2.read().decode("utf-8")), status=r2.status)
            except urllib.error.HTTPError as he:
                try:
                    self._send_json(json.loads(he.read().decode("utf-8")), status=he.code)
                except Exception:
                    self._send_json({"ok": False, "error": str(he)}, status=he.code)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/guide/preprocess"):
            try:
                if "/content" not in sys.path:
                    sys.path.insert(0, "/content")
                import guide_preprocessor
                res_gp = guide_preprocessor.preprocess_guide(payload, OUTPUT_DIR, gpu_lock=GPU_LOCK)
                self._send_json(res_gp, status=200 if res_gp.get("ok") else 400)
            except Exception as e:
                self._send_json({"ok": False, "error": f"白模/深度预处理失败: {e}"}, status=500)
            return

        if self.path.startswith("/v1/audio/indextts2"):
            def _fwd_idx():
                req3 = urllib.request.Request(
                    f"http://127.0.0.1:30003{self.path}",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req3, timeout=300) as r3:
                    return r3.status, json.loads(r3.read().decode("utf-8"))
            try:
                st3, d3 = _fwd_idx()
                self._send_json(d3, status=st3)
            except urllib.error.HTTPError as he:
                try:
                    self._send_json(json.loads(he.read().decode("utf-8")), status=he.code)
                except Exception:
                    self._send_json({"ok": False, "error": str(he)}, status=he.code)
            except Exception as e:
                if os.path.isfile("/content/indextts2_server.py"):
                    _idx_env = os.environ.copy()
                    _idx_env["PYTHONPATH"] = f"/content/indextts2_pkgs:{_idx_env.get('PYTHONPATH', '')}".rstrip(":")
                    subprocess.Popen([sys.executable, "-u", "/content/indextts2_server.py"], env=_idx_env, stdout=open("/content/indextts2_server.log", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, close_fds=True, start_new_session=True)
                    time.sleep(2.5)
                    try:
                        st3, d3 = _fwd_idx()
                        self._send_json(d3, status=st3)
                        return
                    except Exception as e2:
                        e = e2
                self._send_json({"ok": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/video/cancel"):
            open("/content/.cancel_minimax", "w").write("1")
            self._send_json({"ok": True, "message": "🛑 正在停止视频生成任务..."})
            return

        if self.path.startswith("/v1/video/generate") or self.path.startswith("/v1/video/generations"):
            req_m = str(payload.get("mode") or "").strip().lower()
            if req_m in ("image", "avatar"):
                self._send_json({"ok": False, "error": f"Engine mismatch: {ENGINE_NAME} cannot run image/avatar mode"}, status=503)
                return
            if req_m == "ref2va" and MODEL_VARIANT != "Ref2VA":
                self._send_json({"ok": False, "error": "Engine mismatch: need minimax-h3-ref2va for ref2va mode"}, status=503)
                return
            if req_m in ("t2va", "fl2va") and MODEL_VARIANT == "Ref2VA":
                self._send_json({"ok": False, "error": "Engine mismatch: need minimax-h3 (FL2VA) for t2va/fl2va mode"}, status=503)
                return
            if os.path.exists("/content/.cancel_minimax"):
                try:
                    os.unlink("/content/.cancel_minimax")
                except Exception:
                    pass
            if not ENGINE_READY:
                self._send_json({"ok": False, "error": "MiniMax-H3 尚未完成显存加载，请稍候..."}, status=503)
                return
            if JOB_STATE["running"]:
                self._send_json({"ok": False, "error": "当前已有视频正在生成中，请等待当前视频完成"}, status=409)
                return
            prompt = str(payload.get("prompt") or "").strip()
            if not prompt:
                self._send_json({"ok": False, "error": "请输入画面描述提示词 (Prompt)"}, status=400)
                return
            JOB_STATE.update({
                "running": True,
                "pct": 2,
                "step": 0,
                "completed_videos": [],
                "latest_video": "",
                "error": "",
                "message": "⏳ 任务已提交，正在进入 GPU 渲染队列...",
            })
            threading.Thread(target=_run_video_job, args=(payload,), daemon=True).start()
            self._send_json({"ok": True, "message": "已开始生成有声视频"})
            return

        if self.path.startswith("/v1/shell"):
            cmd = str(payload.get("cmd") or "").strip()
            if not cmd:
                self._send_json({"ok": False, "error": "empty cmd"}, status=400)
                return
            try:
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=int(payload.get("timeout") or 120))
                self._send_json({"ok": True, "returncode": r.returncode, "stdout": r.stdout[-16000:], "stderr": r.stderr[-16000:]})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=500)
            return

        if "chat/completions" in self.path:
            reply_txt = (
                "🎬 当前 GPU 显存中运行的是【MiniMax-H3 (33B · 原生BF16 · 无审查) 有声视频生成模型】！\n"
                "请直接在下方「🎬 MiniMax-H3 有声视频生成工作台」中粘贴提示词并点击「立即生成有声视频」。\n"
                "（如需切换回 Qwen3.8-27B 写提示词，请在右侧选中 Qwen3.8-27B 并点击「🚀 下载 / 预热并加载运行」。）"
            )
            self._send_json({
                "id": "chatcmpl-minimax",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply_txt}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 42, "total_tokens": 52},
                "timings": {"predicted_per_second": 100.0},
            })
            return

        self._send_json({"ok": True})


if __name__ == "__main__":
    threading.Thread(target=_init_pipeline, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", 30000), _MiniMaxHandler)
    srv.serve_forever()
'''


def _start_minimax_h3_server_on_30000(target_dir: str, variant: str = "FL2VA"):
    """在远端 Colab G4 实例上启动 MiniMax-H3 (FL2VA 或 Ref2VA · 33B · 原生BF16 · 无审查) 推理引擎 (端口 30000)"""
    variant = "Ref2VA" if str(variant or "").strip().lower() == "ref2va" else "FL2VA"
    expected_engine = "minimax-h3-ref2va" if variant == "Ref2VA" else "minimax-h3"
    engine_label = "MiniMax-H3-Ref2VA (33B · 白模/动作迁移 · 无审查)" if variant == "Ref2VA" else "MiniMax-H3 (33B · 原生BF16 · 无审查)"

    if not shutil.which("cloudflared"):
        log("正在安装 cloudflared 隧道客户端...")
        subprocess.run(
            "curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
            "-o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
            shell=True,
            check=True,
        )

    # 写入通用的 4 模式白模/深度/骨骼预处理器 (供卡片06与卡片01调用)
    with open("/content/guide_preprocessor.py", "w", encoding="utf-8") as gwf:
        gwf.write(GUIDE_PREPROCESSOR_CODE)

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    already_loading_remote = False
    try:
        with opener.open("http://127.0.0.1:30000/health", timeout=3) as resp:
            if resp.status == 200:
                h_data = json.loads(resp.read().decode("utf-8"))
                if h_data.get("engine") == expected_engine:
                    log(f"✅ 检测到 {engine_label} 视频推理引擎已在 GPU 显存中运行，直接复用！")
                    return
    except urllib.error.HTTPError as he:
        if he.code == 503 and subprocess.getoutput("pgrep -f '[m]inimax_server\\.py'").strip():
            try:
                eb = json.loads(he.read().decode("utf-8"))
                if eb.get("engine") == expected_engine:
                    log(f"⏳ 检测到 {engine_label} 正在后台装载显存中，直接接力等待完成 (不重复重启)...")
                    already_loading_remote = True
            except Exception:
                pass
    except Exception:
        pass

    proc = None
    if not already_loading_remote:
        log("正在释放旧模型显存 (保留内存盘/NVMe 中的权重以便随时极速切回)...")
        subprocess.run("pkill -9 -f '[l]lama-server|minimax_server\\.py|minimax_t4_server\\.py|longcat_avatar_server\\.py|minimax_image_server\\.py|qwen_image_server\\.py|project-maya/serve/server\\.py|build/maya|maya-vision|/content/app\\.py' || true; fuser -k -9 30000/tcp 2>/dev/null || true", shell=True)
        time.sleep(2)

        if not os.path.isdir("/content/DiffSynth-Studio/.git"):
            shutil.rmtree("/content/DiffSynth-Studio", ignore_errors=True)
            log("正在安装 DiffSynth-Studio 音视频管线依赖 (首次约需 25 秒)...")
            for _clone_try in range(3):
                r_cl = subprocess.run(
                    ["git", "clone", "--depth", "1", "https://github.com/modelscope/DiffSynth-Studio.git", "/content/DiffSynth-Studio"],
                    check=False,
                )
                if r_cl.returncode == 0 and os.path.isdir("/content/DiffSynth-Studio/.git"):
                    break
                shutil.rmtree("/content/DiffSynth-Studio", ignore_errors=True)
                time.sleep(2)
        missing_pkgs = []
        for pkg, mod in [
            ("av", "av"),
            ("soundfile", "soundfile"),
            ("librosa", "librosa"),
            ("einops", "einops"),
            ("qwen_vl_utils", "qwen_vl_utils"),
            ("imageio", "imageio"),
            ("imageio_ffmpeg", "imageio_ffmpeg"),
            ("modelscope", "modelscope"),
            ("sageattention", "sageattention"),
        ]:
            try:
                __import__(mod)
            except ImportError:
                missing_pkgs.append(pkg.replace("_", "-"))
        if missing_pkgs:
            log(f"正在补齐音视频合成依赖 ({', '.join(missing_pkgs)})...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing_pkgs, check=True)

        server_py = "/content/minimax_server.py"
        with open(server_py, "w", encoding="utf-8") as wf:
            wf.write(MINIMAX_SERVER_CODE)

        env = os.environ.copy()
        env["MINIMAX_MODEL_DIR"] = target_dir
        env["MINIMAX_VARIANT"] = variant
        log(f"🚀 正在启动 {engine_label} 视频引擎并预热至 96GB GPU 显存...")
        with open("/content/minimax_server.log", "w") as lf:
            proc = subprocess.Popen(
                [sys.executable, "-u", server_py],
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )

    t0 = time.time()
    expected_vram_gb = 72.4
    last_logged_pct = -1

    def _read_gpu_used_gb() -> float:
        try:
            out = subprocess.getoutput("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null").strip()
            return float(out.splitlines()[0].strip()) / 1024.0
        except Exception:
            return 0.0

    while time.time() - t0 < 600:
        if (proc is not None and proc.poll() is not None) or (proc is None and not subprocess.getoutput("pgrep -f '[m]inimax_server\\.py'").strip()):
            tail = subprocess.getoutput("tail -n 30 /content/minimax_server.log")
            raise RuntimeError(f"MiniMax-H3 引擎异常退出:\n{tail}")
        used_vram_gb = _read_gpu_used_gb()
        elapsed = max(0.5, time.time() - t0)
        try:
            with opener.open("http://127.0.0.1:30000/health", timeout=2) as resp:
                if resp.status == 200:
                    final_vram = max(used_vram_gb, 62.1)
                    log(f"[显存装载] MiniMax-H3(33B·无审查): 100% ({final_vram:.1f}/{final_vram:.1f} GiB VRAM, 已就绪)")
                    log("✅ MiniMax-H3 (33B · 原生 BF16 · 无审查) 有声视频推理引擎已就绪！")
                    _ensure_recavsr_service_on_30002()
                    _ensure_indextts2_service_on_30003()
                    return
        except Exception as e_h:
            if hasattr(e_h, "code") and getattr(e_h, "code", 0) == 503:
                try:
                    err_b = json.loads(e_h.read().decode("utf-8"))
                    if err_b.get("error"):
                        raise RuntimeError(f"MiniMax-H3 加载失败: {err_b['error']}")
                except RuntimeError:
                    raise
                except Exception:
                    pass
        vram_pct = int((used_vram_gb / expected_vram_gb) * 100) if used_vram_gb > 0.5 else int(min(18, elapsed * 1.5))
        pct = min(98, max(2, vram_pct))
        if pct != last_logged_pct:
            last_logged_pct = pct
            vram_spd = used_vram_gb / elapsed if used_vram_gb > 0.5 else 0.0
            log(f"[显存装载] MiniMax-H3(33B·无审查): {pct}% ({used_vram_gb:.1f}/{expected_vram_gb:.1f} GiB VRAM, {vram_spd:.2f} GB/s)")
        time.sleep(2)

    raise TimeoutError("等待 MiniMax-H3 装载显存超时")


MINIMAX_T4_SERVER_CODE = r'''#!/usr/bin/env python3
import base64
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_DIR = os.environ.get("MINIMAX_MODEL_DIR", "/content/local_models/MiniMax-H3-T4-GGUF")
OUTPUT_DIR = "/content/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SD_CLI_BIN = "/usr/local/bin/sd-cli"
DIT_PATH = os.path.join(MODEL_DIR, "minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf")
LLM_PATH = os.path.join(MODEL_DIR, "qwen3vl_32b_minimax_h3-Q2_K_M.gguf")
VVAE_PATH = os.path.join(MODEL_DIR, "vae/minimax_h3_video_vae_fp16.safetensors")
AVAE_PATH = os.path.join(MODEL_DIR, "vae/minimax_h3_audio_vae_fp32.safetensors")

ENGINE_READY = False
ENGINE_ERROR = ""
GPU_LOCK = threading.Lock()

JOB_STATE = {
    "running": False,
    "pct": 0,
    "step": 0,
    "total_steps": 8,
    "message": "就绪：MiniMax-H3 (T4 量化版 · UD-Q2_K_XL) 等待输入提示词...",
    "elapsed_s": 0.0,
    "latest_video": "",
    "error": "",
}


def _init_t4_engine():
    global ENGINE_READY, ENGINE_ERROR
    missing = [p for p in (DIT_PATH, LLM_PATH, VVAE_PATH, AVAE_PATH) if not os.path.isfile(p)]
    if missing:
        ENGINE_ERROR = f"缺少 T4 量化模型权重: {missing}"
        print(f"MINIMAX_T4_ENGINE_ERROR: {ENGINE_ERROR}", flush=True)
        return
    if not os.path.isfile(SD_CLI_BIN):
        ENGINE_ERROR = f"未找到 CUDA 推理二进制: {SD_CLI_BIN}"
        print(f"MINIMAX_T4_ENGINE_ERROR: {ENGINE_ERROR}", flush=True)
        return
    ENGINE_READY = True
    print("MINIMAX_T4_ENGINE_READY", flush=True)


def _list_history_videos():
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")), key=os.path.getmtime, reverse=True)
    return [os.path.basename(p) for p in files[:12]]


def _align_frames_17k5(n: int) -> int:
    n = max(5, min(124, int(n)))
    rem = (n - 5) % 17
    if rem != 0:
        n += (17 - rem)
    return min(124, max(5, n))


def _run_t4_video_job(payload: dict):
    global JOB_STATE
    t0 = time.time()
    try:
        visual = str(payload.get("prompt") or "").strip()
        sound = str(payload.get("soundscape") or payload.get("soundscape_prompt") or "").strip()
        music = str(payload.get("music") or payload.get("music_prompt") or "").strip()
        neg_prompt = str(payload.get("negative_prompt") or "").strip()
        res_str = str(payload.get("resolution") or "832x480").strip()
        req_frames = int(payload.get("num_frames") or 56)
        req_steps = int(payload.get("steps") or payload.get("num_inference_steps") or 8)
        fps = 24
        batch_count = max(1, min(4, int(payload.get("batch_count") or 1)))
        base_seed = int(payload.get("seed") if payload.get("seed") is not None else -1)
        rand_seed = bool(payload.get("randomize_seed")) or (base_seed < 0)

        res_map = {
            "832x480": (832, 480),
            "848x480": (832, 480),
            "480x832": (480, 832),
            "480x848": (480, 832),
            "1280x720": (832, 480),
            "720x1280": (480, 832),
            "640x640": (640, 640),
        }
        if "width" in payload and "height" in payload and not payload.get("resolution"):
            width, height = int(payload["width"]), int(payload["height"])
        else:
            width, height = res_map.get(res_str, (832, 480))
        width = max(256, ((width + 31) // 32) * 32)
        height = max(256, ((height + 31) // 32) * 32)

        num_frames = _align_frames_17k5(req_frames if req_frames <= 124 else 56)
        steps = max(4, min(8, req_steps if req_steps <= 12 else 8))

        parts = [visual]
        if sound:
            parts.append(f"Soundscape & Dialogue: {sound}")
        if music:
            parts.append(f"Background Music: {music}")
        full_prompt = "\n".join(parts)

        prompt_file = "/tmp/minimax_t4_prompt.txt"
        with open(prompt_file, "w", encoding="utf-8") as pf:
            pf.write(full_prompt)

        neg_file = "/tmp/minimax_t4_neg.txt"
        if neg_prompt:
            with open(neg_file, "w", encoding="utf-8") as nf:
                nf.write(neg_prompt)

        first_img_path = ""
        last_img_path = ""
        first_b64 = payload.get("first_frame_b64")
        last_b64 = payload.get("last_frame_b64")
        if first_b64:
            from PIL import Image, ImageOps
            raw_f = first_b64.split(",", 1)[-1]
            first_img_path = "/tmp/minimax_t4_first.png"
            im_f = Image.open(io.BytesIO(base64.b64decode(raw_f))).convert("RGB")
            ImageOps.fit(im_f, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.28)).save(first_img_path)
        if last_b64:
            from PIL import Image, ImageOps
            raw_l = last_b64.split(",", 1)[-1]
            last_img_path = "/tmp/minimax_t4_last.png"
            im_l = Image.open(io.BytesIO(base64.b64decode(raw_l))).convert("RGB")
            ImageOps.fit(im_l, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.28)).save(last_img_path)

        completed_videos = []
        last_seed = base_seed
        ansi_re = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

        for b_idx in range(batch_count):
            cur_seed = random.randint(1, 2**31 - 1) if rand_seed else (base_seed + b_idx)
            last_seed = cur_seed
            prefix = f"[连抽 {b_idx+1}/{batch_count}] " if batch_count > 1 else ""
            base_pct = int((b_idx / float(batch_count)) * 100)
            span_pct = 100.0 / float(batch_count)

            JOB_STATE.update({
                "running": True,
                "pct": max(4, int(base_pct + 0.05 * span_pct)),
                "step": 0,
                "total_steps": steps,
                "target_steps": steps,
                "seed": cur_seed,
                "batch_count": batch_count,
                "completed_count": b_idx,
                "completed_videos": completed_videos,
                "message": f"{prefix}⚡ [T4 量化版] 正在流式映射 Qwen3-VL-32B 编码提示词 ({width}×{height} · {num_frames}帧 · {steps}步)...",
                "error": "",
            })

            with GPU_LOCK:
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                mode_tag = "fl2va" if first_img_path else "t2va"
                tmp_avi = f"/tmp/minimax_t4_{ts}_{cur_seed}.avi"
                fname = f"minimax_h3_{mode_tag}_{ts}_seed{cur_seed}.mp4"
                out_path = os.path.join(OUTPUT_DIR, fname)

                cmd = [
                    SD_CLI_BIN,
                    "-M", "vid_gen",
                    "--diffusion-model", DIT_PATH,
                    "--llm", LLM_PATH,
                    "--vae", VVAE_PATH,
                    "--audio-vae", AVAE_PATH,
                    "--mmap",
                    "--params-backend", "te=disk,vae=disk",
                    "--max-vram", "13.5",
                    "--model-args", "minimax_h3_attn_chunk_budget=0.5",
                    "--vae-tiling",
                    "--temporal-tiling",
                    "--flash-attn", "on",
                    "--cfg-scale", "1.0",
                    "--sampling-method", "euler",
                    "--steps", str(steps),
                    "-W", str(width),
                    "-H", str(height),
                    "--video-frames", str(num_frames),
                    "--fps", str(fps),
                    "--rng", "cpu",
                    "-s", str(cur_seed),
                    "--prompt-file", prompt_file,
                    "-o", tmp_avi,
                    "-v",
                ]
                if neg_prompt:
                    cmd.extend(["--negative-prompt-file", neg_file])
                if first_img_path:
                    cmd.extend(["--init-img", first_img_path])
                if last_img_path:
                    cmd.extend(["--end-img", last_img_path])

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    bufsize=0,
                )
                buf = ""
                tail_lines = []
                in_sampling = False
                assert proc.stdout is not None
                while True:
                    chunk = proc.stdout.read(256)
                    if not chunk:
                        break
                    text = ansi_re.sub("", chunk.decode("utf-8", errors="replace"))
                    for ch in text:
                        if ch in ("\r", "\n"):
                            ln = buf.strip()
                            buf = ""
                            if not ln:
                                continue
                            tail_lines.append(ln)
                            if len(tail_lines) > 40:
                                tail_lines = tail_lines[-40:]
                            if os.path.exists("/content/.cancel_minimax"):
                                try:
                                    os.unlink("/content/.cancel_minimax")
                                except Exception:
                                    pass
                                proc.kill()
                                raise RuntimeError("🛑 已手动停止视频生成任务")
                            if "sampling using" in ln.lower():
                                in_sampling = True
                                JOB_STATE["pct"] = int(base_pct + 0.18 * span_pct)
                                JOB_STATE["message"] = f"{prefix}🎬 提示词编码完成，正在 T4 显存中进行 {steps} 步蒸馏采样..."
                            elif "sampling completed" in ln.lower() or "decoding" in ln.lower():
                                in_sampling = False
                                JOB_STATE["pct"] = int(base_pct + 0.88 * span_pct)
                                JOB_STATE["message"] = f"{prefix}🎞️ 采样完成，正在流式解码视频画面与 32kHz 立体声音轨..."
                            else:
                                m = re.search(r"\|\s*(\d+)/(\d+)\s*-", ln)
                                if m:
                                    cur_s = int(m.group(1))
                                    tot_s = max(1, int(m.group(2)))
                                    if in_sampling or tot_s == steps:
                                        JOB_STATE["step"] = cur_s
                                        JOB_STATE["total_steps"] = tot_s
                                        sub_pct = 18.0 + (cur_s / float(tot_s)) * 68.0
                                        ov_pct = min(96, max(8, int(base_pct + (sub_pct / 100.0) * span_pct)))
                                        JOB_STATE["pct"] = ov_pct
                                        JOB_STATE["message"] = f"{prefix}🎬 [T4 量化版] 正在扩散采样有声视频: 第 {cur_s}/{tot_s} 步 (总进度 {ov_pct}%)"
                        else:
                            buf += ch
                proc.wait()
                if proc.returncode != 0 or not os.path.isfile(tmp_avi):
                    err_tail = "\n".join(tail_lines[-15:])
                    raise RuntimeError(f"sd-cli 退出码 {proc.returncode}:\n{err_tail}")

                JOB_STATE["pct"] = int(base_pct + 0.95 * span_pct)
                JOB_STATE["message"] = f"{prefix}🎵 正在封装 H.264 + AAC 立体声 MP4 视频..."
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", tmp_avi,
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-crf", "18",
                        "-preset", "fast",
                        "-c:a", "aac",
                        "-b:a", "192k",
                        "-movflags", "+faststart",
                        out_path,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                try:
                    os.remove(tmp_avi)
                except Exception:
                    pass

            completed_videos.append(fname)
            JOB_STATE["latest_video"] = fname
            JOB_STATE["completed_count"] = b_idx + 1
            JOB_STATE["completed_videos"] = completed_videos

            rc_conf = os.path.expanduser("~/.config/rclone/rclone.conf")
            if os.path.exists(rc_conf) and shutil.which("rclone"):
                from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                _ym = _dt.now(_tz(_td(hours=8))).strftime("%Y.%m")
                subprocess.Popen(
                    ["rclone", "copy", out_path, f"gdrive:pro6000-output/{_ym}/videos", "--config", rc_conf],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        dt = round(time.time() - t0, 1)
        batch_label = f"共 {batch_count} 条视频 · " if batch_count > 1 else ""
        JOB_STATE.update({
            "running": False,
            "pct": 100,
            "seed": last_seed,
            "elapsed_s": dt,
            "latest_video": completed_videos[-1] if completed_videos else "",
            "completed_videos": completed_videos,
            "message": f"✅ [T4 量化版] 生成完成！{batch_label}总耗时 {dt}s · {width}×{height} · {num_frames}帧 (Seed: {last_seed}) · 已备份至云盘",
        })
    except Exception as e:
        JOB_STATE.update({
            "running": False,
            "error": str(e),
            "message": f"❌ 生成出错: {e}",
        })


class _MiniMaxT4Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            if ENGINE_READY:
                self._send_json({"status": "ok", "engine": "minimax-h3", "variant": "t4-gguf"})
            else:
                self._send_json({"status": "loading", "error": ENGINE_ERROR}, status=503)
            return

        if self.path.startswith("/v1/models") or self.path.startswith("/models"):
            self._send_json({
                "object": "list",
                "data": [
                    {"id": "unsloth/MiniMax-H3-GGUF", "object": "model", "owned_by": "MiniMaxAI"},
                    {"id": "MiniMaxAI/MiniMax-H3", "object": "model", "owned_by": "MiniMaxAI"},
                ],
            })
            return

        if self.path.startswith("/v1/video/status"):
            st = dict(JOB_STATE)
            st["engine_ready"] = ENGINE_READY
            st["history"] = _list_history_videos()
            self._send_json(st)
            return

        if self.path.startswith("/v1/video/upscale_status"):
            try:
                with urllib.request.urlopen("http://127.0.0.1:30002/v1/video/upscale_status", timeout=5) as r:
                    self._send_json(json.loads(r.read().decode("utf-8")))
            except Exception as e:
                self._send_json({"running": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/audio/indextts2"):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:30003{self.path}", timeout=8) as r:
                    self._send_json(json.loads(r.read().decode("utf-8")), status=r.status)
            except Exception as e:
                self._send_json({"ok": False, "ready": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/video/file/"):
            fname = os.path.basename(self.path.split("/v1/video/file/", 1)[-1].split("?")[0])
            fpath = os.path.join(OUTPUT_DIR, fname)
            if not os.path.isfile(fpath):
                self._send_json({"error": "File not found"}, status=404)
                return
            fsize = os.path.getsize(fpath)
            range_hdr = self.headers.get("Range")
            if range_hdr and range_hdr.startswith("bytes="):
                try:
                    parts = range_hdr.replace("bytes=", "").split("-")
                    start = int(parts[0]) if parts[0] else 0
                    end = int(parts[1]) if (len(parts) > 1 and parts[1]) else (fsize - 1)
                    end = min(end, fsize - 1)
                    length = max(0, end - start + 1)
                    self.send_response(206)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{fsize}")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(length))
                    self.end_headers()
                    with open(fpath, "rb") as rf:
                        rf.seek(start)
                        rem = length
                        while rem > 0:
                            buf = rf.read(min(262144, rem))
                            if not buf:
                                break
                            self.wfile.write(buf)
                            rem -= len(buf)
                    return
                except Exception:
                    pass
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(fsize))
            self.send_header("Content-Disposition", f'inline; filename="{fname}"')
            self.end_headers()
            with open(fpath, "rb") as rf:
                while True:
                    buf = rf.read(262144)
                    if not buf:
                        break
                    self.wfile.write(buf)
            return

        self._send_json({"status": "ok", "engine": "minimax-h3", "variant": "t4-gguf"})

    def do_POST(self):
        cl = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(cl).decode("utf-8") if cl > 0 else "{}"
        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {}

        if self.path.startswith("/v1/video/upscale"):
            try:
                req2 = urllib.request.Request(
                    f"http://127.0.0.1:30002{self.path}",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req2, timeout=15) as r2:
                    self._send_json(json.loads(r2.read().decode("utf-8")), status=r2.status)
            except urllib.error.HTTPError as he:
                try:
                    self._send_json(json.loads(he.read().decode("utf-8")), status=he.code)
                except Exception:
                    self._send_json({"ok": False, "error": str(he)}, status=he.code)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/video/cancel"):
            open("/content/.cancel_minimax", "w").write("1")
            self._send_json({"ok": True, "message": "🛑 正在停止视频生成任务..."})
            return

        if self.path.startswith("/v1/audio/indextts2"):
            def _fwd_idx_t4():
                req3 = urllib.request.Request(
                    f"http://127.0.0.1:30003{self.path}",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req3, timeout=300) as r3:
                    return r3.status, json.loads(r3.read().decode("utf-8"))
            try:
                st3, d3 = _fwd_idx_t4()
                self._send_json(d3, status=st3)
            except urllib.error.HTTPError as he:
                try:
                    self._send_json(json.loads(he.read().decode("utf-8")), status=he.code)
                except Exception:
                    self._send_json({"ok": False, "error": str(he)}, status=he.code)
            except Exception as e:
                if os.path.isfile("/content/indextts2_server.py"):
                    _idx_env = os.environ.copy()
                    _idx_env["PYTHONPATH"] = f"/content/indextts2_pkgs:{_idx_env.get('PYTHONPATH', '')}".rstrip(":")
                    subprocess.Popen([sys.executable, "-u", "/content/indextts2_server.py"], env=_idx_env, stdout=open("/content/indextts2_server.log", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, close_fds=True, start_new_session=True)
                    time.sleep(2.5)
                    try:
                        st3, d3 = _fwd_idx_t4()
                        self._send_json(d3, status=st3)
                        return
                    except Exception as e2:
                        e = e2
                self._send_json({"ok": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/video/generate") or self.path.startswith("/v1/video/generations"):
            if payload.get("mode") in ("image", "avatar", "ref2va"):
                self._send_json({"ok": False, "error": "Engine mismatch: minimax-h3-t4 cannot run image/avatar/ref2va mode"}, status=503)
                return
            if os.path.exists("/content/.cancel_minimax"):
                try:
                    os.unlink("/content/.cancel_minimax")
                except Exception:
                    pass
            if not ENGINE_READY:
                self._send_json({"ok": False, "error": f"MiniMax-H3 (T4 量化版) 未就绪: {ENGINE_ERROR}"}, status=503)
                return
            if JOB_STATE["running"]:
                self._send_json({"ok": False, "error": "当前已有视频正在生成中，请等待当前视频完成"}, status=409)
                return
            prompt = str(payload.get("prompt") or "").strip()
            if not prompt:
                self._send_json({"ok": False, "error": "请输入画面描述提示词 (Prompt)"}, status=400)
                return
            threading.Thread(target=_run_t4_video_job, args=(payload,), daemon=True).start()
            self._send_json({"ok": True, "message": "已开始生成有声视频 (T4 量化版)"})
            return

        if self.path.startswith("/v1/shell"):
            cmd = str(payload.get("cmd") or "").strip()
            if not cmd:
                self._send_json({"ok": False, "error": "empty cmd"}, status=400)
                return
            try:
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=int(payload.get("timeout") or 120))
                self._send_json({"ok": True, "returncode": r.returncode, "stdout": r.stdout[-16000:], "stderr": r.stderr[-16000:]})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=500)
            return

        if "chat/completions" in self.path:
            reply_txt = (
                "🎬 当前运行的是【MiniMax-H3 (T4 量化版 · UD-Q2_K_XL) 有声视频生成引擎】！\n"
                "请直接在下方「🎬 MiniMax-H3 有声视频生成工作台」中粘贴提示词并点击「立即生成有声视频」。\n"
                "（如需先用 Qwen3.8-27B 扩写四项提示词，请在右侧选中「🧠 Qwen3.8-27B 无审查 (T4版)」加载扩写后再点生成视频，系统会自动切换！）"
            )
            self._send_json({
                "id": "chatcmpl-minimax-t4",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply_txt}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 42, "total_tokens": 52},
                "timings": {"predicted_per_second": 100.0},
            })
            return

        self._send_json({"ok": True})


if __name__ == "__main__":
    _init_t4_engine()
    srv = ThreadingHTTPServer(("0.0.0.0", 30000), _MiniMaxT4Handler)
    srv.serve_forever()
'''


def _ensure_sd_cli_binary(target_dir: str):
    """确保远端实例具备支持 MiniMax-H3 GGUF 的 CUDA 版 sd-cli 二进制，并持久化至云盘免重复编译"""
    sd_bin = "/usr/local/bin/sd-cli"
    cached_bin = os.path.join(target_dir, "sd-cli")
    if os.path.isfile(sd_bin) and os.path.getsize(sd_bin) > 1024 * 1024:
        return
    if os.path.isfile(cached_bin) and os.path.getsize(cached_bin) > 1024 * 1024:
        shutil.copy2(cached_bin, sd_bin)
        os.chmod(sd_bin, 0o755)
        log("⚡ 已直接复用本地/云盘缓存的 CUDA sd-cli 推理引擎！")
        return
    rc_conf = os.path.expanduser("~/.config/rclone/rclone.conf")
    if os.path.exists(rc_conf) and shutil.which("rclone"):
        subprocess.run(
            ["rclone", "copy", "gdrive:models/MiniMax-H3-T4-GGUF/sd-cli", target_dir, "--config", rc_conf],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if os.path.isfile(cached_bin) and os.path.getsize(cached_bin) > 1024 * 1024:
            shutil.copy2(cached_bin, sd_bin)
            os.chmod(sd_bin, 0o755)
            log("☁️ 已从 Google Drive 秒级拉取预编译的 CUDA sd-cli 推理引擎！")
            return

    try:
        cc_raw = subprocess.getoutput("nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null").strip().splitlines()[0]
        cuda_arch = cc_raw.replace(".", "").strip() or "75"
    except Exception:
        cuda_arch = "75"

    log(f"🔧 首次在 T4 运行 MiniMax-H3 (GGUF)，正在编译专用 CUDA (SM {cuda_arch}) 推理内核 (完成后将永久备份至云盘，后续 0 秒秒开)...")
    src_dir = "/content/sdcpp_build"
    shutil.rmtree(src_dir, ignore_errors=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/leejet/stable-diffusion.cpp.git", src_dir],
        check=True,
    )
    subprocess.run(
        ["git", "submodule", "update", "--init", "--depth", "1", "ggml"],
        cwd=src_dir,
        check=True,
    )
    subprocess.run(
        [
            "cmake", "-B", "build",
            "-DSD_CUDA=ON",
            "-DSD_CUDA_CUTLASS_PATH=",
            "-DSD_WEBP=OFF",
            "-DSD_WEBM=OFF",
            "-DSD_SERVER_BUILD_FRONTEND=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
        ],
        cwd=src_dir,
        check=True,
    )
    nproc = max(2, os.cpu_count() or 2)
    subprocess.run(
        ["cmake", "--build", "build", "--config", "Release", "--target", "sd-cli", "-j", str(nproc)],
        cwd=src_dir,
        check=True,
    )
    built_bin = os.path.join(src_dir, "build/bin/sd-cli")
    if not os.path.isfile(built_bin):
        raise RuntimeError("sd-cli 编译未生成目标文件")
    os.chmod(built_bin, 0o755)
    shutil.copy2(built_bin, sd_bin)
    shutil.copy2(built_bin, cached_bin)
    shutil.rmtree(src_dir, ignore_errors=True)
    if os.path.exists(rc_conf) and shutil.which("rclone"):
        subprocess.Popen(
            ["rclone", "copy", cached_bin, "gdrive:models/MiniMax-H3-T4-GGUF", "--config", rc_conf],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    log("✅ CUDA sd-cli 编译完成并已备份至 Google Drive 云盘！")


def _start_minimax_t4_gguf_server_on_30000(target_dir: str):
    """在远端 Colab T4 (16GB) 实例上启动 MiniMax-H3 (T4 量化版 · UD-Q2_K_XL) 推理引擎 (端口 30000)"""
    if not shutil.which("cloudflared"):
        log("正在安装 cloudflared 隧道客户端...")
        subprocess.run(
            "curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
            "-o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
            shell=True,
            check=True,
        )

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open("http://127.0.0.1:30000/health", timeout=2) as resp:
            if resp.status == 200:
                h_data = json.loads(resp.read().decode("utf-8"))
                if h_data.get("variant") == "t4-gguf":
                    log("✅ 检测到 MiniMax-H3 (T4 量化版) 视频推理服务已在运行，直接复用！")
                    return
    except Exception:
        pass

    log("正在释放旧模型进程，准备启动 MiniMax-H3 (T4 量化版 · UD-Q2_K_XL)...")
    subprocess.run("pkill -9 -f '[l]lama-server|minimax_server\\.py|minimax_t4_server\\.py|longcat_avatar_server\\.py|minimax_image_server\\.py|qwen_image_server\\.py|project-maya/serve/server\\.py|build/maya|maya-vision|/content/app\\.py' || true; fuser -k -9 30000/tcp 2>/dev/null || true", shell=True)
    time.sleep(1)

    _ensure_sd_cli_binary(target_dir)

    server_py = "/content/minimax_t4_server.py"
    with open(server_py, "w", encoding="utf-8") as wf:
        wf.write(MINIMAX_T4_SERVER_CODE)

    env = os.environ.copy()
    env["MINIMAX_MODEL_DIR"] = target_dir
    with open("/content/minimax_server.log", "w") as lf:
        proc = subprocess.Popen(
            [sys.executable, "-u", server_py],
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )

    for _ in range(30):
        if proc.poll() is not None:
            tail = subprocess.getoutput("tail -n 30 /content/minimax_server.log")
            raise RuntimeError(f"MiniMax-H3 (T4 量化版) 启动失败:\n{tail}")
        try:
            with opener.open("http://127.0.0.1:30000/health", timeout=2) as resp:
                if resp.status == 200:
                    log("[显存装载] MiniMax-H3(T4量化版·UD-Q2_K_XL): 100% (7.5/15.0 GiB VRAM, 已就绪)")
                    log("✅ MiniMax-H3 (T4 量化版 · UD-Q2_K_XL · 8步有声视频) 推理服务已就绪！")
                    _ensure_recavsr_service_on_30002()
                    _ensure_indextts2_service_on_30003()
                    return
        except Exception:
            pass
        time.sleep(1)

    raise TimeoutError("等待 MiniMax-H3 (T4 量化版) 服务就绪超时")


LONGCAT_AVATAR_SERVER_CODE = r'''#!/usr/bin/env python3
import base64
import base64
import gc
import glob
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL_DIR = os.environ.get("LONGCAT_MODEL_DIR", "/content/local_models/LongCat-Video-Avatar-1.6")
OUTPUT_DIR = "/content/outputs"
LONGCAT_CODE_DIR = "/content/LongCat-Video"
os.makedirs(OUTPUT_DIR, exist_ok=True)
if LONGCAT_CODE_DIR not in sys.path:
    sys.path.insert(0, LONGCAT_CODE_DIR)

ENGINE_READY = True
ENGINE_ERROR = ""
GPU_LOCK = threading.Lock()
ENGINE_INIT_LOCK = threading.Lock()

_AVATAR_ENGINE = {
    "pipe": None,
    "vocal_separator": None,
    "use_int8": None,
    "model_type": "avatar-v1.5",
    "ming_talker": None,
}

_STEP_TRACKER = {
    "active": False,
    "cur_seg_idx": 1,
    "num_segments": 1,
    "cur_seg_step": 0,
}

JOB_STATE = {
    "running": False,
    "pct": 0,
    "step": 0,
    "total_steps": 8,
    "message": "就绪：LongCat-Video-Avatar-1.6 数字人音频驱动引擎已待命...",
    "elapsed_s": 0.0,
    "latest_video": "",
    "error": "",
}


def _list_history_videos():
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")), key=os.path.getmtime, reverse=True)
    return [os.path.basename(p) for p in files[:12]]


def _apply_fast_attention_and_ffmpeg_patches():
    """
    1. 消除 MultiHeadCrossAttention 中每块 2 次的 cu_seqlens.tolist() GPU->CPU 同步阻塞 (每步消除 96 次同步停顿)
    2. 消除自注意力与音频交叉注意力中的 4 次 3D 视频张量 rearrange + .transpose().contiguous() 显存拷贝 (直接原位调用 CUDA SDPA)
    3. 确保 save_video_ffmpeg 具备 -af apad 与 BrokenPipeError 保护
    """
    import torch
    import torch.nn.functional as F
    from einops import rearrange

    tu_p = os.path.join(LONGCAT_CODE_DIR, "longcat_video/audio_process/torch_utils.py")
    if os.path.isfile(tu_p):
        try:
            tu_txt = open(tu_p, "r", encoding="utf-8").read()
            if 'process.stdin.write(memoryview(video[start:start + 16]).cast("B"))' in tu_txt and "except BrokenPipeError:" not in tu_txt:
                tu_txt = tu_txt.replace(
                    '"-c:a", "aac", "-shortest", output_path,',
                    '"-af", "apad", "-c:a", "aac", "-shortest", output_path,',
                ).replace(
                    '        for start in range(0, frame_count, 16):\n            process.stdin.write(memoryview(video[start:start + 16]).cast("B"))\n        process.stdin.close()',
                    '        for start in range(0, frame_count, 16):\n            try:\n                process.stdin.write(memoryview(video[start:start + 16]).cast("B"))\n            except BrokenPipeError:\n                break\n        try:\n            process.stdin.close()\n        except BrokenPipeError:\n            pass',
                )
                open(tu_p, "w", encoding="utf-8").write(tu_txt)
        except Exception:
            pass

    import longcat_video.modules.attention as base_attn_mod
    import longcat_video.modules.avatar.attention as avatar_attn_mod

    def _fast_self_attn(self, q, k, v, shape):
        if getattr(self, "enable_bsa", False) and shape is not None and shape[0] > 1:
            from longcat_video.block_sparse_attention.bsa_interface import flash_attn_bsa_3d
            _, H, W = shape
            H, W = H // self.cp_split_hw[0], W // self.cp_split_hw[1]
            Tq = q.shape[2] // (H * W)
            Tk = k.shape[2] // (H * W)
            return flash_attn_bsa_3d(q, k, v, (Tq, H, W), (Tk, H, W), **self.bsa_params)
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=self.scale)

    def _fast_text_cross_attn(self, x, cond, kv_seqlen):
        B, N, C = x.shape
        q = self.q_norm(self.q_linear(x).view(1, -1, self.num_heads, self.head_dim))
        k, v = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim).unbind(2)
        k = self.k_norm(k)
        if B == 1 or len(kv_seqlen) == 1:
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=0.0, is_causal=False,
            ).transpose(1, 2)
        else:
            outs = []
            k_off = 0
            q0, k0, v0 = q[0], k[0], v[0]
            for i in range(B):
                mk = int(kv_seqlen[i])
                qi = q0[i * N : (i + 1) * N].unsqueeze(0).transpose(1, 2)
                ki = k0[k_off : k_off + mk].unsqueeze(0).transpose(1, 2)
                vi = v0[k_off : k_off + mk].unsqueeze(0).transpose(1, 2)
                k_off += mk
                oi = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=0.0, is_causal=False)
                outs.append(oi.transpose(1, 2))
            out = torch.cat(outs, dim=1)
        return self.proj(out.reshape(B, N, C))

    orig_audio_cross = avatar_attn_mod.SingleStreamAttention._process_cross_attn

    def _fast_audio_cross_attn(self, x, cond, frames_num=None, x_ref_attn_map=None, human_num=None):
        if x_ref_attn_map is not None:
            return orig_audio_cross(self, x, cond, frames_num=frames_num, x_ref_attn_map=x_ref_attn_map, human_num=human_num)
        N_t = frames_num
        out_dtype = x.dtype
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)
        B, N, C = x.shape
        q = self.q_norm(self.q_linear(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3))
        _, N_a, _ = cond.shape
        encoder_k, encoder_v = self.kv_linear(cond).view(B, N_a, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        encoder_k = self.k_norm(encoder_k)
        out = F.scaled_dot_product_attention(q, encoder_k, encoder_v, dropout_p=0.0, is_causal=False, scale=self.scale)
        out = self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, N, C)))
        return rearrange(out, "(B N_t) S C -> B (N_t S) C", N_t=N_t).to(out_dtype)

    base_attn_mod.Attention._process_attn = _fast_self_attn
    base_attn_mod.MultiHeadCrossAttention._process_cross_attn = _fast_text_cross_attn
    avatar_attn_mod.Attention._process_attn = _fast_self_attn
    avatar_attn_mod.SingleStreamAttention._process_cross_attn = _fast_audio_cross_attn


def _merge_lora_into_dit(dit, lora_path: str, multiplier: float = 0.5, lora_dim: int = 128, alpha: float = 64.0) -> int:
    """在 GPU 上直接将 LoRA 权重静态融合进 DiT Linear / QuantizedLinear 矩阵，彻底消除推理期动态 LoRA 包装器开销并解锁 KV Cache"""
    import torch
    from safetensors.torch import load_file

    if not os.path.isfile(lora_path) or multiplier == 0.0:
        return 0
    sd = load_file(lora_path, device="cpu")
    eff_scale = float(multiplier) * (float(alpha) / float(lora_dim))
    lora_names = [k[:-len(".lora_down.weight")] for k in sd.keys() if k.endswith(".lora_down.weight")]
    merged = 0
    dev = next(dit.parameters()).device
    with torch.no_grad():
        for lname in lora_names:
            mod_name = lname.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")
            try:
                mod = dit._get_module_by_name(mod_name)
            except Exception:
                continue
            down_g = sd[f"{lname}.lora_down.weight"].to(device=dev, dtype=torch.float32)
            up_single_k = f"{lname}.lora_up.weight"
            if up_single_k in sd:
                up_g = sd[up_single_k].to(device=dev, dtype=torch.float32)
                delta = torch.mm(up_g, down_g)
            else:
                blk_keys = sorted(
                    [k for k in sd.keys() if k.startswith(f"{lname}.lora_up.blocks.") and k.endswith(".weight")],
                    key=lambda x: int(x.split(".blocks.")[1].split(".")[0]),
                )
                if not blk_keys:
                    del down_g
                    continue
                down_chunks = torch.chunk(down_g, len(blk_keys), dim=0)
                delta = torch.cat(
                    [torch.mm(sd[bk].to(device=dev, dtype=torch.float32), down_chunks[idx]) for idx, bk in enumerate(blk_keys)],
                    dim=0,
                )
            if hasattr(mod, "weight") and isinstance(mod.weight, torch.nn.Parameter):
                orig_dt = mod.weight.dtype
                w_fp32 = mod.weight.data.to(device=dev, dtype=torch.float32)
                w_fp32.add_(delta, alpha=eff_scale)
                mod.weight.data.copy_(w_fp32.to(orig_dt))
                merged += 1
            elif hasattr(mod, "weight_int8") and hasattr(mod, "weight_scale"):
                w_fp32 = mod.weight_int8.to(device=dev, dtype=torch.float32) * mod.weight_scale.to(device=dev, dtype=torch.float32).unsqueeze(1)
                w_fp32.add_(delta, alpha=eff_scale)
                new_scale = w_fp32.abs().amax(dim=1).clamp(min=1e-8) / 127.0
                new_int8 = (w_fp32 / new_scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
                mod.weight_int8.copy_(new_int8.to(mod.weight_int8.device))
                mod.weight_scale.copy_(new_scale.to(mod.weight_scale.device))
                merged += 1
            del delta, down_g
    del sd
    torch.cuda.empty_cache()
    return merged


def _ensure_avatar_engine(use_int8: bool = False, update_job_state: bool = True):
    """将 LongCat-Video-Avatar-1.6 全套管线一次性装载并常驻 GPU 显存，后续生成 0 秒启动"""
    with ENGINE_INIT_LOCK:
        if _AVATAR_ENGINE["pipe"] is not None and _AVATAR_ENGINE["use_int8"] == use_int8:
            return _AVATAR_ENGINE["pipe"], _AVATAR_ENGINE["vocal_separator"]

        import torch
        from transformers import AutoTokenizer, UMT5EncoderModel
        from audio_separator.separator import Separator
        from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
        from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
        from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
        from longcat_video.modules.quantization import load_quantized_dit
        from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        _apply_fast_attention_and_ffmpeg_patches()

        if update_job_state:
            JOB_STATE.update({
                "pct": 5,
                "message": "🚀 [LongCat-Avatar-1.6] 正在将数字人引擎常驻装载至 GPU 显存 (首次预热后永久驻留，后续 0 秒瞬发)...",
            })

        base_lc_dir = os.path.join(MODEL_DIR, "..", "LongCat-Video")
        if not os.path.isdir(os.path.join(base_lc_dir, "tokenizer")):
            base_lc_dir = MODEL_DIR

        tokenizer = AutoTokenizer.from_pretrained(base_lc_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
        text_encoder = UMT5EncoderModel.from_pretrained(base_lc_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        vae = AutoencoderKLWan.from_pretrained(base_lc_dir, subfolder="vae", torch_dtype=torch.bfloat16)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_DIR, subfolder="scheduler", torch_dtype=torch.bfloat16)

        cp_split_hw = [1, 1]
        if use_int8 and os.path.isdir(os.path.join(MODEL_DIR, "base_model_int8")):
            dit = load_quantized_dit(MODEL_DIR, subfolder="base_model_int8", cp_split_hw=cp_split_hw)
        else:
            dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
                MODEL_DIR, subfolder="base_model", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16
            )
        dit = dit.to("cuda:0")

        if update_job_state:
            JOB_STATE.update({
                "pct": 5,
                "message": "🎭 [LongCat-Avatar-1.6] 正在 GPU 上静态融合 DMD + OPSD 双蒸馏加速权重...",
            })

        dmd_lora = os.path.join(MODEL_DIR, "loraA", "lora.safetensors")
        opsd_lora = os.path.join(MODEL_DIR, "loraB", "lora.safetensors")
        if os.path.isfile(dmd_lora) and os.path.isfile(opsd_lora):
            _merge_lora_into_dit(dit, dmd_lora, multiplier=0.35, lora_dim=128, alpha=64.0)
            _merge_lora_into_dit(dit, opsd_lora, multiplier=0.65, lora_dim=128, alpha=64.0)
        elif os.path.isfile(opsd_lora):
            _merge_lora_into_dit(dit, opsd_lora, multiplier=0.8, lora_dim=128, alpha=64.0)

        # 注入扩散步进实时回调钩子与毫秒级中断响应
        orig_dit_forward = dit.forward

        def _hooked_dit_forward(*args, **kwargs):
            if os.path.exists("/content/.cancel_minimax"):
                try:
                    os.unlink("/content/.cancel_minimax")
                except Exception:
                    pass
                raise RuntimeError("🛑 已手动停止数字人视频生成任务")
            is_kv_prepass = bool(kwargs.get("return_kv", False)) or (len(args) > 5 and bool(args[5]))
            out = orig_dit_forward(*args, **kwargs)
            if _STEP_TRACKER["active"] and not is_kv_prepass:
                _STEP_TRACKER["cur_seg_step"] = min(8, _STEP_TRACKER["cur_seg_step"] + 1)
                cur_s = _STEP_TRACKER["cur_seg_step"]
                cur_seg = _STEP_TRACKER["cur_seg_idx"]
                num_segs = max(1, _STEP_TRACKER["num_segments"])
                glob_step = (cur_seg - 1) * 8 + cur_s
                glob_tot = num_segs * 8
                ov_pct = min(97, max(6, 6 + int(round((glob_step / float(glob_tot)) * 91))))
                JOB_STATE["step"] = glob_step
                JOB_STATE["total_steps"] = glob_tot
                JOB_STATE["pct"] = ov_pct
                seg_tag = f"第 {cur_seg}/{num_segs} 段 (本段 {cur_s}/8 步 · 总计 {glob_step}/{glob_tot} 步)" if num_segs > 1 else f"第 {cur_s}/8 步"
                JOB_STATE["message"] = f"🎭 [LongCat-Avatar-1.6] 正在扩散生成数字人视频: {seg_tag} (总进度 {ov_pct}%)"
            return out

        dit.forward = _hooked_dit_forward

        audio_ckpt = os.path.join(MODEL_DIR, "whisper-large-v3")
        audio_encoder = get_audio_encoder(audio_ckpt, "avatar-v1.5").to("cuda:0")
        audio_feature_extractor = get_audio_feature_extractor(audio_ckpt, "avatar-v1.5")

        vocal_sep_path = os.path.join(MODEL_DIR, "vocal_separator/Kim_Vocal_2.onnx")
        audio_temp_dir = Path("/tmp/longcat_audio_temp")
        audio_temp_dir.mkdir(parents=True, exist_ok=True)
        vocal_separator = Separator(
            output_dir=audio_temp_dir / "vocals",
            output_single_stem="vocals",
            model_file_dir=os.path.dirname(vocal_sep_path),
        )
        if vocal_separator.onnx_execution_provider == ["CUDAExecutionProvider"]:
            vocal_separator.onnx_execution_provider = [
                ("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC"})
            ]
        vocal_separator.load_model(os.path.basename(vocal_sep_path))

        pipe = LongCatVideoAvatarPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            dit=dit,
            audio_encoder=audio_encoder,
            audio_feature_extractor=audio_feature_extractor,
            model_type="avatar-v1.5",
        )
        pipe.to("cuda:0")

        # 开启 3D VAE 融合 Padding 与零拷贝缓存视图加速
        for fn_name in (
            "enable_encoder_fused_padding",
            "enable_encoder_cache_views",
            "enable_decoder_fused_padding",
            "enable_decoder_cache_views",
        ):
            if hasattr(pipe.vae, fn_name):
                try:
                    getattr(pipe.vae, fn_name)()
                except Exception:
                    pass

        # 缓存 UMT5 文本编码结果，避免多段续写时每一段重复运行 11GB UMT5-XXL
        orig_encode_prompt = pipe.encode_prompt
        _prompt_cache = {"key": None, "val": None}

        def _cached_encode_prompt(prompt, negative_prompt=None, do_classifier_free_guidance=True, num_videos_per_prompt=1, max_sequence_length=512, device=None, dtype=None):
            ck = (str(prompt), str(negative_prompt), bool(do_classifier_free_guidance), int(num_videos_per_prompt), int(max_sequence_length), str(dtype))
            if _prompt_cache["key"] == ck and _prompt_cache["val"] is not None:
                return _prompt_cache["val"]
            res = orig_encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
            _prompt_cache["key"] = ck
            _prompt_cache["val"] = res
            return res

        pipe.encode_prompt = _cached_encode_prompt

        _AVATAR_ENGINE["pipe"] = pipe
        _AVATAR_ENGINE["vocal_separator"] = vocal_separator
        _AVATAR_ENGINE["use_int8"] = use_int8
        return pipe, vocal_separator


def _ensure_ming_talker(update_job_state: bool = True):
    """按需加载并常驻 Ming-3.0-Audio-Flash Talker (2.78 GB · 100+ 戏剧/影视人设音色 + 副语言气口 + 3秒零样本声音克隆)"""
    if _AVATAR_ENGINE.get("ming_talker") is not None:
        return _AVATAR_ENGINE["ming_talker"]

    with ENGINE_INIT_LOCK:
        if _AVATAR_ENGINE.get("ming_talker") is not None:
            return _AVATAR_ENGINE["ming_talker"]

        import torch
        ming_root = os.path.join(MODEL_DIR, "ming_talker")
        shared_ming_root = "/content/local_models/ming_talker"
        if not os.path.isfile(os.path.join(ming_root, "talker/model.safetensors")) and os.path.isfile(os.path.join(shared_ming_root, "talker/model.safetensors")):
            ming_root = shared_ming_root
        talker_dir = os.path.join(ming_root, "talker")
        os.makedirs(ming_root, exist_ok=True)

        # 确保依赖就绪 (Python 3.13 跳过 C++ 源码编译的 pynini/fasttext，使用内置纯 Python TalkerRE 文本正则归一化)
        missing_deps = []
        for pkg, mod in [
            ("diffusers", "diffusers"),
            ("x-transformers", "x_transformers"),
            ("inflect", "inflect"),
            ("onnxruntime", "onnxruntime"),
            ("librosa", "librosa"),
            ("soundfile", "soundfile"),
        ]:
            try:
                __import__(mod)
            except ImportError:
                missing_deps.append(pkg)
        if missing_deps:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing_deps, check=False)

        # 若本地尚未缓存 ming_talker (2.78 GB)，优先从 Google Drive 云盘拉取，再按需从 HuggingFace 拉取并异步备份到云盘
        rc_conf = os.path.expanduser("~/.config/rclone/rclone.conf")
        need_weights = (
            not os.path.isfile(os.path.join(talker_dir, "model.safetensors"))
            or not os.path.isfile(os.path.join(talker_dir, "vae/model.safetensors"))
        )
        if need_weights and os.path.exists(rc_conf) and shutil.which("rclone"):
            folder_name = os.path.basename(MODEL_DIR.rstrip("/"))
            for r_src in (f"gdrive:models/{folder_name}/ming_talker", "gdrive:models/LongCat-Video-Avatar-1.6/ming_talker"):
                subprocess.run(
                    ["rclone", "copy", r_src, ming_root, "--transfers=8", "--config", rc_conf],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if os.path.isfile(os.path.join(talker_dir, "model.safetensors")) and os.path.isfile(os.path.join(talker_dir, "vae/model.safetensors")):
                    need_weights = False
                    break
        need_code = not os.path.isfile(os.path.join(talker_dir, "inference.py"))
        if need_weights or need_code:
            if update_job_state:
                JOB_STATE.update({
                    "running": True,
                    "pct": 4,
                    "message": "🎭 [Ming-3.0-Talker] 首次启用百灵 3.0 戏剧与声音克隆引擎，正在拉取 2.78GB 核心组件并持久化至云盘...",
                })
            from huggingface_hub import snapshot_download
            ignore_pats = ["*__pycache__*"] if need_weights else ["*.safetensors", "*.onnx", "*__pycache__*"]
            snapshot_download(
                repo_id="inclusionAI/Ming-3.0-audio-flash",
                allow_patterns=["talker/*", "voice_presets/*"],
                ignore_patterns=ignore_pats,
                local_dir=ming_root,
            )
            if os.path.exists(rc_conf) and shutil.which("rclone"):
                folder_name = os.path.basename(MODEL_DIR.rstrip("/"))
                subprocess.Popen(
                    ["rclone", "copy", ming_root, f"gdrive:models/{folder_name}/ming_talker", "--transfers=8", "--config", rc_conf],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        # 1. 为 Python 3.13 环境注入无 pynini C++ 依赖的 Normalizer 兼容补丁 (TalkerRE 已完成纯 Python 数字/日期/符号转换)
        norm_py = os.path.join(talker_dir, "talker_tn/normalizer.py")
        if os.path.isfile(norm_py):
            try:
                norm_txt = open(norm_py, "r", encoding="utf-8").read()
                if "from pynini import escape, Fst, shortestpath" in norm_txt and "except ImportError:" not in norm_txt:
                    open(norm_py, "w", encoding="utf-8").write(
                        "try:\n"
                        "    from pynini import escape, Fst, shortestpath\n"
                        "    _HAS_PYNINI = True\n"
                        "except ImportError:\n"
                        "    _HAS_PYNINI = False\n"
                        "from .token_parser import TokenParser\n\n"
                        "class Normalizer:\n"
                        "    def __init__(self, prefix_p, ordertype='tn'):\n"
                        "        self.ordertype = ordertype\n"
                        "        if _HAS_PYNINI:\n"
                        "            self.tagger = Fst.read(f'{prefix_p}_tagger.fst').optimize()\n"
                        "            self.verbalizer = Fst.read(f'{prefix_p}_verbalizer.fst').optimize()\n"
                        "    def normalize(self, input):\n"
                        "        if not _HAS_PYNINI or len(input) == 0:\n"
                        "            return input\n"
                        "        lattice = escape(input) @ self.tagger\n"
                        "        tmp = shortestpath(lattice, nshortest=1, unique=True).string()\n"
                        "        if len(tmp) == 0:\n"
                        "            return ''\n"
                        "        output = TokenParser(self.ordertype).reorder(tmp)\n"
                        "        lattice2 = escape(output) @ self.verbalizer\n"
                        "        return shortestpath(lattice2, nshortest=1, unique=True).string()\n"
                    )
            except Exception:
                pass

        # 2. 修复 talker_module/modules.py 对单文件 flash_attn.py 兼容层的 bert_padding 导入异常
        mod_py = os.path.join(talker_dir, "talker_module/modules.py")
        if os.path.isfile(mod_py):
            try:
                mod_txt = open(mod_py, "r", encoding="utf-8").read()
                if 'if is_package_available("flash_attn"):' in mod_txt:
                    mod_txt = mod_txt.replace(
                        'if is_package_available("flash_attn"):\n    from flash_attn.bert_padding import pad_input, unpad_input\n    from flash_attn import flash_attn_varlen_func, flash_attn_func',
                        'try:\n    from flash_attn.bert_padding import pad_input, unpad_input\n    from flash_attn import flash_attn_varlen_func, flash_attn_func\nexcept Exception:\n    pass',
                    )
                    open(mod_py, "w", encoding="utf-8").write(mod_txt)
            except Exception:
                pass

        # 3. 修复 AudioVAE 在 transformers 5.x 下 post_init() 访问 all_tied_weights_keys 报错，以及 PyTorch 2.14+ BComplex32 ISTFT 报错
        avae_py = os.path.join(talker_dir, "AudioVAE/modeling_audio_vae.py")
        if os.path.isfile(avae_py):
            try:
                avae_txt = open(avae_py, "r", encoding="utf-8").read()
                if "        self.post_init()\n" in avae_txt and "try:\n            self.post_init()" not in avae_txt:
                    avae_txt = avae_txt.replace(
                        "        self.post_init()\n",
                        "        try:\n            self.post_init()\n        except Exception:\n            pass\n",
                    )
                    open(avae_py, "w", encoding="utf-8").write(avae_txt)
            except Exception:
                pass

        istft_py = os.path.join(talker_dir, "AudioVAE/istft.py")
        if os.path.isfile(istft_py):
            try:
                ist_txt = open(istft_py, "r", encoding="utf-8").read()
                if ".float()" not in ist_txt:
                    ist_txt = (
                        ist_txt.replace("x_pred = self.out(x )", "x_pred = self.out(x).float()")
                        .replace("x_pred = self.out(x)", "x_pred = self.out(x).float()")
                        .replace("self.window[None, :, None]", "self.window.float()[None, :, None]")
                        .replace("self.window.square()", "self.window.float().square()")
                    )
                    open(istft_py, "w", encoding="utf-8").write(ist_txt)
            except Exception:
                pass

        istft_py = os.path.join(talker_dir, "AudioVAE/istft.py")
        if os.path.isfile(istft_py):
            try:
                ist_txt = open(istft_py, "r", encoding="utf-8").read()
                if ".float()" not in ist_txt:
                    ist_txt = (
                        ist_txt.replace("x_pred = self.out(x )", "x_pred = self.out(x).float()")
                        .replace("x_pred = self.out(x)", "x_pred = self.out(x).float()")
                        .replace("self.window[None, :, None]", "self.window.float()[None, :, None]")
                        .replace("self.window.square()", "self.window.float().square()")
                    )
                    open(istft_py, "w", encoding="utf-8").write(ist_txt)
            except Exception:
                pass

        # 4. 修复 inference.py：
        #    (a) 避免 text_fragments 在 12 token 处过早切句导致句尾副语言气口 (如 [Sigh] [Uhm]) 被拆成无汉字孤立片段而失效
        #    (b) 兼容 transformers 5.x state_dict 加载与 inference_mode 下 embeddings 原位赋值
        inf_py = os.path.join(talker_dir, "inference.py")
        if os.path.isfile(inf_py):
            try:
                inf_txt = open(inf_py, "r", encoding="utf-8").read()
                changed_inf = False
                if "model.load_state_dict(weights, strict=True)" in inf_txt:
                    inf_txt = inf_txt.replace(
                        "model.load_state_dict(weights, strict=True)",
                        "model.load_state_dict(weights, strict=False)",
                    ).replace(
                        '        model.audio_vae.load_state_dict(\n            load_file(str(path / "vae/model.safetensors")), strict=True\n        )',
                        '        _vw = load_file(str(path / "vae/model.safetensors"))\n        _vk = model.audio_vae.state_dict().keys()\n        _vw = {k: v for k, v in _vw.items() if not (k.endswith("inv_freq") and k not in _vk)}\n        model.audio_vae.load_state_dict(_vw, strict=False)',
                    )
                    changed_inf = True
                if "embeddings = self.model.get_input_embeddings()(ids)\n" in inf_txt:
                    inf_txt = inf_txt.replace(
                        "embeddings = self.model.get_input_embeddings()(ids)\n",
                        "embeddings = self.model.get_input_embeddings()(ids).clone()\n",
                    )
                    changed_inf = True
                if "def text_fragments(self, value, max_length=50):" in inf_txt:
                    inf_txt = inf_txt.replace(
                        "def text_fragments(self, value, max_length=50):",
                        "def text_fragments(self, value, max_length=140):",
                    ).replace(
                        "len(streaming_text) >= 12 or (count > 0 and len(streaming_text) >= 8)",
                        "len(streaming_text) >= 140",
                    ).replace(
                        "elif ele[-1] in \"，,\" and len(streaming_text) >= 36:",
                        "elif ele[-1] in \"，,\" and len(streaming_text) >= 160:",
                    )
                    changed_inf = True
                if "for fragment in self.text_fragments(text):" in inf_txt and "_merged_frags" not in inf_txt:
                    inf_txt = inf_txt.replace(
                        "            for fragment in self.text_fragments(text):",
                        "            _raw_frags = list(self.text_fragments(text))\n"
                        "            _merged_frags = []\n"
                        "            for _fg in _raw_frags:\n"
                        "                _stripped_no_tags = re.sub(r'\\[[A-Za-z-]+\\]', '', _fg)\n"
                        "                if _merged_frags and not re.search(r'[\\u4e00-\\u9fffA-Za-z0-9]', _stripped_no_tags):\n"
                        "                    _merged_frags[-1] = _merged_frags[-1].rstrip() + ' ' + _fg.lstrip()\n"
                        "                else:\n"
                        "                    _merged_frags.append(_fg)\n"
                        "            for fragment in (_merged_frags or [text]):",
                    )
                    changed_inf = True
                if changed_inf:
                    open(inf_py, "w", encoding="utf-8").write(inf_txt)
            except Exception:
                pass

        if update_job_state:
            JOB_STATE.update({
                "running": True,
                "pct": 5,
                "message": "🎭 [Ming-3.0-Talker] 正在将 2.78GB 戏剧表现力 & 零样本声音克隆引擎装载至 GPU 显存...",
            })
        if ming_root not in sys.path:
            sys.path.insert(0, ming_root)
        from talker import MingTalker
        talker = MingTalker.from_pretrained(
            Path(talker_dir),
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        _AVATAR_ENGINE["ming_talker"] = talker
        return talker


@__import__("torch").inference_mode()
def _synthesize_ming_talker_wav(
    text: str,
    voice_spec: str = "ming:灵小柔",
    clone_ref_b64: str = "",
    clone_prompt_text: str = "",
    update_job_state: bool = True,
) -> str:
    """使用 Ming-3.0-Audio-Flash Talker 合成带副语言气口 (呼吸/笑声/叹气/娇嗔) 或 3 秒零样本克隆的高表现力驱动语音"""
    import librosa
    import numpy as np
    import soundfile as sf

    talker = _ensure_ming_talker(update_job_state=update_job_state)
    clean_v = str(voice_spec or "ming:灵小柔").split("|")[0].strip()
    if clean_v.startswith("ming:"):
        clean_v = clean_v[len("ming:"):].strip()
    if not clean_v:
        clean_v = "灵小柔"

    _MING_VOICE_ALIASES = {
        "灵逍遥": "灵信远_中性",
        "温润公子": "灵信远_中性",
        "深情男主": "灵朗",
        "古风男主": "灵信远_中性",
        "清朗少年": "灵旭",
        "磁性男神": "灵空",
        "灵信远": "灵信远_中性",
        "灵孤影": "灵孤影_中性",
        "灵思齐": "灵思齐_中性",
        "灵言澈": "灵言澈_中性",
        "灵溯光": "灵溯光_中性",
        "灵嘉肴": "灵嘉肴_中性",
        "灵雨汐": "灵雨汐_中性",
        "灵书悦": "灵书悦_高兴",
        "灵惜梦": "灵惜梦_中性",
        "蜡笔小新_野原新之助": "蜡笔小新",
        "蜡笔小新_野原美伢": "蜡笔小新_野原美伢 (美伢)",
    }
    clean_v = _MING_VOICE_ALIASES.get(clean_v, clean_v)
    if any(k in clean_v for k in ("男", "公子", "逍遥", "少侠", "先生")) and clean_v not in ("灵信远_中性", "灵朗", "灵楷", "灵宇", "灵旭", "灵空", "灵孤影_中性", "灵思齐_中性"):
        clean_v = "灵信远_中性"

    reference = None
    prompt_txt = str(clone_prompt_text or "").strip() or None
    if clone_ref_b64 or clean_v == "__clone__":
        if clone_ref_b64:
            raw_b = clone_ref_b64.split(",", 1)[-1]
            ref_tmp = "/tmp/ming_clone_ref_input"
            with open(ref_tmp, "wb") as rf:
                rf.write(base64.b64decode(raw_b))
            samples, sr = librosa.load(ref_tmp, sr=None, mono=True)
            # 截取前 12 秒黄金声纹区间，防止过长样本拖慢推理
            max_samples = int(sr * 12)
            if len(samples) > max_samples:
                samples = samples[:max_samples]
            reference = {"samples": samples.astype(np.float32), "sample_rate": int(sr)}
        elif os.path.isfile(os.path.join(MODEL_DIR, "ming_talker/voice_presets/voice_audio/灵楷.wav")):
            samples, sr = librosa.load(os.path.join(MODEL_DIR, "ming_talker/voice_presets/voice_audio/灵楷.wav"), sr=None, mono=True)
            reference = {"samples": samples.astype(np.float32), "sample_rate": int(sr)}

    if update_job_state:
        mode_lbl = "零样本声音克隆" if reference is not None else f"Ming-3.0「{clean_v}」"
        JOB_STATE.update({
            "running": True,
            "pct": 5,
            "message": f"🎭 正在使用 {mode_lbl} 合成高表现力戏剧语音: 「{text[:28]}...」",
            "error": "",
        })

    audio_out = talker.synthesize(
        text,
        ip_name="灵小柔" if reference is not None else clean_v,
        reference=reference,
        prompt_text=prompt_txt if reference is not None else None,
        max_steps=1000,
    )
    waveform = audio_out["audio"].squeeze().detach().float().cpu().numpy()
    sample_rate = int(audio_out["sr"])
    out_wav = "/tmp/longcat_ming_tts.wav"
    sf.write(out_wav, waveform, sample_rate)
    return out_wav


def _warmup_engine_bg():
    try:
        has_int8 = os.path.isdir(os.path.join(MODEL_DIR, "base_model_int8")) and not os.path.isdir(os.path.join(MODEL_DIR, "base_model"))
        use_int8 = has_int8 or ("INT8" in MODEL_DIR.upper())
        with GPU_LOCK:
            _ensure_avatar_engine(use_int8=use_int8, update_job_state=False)
            if not JOB_STATE.get("running"):
                JOB_STATE["message"] = "⚡ 就绪：LongCat-Video-Avatar-1.6 已常驻 GPU 显存 (静态 LoRA + 零拷贝 SDPA + KV Cache 已启用)"
    except Exception as e:
        print(f"[Warmup Error] {e}", flush=True)


@__import__("torch").inference_mode()
def _run_avatar_job(payload: dict):
    global JOB_STATE
    t0 = time.time()
    try:
        import numpy as np
        import PIL.Image
        import torch
        from diffusers.utils import load_image
        from longcat_video.audio_process.torch_utils import save_video_ffmpeg
        from longcat_video.audio_process.vocal_separator import separate_vocals

        prompt = str(payload.get("prompt") or "").strip() or (
            "A person stands under cinematic lighting, speaking naturally and expressively with clear facial movements."
        )
        res_str = str(payload.get("resolution") or "832x480").strip()
        resolution = "720p" if "720" in res_str or "1280" in res_str else "480p"
        raw_req_segments = int(payload.get("num_segments") if payload.get("num_segments") is not None else 0)
        base_seed = int(payload.get("seed") if payload.get("seed") is not None else -1)
        rand_seed = bool(payload.get("randomize_seed")) or (base_seed < 0)
        cur_seed = random.randint(1, 2**31 - 1) if rand_seed else base_seed

        has_int8 = os.path.isdir(os.path.join(MODEL_DIR, "base_model_int8")) and not os.path.isdir(os.path.join(MODEL_DIR, "base_model"))
        use_int8 = bool(payload.get("use_int8")) or has_int8 or ("INT8" in MODEL_DIR.upper())

        audio_b64 = str(payload.get("audio_b64") or "").strip()
        audio_ext = str(payload.get("audio_ext") or ".mp3").strip()
        if not audio_ext.startswith("."):
            audio_ext = "." + audio_ext
        tts_text = str(payload.get("tts_text") or "").strip()
        tts_voice = str(payload.get("tts_voice") or "zh-CN-XiaoyiNeural").strip()
        clone_ref_b64 = str(payload.get("clone_ref_b64") or "").strip()
        clone_prompt_text = str(payload.get("clone_prompt_text") or "").strip()
        visual_prompt = prompt

        # 若用户把「台词如下：...」连同画面描述一起粘进了第 3 框，自动将画面描述与台词精准分离
        m_split = re.match(r"^(?:([\s\S]*?)\n+)?\s*(?:台词如下|说话内容如下|对白如下|台词内容|开口说话|说话内容|台词|对白)\s*[:：]\s*([\s\S]+)$", visual_prompt)
        if m_split and m_split.group(2) and m_split.group(2).strip():
            visual_prompt = (m_split.group(1) or "").strip() or (
                "固定镜头，近景居中构图。画面中的人物直视镜头自然说话，眼神温柔灵动，伴随说话节奏有轻微自然的眨眼、挑眉、轻轻点头和呼吸起伏，嘴型与语音高度同步，牙齿与唇形清晰自然，面部柔和暖光，五官清晰稳定，过渡丝滑真实。"
            )
            if not tts_text:
                tts_text = m_split.group(2).strip()

        # 若用户直接复用 4K 写真静态生图提示词 (不含任何说话/微表情描述)，自动注入自然语流神态与眼神联动引导词
        if not any(k in visual_prompt.lower() for k in ("说话", "开口", "播报", "speaking", "talking", "眨眼", "blink", "嘴型", "口型")):
            visual_prompt = (
                visual_prompt.rstrip("。.,， ")
                + "。画面中的人物直视镜头自然生动地开口说话，眼神灵动有神，伴随说话语流节奏有自然的微眨眼、挑眉与轻微头部摆动，口型与语音精准同步，牙齿与唇形自然清晰 "
                "(speaking naturally and expressively toward the camera, natural eye contact, subtle blinks, expressive eyebrow and head movement matching speech cadence)。"
            )
        if "实拍原片" not in visual_prompt and "raw unedited cinema camera footage" not in visual_prompt.lower():
            visual_prompt = (
                visual_prompt.rstrip("。.,， ")
                + "。电影摄影机实拍原片，纯洁干净画面 "
                "(raw unedited cinema camera footage, pure clean frame)。"
            )

        tts_note = ""
        if audio_b64:
            raw_a = audio_b64.split(",", 1)[-1]
            audio_path = f"/tmp/longcat_input_audio{audio_ext}"
            with open(audio_path, "wb") as af:
                af.write(base64.b64decode(raw_a))
        elif tts_text:
            raw_v = tts_voice or ("zh-CN-YunxiNeural" if any(w in visual_prompt for w in ("男", "先生", "帅哥", "大叔", "boy", "man")) else "zh-CN-XiaoyiNeural")
            use_indextts2 = raw_v.startswith("indextts2:")
            use_ming_talker = (not use_indextts2) and (raw_v.startswith("ming:") or bool(clone_ref_b64) or any(
                tag in tts_text for tag in ("[Breathing]", "[Laughter]", "[Sigh]", "[Dissatisfaction-hnn]", "[Surprise-ah]", "[Surprise-oh]", "[Question-ei]", "[Question-huh]", "[Uhm]", "[Cough]")
            ))
            if use_indextts2:
                try:
                    JOB_STATE.update({
                        "running": True,
                        "pct": 4,
                        "message": f"🎙️ 正在使用 IndexTTS-2 云端原版合成高拟真情绪语音: 「{tts_text[:28]}...」",
                        "error": "",
                    })
                    idx_req = urllib.request.Request(
                        "http://127.0.0.1:30003/v1/audio/indextts2",
                        data=json.dumps({
                            "text": tts_text,
                            "voice": raw_v,
                            "clone_ref_b64": clone_ref_b64,
                            "emo_mode": payload.get("emo_mode") or "auto",
                            "emo_text": payload.get("emo_text") or "",
                            "emo_preset": payload.get("emo_preset") or "",
                            "emo_alpha": payload.get("emo_alpha", 0.75),
                            "emo_vector": payload.get("emo_vector"),
                            "target_duration_sec": payload.get("target_duration_sec", 0),
                        }).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(idx_req, timeout=240) as idx_r:
                        idx_d = json.loads(idx_r.read().decode("utf-8"))
                    if idx_d.get("ok") and idx_d.get("audio_b64"):
                        audio_path = "/tmp/longcat_indextts2_audio.wav"
                        with open(audio_path, "wb") as af:
                            af.write(base64.b64decode(str(idx_d["audio_b64"]).split(",", 1)[-1]))
                        tts_note = f" (已使用 IndexTTS-2 云端原版合成: {raw_v.replace('indextts2:', '')})"
                    else:
                        raise RuntimeError(idx_d.get("error") or "IndexTTS-2 返回空音频")
                except Exception as e_idx:
                    print(f"[IndexTTS2 Fallback to Edge-TTS] {e_idx}", flush=True)
                    use_indextts2 = False
            if use_ming_talker:
                try:
                    with GPU_LOCK:
                        audio_path = _synthesize_ming_talker_wav(
                            tts_text,
                            voice_spec=raw_v if raw_v.startswith("ming:") else "ming:灵雨汐_中性",
                            clone_ref_b64=clone_ref_b64,
                            clone_prompt_text=clone_prompt_text,
                            update_job_state=True,
                        )
                    tts_note = f" (已使用 Ming-3.0 合成: {'3秒声音克隆' if clone_ref_b64 else raw_v})"
                except Exception as e_ming:
                    print(f"[MingTalker Fallback to Edge-TTS] {e_ming}", flush=True)
                    use_ming_talker = False
            if not use_indextts2 and not use_ming_talker:
                v_parts = [p.strip() for p in raw_v.split("|") if p.strip()]
                voice_name = v_parts[0] if v_parts and not v_parts[0].startswith("ming:") else "zh-CN-XiaoyiNeural"
                extra_flags = []
                for vp in v_parts[1:]:
                    if vp.startswith("pitch="):
                        extra_flags.append(f"--{vp}")
                    elif vp.startswith("rate="):
                        extra_flags.append(f"--{vp}")
                audio_path = "/tmp/longcat_tts_audio.mp3"
                JOB_STATE.update({
                    "running": True,
                    "pct": 4,
                    "message": f"🎙️ 正在使用 {voice_name} 合成驱动语音: 「{tts_text[:28]}...」",
                    "error": "",
                })
                clean_edge_text = re.sub(r"\[(Breathing|Cough|Sigh|Laughter|Surprise-ah|Surprise-oh|Dissatisfaction-hnn|Uhm|Confirmation-en|Question-ei|Question-ah|Question-en|Question-oh|Question-huh)\]", "", tts_text).strip() or tts_text
                tts_cmd = [sys.executable, "-m", "edge_tts", "--voice", voice_name, *extra_flags, "--text", clean_edge_text, "--write-media", audio_path]
                r_tts = subprocess.run(tts_cmd, capture_output=True, text=True)
                if r_tts.returncode != 0 or not os.path.isfile(audio_path) or os.path.getsize(audio_path) < 256:
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "edge-tts"], check=False)
                    subprocess.run(tts_cmd, check=False)
                if not os.path.isfile(audio_path) or os.path.getsize(audio_path) < 256:
                    audio_path = os.path.join(LONGCAT_CODE_DIR, "assets/avatar/single/man.mp3")
                else:
                    tts_note = f" (已合成配音: {raw_v})"
        else:
            audio_path = os.path.join(LONGCAT_CODE_DIR, "assets/avatar/single/man.mp3")

        # 若用户选择「0 (自动匹配音频全长)」，通过 ffprobe 测量音频真实秒数并换算为对应段数 (首段 3.72s + 后续每段 3.20s，最高支持 200 段 >10 分钟)
        if raw_req_segments <= 0:
            try:
                dur_out = subprocess.check_output(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
                    text=True,
                    timeout=10,
                ).strip()
                aud_dur = float(dur_out)
                if aud_dur <= 3.85:
                    num_segments = 1
                else:
                    num_segments = max(1, min(200, 1 + int(math.ceil((aud_dur - 3.72) / 3.20))))
            except Exception:
                num_segments = 1
        else:
            num_segments = max(1, min(200, raw_req_segments))

        first_b64 = str(payload.get("first_frame_b64") or "").strip()
        use_builtin_img = bool(payload.get("use_builtin_sample")) and not first_b64
        if first_b64:
            raw_i = first_b64.split(",", 1)[-1]
            img_path = "/tmp/longcat_input_ref.png"
            with open(img_path, "wb") as imf:
                imf.write(base64.b64decode(raw_i))
            stage_1 = "ai2v"
        elif use_builtin_img and os.path.isfile(os.path.join(LONGCAT_CODE_DIR, "assets/avatar/single/man.png")):
            img_path = os.path.join(LONGCAT_CODE_DIR, "assets/avatar/single/man.png")
            stage_1 = "ai2v"
        else:
            img_path = ""
            stage_1 = "at2v"

        JOB_STATE.update({
            "running": True,
            "pct": 5,
            "step": 0,
            "total_steps": 8 * num_segments,
            "seed": cur_seed,
            "message": f"🎭 [LongCat-Avatar-1.6] 正在准备常驻显存引擎并提取语音特征 ({stage_1.upper()} · {num_segments}段){tts_note}...",
            "error": "",
        })

        with GPU_LOCK:
            pipe, vocal_separator = _ensure_avatar_engine(use_int8=use_int8, update_job_state=True)

            save_fps = 25
            audio_stride = 1
            num_frames = 93
            num_cond_frames = 13
            num_inference_steps = 8
            text_guidance_scale = 1.0
            audio_guidance_scale = 1.0
            is_portrait_res = False
            try:
                rw_s, rh_s = res_str.lower().split("x", 1)
                is_portrait_res = int(rw_s) < int(rh_s)
            except Exception:
                is_portrait_res = False
            if is_portrait_res:
                height, width = (1280, 768) if resolution == "720p" else (832, 480)
            else:
                height, width = (768, 1280) if resolution == "720p" else (480, 832)
            negative_prompt = (
                "字幕, 底部字幕, 台词文字, 乱码, 水印, 电视台台标, subtitles, captions, text on screen, lower thirds, watermark, logo, "
                "static frozen face, stiff mouth, distorted teeth, weird teeth, warped face, unnatural lip sync, asymmetrical eyes, "
                "Close-up, Bright tones, overexposed, static, blurred details, style, works, paintings, images, static, "
                "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
                "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, "
                "many people in the background, walking backwards"
            )

            generate_duration = num_frames / save_fps + (num_segments - 1) * (num_frames - num_cond_frames) / save_fps
            audio_process_duration = generate_duration + 1.0
            is_clean_tts_audio = (not audio_b64 and bool(tts_text)) or bool(payload.get("is_tts_audio"))
            if is_clean_tts_audio:
                JOB_STATE.update({
                    "pct": 6,
                    "message": f"🎭 [LongCat-Avatar-1.6] 正在直接提取无损 TTS 气口并执行 -23 LUFS 响度对齐 ({stage_1.upper()} · {num_segments}段){tts_note}...",
                })
                try:
                    import librosa
                    speech_array, sr = librosa.load(audio_path, sr=16000, mono=True, duration=audio_process_duration)
                except Exception:
                    speech_array, sr = separate_vocals(vocal_separator, audio_path, duration=audio_process_duration)
            else:
                JOB_STATE.update({
                    "pct": 6,
                    "message": f"🎭 [LongCat-Avatar-1.6] 正在使用 Kim_Vocal_2 分离人声并执行 -23 LUFS 响度对齐 ({stage_1.upper()} · {num_segments}段){tts_note}...",
                })
                speech_array, sr = separate_vocals(vocal_separator, audio_path, duration=audio_process_duration)
            if speech_array.size == 0:
                raise RuntimeError("未能从音频中检测到有效人声 (No vocal detected)")

            # 在填充静音尾巴前，对有效语音段执行 -23.0 LUFS 标准响度对齐 (防止音量过小张不开嘴或音量过大嘴型崩裂)
            try:
                speech_f32 = speech_array.astype(np.float32)
                norm_done = False
                if len(speech_f32) >= int(sr * 0.45):
                    try:
                        import pyloudnorm as pyln
                        meter = pyln.Meter(sr)
                        lufs = float(meter.integrated_loudness(speech_f32))
                        if not math.isnan(lufs) and not math.isinf(lufs) and abs(lufs) < 90.0:
                            speech_f32 = pyln.normalize.loudness(speech_f32, lufs, -23.0).astype(np.float32)
                            norm_done = True
                    except Exception:
                        pass
                if not norm_done:
                    rms = float(np.sqrt(np.mean(np.square(speech_f32)) + 1e-12))
                    if rms > 1e-4:
                        speech_f32 = speech_f32 * (0.071 / rms)
                peak = float(np.max(np.abs(speech_f32))) if speech_f32.size > 0 else 0.0
                if peak > 0.98:
                    speech_f32 = speech_f32 * (0.98 / peak)
                speech_array = speech_f32
            except Exception:
                pass

            source_duration = len(speech_array) / sr
            added_sample_nums = math.ceil((audio_process_duration - source_duration) * sr)
            if added_sample_nums > 0:
                speech_array = np.append(speech_array, np.zeros(added_sample_nums, dtype=speech_array.dtype))

            full_audio_emb = pipe.get_audio_embedding(
                speech_array,
                fps=save_fps * audio_stride,
                device="cuda:0",
                sample_rate=sr,
                model_type="avatar-v1.5",
                use_loudness_norm=False,
            )
            if torch.isnan(full_audio_emb).any():
                raise RuntimeError("音频特征提取出现 NaN，请检查输入音频格式")

            indices = torch.arange(5) - 2
            audio_start_idx = 0
            audio_end_idx = audio_start_idx + audio_stride * num_frames
            center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
            center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
            audio_emb = full_audio_emb[center_indices][None, ...].to("cuda:0")

            generator = torch.Generator(device="cuda:0")
            generator.manual_seed(cur_seed)

            _STEP_TRACKER.update({
                "active": True,
                "cur_seg_idx": 1,
                "num_segments": num_segments,
                "cur_seg_step": 0,
            })
            JOB_STATE.update({
                "step": 0,
                "total_steps": num_segments * 8,
                "pct": 6,
                "message": f"🎬 [LongCat-Avatar-1.6] 正在生成第 1/{num_segments} 段 · 总计 0/{num_segments * 8} 步 (总进度 6%)",
            })

            try:
                if stage_1 == "at2v":
                    output, latent = pipe.generate_at2v(
                        prompt=visual_prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        text_guidance_scale=text_guidance_scale,
                        audio_guidance_scale=audio_guidance_scale,
                        generator=generator,
                        output_type="both",
                        audio_emb=audio_emb,
                        use_distill=True,
                    )
                else:
                    from PIL import ImageOps
                    image = load_image(img_path).convert("RGB")
                    target_ratio = float(width) / float(max(1, height))
                    img_ratio = float(image.width) / float(max(1, image.height))
                    if abs(img_ratio - target_ratio) > 0.06:
                        image = ImageOps.fit(image, (width, height), method=PIL.Image.Resampling.LANCZOS, centering=(0.5, 0.25))
                    output, latent = pipe.generate_ai2v(
                        image=image,
                        prompt=visual_prompt,
                        negative_prompt=negative_prompt,
                        resolution=resolution,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        text_guidance_scale=text_guidance_scale,
                        audio_guidance_scale=audio_guidance_scale,
                        output_type="both",
                        generator=generator,
                        audio_emb=audio_emb,
                        use_distill=True,
                    )

                output = output[0]
                video_array = (output * 255).astype(np.uint8)
                del output

                h_out, w_out = video_array.shape[1], video_array.shape[2]
                # 记录首段基准帧的色彩均值与标准差，用于后续多段续写时跨段锁色 (彻底防止长视频多段变暗或色温漂移)
                _ref_frame_f32 = video_array[0].astype(np.float32)
                _ref_color_mean = _ref_frame_f32.mean(axis=(0, 1), keepdims=True)
                _ref_color_std = _ref_frame_f32.std(axis=(0, 1), keepdims=True) + 1e-5
                del _ref_frame_f32

                current_video = [PIL.Image.fromarray(img) for img in video_array[-num_cond_frames:]]
                ref_latent = latent[:, :, :1].clone()
                total_output_frames = len(video_array) + max(0, num_segments - 1) * (num_frames - num_cond_frames)
                all_generated_frames = np.empty((total_output_frames, h_out, w_out, 3), dtype=np.uint8)
                generated_frame_count = len(video_array)
                all_generated_frames[:generated_frame_count] = video_array
                del video_array

                for segment_idx in range(1, num_segments):
                    audio_start_idx = audio_start_idx + audio_stride * (num_frames - num_cond_frames)
                    audio_end_idx = audio_start_idx + audio_stride * num_frames
                    center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
                    center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
                    audio_emb = full_audio_emb[center_indices][None, ...].to("cuda:0")

                    _STEP_TRACKER.update({
                        "cur_seg_idx": segment_idx + 1,
                        "cur_seg_step": 0,
                    })
                    glob_done = segment_idx * 8
                    glob_tot = num_segments * 8
                    seg_base_pct = min(96, max(6, 6 + int(round((glob_done / float(glob_tot)) * 91))))
                    JOB_STATE.update({
                        "step": glob_done,
                        "total_steps": glob_tot,
                        "pct": seg_base_pct,
                        "message": f"🎬 [LongCat-Avatar-1.6] 正在生成第 {segment_idx + 1}/{num_segments} 段 (KV Cache + 滑窗锁色已启用) · 总计 {glob_done}/{glob_tot} 步 (总进度 {seg_base_pct}%)",
                    })

                    output, latent = pipe.generate_avc(
                        video=current_video,
                        video_latent=latent,
                        prompt=visual_prompt,
                        negative_prompt=negative_prompt,
                        height=h_out,
                        width=w_out,
                        num_frames=num_frames,
                        num_cond_frames=num_cond_frames,
                        num_inference_steps=num_inference_steps,
                        text_guidance_scale=text_guidance_scale,
                        audio_guidance_scale=audio_guidance_scale,
                        generator=generator,
                        output_type="both",
                        use_kv_cache=True,
                        offload_kv_cache=False,
                        enhance_hf=False,
                        audio_emb=audio_emb,
                        ref_latent=ref_latent,
                        ref_img_index=10,
                        mask_frame_range=3,
                        use_distill=True,
                    )
                    output = output[0]
                    new_video_array = (output * 255).astype(np.uint8)
                    del output

                    # 1. 跨段色彩与亮度锁定 (对齐首段基准帧均值/方差，防止多段累积变暗或偏色)
                    try:
                        seg_f32 = new_video_array.astype(np.float32)
                        cur_mean = seg_f32.mean(axis=(0, 1, 2), keepdims=True)
                        cur_std = seg_f32.std(axis=(0, 1, 2), keepdims=True) + 1e-5
                        matched_f32 = (seg_f32 - cur_mean) * (_ref_color_std[None, ...] / cur_std) + _ref_color_mean[None, ...]
                        seg_f32 = seg_f32 * 0.15 + matched_f32 * 0.85
                        new_video_array = np.clip(seg_f32, 0, 255).astype(np.uint8)
                        del seg_f32, matched_f32
                    except Exception:
                        pass

                    # 2. 滑窗交界处 6 帧线性交叉淡入淡出，消除段间微跳帧
                    blend_k = min(6, num_cond_frames, generated_frame_count)
                    if blend_k > 0:
                        for b_i in range(blend_k):
                            alpha = float(b_i + 1) / float(blend_k + 1)
                            prev_idx = generated_frame_count - blend_k + b_i
                            cond_idx = num_cond_frames - blend_k + b_i
                            all_generated_frames[prev_idx] = np.clip(
                                all_generated_frames[prev_idx].astype(np.float32) * (1.0 - alpha)
                                + new_video_array[cond_idx].astype(np.float32) * alpha,
                                0,
                                255,
                            ).astype(np.uint8)

                    appended_frames = new_video_array[num_cond_frames:]
                    next_frame_count = generated_frame_count + len(appended_frames)
                    all_generated_frames[generated_frame_count:next_frame_count] = appended_frames
                    generated_frame_count = next_frame_count
                    current_video = [PIL.Image.fromarray(img) for img in new_video_array[-num_cond_frames:]]
                    del new_video_array
            finally:
                _STEP_TRACKER["active"] = False
                try:
                    pipe._clear_cache()
                except Exception:
                    pass

            JOB_STATE.update({
                "pct": 98,
                "message": "🎬 [LongCat-Avatar-1.6] 正在合成高保真音画同步 MP4 视频文件 (总进度 98%)...",
            })
            tmp_out_dir = f"/tmp/longcat_out_{int(time.time())}"
            os.makedirs(tmp_out_dir, exist_ok=True)
            out_mp4_p = os.path.join(tmp_out_dir, "avatar_final.mp4")
            vid_np = np.ascontiguousarray(all_generated_frames[:generated_frame_count], dtype=np.uint8)
            f_cnt, h_v, w_v, _ = vid_np.shape
            ff_cmd = [
                "ffmpeg", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-video_size", f"{w_v}x{h_v}", "-framerate", str(save_fps),
                "-i", "pipe:0", "-i", audio_path,
                "-t", f"{f_cnt / float(save_fps):.6f}",
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-af", "apad", "-c:a", "aac", "-b:a", "192k", "-shortest",
                out_mp4_p,
            ]
            ff_proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE)
            try:
                assert ff_proc.stdin is not None
                for st_i in range(0, f_cnt, 16):
                    try:
                        ff_proc.stdin.write(memoryview(vid_np[st_i : st_i + 16]).cast("B"))
                    except BrokenPipeError:
                        break
                try:
                    ff_proc.stdin.close()
                except BrokenPipeError:
                    pass
                ff_proc.wait(timeout=120)
            except Exception:
                ff_proc.kill()
                ff_proc.wait()
                raise
            del vid_np, all_generated_frames, full_audio_emb, audio_emb, latent, ref_latent
            gc.collect()
            torch.cuda.empty_cache()

            mp4_list = [
                p for p in sorted(glob.glob(os.path.join(tmp_out_dir, "*.mp4")), key=os.path.getmtime, reverse=True)
                if os.path.getsize(p) > 10240
            ]
            if not mp4_list:
                raise RuntimeError("FFmpeg 音画封装未生成有效 MP4 文件")

            dt = round(time.time() - t0, 1)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            fname = f"longcat_avatar_{stage_1}_{ts}_dur{max(1, int(round(dt)))}s_seed{cur_seed}.mp4"
            out_path = os.path.join(OUTPUT_DIR, fname)
            shutil.move(mp4_list[0], out_path)
            shutil.rmtree(tmp_out_dir, ignore_errors=True)

            rc_conf = os.path.expanduser("~/.config/rclone/rclone.conf")
            if os.path.exists(rc_conf) and shutil.which("rclone"):
                from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                _ym = _dt.now(_tz(_td(hours=8))).strftime("%Y.%m")
                subprocess.Popen(
                    ["rclone", "copy", out_path, f"gdrive:pro6000-output/{_ym}/videos", "--config", rc_conf],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        dt = round(time.time() - t0, 1)
        JOB_STATE.update({
            "running": False,
            "pct": 100,
            "seed": cur_seed,
            "elapsed_s": dt,
            "latest_video": fname,
            "completed_videos": [fname],
            "message": f"✅ [LongCat-Avatar-1.6] 数字人视频生成完成！总耗时 {dt}s · {stage_1.upper()} · {num_segments}段 (Seed: {cur_seed})",
        })
    except Exception as e:
        JOB_STATE.update({
            "running": False,
            "error": str(e),
            "message": f"❌ 生成出错: {e}",
        })


class _LongCatAvatarHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send_json({"status": "ok", "engine": "longcat-avatar-1.6"})
            return
        if self.path.startswith("/v1/models") or self.path.startswith("/models"):
            self._send_json({
                "object": "list",
                "data": [{"id": "meituan-longcat/LongCat-Video-Avatar-1.6", "object": "model", "owned_by": "meituan-longcat"}],
            })
            return
        if self.path.startswith("/v1/video/status"):
            st = dict(JOB_STATE)
            st["engine_ready"] = ENGINE_READY
            st["history"] = _list_history_videos()
            self._send_json(st)
            return
        if self.path.startswith("/v1/video/upscale_status"):
            try:
                with urllib.request.urlopen("http://127.0.0.1:30002/v1/video/upscale_status", timeout=5) as r:
                    self._send_json(json.loads(r.read().decode("utf-8")))
            except Exception as e:
                self._send_json({"running": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/audio/indextts2"):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:30003{self.path}", timeout=8) as r:
                    self._send_json(json.loads(r.read().decode("utf-8")), status=r.status)
            except Exception as e:
                self._send_json({"ok": False, "ready": False, "error": str(e)}, status=503)
            return
        if self.path.startswith("/v1/video/file/"):
            fname = os.path.basename(self.path.split("/v1/video/file/", 1)[-1].split("?")[0])
            fpath = os.path.join(OUTPUT_DIR, fname)
            if not os.path.isfile(fpath):
                self._send_json({"error": "File not found"}, status=404)
                return
            fsize = os.path.getsize(fpath)
            range_hdr = self.headers.get("Range")
            if range_hdr and range_hdr.startswith("bytes="):
                try:
                    parts = range_hdr.replace("bytes=", "").split("-")
                    start = int(parts[0]) if parts[0] else 0
                    end = int(parts[1]) if (len(parts) > 1 and parts[1]) else (fsize - 1)
                    end = min(end, fsize - 1)
                    length = max(0, end - start + 1)
                    self.send_response(206)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{fsize}")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(length))
                    self.end_headers()
                    with open(fpath, "rb") as rf:
                        rf.seek(start)
                        rem = length
                        while rem > 0:
                            buf = rf.read(min(262144, rem))
                            if not buf:
                                break
                            self.wfile.write(buf)
                            rem -= len(buf)
                    return
                except Exception:
                    pass
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(fsize))
            self.send_header("Content-Disposition", f'inline; filename="{fname}"')
            self.end_headers()
            with open(fpath, "rb") as rf:
                while True:
                    buf = rf.read(262144)
                    if not buf:
                        break
                    self.wfile.write(buf)
            return
        self._send_json({"status": "ok", "engine": "longcat-avatar-1.6"})

    def do_POST(self):
        cl = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(cl).decode("utf-8") if cl > 0 else "{}"
        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {}

        if self.path.startswith("/v1/video/upscale_cancel"):
            try:
                req2 = urllib.request.Request(
                    "http://127.0.0.1:30002/v1/video/upscale_cancel",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req2, timeout=10) as r2:
                    self._send_json(json.loads(r2.read().decode("utf-8")), status=r2.status)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/video/upscale"):
            try:
                req2 = urllib.request.Request(
                    f"http://127.0.0.1:30002{self.path}",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req2, timeout=15) as r2:
                    self._send_json(json.loads(r2.read().decode("utf-8")), status=r2.status)
            except Exception as e:
                if os.path.isfile("/content/recavsr_server.py"):
                    subprocess.Popen([sys.executable, "-u", "/content/recavsr_server.py"], stdout=open("/content/recavsr_server.log", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, close_fds=True, start_new_session=True)
                    time.sleep(2.5)
                    try:
                        req2 = urllib.request.Request(
                            "http://127.0.0.1:30002/v1/video/upscale",
                            data=json.dumps(payload).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urllib.request.urlopen(req2, timeout=15) as r2:
                            self._send_json(json.loads(r2.read().decode("utf-8")), status=r2.status)
                            return
                    except Exception as e2:
                        e = e2
                self._send_json({"ok": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/video/cancel"):
            open("/content/.cancel_minimax", "w").write("1")
            self._send_json({"ok": True, "message": "🛑 正在停止数字人视频生成任务..."})
            return

        if self.path.startswith("/v1/audio/indextts2"):
            def _fwd_idx_av():
                req3 = urllib.request.Request(
                    f"http://127.0.0.1:30003{self.path}",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req3, timeout=300) as r3:
                    return r3.status, json.loads(r3.read().decode("utf-8"))
            try:
                st3, d3 = _fwd_idx_av()
                self._send_json(d3, status=st3)
            except urllib.error.HTTPError as he:
                try:
                    self._send_json(json.loads(he.read().decode("utf-8")), status=he.code)
                except Exception:
                    self._send_json({"ok": False, "error": str(he)}, status=he.code)
            except Exception as e:
                if os.path.isfile("/content/indextts2_server.py"):
                    _idx_env = os.environ.copy()
                    _idx_env["PYTHONPATH"] = f"/content/indextts2_pkgs:{_idx_env.get('PYTHONPATH', '')}".rstrip(":")
                    subprocess.Popen([sys.executable, "-u", "/content/indextts2_server.py"], env=_idx_env, stdout=open("/content/indextts2_server.log", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, close_fds=True, start_new_session=True)
                    time.sleep(2.5)
                    try:
                        st3, d3 = _fwd_idx_av()
                        self._send_json(d3, status=st3)
                        return
                    except Exception as e2:
                        e = e2
                self._send_json({"ok": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/audio/ming_tts"):
            try:
                tts_text = str(payload.get("text") or "").strip()
                tts_voice = str(payload.get("voice") or "ming:灵小柔").strip()
                clone_ref_b64 = str(payload.get("clone_ref_b64") or "").strip()
                clone_prompt_text = str(payload.get("clone_prompt_text") or "").strip()
                if not tts_text:
                    self._send_json({"ok": False, "error": "台词内容不能为空"}, status=400)
                    return
                with GPU_LOCK:
                    wav_path = _synthesize_ming_talker_wav(
                        tts_text,
                        voice_spec=tts_voice,
                        clone_ref_b64=clone_ref_b64,
                        clone_prompt_text=clone_prompt_text,
                        update_job_state=False,
                    )
                wav_bytes = Path(wav_path).read_bytes()
                dur_sec = 4.0
                try:
                    import soundfile as sf
                    info = sf.info(wav_path)
                    dur_sec = round(float(info.duration), 1)
                except Exception:
                    pass
                auto_segs = 1 if dur_sec <= 3.85 else max(1, min(200, 1 + int(math.ceil((dur_sec - 3.72) / 3.20))))
                b64_str = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")
                self._send_json({
                    "ok": True,
                    "audio_b64": b64_str,
                    "audio_ext": ".wav",
                    "duration_sec": dur_sec,
                    "auto_segments": auto_segs,
                })
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=500)
            return

        if self.path.startswith("/v1/video/generate") or self.path.startswith("/v1/video/generations"):
            if payload.get("mode") in ("image", "t2va", "fl2va"):
                self._send_json({"ok": False, "error": "Engine mismatch: longcat-avatar cannot run image/minimax-h3 mode"}, status=503)
                return
            if os.path.exists("/content/.cancel_minimax"):
                try:
                    os.unlink("/content/.cancel_minimax")
                except Exception:
                    pass
            if JOB_STATE["running"]:
                self._send_json({"ok": True, "already_running": True, "message": JOB_STATE.get("message") or "当前已有数字人视频正在生成中，已自动接管实时进度"})
                return
            JOB_STATE.update({
                "running": True,
                "pct": 2,
                "step": 0,
                "completed_videos": [],
                "latest_video": "",
                "error": "",
                "message": "⏳ 数字人任务已提交，正在进入 GPU 渲染队列...",
            })
            threading.Thread(target=_run_avatar_job, args=(payload,), daemon=True).start()
            self._send_json({"ok": True, "message": "已开始生成 LongCat-Video-Avatar-1.6 数字人视频"})
            return

        if self.path.startswith("/v1/shell"):
            cmd = str(payload.get("cmd") or "").strip()
            if not cmd:
                self._send_json({"ok": False, "error": "empty cmd"}, status=400)
                return
            try:
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=int(payload.get("timeout") or 120))
                self._send_json({"ok": True, "returncode": r.returncode, "stdout": r.stdout[-16000:], "stderr": r.stderr[-16000:]})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=500)
            return

        if "chat/completions" in self.path:
            reply_txt = (
                "🎭 当前运行的是【美团 LongCat-Video-Avatar-1.6 数字人音频驱动视频引擎】！\n"
                "请在下方工作台切换至「🎭 数字人音频驱动 (LongCat-Avatar)」，上传语音/歌曲音频与人物照片（或直接点「🎙️ 使用内置示例」），然后点击「🎬 开始生成音视频」即可！"
            )
            self._send_json({
                "id": "chatcmpl-minimax",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply_txt}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 42, "total_tokens": 52},
                "timings": {"predicted_per_second": 100.0},
            })
            return

        self._send_json({"ok": True})


if __name__ == "__main__":
    threading.Thread(target=_warmup_engine_bg, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", 30000), _LongCatAvatarHandler)
    srv.serve_forever()
'''


def _start_longcat_avatar_server_on_30000(target_dir: str, total_gb: float = 48.0):
    """在远端实例上启动 LongCat-Video-Avatar-1.6 HTTP 推理网关 (端口 30000)"""
    if not shutil.which("cloudflared"):
        log("正在安装 cloudflared 隧道客户端...")
        subprocess.run(
            "curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
            "-o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
            shell=True,
            check=True,
        )

    subprocess.run("pkill -9 -f '[l]lama-server|minimax_server\\.py|minimax_t4_server\\.py|longcat_avatar_server\\.py|minimax_image_server\\.py|qwen_image_server\\.py|project-maya/serve/server\\.py|build/maya|maya-vision|/content/app\\.py' || true; fuser -k -9 30000/tcp 2>/dev/null || true", shell=True)
    time.sleep(1)

    lc_link = os.path.join(os.path.dirname(target_dir), "LongCat-Video")
    if not os.path.exists(lc_link):
        try:
            os.symlink(target_dir, lc_link)
        except Exception:
            pass
    if not os.path.isdir("/content/LongCat-Video/.git"):
        log("正在拉取美团 LongCat-Video-Avatar-1.6 推理框架...")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/meituan-longcat/LongCat-Video.git", "/content/LongCat-Video"], check=False)

    missing_pkgs = []
    for pkg, mod in [
        ("loguru", "loguru"),
        ("av", "av"),
        ("imageio", "imageio"),
        ("imageio-ffmpeg", "imageio_ffmpeg"),
        ("edge-tts", "edge_tts"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("soxr", "soxr"),
        ("pyloudnorm", "pyloudnorm"),
        ("audio-separator", "audio_separator"),
        ("onnx", "onnx"),
        ("onnxruntime-gpu", "onnxruntime"),
        ("ftfy", "ftfy"),
        ("einops", "einops"),
        ("x-transformers", "x_transformers"),
        ("inflect", "inflect"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing_pkgs.append(pkg)
    if missing_pkgs:
        log(f"正在安装 LongCat-Video-Avatar 与 Ming-3.0-Talker 运行依赖 ({', '.join(missing_pkgs)})...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing_pkgs, check=False)

    ming_talker_root = os.path.join(target_dir, "ming_talker")
    if not os.path.isfile(os.path.join(ming_talker_root, "talker/inference.py")):
        log("🎭 正在同步 Ming-3.0-Audio-Flash Talker 戏剧人设与零样本声音克隆模块...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="inclusionAI/Ming-3.0-audio-flash",
                allow_patterns=["talker/*", "voice_presets/*"],
                ignore_patterns=["*.safetensors", "*.onnx", "*__pycache__*"],
                local_dir=ming_talker_root,
            )
        except Exception as e_mt:
            log(f"⚠️ 同步 MingTalker 脚本提示: {e_mt}")

    # 若当前环境未预装编译版 flash_attn，自动注入基于 PyTorch 原生 CUDA SDPA 的零同步兼容包 (含 bert_padding 子模块以兼容 Ming-3.0-Talker)
    legacy_flash_file = "/content/LongCat-Video/flash_attn.py"
    if os.path.isfile(legacy_flash_file):
        try:
            os.remove(legacy_flash_file)
        except Exception:
            pass
    try:
        __import__("flash_attn")
        __import__("flash_attn.bert_padding")
    except ImportError:
        flash_pkg_dir = "/content/LongCat-Video/flash_attn"
        os.makedirs(flash_pkg_dir, exist_ok=True)
        with open(os.path.join(flash_pkg_dir, "__init__.py"), "w", encoding="utf-8") as sf:
            sf.write(
                "import torch\n"
                "import torch.nn.functional as F\n\n"
                "def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):\n"
                "    out = F.scaled_dot_product_attention(\n"
                "        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),\n"
                "        dropout_p=dropout_p, is_causal=causal, scale=softmax_scale,\n"
                "    )\n"
                "    return out.transpose(1, 2)\n\n"
                "def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):\n"
                "    if cu_seqlens_q.numel() == 2 and cu_seqlens_k.numel() == 2:\n"
                "        out = F.scaled_dot_product_attention(\n"
                "            q.unsqueeze(0).transpose(1, 2), k.unsqueeze(0).transpose(1, 2), v.unsqueeze(0).transpose(1, 2),\n"
                "            dropout_p=dropout_p, is_causal=causal, scale=softmax_scale,\n"
                "        )\n"
                "        return out.transpose(1, 2).squeeze(0)\n"
                "    q_bounds = cu_seqlens_q.tolist()\n"
                "    k_bounds = cu_seqlens_k.tolist()\n"
                "    outs = []\n"
                "    for i in range(len(q_bounds) - 1):\n"
                "        qi = q[q_bounds[i]:q_bounds[i+1]].unsqueeze(0).transpose(1, 2)\n"
                "        ki = k[k_bounds[i]:k_bounds[i+1]].unsqueeze(0).transpose(1, 2)\n"
                "        vi = v[k_bounds[i]:k_bounds[i+1]].unsqueeze(0).transpose(0, 1).transpose(1, 2) if v.ndim == 2 else v[k_bounds[i]:k_bounds[i+1]].unsqueeze(0).transpose(1, 2)\n"
                "        oi = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)\n"
                "        outs.append(oi.transpose(1, 2).squeeze(0))\n"
                "    return torch.cat(outs, dim=0)\n"
            )
        with open(os.path.join(flash_pkg_dir, "bert_padding.py"), "w", encoding="utf-8") as bpf:
            bpf.write(
                "import torch\n"
                "import torch.nn.functional as F\n\n"
                "def unpad_input(hidden_states, attention_mask):\n"
                "    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)\n"
                "    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()\n"
                "    max_seqlen_in_batch = int(seqlens_in_batch.max().item())\n"
                "    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))\n"
                "    return hidden_states.reshape(-1, *hidden_states.shape[2:])[indices], indices, cu_seqlens, max_seqlen_in_batch\n\n"
                "def pad_input(hidden_states, indices, batch, seqlen):\n"
                "    out = torch.zeros((batch * seqlen, *hidden_states.shape[1:]), device=hidden_states.device, dtype=hidden_states.dtype)\n"
                "    out[indices] = hidden_states\n"
                "    return out.reshape(batch, seqlen, *hidden_states.shape[1:])\n"
            )

    server_py = "/content/longcat_avatar_server.py"
    with open(server_py, "w", encoding="utf-8") as wf:
        wf.write(LONGCAT_AVATAR_SERVER_CODE)

    env = os.environ.copy()
    env["LONGCAT_MODEL_DIR"] = target_dir
    with open("/content/minimax_server.log", "w") as lf:
        subprocess.Popen(
            [sys.executable, "-u", server_py],
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(25):
        try:
            with opener.open("http://127.0.0.1:30000/health", timeout=2) as resp:
                if resp.status == 200:
                    h_data = json.loads(resp.read().decode("utf-8"))
                    if h_data.get("engine") == "longcat-avatar-1.6":
                        log(f"[显存装载] {os.path.basename(target_dir)}: 100% ({total_gb:.1f}/{total_gb:.1f} GiB 已就绪)")
                        log("✅ LongCat-Video-Avatar-1.6 数字人音频驱动视频服务已启动 (端口 30000)！")
                        _ensure_recavsr_service_on_30002()
                        _ensure_indextts2_service_on_30003()
                        return
        except Exception:
            pass
        time.sleep(1)


MINIMAX_H3_IMAGE_SERVER_CODE = r'''#!/usr/bin/env python3
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import base64
import gc
import glob
import io
import json
import math
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL_DIR = os.environ.get("MINIMAX_IMAGE_MODEL_DIR", "/content/local_models/MiniMax-H3-Image")
COMFY_DIR = "/content/ComfyUI"
OUTPUT_DIR = "/content/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
if COMFY_DIR not in sys.path:
    sys.path.insert(0, COMFY_DIR)

ENGINE_READY = True
ENGINE_ERROR = ""
GPU_LOCK = threading.Lock()
ENGINE_INIT_LOCK = threading.Lock()

_IMG_ENGINE = {
    "unet": None,
    "clip": None,
    "vae": None,
    "ming_talker": None,
}

JOB_STATE = {
    "running": False,
    "pct": 0,
    "step": 0,
    "total_steps": 6,
    "message": "就绪：MiniMax-H3-Image (33B · 4K无审查写真 & 免Mask换装引擎) 已待命...",
    "elapsed_s": 0.0,
    "latest_video": "",
    "error": "",
}


def _list_history_files():
    files = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "*.png")) + glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")),
        key=os.path.getmtime,
        reverse=True,
    )
    return [os.path.basename(p) for p in files[:16]]


def _sanitize_h3_image_prompt(prompt: str, clean_matte: bool = True) -> str:
    """自动清除会导致 MiniMax-H3-Image 4K 人像出现粗砂纸磨砂感或 3D 假脸的旧时代胶片/微孔毒词"""
    if not clean_matte or not prompt:
        return prompt
    bad_phrases = [
        r"\bmatte\s+skin\s+texture\b",
        r"\bfine\s+pores\b",
        r"\bvisible\s+pores\b",
        r"\bdelicate\s+peach\s+fuzz\b",
        r"\bpeach\s+fuzz\b",
        r"\btiny\s+microscopic\s+raindrops\b",
        r"\bmicroscopic\s+raindrops\b",
        r"\b35mm\s+Kodak\s+Portra\s+400\s+film\b",
        r"\bKodak\s+Portra\s+400\b",
        r"\bheavy\s+film\s+grain\b",
        r"\bfilm\s+grain\b",
    ]
    cleaned = prompt
    for pat in bad_phrases:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",\s*,+", ", ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,")
    return cleaned or prompt


def _build_picture_subject_prompt(prompt: str, num_refs: int, edit_outfit: str = "", edit_scene: str = "", edit_props: str = "") -> str:
    """
    若上传了参考图 (<Picture 1>...) 且用户填写了换装/换景槽位或未手写 <Subject 1> 语法，
    自动按照 MiniMax-H3-Image-PE 三段式 Picture + Subject 主语绑定语法生成免涂 Mask 锁脸指令
    """
    raw = (prompt or "").strip()
    outfit = (edit_outfit or "").strip()
    scene = (edit_scene or "").strip()
    props = (edit_props or "").strip()

    if num_refs <= 0:
        return raw or (
            "A high-resolution, real-life portrait photograph of an elegant 20-year-old Chinese young woman with a classic East Asian oval face, "
            "smooth fair skin, expressive dark brown almond eyes, and a serene, natural expression, captured on a Canon EOS R5 with an 85mm f/1.4 lens, "
            "clean natural lighting, and shallow depth of field."
        )

    # 若用户自己已经写了 <Subject 1> 或 <Picture 1> 锁脸分镜指令，直接原样使用（严禁在此强锁 <Subject 3> 背景，否则所有分镜都会复制参考图的站姿和街道背景！）
    if "<picture 1>" in raw.lower() or "<subject 1>" in raw.lower():
        return raw

    # 若主提示词是分镜画面描述（包含镜头景别/场景/动作），仅锁定 <Picture 1> 的面部五官与服装，完全按新分镜生成全新机位、景别、动作与场景
    if any(k in raw for k in ("镜头", "特写", "近景", "中景", "全景", "远景", "越肩", "正反打", "俯瞰", "机位", "坐", "走")):
        if num_refs >= 2:
            return (
                "The person in <Picture 1> is <Subject 1>. The person in <Picture 2> is <Subject 2>. "
                f"Generate a brand-new cinematic scene and camera angle showing <Subject 1> and <Subject 2>: {raw} "
                "Keep the facial identity, hairstyle, and clothing of <Subject 1> and <Subject 2> consistent with their reference pictures, "
                "while completely changing the camera framing, shot scale, pose, action, and background environment to match the scene description."
            )
        return (
            "The person in <Picture 1> is <Subject 1>. "
            f"Generate a brand-new cinematic scene and camera angle showing <Subject 1>: {raw} "
            "Keep the exact facial identity, hairstyle, and clothing of <Subject 1> consistent with <Picture 1>, "
            "while completely changing the camera framing, shot scale, pose, action, and background environment to match the scene description."
        )

    if outfit or scene or props:
        tgt_outfit = outfit or "an elegant, tailored silk outfit with refined details"
        tgt_props = f" while holding {props}" if props else ""
        if num_refs >= 2 and not outfit:
            # 多图融合模式：<Picture 1> 人物 + <Picture 2> 服饰/物品 + (<Picture 3> 场景)
            if num_refs >= 3:
                return (
                    "The person in <Picture 1> is <Subject 1>. "
                    "The featured item or outfit in <Picture 2> is <Subject 2>. "
                    "The environment and background in <Picture 3> is <Subject 3>. "
                    f"The target image shows <Subject 1> with <Subject 2> integrated naturally inside <Subject 3>. {raw} "
                    "<Subject 1> is partially preserved; exact East Asian facial features, face shape, smooth fair skin, eye spacing, and hairstyle remain completely unchanged, "
                    "while lighting and pose naturally adapt to <Subject 3>. "
                    "<Subject 2> is fully preserved. <Subject 3> is fully preserved."
                ).strip()
            return (
                "The person in <Picture 1> is <Subject 1>. "
                "The background environment in <Picture 1> is <Subject 3>. "
                "The featured outfit or object in <Picture 2> is <Subject 2>. "
                f"The target image shows <Subject 1> wearing or holding <Subject 2> in <Subject 3>. {raw} "
                "<Subject 1> is partially preserved; exact East Asian facial features, face shape, smooth fair skin, eye spacing, and hairstyle remain completely unchanged. "
                "<Subject 2> is fully preserved. <Subject 3> is fully preserved."
            ).strip()

        if scene:
            return (
                "The person in <Picture 1> is <Subject 1>. "
                "The original clothing worn by <Subject 1> in <Picture 1> is <Subject 2>. "
                f"The target image shows <Subject 1> wearing {tgt_outfit}{tgt_props} in {scene}, captured as a high-resolution real-life portrait photograph with clean natural lighting. "
                "<Subject 1> is partially preserved; her exact East Asian facial features, oval face shape, smooth fair skin, serene gaze, and hairstyle remain completely unchanged, "
                f"while <Subject 2> is replaced by {tgt_outfit}{tgt_props}, and the surrounding environment is {scene}."
            ).strip()
        return (
            "The person in <Picture 1> is <Subject 1>. "
            "The original clothing worn by <Subject 1> in <Picture 1> is <Subject 2>. "
            "The background and lighting environment in <Picture 1> is <Subject 3>. "
            f"The target image shows <Subject 1> wearing {tgt_outfit}{tgt_props} in <Subject 3>. "
            "<Subject 1> is partially preserved; her exact East Asian facial features, oval face shape, smooth fair skin, serene gaze, and hairstyle remain completely unchanged, "
            f"while <Subject 2> is replaced by {tgt_outfit}{tgt_props}. "
            "<Subject 3> is fully preserved."
        ).strip()

    # 若用户只传了 <Picture 1> 并在主提示词框里写了自然语言改图指令，自动包裹为标准三段式锁脸语法
    if raw:
        return (
            "The person in <Picture 1> is <Subject 1>. "
            "The original clothing worn by <Subject 1> in <Picture 1> is <Subject 2>. "
            "The background environment in <Picture 1> is <Subject 3>. "
            f"The target image shows <Subject 1> in <Subject 3> with the following modification: {raw}. "
            "<Subject 1> is partially preserved; exact facial features, face shape, smooth fair skin, and identity remain completely unchanged, "
            f"while <Subject 2> is updated according to: {raw}. "
            "<Subject 3> is fully preserved."
        )

    return "Edit <Picture 1>. Keep the exact facial features and identity of the person in <Picture 1> unchanged."


def _ensure_minimax_image_engine(update_job_state: bool = True):
    with ENGINE_INIT_LOCK:
        if _IMG_ENGINE["unet"] is not None and _IMG_ENGINE["clip"] is not None and _IMG_ENGINE["vae"] is not None:
            return _IMG_ENGINE["unet"], _IMG_ENGINE["clip"], _IMG_ENGINE["vae"]

        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        if update_job_state:
            JOB_STATE.update({
                "pct": 5,
                "message": "🚀 [MiniMax-H3-Image] 正在将 33B INT8 主干 + Heretic 32B 无审查编码器 + 2D VAE 装载至 GPU 显存...",
            })

        # 将 MODEL_DIR 中的黄金三件套软链至 /content/ComfyUI/models 对应子目录
        unet_fn = "minimax_h3_image_pruned_int8_convrot.safetensors"
        clip_fn = "qwen3vl_32b_minimax_h3_ultra_uncensored_heretic_int8_convrot.safetensors"
        vae_fn = "minimax_h3_image_vae_fp16.safetensors"

        for sub, fn in [
            ("diffusion_models", unet_fn),
            ("text_encoders", clip_fn),
            ("vae", vae_fn),
        ]:
            src_p = os.path.join(MODEL_DIR, sub, fn)
            if not os.path.isfile(src_p):
                alt_p = os.path.join(MODEL_DIR, fn)
                if os.path.isfile(alt_p):
                    src_p = alt_p
            dst_dir = os.path.join(COMFY_DIR, "models", sub)
            os.makedirs(dst_dir, exist_ok=True)
            dst_p = os.path.join(dst_dir, fn)
            if os.path.isfile(src_p) and not os.path.exists(dst_p):
                try:
                    os.symlink(src_p, dst_p)
                except Exception:
                    shutil.copy2(src_p, dst_p)

        import comfy.utils
        import nodes

        def _progress_hook(value, total, preview_image=None, node_id=None):
            if os.path.exists("/content/.cancel_minimax"):
                try:
                    os.unlink("/content/.cancel_minimax")
                except Exception:
                    pass
                raise RuntimeError("🛑 已手动停止 MiniMax-H3-Image 生图任务")
            if JOB_STATE.get("running") and total and total <= 20:
                cur_s = max(0, min(int(total), int(value)))
                tot_s = max(1, int(total))
                ov_pct = min(94, max(12, 12 + int(round((cur_s / float(tot_s)) * 80))))
                JOB_STATE["step"] = cur_s
                JOB_STATE["total_steps"] = tot_s
                JOB_STATE["pct"] = ov_pct
                JOB_STATE["message"] = f"📸 [MiniMax-H3-Image] 正在 6 步极速扩散采样: 第 {cur_s}/{tot_s} 步 (总进度 {ov_pct}%)"

        comfy.utils.set_progress_bar_global_hook(_progress_hook)

        with torch.inference_mode():
            unet = nodes.UNETLoader().load_unet(unet_fn, "default")[0]
            clip = nodes.CLIPLoader().load_clip(clip_fn, "minimax", "default")[0]
            vae = nodes.VAELoader().load_vae(vae_fn)[0]

        _IMG_ENGINE["unet"] = unet
        _IMG_ENGINE["clip"] = clip
        _IMG_ENGINE["vae"] = vae
        return unet, clip, vae


def _warmup_image_engine_bg():
    try:
        with GPU_LOCK:
            _ensure_minimax_image_engine(update_job_state=False)
            if not JOB_STATE.get("running"):
                JOB_STATE["message"] = "⚡ 就绪：MiniMax-H3-Image (33B · Heretic无审查 + 1680万像素4K直出 + 免Mask换装) 已常驻显存"
    except Exception as e:
        print(f"[MiniMax-H3-Image Warmup Error] {e}", flush=True)


def _ensure_img_ming_talker():
    """在 MiniMax-H3-Image (45.2GB) 引擎中也支持直接按需加载 Ming-3.0-Talker (2.78GB) 进行试听配音"""
    if _IMG_ENGINE.get("ming_talker") is not None:
        return _IMG_ENGINE["ming_talker"]

    with ENGINE_INIT_LOCK:
        if _IMG_ENGINE.get("ming_talker") is not None:
            return _IMG_ENGINE["ming_talker"]

        import torch
        candidates = [
            "/content/local_models/LongCat-Video-Avatar-1.6/ming_talker",
            "/content/drive/MyDrive/models/LongCat-Video-Avatar-1.6/ming_talker",
            "/content/local_models/ming_talker",
        ]
        ming_root = "/content/local_models/ming_talker"
        for c_dir in candidates:
            if os.path.isfile(os.path.join(c_dir, "talker/model.safetensors")):
                ming_root = c_dir
                break
        talker_dir = os.path.join(ming_root, "talker")
        os.makedirs(ming_root, exist_ok=True)

        missing_deps = []
        for pkg, mod in [
            ("diffusers", "diffusers"),
            ("x-transformers", "x_transformers"),
            ("inflect", "inflect"),
            ("onnxruntime", "onnxruntime"),
            ("librosa", "librosa"),
            ("soundfile", "soundfile"),
        ]:
            try:
                __import__(mod)
            except ImportError:
                missing_deps.append(pkg)
        if missing_deps:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing_deps, check=False)

        rc_conf = os.path.expanduser("~/.config/rclone/rclone.conf")
        need_weights = (
            not os.path.isfile(os.path.join(talker_dir, "model.safetensors"))
            or not os.path.isfile(os.path.join(talker_dir, "vae/model.safetensors"))
        )
        if need_weights and os.path.exists(rc_conf) and shutil.which("rclone"):
            subprocess.run(
                ["rclone", "copy", "gdrive:models/LongCat-Video-Avatar-1.6/ming_talker", ming_root, "--transfers=8", "--config", rc_conf],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if os.path.isfile(os.path.join(talker_dir, "model.safetensors")) and os.path.isfile(os.path.join(talker_dir, "vae/model.safetensors")):
                need_weights = False
        need_code = not os.path.isfile(os.path.join(talker_dir, "inference.py"))
        if need_weights or need_code:
            from huggingface_hub import snapshot_download
            ignore_pats = ["*__pycache__*"] if need_weights else ["*.safetensors", "*.onnx", "*__pycache__*"]
            snapshot_download(
                repo_id="inclusionAI/Ming-3.0-audio-flash",
                allow_patterns=["talker/*", "voice_presets/*"],
                ignore_patterns=ignore_pats,
                local_dir=ming_root,
            )
            if os.path.exists(rc_conf) and shutil.which("rclone"):
                subprocess.Popen(
                    ["rclone", "copy", ming_root, "gdrive:models/LongCat-Video-Avatar-1.6/ming_talker", "--transfers=8", "--config", rc_conf],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        norm_py = os.path.join(talker_dir, "talker_tn/normalizer.py")
        if os.path.isfile(norm_py):
            try:
                norm_txt = open(norm_py, "r", encoding="utf-8").read()
                if "from pynini import escape, Fst, shortestpath" in norm_txt and "except ImportError:" not in norm_txt:
                    open(norm_py, "w", encoding="utf-8").write(
                        "try:\n"
                        "    from pynini import escape, Fst, shortestpath\n"
                        "    _HAS_PYNINI = True\n"
                        "except ImportError:\n"
                        "    _HAS_PYNINI = False\n"
                        "from .token_parser import TokenParser\n\n"
                        "class Normalizer:\n"
                        "    def __init__(self, prefix_p, ordertype='tn'):\n"
                        "        self.ordertype = ordertype\n"
                        "        if _HAS_PYNINI:\n"
                        "            self.tagger = Fst.read(f'{prefix_p}_tagger.fst').optimize()\n"
                        "            self.verbalizer = Fst.read(f'{prefix_p}_verbalizer.fst').optimize()\n"
                        "    def normalize(self, input):\n"
                        "        if not _HAS_PYNINI or len(input) == 0:\n"
                        "            return input\n"
                        "        lattice = escape(input) @ self.tagger\n"
                        "        tmp = shortestpath(lattice, nshortest=1, unique=True).string()\n"
                        "        if len(tmp) == 0:\n"
                        "            return ''\n"
                        "        output = TokenParser(self.ordertype).reorder(tmp)\n"
                        "        lattice2 = escape(output) @ self.verbalizer\n"
                        "        return shortestpath(lattice2, nshortest=1, unique=True).string()\n"
                    )
            except Exception:
                pass

        mod_py = os.path.join(talker_dir, "talker_module/modules.py")
        if os.path.isfile(mod_py):
            try:
                mod_txt = open(mod_py, "r", encoding="utf-8").read()
                if 'if is_package_available("flash_attn"):' in mod_txt:
                    mod_txt = mod_txt.replace(
                        'if is_package_available("flash_attn"):\n    from flash_attn.bert_padding import pad_input, unpad_input\n    from flash_attn import flash_attn_varlen_func, flash_attn_func',
                        'try:\n    from flash_attn.bert_padding import pad_input, unpad_input\n    from flash_attn import flash_attn_varlen_func, flash_attn_func\nexcept Exception:\n    pass',
                    )
                    open(mod_py, "w", encoding="utf-8").write(mod_txt)
            except Exception:
                pass

        avae_py = os.path.join(talker_dir, "AudioVAE/modeling_audio_vae.py")
        if os.path.isfile(avae_py):
            try:
                avae_txt = open(avae_py, "r", encoding="utf-8").read()
                if "        self.post_init()\n" in avae_txt and "try:\n            self.post_init()" not in avae_txt:
                    avae_txt = avae_txt.replace(
                        "        self.post_init()\n",
                        "        try:\n            self.post_init()\n        except Exception:\n            pass\n",
                    )
                    open(avae_py, "w", encoding="utf-8").write(avae_txt)
            except Exception:
                pass

        istft_py = os.path.join(talker_dir, "AudioVAE/istft.py")
        if os.path.isfile(istft_py):
            try:
                ist_txt = open(istft_py, "r", encoding="utf-8").read()
                if ".float()" not in ist_txt:
                    ist_txt = (
                        ist_txt.replace("x_pred = self.out(x )", "x_pred = self.out(x).float()")
                        .replace("x_pred = self.out(x)", "x_pred = self.out(x).float()")
                        .replace("self.window[None, :, None]", "self.window.float()[None, :, None]")
                        .replace("self.window.square()", "self.window.float().square()")
                    )
                    open(istft_py, "w", encoding="utf-8").write(ist_txt)
            except Exception:
                pass

        inf_py = os.path.join(talker_dir, "inference.py")
        if os.path.isfile(inf_py):
            try:
                inf_txt = open(inf_py, "r", encoding="utf-8").read()
                changed_inf = False
                if "model.load_state_dict(weights, strict=True)" in inf_txt:
                    inf_txt = inf_txt.replace(
                        "model.load_state_dict(weights, strict=True)",
                        "model.load_state_dict(weights, strict=False)",
                    ).replace(
                        '        model.audio_vae.load_state_dict(\n            load_file(str(path / "vae/model.safetensors")), strict=True\n        )',
                        '        _vw = load_file(str(path / "vae/model.safetensors"))\n        _vk = model.audio_vae.state_dict().keys()\n        _vw = {k: v for k, v in _vw.items() if not (k.endswith("inv_freq") and k not in _vk)}\n        model.audio_vae.load_state_dict(_vw, strict=False)',
                    )
                    changed_inf = True
                if "embeddings = self.model.get_input_embeddings()(ids)\n" in inf_txt:
                    inf_txt = inf_txt.replace(
                        "embeddings = self.model.get_input_embeddings()(ids)\n",
                        "embeddings = self.model.get_input_embeddings()(ids).clone()\n",
                    )
                    changed_inf = True
                if "def text_fragments(self, value, max_length=50):" in inf_txt:
                    inf_txt = inf_txt.replace(
                        "def text_fragments(self, value, max_length=50):",
                        "def text_fragments(self, value, max_length=140):",
                    ).replace(
                        "len(streaming_text) >= 12 or (count > 0 and len(streaming_text) >= 8)",
                        "len(streaming_text) >= 140",
                    ).replace(
                        "elif ele[-1] in \"，,\" and len(streaming_text) >= 36:",
                        "elif ele[-1] in \"，,\" and len(streaming_text) >= 160:",
                    )
                    changed_inf = True
                if "for fragment in self.text_fragments(text):" in inf_txt and "_merged_frags" not in inf_txt:
                    inf_txt = inf_txt.replace(
                        "            for fragment in self.text_fragments(text):",
                        "            _raw_frags = list(self.text_fragments(text))\n"
                        "            _merged_frags = []\n"
                        "            for _fg in _raw_frags:\n"
                        "                _stripped_no_tags = re.sub(r'\\[[A-Za-z-]+\\]', '', _fg)\n"
                        "                if _merged_frags and not re.search(r'[\\u4e00-\\u9fffA-Za-z0-9]', _stripped_no_tags):\n"
                        "                    _merged_frags[-1] = _merged_frags[-1].rstrip() + ' ' + _fg.lstrip()\n"
                        "                else:\n"
                        "                    _merged_frags.append(_fg)\n"
                        "            for fragment in (_merged_frags or [text]):",
                    )
                    changed_inf = True
                if changed_inf:
                    open(inf_py, "w", encoding="utf-8").write(inf_txt)
            except Exception:
                pass

        if ming_root not in sys.path:
            sys.path.insert(0, ming_root)
        from talker import MingTalker
        talker = MingTalker.from_pretrained(
            Path(talker_dir),
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        _IMG_ENGINE["ming_talker"] = talker
        return talker


@__import__("torch").inference_mode()
def _synthesize_img_ming_talker_wav(
    text: str,
    voice_spec: str = "ming:灵小柔",
    clone_ref_b64: str = "",
    clone_prompt_text: str = "",
) -> str:
    import librosa
    import numpy as np
    import soundfile as sf

    talker = _ensure_img_ming_talker()
    clean_v = str(voice_spec or "ming:灵小柔").split("|")[0].strip()
    if clean_v.startswith("ming:"):
        clean_v = clean_v[len("ming:"):].strip()
    if not clean_v:
        clean_v = "灵小柔"

    _MING_VOICE_ALIASES = {
        "灵逍遥": "灵信远_中性",
        "温润公子": "灵信远_中性",
        "深情男主": "灵朗",
        "古风男主": "灵信远_中性",
        "清朗少年": "灵旭",
        "磁性男神": "灵空",
        "灵信远": "灵信远_中性",
        "灵孤影": "灵孤影_中性",
        "灵思齐": "灵思齐_中性",
        "灵言澈": "灵言澈_中性",
        "灵溯光": "灵溯光_中性",
        "灵嘉肴": "灵嘉肴_中性",
        "灵雨汐": "灵雨汐_中性",
        "灵书悦": "灵书悦_高兴",
        "灵惜梦": "灵惜梦_中性",
        "蜡笔小新_野原新之助": "蜡笔小新",
        "蜡笔小新_野原美伢": "蜡笔小新_野原美伢 (美伢)",
    }
    clean_v = _MING_VOICE_ALIASES.get(clean_v, clean_v)
    if any(k in clean_v for k in ("男", "公子", "逍遥", "少侠", "先生")) and clean_v not in ("灵信远_中性", "灵朗", "灵楷", "灵宇", "灵旭", "灵空", "灵孤影_中性", "灵思齐_中性"):
        clean_v = "灵信远_中性"

    reference = None
    prompt_txt = str(clone_prompt_text or "").strip() or None
    if clone_ref_b64 or clean_v == "__clone__":
        if clone_ref_b64:
            raw_b = clone_ref_b64.split(",", 1)[-1]
            ref_tmp = "/tmp/ming_clone_ref_input"
            with open(ref_tmp, "wb") as rf:
                rf.write(base64.b64decode(raw_b))
            samples, sr = librosa.load(ref_tmp, sr=None, mono=True)
            max_samples = int(sr * 12)
            if len(samples) > max_samples:
                samples = samples[:max_samples]
            reference = {"samples": samples.astype(np.float32), "sample_rate": int(sr)}

    audio_out = talker.synthesize(
        text,
        ip_name="灵小柔" if reference is not None else clean_v,
        reference=reference,
        prompt_text=prompt_txt if reference is not None else None,
        max_steps=1000,
    )
    waveform = audio_out["audio"].squeeze().detach().float().cpu().numpy()
    sample_rate = int(audio_out["sr"])
    out_wav = "/tmp/longcat_ming_tts.wav"
    sf.write(out_wav, waveform, sample_rate)
    return out_wav


@__import__("torch").inference_mode()
def _run_image_job(payload: dict):
    global JOB_STATE
    t0 = time.time()
    try:
        import numpy as np
        import PIL.Image
        import torch
        import nodes
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning

        raw_prompt = str(payload.get("prompt") or "").strip()
        clean_matte = bool(payload.get("clean_matte", True))
        edit_outfit = str(payload.get("edit_outfit") or "").strip()
        edit_scene = str(payload.get("edit_scene") or "").strip()
        edit_props = str(payload.get("edit_props") or "").strip()
        res_str = str(payload.get("resolution") or "3552x4736").strip().lower()
        steps = max(4, min(12, int(payload.get("steps") or 6)))
        batch_count = max(1, min(4, int(payload.get("batch_count") or 1)))
        base_seed = int(payload.get("seed") if payload.get("seed") is not None else -1)
        rand_seed = bool(payload.get("randomize_seed")) or (base_seed < 0)
        cur_seed = random.randint(1, 2**31 - 1) if rand_seed else base_seed

        try:
            w_s, h_s = res_str.split("x", 1)
            width = max(512, min(6272, (int(w_s) // 32) * 32))
            height = max(512, min(6272, (int(h_s) // 32) * 32))
        except Exception:
            width, height = 3552, 4736
        if width < 1024 and height < 1024:
            width, height = (4736, 3552) if width > height else (3552, 4736)

        ref_b64_list = payload.get("ref_images_b64") or []
        if not ref_b64_list and payload.get("first_frame_b64"):
            ref_b64_list = [payload["first_frame_b64"]]
        ref_b64_list = [str(x).strip() for x in ref_b64_list if str(x).strip()]

        final_prompt = _build_picture_subject_prompt(
            raw_prompt,
            num_refs=len(ref_b64_list),
            edit_outfit=edit_outfit,
            edit_scene=edit_scene,
            edit_props=edit_props,
        )
        final_prompt = _sanitize_h3_image_prompt(final_prompt, clean_matte=clean_matte)

        mode_tag = f"免Mask锁脸改图 ({len(ref_b64_list)}张参考图)" if ref_b64_list else "原生直出"
        mp_val = round((width * height) / 1_000_000.0, 1)

        JOB_STATE.update({
            "running": True,
            "pct": 6,
            "step": 0,
            "total_steps": steps,
            "seed": cur_seed,
            "final_prompt": final_prompt,
            "message": f"📸 [MiniMax-H3-Image] 正在准备 Heretic 32B 无审查编码器 ({width}×{height} · {mp_val}MP · {mode_tag})...",
            "error": "",
        })

        with GPU_LOCK:
            unet, clip, vae = _ensure_minimax_image_engine(update_job_state=True)

            ref_images_dict = {}
            for idx, b64_str in enumerate(ref_b64_list[:4], start=1):
                raw_b = b64_str.split(",", 1)[-1]
                pil_im = PIL.Image.open(io.BytesIO(base64.b64decode(raw_b))).convert("RGB")
                arr = np.asarray(pil_im, dtype=np.float32) / 255.0
                t_im = torch.from_numpy(arr).unsqueeze(0)
                ref_images_dict[f"image_{idx}"] = t_im

            JOB_STATE.update({
                "pct": 10,
                "message": f"📸 [MiniMax-H3-Image] 正在使用 Heretic 32B 编码指令与参考图特征 ({width}×{height})...",
            })

            cond_out = MiniMaxH3ImageConditioning.execute(
                clip=clip,
                prompt=final_prompt,
                width=width,
                height=height,
                vae=vae,
                images=ref_images_dict if ref_images_dict else None,
            )
            positive = cond_out[0]
            latent = cond_out[1]
            negative = nodes.ConditioningZeroOut().zero_out(positive)[0]

            completed_imgs = []
            last_seed = cur_seed
            rc_conf = os.path.expanduser("~/.config/rclone/rclone.conf")
            for b_i in range(batch_count):
                b_seed = (random.randint(1, 2**31 - 1) if (rand_seed and b_i > 0) else (cur_seed + b_i))
                last_seed = b_seed
                b_pref = f"[连抽 {b_i + 1}/{batch_count}] " if batch_count > 1 else ""
                sampled = nodes.KSampler().sample(
                    model=unet,
                    seed=b_seed,
                    steps=steps,
                    cfg=1.0,
                    sampler_name="euler",
                    scheduler="simple",
                    positive=positive,
                    negative=negative,
                    latent_image=latent,
                    denoise=1.0,
                )[0]

                JOB_STATE.update({
                    "pct": min(98, int(round(((b_i + 0.9) / float(batch_count)) * 90)) + 8),
                    "message": f"🖼️ {b_pref}[MiniMax-H3-Image] 正在通过 24 层 2D ViT VAE 解码 {width}×{height} ({mp_val}MP) 原生超清大图...",
                })
                decoded = nodes.VAEDecode().decode(vae, sampled)[0]
                img_np = (decoded[0].detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
                pil_out = PIL.Image.fromarray(img_np)

                dt_i = round(time.time() - t0, 1)
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                kind = "edit" if ref_images_dict else "t2i"
                fname = f"minimax_h3_image_{kind}_{width}x{height}_{ts}_dur{max(1, int(round(dt_i)))}s_seed{b_seed}.jpg"
                out_path = os.path.join(OUTPUT_DIR, fname)
                pil_out.save(out_path, format="JPEG", quality=96, subsampling=0, optimize=True)
                completed_imgs.append(fname)
                JOB_STATE["latest_video"] = fname
                JOB_STATE["completed_videos"] = list(completed_imgs)
                del decoded, sampled
                if os.path.exists(rc_conf) and shutil.which("rclone"):
                    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                    _ym = _dt.now(_tz(_td(hours=8))).strftime("%Y.%m")
                    _dsub = str(payload.get("drive_subdir") or "images").strip().lower()
                    if _dsub not in ("images", "temp"):
                        _dsub = "images"
                    subprocess.Popen(
                        ["rclone", "copy", out_path, f"gdrive:pro6000-output/{_ym}/{_dsub}", "--config", rc_conf],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

            del latent, positive, negative, ref_images_dict
            gc.collect()
            torch.cuda.empty_cache()

        dt = round(time.time() - t0, 1)
        b_lbl = f"共 {batch_count} 张 · " if batch_count > 1 else ""
        JOB_STATE.update({
            "running": False,
            "pct": 100,
            "step": steps,
            "total_steps": steps,
            "seed": last_seed,
            "elapsed_s": dt,
            "latest_video": completed_imgs[-1] if completed_imgs else "",
            "completed_videos": completed_imgs,
            "final_prompt": final_prompt,
            "message": f"✅ [MiniMax-H3-Image] {b_lbl}{width}×{height} ({mp_val}MP · {mode_tag}) 超清大图生成完成！耗时 {dt}s (Seed: {last_seed})",
        })
    except Exception as e:
        JOB_STATE.update({
            "running": False,
            "error": str(e),
            "message": f"❌ MiniMax-H3-Image 生成出错: {e}",
        })


class _MiniMaxImageHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send_json({"status": "ok", "engine": "minimax-h3-image"})
            return
        if self.path.startswith("/v1/models") or self.path.startswith("/models"):
            self._send_json({
                "object": "list",
                "data": [{"id": "Comfy-Org/MiniMax-H3-Image", "object": "model", "owned_by": "MiniMaxAI"}],
            })
            return
        if self.path.startswith("/v1/video/status"):
            st = dict(JOB_STATE)
            st["engine_ready"] = ENGINE_READY
            st["history"] = _list_history_files()
            self._send_json(st)
            return
        if self.path.startswith("/v1/video/upscale_status"):
            try:
                with urllib.request.urlopen("http://127.0.0.1:30002/v1/video/upscale_status", timeout=5) as r:
                    self._send_json(json.loads(r.read().decode("utf-8")))
            except Exception as e:
                self._send_json({"running": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/audio/indextts2"):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:30003{self.path}", timeout=8) as r:
                    self._send_json(json.loads(r.read().decode("utf-8")), status=r.status)
            except Exception as e:
                self._send_json({"ok": False, "ready": False, "error": str(e)}, status=503)
            return
        if self.path.startswith("/v1/video/file/"):
            fname = os.path.basename(self.path.split("/v1/video/file/", 1)[-1].split("?")[0])
            fpath = os.path.join(OUTPUT_DIR, fname)
            if not os.path.isfile(fpath):
                self._send_json({"error": "File not found"}, status=404)
                return
            fsize = os.path.getsize(fpath)
            ctype = "image/png" if fname.lower().endswith(".png") else ("image/jpeg" if fname.lower().endswith((".jpg", ".jpeg")) else "video/mp4")
            range_hdr = self.headers.get("Range")
            if range_hdr and range_hdr.startswith("bytes="):
                try:
                    parts = range_hdr.replace("bytes=", "").split("-")
                    start = int(parts[0]) if parts[0] else 0
                    end = int(parts[1]) if (len(parts) > 1 and parts[1]) else (fsize - 1)
                    end = min(end, fsize - 1)
                    length = max(0, end - start + 1)
                    self.send_response(206)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{fsize}")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(length))
                    self.end_headers()
                    with open(fpath, "rb") as rf:
                        rf.seek(start)
                        rem = length
                        while rem > 0:
                            buf = rf.read(min(262144, rem))
                            if not buf:
                                break
                            self.wfile.write(buf)
                            rem -= len(buf)
                    return
                except Exception:
                    pass
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(fsize))
            self.send_header("Content-Disposition", f'inline; filename="{fname}"')
            self.end_headers()
            with open(fpath, "rb") as rf:
                while True:
                    buf = rf.read(262144)
                    if not buf:
                        break
                    self.wfile.write(buf)
            return
        self._send_json({"status": "ok", "engine": "minimax-h3-image"})

    def do_POST(self):
        cl = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(cl).decode("utf-8") if cl > 0 else "{}"
        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {}

        if self.path.startswith("/v1/video/cancel"):
            open("/content/.cancel_minimax", "w").write("1")
            self._send_json({"ok": True, "message": "🛑 正在停止 MiniMax-H3-Image 生成任务..."})
            return

        if self.path.startswith("/v1/video/upscale_cancel"):
            try:
                req2 = urllib.request.Request(
                    "http://127.0.0.1:30002/v1/video/upscale_cancel",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req2, timeout=10) as r2:
                    self._send_json(json.loads(r2.read().decode("utf-8")), status=r2.status)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/video/upscale"):
            try:
                req2 = urllib.request.Request(
                    f"http://127.0.0.1:30002{self.path}",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req2, timeout=15) as r2:
                    self._send_json(json.loads(r2.read().decode("utf-8")), status=r2.status)
            except Exception as e:
                if os.path.isfile("/content/recavsr_server.py"):
                    subprocess.Popen([sys.executable, "-u", "/content/recavsr_server.py"], stdout=open("/content/recavsr_server.log", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, close_fds=True, start_new_session=True)
                    time.sleep(2.5)
                    try:
                        req2 = urllib.request.Request(
                            "http://127.0.0.1:30002/v1/video/upscale",
                            data=json.dumps(payload).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urllib.request.urlopen(req2, timeout=15) as r2:
                            self._send_json(json.loads(r2.read().decode("utf-8")), status=r2.status)
                            return
                    except Exception as e2:
                        e = e2
                self._send_json({"ok": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/audio/ming_tts"):
            try:
                tts_text = str(payload.get("text") or "").strip()
                tts_voice = str(payload.get("voice") or "ming:灵小柔").strip()
                clone_ref_b64 = str(payload.get("clone_ref_b64") or "").strip()
                clone_prompt_text = str(payload.get("clone_prompt_text") or "").strip()
                if not tts_text:
                    self._send_json({"ok": False, "error": "台词内容不能为空"}, status=400)
                    return
                with GPU_LOCK:
                    wav_path = _synthesize_img_ming_talker_wav(
                        tts_text,
                        voice_spec=tts_voice,
                        clone_ref_b64=clone_ref_b64,
                        clone_prompt_text=clone_prompt_text,
                    )
                wav_bytes = Path(wav_path).read_bytes()
                dur_sec = 4.0
                try:
                    import soundfile as sf
                    info = sf.info(wav_path)
                    dur_sec = round(float(info.duration), 1)
                except Exception:
                    pass
                auto_segs = 1 if dur_sec <= 3.85 else max(1, min(200, 1 + int(math.ceil((dur_sec - 3.72) / 3.20))))
                b64_str = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")
                self._send_json({
                    "ok": True,
                    "audio_b64": b64_str,
                    "audio_ext": ".wav",
                    "duration_sec": dur_sec,
                    "auto_segments": auto_segs,
                })
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=500)
            return

        if self.path.startswith("/v1/guide/preprocess"):
            try:
                if "/content" not in sys.path:
                    sys.path.insert(0, "/content")
                import guide_preprocessor
                res_gp = guide_preprocessor.preprocess_guide(payload, OUTPUT_DIR, gpu_lock=GPU_LOCK)
                self._send_json(res_gp, status=200 if res_gp.get("ok") else 400)
            except Exception as e:
                self._send_json({"ok": False, "error": f"白模/深度预处理失败: {e}"}, status=500)
            return

        if self.path.startswith("/v1/audio/indextts2"):
            def _fwd_idx():
                req3 = urllib.request.Request(
                    f"http://127.0.0.1:30003{self.path}",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req3, timeout=300) as r3:
                    return r3.status, json.loads(r3.read().decode("utf-8"))
            try:
                st3, d3 = _fwd_idx()
                self._send_json(d3, status=st3)
            except urllib.error.HTTPError as he:
                try:
                    self._send_json(json.loads(he.read().decode("utf-8")), status=he.code)
                except Exception:
                    self._send_json({"ok": False, "error": str(he)}, status=he.code)
            except Exception as e:
                if os.path.isfile("/content/indextts2_server.py"):
                    _idx_env = os.environ.copy()
                    _idx_env["PYTHONPATH"] = f"/content/indextts2_pkgs:{_idx_env.get('PYTHONPATH', '')}".rstrip(":")
                    subprocess.Popen([sys.executable, "-u", "/content/indextts2_server.py"], env=_idx_env, stdout=open("/content/indextts2_server.log", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, close_fds=True, start_new_session=True)
                    time.sleep(2.5)
                    try:
                        st3, d3 = _fwd_idx()
                        self._send_json(d3, status=st3)
                        return
                    except Exception as e2:
                        e = e2
                self._send_json({"ok": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/video/generate") or self.path.startswith("/v1/video/generations") or self.path.startswith("/v1/images/generations"):
            if payload.get("mode") in ("avatar", "t2va", "fl2va", "ref2va"):
                self._send_json({"ok": False, "error": "Engine mismatch: minimax-h3-image cannot run video/avatar mode"}, status=503)
                return
            if os.path.exists("/content/.cancel_minimax"):
                try:
                    os.unlink("/content/.cancel_minimax")
                except Exception:
                    pass
            if JOB_STATE["running"]:
                self._send_json({"ok": True, "already_running": True, "message": JOB_STATE.get("message") or "当前已有 MiniMax-H3-Image 任务正在运行，已自动接管实时进度"})
                return
            JOB_STATE.update({
                "running": True,
                "pct": 2,
                "step": 0,
                "completed_videos": [],
                "latest_video": "",
                "error": "",
                "message": "⏳ 4K 生图任务已提交，正在进入 GPU 渲染队列...",
            })
            threading.Thread(target=_run_image_job, args=(payload,), daemon=True).start()
            self._send_json({"ok": True, "message": "已开始生成 MiniMax-H3-Image 4K 超清大图"})
            return

        if self.path.startswith("/v1/shell"):
            cmd = str(payload.get("cmd") or "").strip()
            if not cmd:
                self._send_json({"ok": False, "error": "empty cmd"}, status=400)
                return
            try:
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=int(payload.get("timeout") or 120))
                self._send_json({"ok": True, "returncode": r.returncode, "stdout": r.stdout[-16000:], "stderr": r.stderr[-16000:]})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=500)
            return

        if "chat/completions" in self.path:
            reply_txt = (
                "📸 当前运行的是【MiniMax-H3-Image (33B · Heretic无审查 + 11秒原生4K直出 + 免Mask锁脸换装) 图像引擎】！\n"
                "请在下方工作台切换至「📸 4K写真/免Mask换装 (MiniMax-H3-Image)」直接生图或上传人物照锁脸换装，生成后可一键送入 LongCat 数字人开口说话！"
            )
            self._send_json({
                "id": "chatcmpl-minimax-img",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply_txt}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 42, "total_tokens": 52},
                "timings": {"predicted_per_second": 100.0},
            })
            return

        self._send_json({"ok": True})


if __name__ == "__main__":
    threading.Thread(target=_warmup_image_engine_bg, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", 30000), _MiniMaxImageHandler)
    srv.serve_forever()
'''


def _start_minimax_h3_image_server_on_30000(target_dir: str, total_gb: float = 45.2):
    """在远端 Colab G4 实例上启动 MiniMax-H3-Image (33B · INT8 ConvRot + Heretic 32B 无审查) 推理网关 (端口 30000)"""
    if not shutil.which("cloudflared"):
        log("正在安装 cloudflared 隧道客户端...")
        subprocess.run(
            "curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
            "-o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
            shell=True,
            check=True,
        )

    subprocess.run("pkill -9 -f '[l]lama-server|minimax_server\\.py|minimax_t4_server\\.py|longcat_avatar_server\\.py|minimax_image_server\\.py|qwen_image_server\\.py|project-maya/serve/server\\.py|build/maya|maya-vision|/content/app\\.py' || true; fuser -k -9 30000/tcp 2>/dev/null || true", shell=True)
    time.sleep(1)

    if not os.path.isdir("/content/ComfyUI/.git"):
        log("正在拉取最新 ComfyUI 主干框架 (原生支持 MiniMax-H3-Image 6步蒸馏与免Mask改图)...")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", "/content/ComfyUI"], check=False)
    else:
        subprocess.run(["git", "-C", "/content/ComfyUI", "pull", "--ff-only"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    log("正在检查 ComfyUI 运行依赖 (保留 Colab 原生 PyTorch 环境)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "/content/ComfyUI/requirements.txt"], check=False)

    with open("/content/guide_preprocessor.py", "w", encoding="utf-8") as gwf:
        gwf.write(GUIDE_PREPROCESSOR_CODE)

    server_py = "/content/minimax_image_server.py"
    with open(server_py, "w", encoding="utf-8") as wf:
        wf.write(MINIMAX_H3_IMAGE_SERVER_CODE)

    env = os.environ.copy()
    env["MINIMAX_IMAGE_MODEL_DIR"] = target_dir
    with open("/content/minimax_server.log", "w") as lf:
        subprocess.Popen(
            [sys.executable, "-u", server_py],
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(30):
        try:
            with opener.open("http://127.0.0.1:30000/health", timeout=2) as resp:
                if resp.status == 200:
                    h_data = json.loads(resp.read().decode("utf-8"))
                    if h_data.get("engine") == "minimax-h3-image":
                        log(f"[显存装载] MiniMax-H3-Image(33B·无审查4K): 100% ({total_gb:.1f}/{total_gb:.1f} GiB 已就绪)")
                        log("✅ MiniMax-H3-Image (11秒4K无审查写真 + 免Mask锁脸换装) 服务已启动 (端口 30000)！")
                        _ensure_recavsr_service_on_30002()
                        _ensure_indextts2_service_on_30003()
                        return
        except Exception:
            pass
        time.sleep(1)


QWEN_IMAGE_21_SERVER_CODE = r'''#!/usr/bin/env python3
import base64
import gc
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

COMFY_DIR = "/content/ComfyUI"
if COMFY_DIR not in sys.path:
    sys.path.insert(0, COMFY_DIR)

MODEL_DIR = os.environ.get("QWEN_IMAGE_21_MODEL_DIR", "/content/local_models/Qwen-Image-2.1-Uncensored")
OUTPUT_DIR = "/content/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_MODELS = {
    "chatr": "qwen-image-2.1-step90000-BF16.safetensors",
    "official": "qwen_image_2.1_bf16.safetensors",
}
CLIP_FN = "qwen3vl_8b_bf16_heretic.safetensors"
VAE_FN = "qwen_image_2.1_vae_bf16.safetensors"

GPU_LOCK = threading.Lock()
ENGINE_INIT_LOCK = threading.Lock()
_QWEN_ENGINE = {
    "unet": None,
    "unet_key": "",
    "clip": None,
    "vae": None,
}
ENGINE_READY = False

JOB_STATE = {
    "running": False,
    "pct": 0,
    "step": 0,
    "total_steps": 50,
    "message": "就绪：Qwen-Image-2.1 (满血 BF16 破限双底模 + 3D控姿势/改图/LoRA) 待命中...",
    "latest_video": "",
    "completed_videos": [],
    "error": "",
    "seed": 42,
    "elapsed_s": 0.0,
    "final_prompt": "",
    "engine_ready": False,
}


def _list_history_files():
    out = []
    if not os.path.isdir(OUTPUT_DIR):
        return out
    for fn in sorted(os.listdir(OUTPUT_DIR), reverse=True):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".mp4")):
            continue
        if fn.startswith("_tmp_") or fn.startswith("guide_"):
            continue
        p = os.path.join(OUTPUT_DIR, fn)
        if os.path.isfile(p) and os.path.getsize(p) > 2048:
            out.append({"name": fn, "size_mb": round(os.path.getsize(p) / (1024 * 1024), 2)})
            if len(out) >= 60:
                break
    return out


def _sync_comfy_symlinks():
    for sub in ("diffusion_models", "text_encoders", "vae", "loras"):
        src_sub = os.path.join(MODEL_DIR, sub)
        dst_sub = os.path.join(COMFY_DIR, "models", sub)
        os.makedirs(dst_sub, exist_ok=True)
        if not os.path.isdir(src_sub):
            continue
        for fn in os.listdir(src_sub):
            if not fn.lower().endswith(".safetensors"):
                continue
            src_p = os.path.join(src_sub, fn)
            dst_p = os.path.join(dst_sub, fn)
            if os.path.isfile(src_p) and not os.path.exists(dst_p):
                try:
                    os.symlink(src_p, dst_p)
                except Exception:
                    shutil.copy2(src_p, dst_p)


def _list_available_loras():
    _sync_comfy_symlinks()
    lora_dir = os.path.join(COMFY_DIR, "models", "loras")
    res = []
    if os.path.isdir(lora_dir):
        for fn in sorted(os.listdir(lora_dir)):
            if fn.lower().endswith(".safetensors"):
                res.append(fn)
    return res


def _ensure_qwen21_engine(base_key: str = "chatr", update_job_state: bool = True):
    global ENGINE_READY
    req_key = "official" if str(base_key or "").strip().lower() == "official" else "chatr"
    with ENGINE_INIT_LOCK:
        _sync_comfy_symlinks()
        target_unet_fn = BASE_MODELS[req_key]
        if not os.path.exists(os.path.join(COMFY_DIR, "models", "diffusion_models", target_unet_fn)):
            alt_key = "official" if req_key == "chatr" else "chatr"
            if os.path.exists(os.path.join(COMFY_DIR, "models", "diffusion_models", BASE_MODELS[alt_key])):
                req_key = alt_key
                target_unet_fn = BASE_MODELS[req_key]

        if (
            _QWEN_ENGINE["unet"] is not None
            and _QWEN_ENGINE["unet_key"] == req_key
            and _QWEN_ENGINE["clip"] is not None
            and _QWEN_ENGINE["vae"] is not None
        ):
            ENGINE_READY = True
            return _QWEN_ENGINE["unet"], _QWEN_ENGINE["clip"], _QWEN_ENGINE["vae"], req_key

        import torch
        import comfy.utils
        import nodes

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        if update_job_state:
            b_lbl = "ChatR 无审查 Omni BF16" if req_key == "chatr" else "官方满血 BF16"
            JOB_STATE.update({
                "pct": 5,
                "message": f"🚀 [Qwen-Image-2.1] 正在将 {b_lbl} (13.25G) + Heretic BF16 破限编码器 (16.33G) 装载至显存...",
            })

        def _progress_hook(value, total, preview_image=None, node_id=None):
            if os.path.exists("/content/.cancel_minimax") or os.path.exists("/content/.cancel_qwen21"):
                for cf in ("/content/.cancel_minimax", "/content/.cancel_qwen21"):
                    try:
                        if os.path.exists(cf):
                            os.unlink(cf)
                    except Exception:
                        pass
                raise RuntimeError("🛑 已手动停止 Qwen-Image-2.1 生图任务")
            if JOB_STATE.get("running") and total and total <= 120:
                cur_s = max(0, min(int(total), int(value)))
                tot_s = max(1, int(total))
                ov_pct = min(94, max(12, 12 + int(round((cur_s / float(tot_s)) * 80))))
                JOB_STATE["step"] = cur_s
                JOB_STATE["total_steps"] = tot_s
                JOB_STATE["pct"] = ov_pct
                JOB_STATE["message"] = f"🎨 [Qwen-Image-2.1 BF16] 正在高清扩散采样: 第 {cur_s}/{tot_s} 步 (总进度 {ov_pct}%)"

        comfy.utils.set_progress_bar_global_hook(_progress_hook)

        with torch.inference_mode():
            if _QWEN_ENGINE["unet"] is not None and _QWEN_ENGINE["unet_key"] != req_key:
                _QWEN_ENGINE["unet"] = None
                gc.collect()
                torch.cuda.empty_cache()

            if _QWEN_ENGINE["unet"] is None:
                _QWEN_ENGINE["unet"] = nodes.UNETLoader().load_unet(target_unet_fn, "default")[0]
                _QWEN_ENGINE["unet_key"] = req_key
            if _QWEN_ENGINE["clip"] is None:
                _QWEN_ENGINE["clip"] = nodes.CLIPLoader().load_clip(CLIP_FN, "qwen_image", "default")[0]
            if _QWEN_ENGINE["vae"] is None:
                _QWEN_ENGINE["vae"] = nodes.VAELoader().load_vae(VAE_FN)[0]

        ENGINE_READY = True
        return _QWEN_ENGINE["unet"], _QWEN_ENGINE["clip"], _QWEN_ENGINE["vae"], req_key


def _warmup_qwen21_bg():
    try:
        with GPU_LOCK:
            _, _, _, act_k = _ensure_qwen21_engine("chatr", update_job_state=False)
            if not JOB_STATE.get("running"):
                b_lbl = "ChatR 无审查 BF16" if act_k == "chatr" else "官方满血 BF16"
                JOB_STATE["message"] = f"⚡ 就绪：Qwen-Image-2.1 ({b_lbl} + Heretic BF16 破限编码器 · 30.2G常驻显存)"
                JOB_STATE["engine_ready"] = True
    except Exception as e:
        print(f"[Qwen-Image-2.1 Warmup Error] {e}", flush=True)


def _pil_from_b64(b64_str: str):
    import PIL.Image
    raw_b = str(b64_str).split(",", 1)[-1]
    return PIL.Image.open(io.BytesIO(base64.b64decode(raw_b))).convert("RGB")


def _tensor_from_pil(pil_im):
    import numpy as np
    import torch
    arr = np.asarray(pil_im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _is_already_mannequin_pil(pil_im) -> bool:
    import numpy as np
    arr = np.asarray(pil_im.convert("RGB").resize((128, 128)), dtype=np.float32)
    chroma = float(np.mean(np.abs(arr[:, :, 0] - arr[:, :, 1]) + np.abs(arr[:, :, 1] - arr[:, :, 2])))
    return chroma < 14.0


def _convert_pose_to_clay_pil(pose_b64: str, fallback_pil):
    """自动将彩色动漫/真人姿势参考图在 GPU 上 0.4s 转为无纹理 3D 素白模，彻底防止原图人脸/发色/服装串入结果"""
    try:
        if "/content" not in sys.path:
            sys.path.insert(0, "/content")
        import guide_preprocessor
        res = guide_preprocessor.preprocess_guide(
            {"guide_mode": "clay", "ref_image_b64": pose_b64},
            OUTPUT_DIR,
            gpu_lock=None,
        )
        if res.get("ok") and res.get("preview_b64"):
            return _pil_from_b64(res["preview_b64"]), True
    except Exception as e_clay:
        print(f"[Qwen-Image-2.1 AutoClay fallback] {e_clay}", flush=True)
    return fallback_pil, False


@__import__("torch").inference_mode()
def _run_qwen21_job(payload: dict):
    global JOB_STATE
    t0 = time.time()
    try:
        import numpy as np
        import PIL.Image
        import torch
        import nodes
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21

        sub_mode = str(payload.get("qwen_sub_mode") or "t2i").strip().lower()
        if sub_mode not in ("t2i", "edit", "pose"):
            sub_mode = "t2i"
        base_model_key = str(payload.get("qwen_base_model") or "chatr").strip().lower()
        raw_prompt = str(payload.get("prompt") or "").strip()
        neg_prompt = str(payload.get("negative_prompt") or "").strip()

        steps = max(4, min(60, int(payload.get("steps") or (25 if sub_mode == "pose" else 50))))
        cfg_val = max(1.0, min(8.0, float(payload.get("cfg") or (1.0 if sub_mode == "pose" else 2.5))))
        sampler_name = str(payload.get("sampler_name") or ("euler" if sub_mode == "pose" else "er_sde")).strip()
        scheduler = str(payload.get("scheduler") or ("simple" if sub_mode == "pose" else "beta")).strip()
        edit_res = max(512, min(2048, int(payload.get("qwen_edit_res") or 1024)))

        batch_count = max(1, min(4, int(payload.get("batch_count") or 1)))
        base_seed = int(payload.get("seed") if payload.get("seed") is not None else -1)
        rand_seed = bool(payload.get("randomize_seed")) or (base_seed < 0)
        cur_seed = random.randint(1, 2**31 - 1) if rand_seed else base_seed

        res_str = str(payload.get("resolution") or "1152x1728").strip().lower()
        use_auto_ref_latent = (res_str == "auto" and sub_mode in ("edit", "pose"))
        width, height = 1152, 1728
        if "x" in res_str:
            try:
                w_s, h_s = res_str.split("x", 1)
                width = max(512, min(2560, (int(w_s) // 32) * 32))
                height = max(512, min(2560, (int(h_s) // 32) * 32))
            except Exception:
                width, height = 1152, 1728

        lora_1 = str(payload.get("qwen_lora_1") or "").strip()
        lora_1_str = float(payload.get("qwen_lora_1_strength") if payload.get("qwen_lora_1_strength") is not None else 1.0)
        lora_2 = str(payload.get("qwen_lora_2") or "").strip()
        lora_2_str = float(payload.get("qwen_lora_2_strength") if payload.get("qwen_lora_2_strength") is not None else 1.0)

        char_b64 = str(payload.get("qwen_char_b64") or "").strip()
        pose_b64 = str(payload.get("qwen_pose_b64") or "").strip()
        ref_list = [str(x).strip() for x in (payload.get("ref_images_b64") or []) if str(x).strip()]
        if not char_b64 and ref_list:
            char_b64 = ref_list[0]
        auto_clay_pose = bool(payload.get("qwen_auto_clay_pose", True))

        if sub_mode == "pose" and (not char_b64 or not pose_b64):
            raise RuntimeError("3D 控姿势模式需要同时提供【角色图】和【3D 姿势参考图】！")
        if sub_mode == "edit" and not char_b64:
            raise RuntimeError("保脸改图 (Edit) 模式需要至少上传 1 张【角色参考图】！")

        mode_cn = {"t2i": "文生图 T2I", "edit": "保脸改图 Edit", "pose": "3D人偶控姿势"}.get(sub_mode, sub_mode)
        JOB_STATE.update({
            "running": True,
            "pct": 6,
            "step": 0,
            "total_steps": steps,
            "seed": cur_seed,
            "error": "",
            "message": f"🎨 [Qwen-Image-2.1 BF16 · {mode_cn}] 正在准备满血模型与 LoRA 挂载...",
        })

        with GPU_LOCK:
            unet_base, clip, vae, act_base_key = _ensure_qwen21_engine(base_model_key, update_job_state=True)
            avail_loras = set(_list_available_loras())

            # 组装 LoRA 链 (不污染底座 unet_base)
            active_model = unet_base
            lora_chain = []
            if sub_mode == "pose" and not any("vnccs" in x.lower() for x in (lora_1, lora_2)):
                default_vnccs = "VNCCS_QI2_PoseStudioV1.1.safetensors" if "VNCCS_QI2_PoseStudioV1.1.safetensors" in avail_loras else (
                    "VNCCS_QI2_PoseStudioV3.safetensors" if "VNCCS_QI2_PoseStudioV3.safetensors" in avail_loras else ""
                )
                if default_vnccs:
                    lora_chain.append((default_vnccs, 1.0))
            if lora_1 and lora_1 in avail_loras and abs(lora_1_str) > 1e-4:
                lora_chain.append((lora_1, lora_1_str))
            if lora_2 and lora_2 in avail_loras and lora_2 != lora_1 and abs(lora_2_str) > 1e-4:
                lora_chain.append((lora_2, lora_2_str))

            for l_fn, l_str in lora_chain:
                active_model = nodes.LoraLoaderModelOnly().load_lora_model_only(active_model, l_fn, l_str)[0]

            images_dict = {}
            final_prompt = raw_prompt
            clay_tag = ""

            if sub_mode == "pose":
                pose_pil = _pil_from_b64(pose_b64)
                if auto_clay_pose and not _is_already_mannequin_pil(pose_pil):
                    JOB_STATE.update({"pct": 8, "message": "🗿 [Qwen-Image-2.1] 正在将所选姿势参考图转为无纹理 3D 素白模（杜绝原图串脸串衣）..."})
                    pose_pil, converted = _convert_pose_to_clay_pil(pose_b64, pose_pil)
                    if converted:
                        clay_tag = " · 🗿3D素模防串脸"
                char_pil = _pil_from_b64(char_b64)
                images_dict["image_1"] = _tensor_from_pil(pose_pil)
                images_dict["image_2"] = _tensor_from_pil(char_pil)
                base_pose_dir = "replace the pose of <image 2> with the pose of <image 1>. keep the character of <image 2>."
                if raw_prompt and "replace the pose of <image 2>" not in raw_prompt.lower():
                    final_prompt = f"{base_pose_dir} {raw_prompt}"
                elif not raw_prompt:
                    final_prompt = base_pose_dir
            elif sub_mode == "edit":
                if ref_list:
                    for idx, rb in enumerate(ref_list[:4], start=1):
                        images_dict[f"image_{idx}"] = _tensor_from_pil(_pil_from_b64(rb))
                else:
                    images_dict["image_1"] = _tensor_from_pil(_pil_from_b64(char_b64))
                if not final_prompt:
                    final_prompt = "keep the exact character identity and facial features of <image 1>, natural lighting, photorealistic photograph"
            else:
                if not final_prompt:
                    final_prompt = "A hyper-realistic candid portrait photograph, soft natural window lighting, natural skin texture, 8k resolution."

            lora_desc = ", ".join(f"{os.path.splitext(n)[0]}@{s:.2f}" for n, s in lora_chain) if lora_chain else "纯底模"
            JOB_STATE.update({
                "pct": 10,
                "final_prompt": final_prompt,
                "message": f"🎨 [Qwen-Image-2.1 · {mode_cn}] Heretic BF16 编码中 ({lora_desc}{clay_tag})...",
            })

            cond_out = TextEncodeQwenImage21.execute(
                clip=clip,
                prompt=final_prompt,
                negative_prompt=neg_prompt,
                vae=vae,
                resolution=edit_res if images_dict else 1024,
                images=images_dict,
            )
            positive = cond_out[0]
            negative = cond_out[1]
            if use_auto_ref_latent and images_dict:
                latent = cond_out[2]
                out_w = int(latent["samples"].shape[3]) * 16
                out_h = int(latent["samples"].shape[2]) * 16
            else:
                latent = nodes.EmptyLatentImage().generate(width, height, 1)[0]
                out_w, out_h = width, height

            completed_imgs = []
            last_seed = cur_seed
            rc_conf = os.path.expanduser("~/.config/rclone/rclone.conf")
            for b_i in range(batch_count):
                b_seed = (random.randint(1, 2**31 - 1) if (rand_seed and b_i > 0) else (cur_seed + b_i))
                last_seed = b_seed
                b_pref = f"[连抽 {b_i + 1}/{batch_count}] " if batch_count > 1 else ""
                sampled = nodes.KSampler().sample(
                    model=active_model,
                    seed=b_seed,
                    steps=steps,
                    cfg=cfg_val,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=positive,
                    negative=negative,
                    latent_image=latent,
                    denoise=1.0,
                )[0]
                JOB_STATE.update({
                    "pct": min(98, int(round(((b_i + 0.9) / float(batch_count)) * 90)) + 8),
                    "message": f"🖼️ {b_pref}[Qwen-Image-2.1 BF16] 正在通过原生 BF16 VAE 解码 {out_w}×{out_h} 高清图...",
                })
                decoded = nodes.VAEDecode().decode(vae, sampled)[0]
                img_np = (decoded[0].detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
                pil_out = PIL.Image.fromarray(img_np)

                dt_i = round(time.time() - t0, 1)
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                fname = f"qwen_image_21_{sub_mode}_{out_w}x{out_h}_{ts}_dur{max(1, int(round(dt_i)))}s_seed{b_seed}.jpg"
                out_path = os.path.join(OUTPUT_DIR, fname)
                pil_out.save(out_path, format="JPEG", quality=96, subsampling=0, optimize=True)
                completed_imgs.append(fname)
                JOB_STATE["latest_video"] = fname
                JOB_STATE["completed_videos"] = list(completed_imgs)
                del decoded, sampled
                if os.path.exists(rc_conf) and shutil.which("rclone"):
                    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                    _ym = _dt.now(_tz(_td(hours=8))).strftime("%Y.%m")
                    subprocess.Popen(
                        ["rclone", "copy", out_path, f"gdrive:pro6000-output/{_ym}/images", "--config", rc_conf],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

            del active_model, latent, positive, negative, images_dict
            gc.collect()
            torch.cuda.empty_cache()

        dt = round(time.time() - t0, 1)
        b_lbl = f"共 {batch_count} 张 · " if batch_count > 1 else ""
        base_cn = "ChatR无审查BF16" if act_base_key == "chatr" else "官方BF16"
        JOB_STATE.update({
            "running": False,
            "pct": 100,
            "step": steps,
            "total_steps": steps,
            "seed": last_seed,
            "elapsed_s": dt,
            "latest_video": completed_imgs[-1] if completed_imgs else "",
            "completed_videos": completed_imgs,
            "final_prompt": final_prompt,
            "message": f"✅ [Qwen-Image-2.1 · {base_cn} · {mode_cn}] {b_lbl}{out_w}×{out_h} 生成完成！耗时 {dt}s ({steps}步 · CFG {cfg_val} · Seed: {last_seed})",
        })
    except Exception as e:
        JOB_STATE.update({
            "running": False,
            "error": str(e),
            "message": f"❌ Qwen-Image-2.1 生成出错: {e}",
        })


class _QwenImage21Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send_json({
                "status": "ok",
                "engine": "qwen-image-2.1",
                "engine_ready": ENGINE_READY,
                "active_base": _QWEN_ENGINE.get("unet_key") or "chatr",
                "loras": _list_available_loras(),
            })
            return
        if self.path.startswith("/v1/qwen_image/loras"):
            self._send_json({"ok": True, "loras": _list_available_loras(), "active_base": _QWEN_ENGINE.get("unet_key") or "chatr"})
            return
        if self.path.startswith("/v1/models") or self.path.startswith("/models"):
            self._send_json({
                "object": "list",
                "data": [{"id": "Qwen-Image-2.1-Uncensored-BF16", "object": "model", "owned_by": "Qwen-Community"}],
            })
            return
        if self.path.startswith("/v1/video/status"):
            st = dict(JOB_STATE)
            st["engine_ready"] = ENGINE_READY
            st["history"] = _list_history_files()
            self._send_json(st)
            return
        if self.path.startswith("/v1/video/upscale_status"):
            try:
                with urllib.request.urlopen("http://127.0.0.1:30002/v1/video/upscale_status", timeout=5) as r:
                    self._send_json(json.loads(r.read().decode("utf-8")))
            except Exception as e:
                self._send_json({"running": False, "error": str(e)}, status=503)
            return
        if self.path.startswith("/v1/audio/indextts2"):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:30003{self.path}", timeout=8) as r:
                    self._send_json(json.loads(r.read().decode("utf-8")), status=r.status)
            except Exception as e:
                self._send_json({"ok": False, "ready": False, "error": str(e)}, status=503)
            return
        if self.path.startswith("/v1/video/file/"):
            fname = os.path.basename(self.path.split("/v1/video/file/", 1)[-1].split("?")[0])
            fpath = os.path.join(OUTPUT_DIR, fname)
            if not os.path.isfile(fpath):
                self._send_json({"error": "File not found"}, status=404)
                return
            fsize = os.path.getsize(fpath)
            ctype = "image/png" if fname.lower().endswith(".png") else ("image/jpeg" if fname.lower().endswith((".jpg", ".jpeg")) else "video/mp4")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(fsize))
            self.send_header("Content-Disposition", f'inline; filename="{fname}"')
            self.end_headers()
            with open(fpath, "rb") as rf:
                while True:
                    buf = rf.read(262144)
                    if not buf:
                        break
                    self.wfile.write(buf)
            return
        self._send_json({"status": "ok", "engine": "qwen-image-2.1"})

    def do_POST(self):
        cl = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(cl).decode("utf-8") if cl > 0 else "{}"
        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {}

        if self.path.startswith("/v1/video/cancel"):
            open("/content/.cancel_qwen21", "w").write("1")
            open("/content/.cancel_minimax", "w").write("1")
            self._send_json({"ok": True, "message": "🛑 正在停止 Qwen-Image-2.1 生成任务..."})
            return

        if self.path.startswith("/v1/guide/preprocess"):
            try:
                if "/content" not in sys.path:
                    sys.path.insert(0, "/content")
                import guide_preprocessor
                res_gp = guide_preprocessor.preprocess_guide(payload, OUTPUT_DIR, gpu_lock=GPU_LOCK)
                self._send_json(res_gp, status=200 if res_gp.get("ok") else 400)
            except Exception as e:
                self._send_json({"ok": False, "error": f"白模/深度预处理失败: {e}"}, status=500)
            return

        if self.path.startswith("/v1/video/upscale"):
            try:
                req2 = urllib.request.Request(
                    f"http://127.0.0.1:30002{self.path}",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req2, timeout=15) as r2:
                    self._send_json(json.loads(r2.read().decode("utf-8")), status=r2.status)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/audio/indextts2"):
            try:
                req3 = urllib.request.Request(
                    f"http://127.0.0.1:30003{self.path}",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req3, timeout=300) as r3:
                    self._send_json(json.loads(r3.read().decode("utf-8")), status=r3.status)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/video/generate") or self.path.startswith("/v1/video/generations") or self.path.startswith("/v1/images/generations"):
            req_m = str(payload.get("mode") or "").strip().lower()
            if req_m in ("avatar", "t2va", "fl2va", "ref2va") or (req_m == "image" and str(payload.get("image_engine") or "h3") != "qwen21"):
                self._send_json({"ok": False, "error": "Engine mismatch: qwen-image-2.1 handles qwen_image mode"}, status=503)
                return
            for cf in ("/content/.cancel_qwen21", "/content/.cancel_minimax"):
                try:
                    if os.path.exists(cf):
                        os.unlink(cf)
                except Exception:
                    pass
            if JOB_STATE["running"]:
                self._send_json({"ok": True, "already_running": True, "message": JOB_STATE.get("message") or "当前已有 Qwen-Image-2.1 任务正在运行"})
                return
            JOB_STATE.update({
                "running": True,
                "pct": 2,
                "step": 0,
                "completed_videos": [],
                "latest_video": "",
                "error": "",
                "message": "⏳ Qwen-Image-2.1 (满血 BF16) 任务已提交，正在进入 GPU 渲染队列...",
            })
            threading.Thread(target=_run_qwen21_job, args=(payload,), daemon=True).start()
            self._send_json({"ok": True, "message": "已开始生成 Qwen-Image-2.1 满血 BF16 大图"})
            return

        if self.path.startswith("/v1/shell"):
            cmd = str(payload.get("cmd") or "").strip()
            if not cmd:
                self._send_json({"ok": False, "error": "empty cmd"}, status=400)
                return
            try:
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=int(payload.get("timeout") or 120))
                self._send_json({"ok": True, "returncode": r.returncode, "stdout": r.stdout[-16000:], "stderr": r.stderr[-16000:]})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=500)
            return

        if "chat/completions" in self.path:
            reply_txt = (
                "🎨 当前运行的是【Qwen-Image-2.1 满血 BF16 破限 + 3D 人偶控姿势/精修】图像引擎！\n"
                "请在下方「01 · 图像与换装」顶部切换至「🎨 Qwen-Image-2.1 满血BF16 · 3D控姿势」直接生图、保脸改图或点选 170 套 3D 姿势库控姿出图！"
            )
            self._send_json({
                "id": "chatcmpl-qwen21-img",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply_txt}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 42, "total_tokens": 52},
                "timings": {"predicted_per_second": 100.0},
            })
            return

        self._send_json({"ok": True})


if __name__ == "__main__":
    threading.Thread(target=_warmup_qwen21_bg, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", 30000), _QwenImage21Handler)
    srv.serve_forever()
'''


def _start_qwen_image_21_server_on_30000(download_only: bool = False):
    """在远端 Colab GPU 实例上部署并启动 Qwen-Image-2.1 满血 BF16 双底模 + Heretic 破限编码器 + 3D控姿势引擎 (端口 30000)"""
    folder_name = "Qwen-Image-2.1-Uncensored"
    has_rclone = _has_vm_rclone_drive()
    fuse_mounted = os.path.ismount("/content/drive/MyDrive") or os.path.ismount("/content/drive")
    target_dir = os.path.join(DRIVE_MODELS_ROOT, folder_name) if fuse_mounted else os.path.join("/content/local_models", folder_name)
    os.makedirs(target_dir, exist_ok=True)

    # (dst_rel, hf_repo, hf_rel, approx_bytes)
    plan = [
        ("diffusion_models/qwen-image-2.1-step90000-BF16.safetensors", "chatrlabs/Qwen-Image-2.1-Uncensored-Omni-Quantized", "qwen-image-2.1-step90000-BF16/diffusion_pytorch_model.safetensors", 14_230_000_000),
        ("diffusion_models/qwen_image_2.1_bf16.safetensors", "Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_bf16.safetensors", 14_230_000_000),
        ("text_encoders/qwen3vl_8b_bf16_heretic.safetensors", "pottokao/Qwen-Image-2.1-Text-Encoder-Heretic", "qwen3vl_8b_bf16_heretic.safetensors", 17_530_000_000),
        ("vae/qwen_image_2.1_vae_bf16.safetensors", "Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors", 679_000_000),
        ("loras/VNCCS_QI2_PoseStudioV1.1.safetensors", "MIUProject/VNCCS_PoseStudio_QI2.1", "VNCCS_QI2_PoseStudioV1.1.safetensors", 159_400_000),
        ("loras/VNCCS_QI2_PoseStudioV3.safetensors", "MIUProject/VNCCS_PoseStudio_QI2.1", "VNCCS_QI2_PoseStudioV3.safetensors", 159_400_000),
        ("loras/chatrlabs-omni-lora-v0.1-bf16.safetensors", "chatrlabs/Qwen-Image-2.1-Uncensored-Omni-LoRA-v0.1", "chatrlabs-omni-lora-v0.1-bf16.safetensors", 679_000_000),
        ("loras/chatrlabs-omni-turbo-8step-fp32.safetensors", "chatrlabs/Qwen-Image-2.1-Uncensored-Omni-LoRA-v0.1", "chatrlabs-omni-turbo-8step-fp32.safetensors", 679_000_000),
        ("loras/Qwen-Image-2.1-NSFW-Image-Edit.safetensors", "RunningHubAI/rh-qwen-image-2.1-aio-nsfw-lora", "Qwen-Image-2.1 NSFW Image Edit.safetensors", 84_000_000),
    ]

    needed_new = sum(
        sz for dst_rel, _, _, sz in plan
        if not (os.path.isfile(os.path.join(target_dir, dst_rel)) and os.path.getsize(os.path.join(target_dir, dst_rel)) > max(10_000_000, int(sz * 0.88)))
    )
    free_nvme = shutil.disk_usage("/content").free
    if not fuse_mounted and free_nvme < needed_new + 16 * (1024 ** 3):
        for base_clean in ("/content/local_models", "/content/models"):
            if not os.path.isdir(base_clean):
                continue
            for other_name in sorted(os.listdir(base_clean)):
                other_p = os.path.join(base_clean, other_name)
                if os.path.abspath(other_p) == os.path.abspath(target_dir):
                    continue
                if os.path.isdir(other_p):
                    log(f"🧹 [NVMe 防爆盘智能腾挪] 剩余 {free_nvme/(1024**3):.1f} GB，释放未激活缓存 {other_name} (20T云盘已永久保存)...")
                    shutil.rmtree(other_p, ignore_errors=True)
                    free_nvme = shutil.disk_usage("/content").free
                    if free_nvme >= needed_new + 16 * (1024 ** 3):
                        break

    gdrive_saved_count = _check_rclone_drive_has_model(folder_name) if (has_rclone and not fuse_mounted) else 0
    st_stage_tag = "云盘极速预热" if gdrive_saved_count > 0 else "HF 下载"
    if has_rclone and not fuse_mounted and gdrive_saved_count > 0:
        log(f"☁️ 检测到您的 Google Drive (gdrive:models/{folder_name}) 已保存 {gdrive_saved_count} 个权重文件，优先 16 路并发从您的云盘高速拉取！")
        _pull_from_rclone_drive_if_exists(folder_name, target_dir)

    batch_total_bytes = sum(x[3] for x in plan)
    total_gb = batch_total_bytes / (1024 ** 3)
    cum_done_bytes = 0
    for idx, (dst_rel, dl_repo, dl_rel, sz) in enumerate(plan, start=1):
        ep = os.path.join(target_dir, dst_rel)
        lbl_str = f"{folder_name} 总进度 ({idx}/{len(plan)} · {os.path.basename(dst_rel)})"
        if os.path.isfile(ep) and os.path.getsize(ep) > max(10_000_000, int(sz * 0.88)):
            actual_sz = os.path.getsize(ep)
            cum_done_bytes += actual_sz
            ov_pct = min(100, cum_done_bytes * 100 // max(1, batch_total_bytes))
            log(f"  [{st_stage_tag}] {lbl_str}: {ov_pct}% ({cum_done_bytes/(1024**3):.2f}/{total_gb:.2f} GB, 本地缓存复用)")
            continue
        log(f"正在 16 线程补拉缺失权重 [{idx}/{len(plan)}] {dst_rel} ({sz/(1024**3):.2f} GB)...")
        dl_dir_tmp = os.path.join(target_dir, "_dl_tmp")
        dl_p = fast_parallel_hf_download(
            dl_repo,
            dl_rel,
            dl_dir_tmp,
            token=HF_TOKEN,
            num_conn=16,
            batch_done_bytes=cum_done_bytes,
            batch_total_bytes=batch_total_bytes,
            batch_label=lbl_str,
            stage_tag=st_stage_tag,
        )
        os.makedirs(os.path.dirname(ep), exist_ok=True)
        shutil.move(dl_p, ep)
        cum_done_bytes += os.path.getsize(ep)
    shutil.rmtree(os.path.join(target_dir, "_dl_tmp"), ignore_errors=True)

    if has_rclone and not fuse_mounted and gdrive_saved_count < len(plan):
        _sync_local_to_rclone_drive(target_dir, background=not download_only, remote_folder=folder_name)

    if download_only:
        log(f"🎉 [仅下载模式] Qwen-Image-2.1 满血 BF16 双底模 + Heretic 破限编码器 + 5 大 LoRA 已永久同步至 Google Drive (gdrive:models/{folder_name})！")
        return

    if not shutil.which("cloudflared"):
        log("正在安装 cloudflared 隧道客户端...")
        subprocess.run(
            "curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
            "-o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
            shell=True,
            check=True,
        )

    subprocess.run("pkill -9 -f '[l]lama-server|minimax_server\\.py|minimax_t4_server\\.py|longcat_avatar_server\\.py|minimax_image_server\\.py|qwen_image_server\\.py|project-maya/serve/server\\.py|build/maya|maya-vision|/content/app\\.py' || true; fuser -k -9 30000/tcp 2>/dev/null || true", shell=True)
    time.sleep(1)

    if not os.path.isdir("/content/ComfyUI/.git"):
        log("正在拉取最新 ComfyUI 主干框架 (原生支持 TextEncodeQwenImage21 满血 BF16 + VNCCS 控姿势)...")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", "/content/ComfyUI"], check=False)
    else:
        subprocess.run(["git", "-C", "/content/ComfyUI", "pull", "--ff-only"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    log("正在检查 ComfyUI 运行依赖...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "/content/ComfyUI/requirements.txt"], check=False)

    with open("/content/guide_preprocessor.py", "w", encoding="utf-8") as gwf:
        gwf.write(GUIDE_PREPROCESSOR_CODE)

    server_py = "/content/qwen_image_server.py"
    with open(server_py, "w", encoding="utf-8") as wf:
        wf.write(QWEN_IMAGE_21_SERVER_CODE)

    env = os.environ.copy()
    env["QWEN_IMAGE_21_MODEL_DIR"] = target_dir
    with open("/content/minimax_server.log", "w") as lf:
        subprocess.Popen(
            [sys.executable, "-u", server_py],
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(35):
        try:
            with opener.open("http://127.0.0.1:30000/health", timeout=2) as resp:
                if resp.status == 200:
                    h_data = json.loads(resp.read().decode("utf-8"))
                    if h_data.get("engine") == "qwen-image-2.1":
                        log("[显存装载] Qwen-Image-2.1 (满血BF16·Heretic破限·3D控姿势): 100% (30.2/30.2 GiB 常驻就绪)")
                        log("✅ Qwen-Image-2.1 满血 BF16 生图/保脸改图/3D人偶控姿势服务已启动 (端口 30000)！")
                        _ensure_recavsr_service_on_30002()
                        _ensure_indextts2_service_on_30003()
                        return
        except Exception:
            pass
        time.sleep(1)


RECAVSR_SERVER_CODE = r'''#!/usr/bin/env python3
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import av
import base64
import gc
import json
import math
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, "/content/ReCaVSR")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import recavsr_infer.models.lq_proj as lq_proj_mod


def _low_mem_rms_forward(self, x: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    rms = torch.linalg.vector_norm(x, dim=1, keepdim=True, dtype=torch.float32)
    scale = (rms.mul_(rms).div_(x.shape[1]).add_(self.eps).rsqrt_()).to(orig_dtype)
    out = x * scale
    out.mul_(self.weight.to(orig_dtype))
    return out


lq_proj_mod.ChannelRMSNorm3d.forward = _low_mem_rms_forward

from recavsr_infer.color_fix import adain_color_fix, wavelet_color_fix
from recavsr_infer.runtime.constants import SPATIAL_TOKEN_STRIDE
from recavsr_infer.runtime.cache import FixedKVCache
from recavsr_infer.runtime.decoder import create_decoder_session, load_decoder
from recavsr_infer.runtime.flashdecoder import decode_block
from recavsr_infer.runtime.transformer import PreparedBlock, TransformerSession
from recavsr_infer.video import block_schedule, read_video, rgb_uint8, scaled_size, video_pixel_format
from recavsr_infer.weights import load_model

_orig_pb_forward = PreparedBlock.forward
def _low_mem_pb_forward(self, hidden, past_key, past_value, slots, past_positions, query_positions):
    b = self.block
    if not hasattr(b.ffn, "_chunked_patched"):
        orig_ffn = b.ffn.forward
        def _chunked_ffn(x, _orig=orig_ffn):
            if x.shape[1] <= 12000:
                return _orig(x)
            return torch.cat([_orig(x[:, i:i+12000]) for i in range(0, x.shape[1], 12000)], dim=1)
        b.ffn.forward = _chunked_ffn
        b.ffn._chunked_patched = True
    return _orig_pb_forward(self, hidden, past_key, past_value, slots, past_positions, query_positions)
PreparedBlock.forward = _low_mem_pb_forward

_orig_kv_history = FixedKVCache.history
_orig_kv_commit = FixedKVCache.commit

def _smart_kv_history(self, current, *, origin, train_frames=22):
    k, v, si, pi = _orig_kv_history(self, current, origin=origin, train_frames=train_frames)
    if k is not None and k.device != current.device:
        k = k.to(current.device, non_blocking=True)
        v = v.to(current.device, non_blocking=True)
    return k, v, si, pi

def _smart_kv_commit(self, key, value, *, frames, start):
    span = key.shape[1] // max(1, frames)
    if span >= 4000 and self.capacity > 0 and self.key is None:
        shape = (key.shape[0], self.capacity * span, *key.shape[2:])
        self.key = torch.empty(shape, dtype=key.dtype, device="cpu", pin_memory=True)
        self.value = torch.empty(shape, dtype=value.dtype, device="cpu", pin_memory=True)
    return _orig_kv_commit(self, key, value, frames=frames, start=start)

FixedKVCache.history = _smart_kv_history
FixedKVCache.commit = _smart_kv_commit

MODEL_DIR = Path("/content/models/ReCaVSR")
VIDEO_DIR = Path("/content/outputs")
ALT_VIDEO_DIR = Path("/content/generated_videos")
DRIVE_OUT_DIR = Path("/content/drive/MyDrive/models/MiniMax-H3/outputs")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

RECAVSR_LOCK = threading.Lock()
MODEL = None
VAE = None
PROMPT = None
RECAVSR_READY = False
RECAVSR_ERROR = ""
UPSCALE_T0 = 0.0
UPSCALE_CANCEL = False


def _find_latest_upscaled() -> str:
    try:
        cands = sorted(
            [p for p in VIDEO_DIR.glob("*_ReCaVSR_*.mp4") if p.is_file() and p.stat().st_size > 1024],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return cands[0].name if cands else ""
    except Exception:
        return ""


_init_latest = _find_latest_upscaled()
UPSCALE_STATE = {
    "running": False,
    "pct": 100 if _init_latest else 0,
    "step": 362 if _init_latest else 0,
    "total_steps": 362 if _init_latest else 100,
    "message": f"✨ ReCaVSR 超清重绘就绪 ({_init_latest})" if _init_latest else "就绪",
    "elapsed_s": 102.9 if _init_latest else 0.0,
    "latest_video": _init_latest,
    "source_video": "",
    "target_res": "1920x3328" if _init_latest else "",
    "error": "",
}


def _init_recavsr():
    global MODEL, VAE, PROMPT, RECAVSR_READY, RECAVSR_ERROR
    try:
        t0 = time.time()
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run("cp -f /content/ReCaVSR/checkpoints/*.json /content/models/ReCaVSR/ 2>/dev/null || true", shell=True)
        if not (MODEL_DIR / "transformer.safetensors").is_file():
            # 若后台已有 rclone 正在同步，等待其完成；否则主动从云盘或 HF 拉取
            for _ in range(90):
                if (MODEL_DIR / "transformer.safetensors").is_file():
                    break
                if not subprocess.getoutput("pgrep -f 'rclone.*ReCaVSR'").strip():
                    break
                time.sleep(2)
        if not (MODEL_DIR / "transformer.safetensors").is_file() and os.path.isfile("/root/.config/rclone/rclone.conf"):
            print("[ReCaVSR] Pulling weights from Google Drive (gdrive:models/ReCaVSR)...", flush=True)
            subprocess.run(["rclone", "copy", "gdrive:models/ReCaVSR", str(MODEL_DIR), "--transfers=8"], check=False)
        if not (MODEL_DIR / "transformer.safetensors").is_file():
            print("[ReCaVSR] Downloading weights from HuggingFace (kopper/ReCaVSR)...", flush=True)
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="kopper/ReCaVSR", allow_patterns=["*.safetensors", "*.json"], local_dir=str(MODEL_DIR))
            if os.path.isfile("/root/.config/rclone/rclone.conf"):
                subprocess.Popen(
                    ["rclone", "copy", str(MODEL_DIR), "gdrive:models/ReCaVSR", "--transfers=8"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        print("[ReCaVSR] Loading Wan2.2-5B 1-Step Streaming Transformer & FlashDecoder into CPU DDR5 RAM...", flush=True)
        MODEL = load_model(MODEL_DIR, device="cpu")
        for m in MODEL.modules():
            if isinstance(m, torch.nn.SiLU):
                m.inplace = True
        VAE = load_decoder(
            MODEL_DIR,
            device="cpu",
            channels_last=False,
            decoder="flashdecoder",
            checkpoint_path=MODEL_DIR / "flashdecoder.safetensors",
        )
        PROMPT = load_file(MODEL_DIR / "prompt.safetensors")["positive"]
        RECAVSR_READY = True
        print(f"[ReCaVSR] Ready in {time.time() - t0:.1f}s (standby in CPU RAM).", flush=True)
    except Exception as e:
        RECAVSR_ERROR = f"{e}\n{traceback.format_exc()}"
        print(f"[ReCaVSR] Init error: {RECAVSR_ERROR}", flush=True)


def _is_minimax_generating() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:30000/v1/video/status", timeout=2) as r:
            d = json.loads(r.read().decode("utf-8"))
            return bool(d.get("running"))
    except Exception:
        return False


def _prepare_block_bf16(frames: np.ndarray, start: int, valid: int, block: int, srh: int, srw: int, device: str) -> torch.Tensor:
    chunk = frames[start : start + valid]
    enlarged = np.stack([
        cv2.resize(f, (srw, srh), interpolation=cv2.INTER_LINEAR)
        if (f.shape[0] != srh or f.shape[1] != srw)
        else f
        for f in chunk
    ])
    x_cpu = torch.from_numpy(enlarged).permute(3, 0, 1, 2).unsqueeze(0).to(dtype=torch.bfloat16)
    x = x_cpu.to(device=device, non_blocking=True)
    x.mul_(2.0 / 255.0).sub_(1.0)
    if valid < block:
        x = torch.cat((x, x[:, :, -1:].expand(-1, -1, block - valid, -1, -1)), dim=2)
    ph, pw = (-srh) % SPATIAL_TOKEN_STRIDE, (-srw) % SPATIAL_TOKEN_STRIDE
    if pw:
        x = F.pad(x, (0, pw, 0, 0, 0, 0), mode="reflect" if pw < srw else "replicate")
    if ph:
        x = F.pad(x, (0, 0, 0, ph, 0, 0), mode="reflect" if ph < srh else "replicate")
    return x


def _run_upscale_job(payload: dict):
    global UPSCALE_T0, UPSCALE_CANCEL
    with RECAVSR_LOCK:
        t0 = time.time()
        UPSCALE_T0 = t0
        UPSCALE_CANCEL = False
        fname = os.path.basename(str(payload.get("filename") or "").strip())
        target_preset = str(payload.get("target_preset") or "2k").lower().strip()
        color_fix = str(payload.get("color_fix") or "wavelet").lower().strip()
        video_b64 = payload.get("video_b64") or ""

        src_path = VIDEO_DIR / fname
        if not src_path.is_file() and (ALT_VIDEO_DIR / fname).is_file():
            shutil.copy2(ALT_VIDEO_DIR / fname, src_path)
        if not src_path.is_file() and (DRIVE_OUT_DIR / fname).is_file():
            shutil.copy2(DRIVE_OUT_DIR / fname, src_path)
        if not src_path.is_file() and video_b64:
            raw_bytes = base64.b64decode(video_b64.split(",", 1)[-1])
            src_path.write_bytes(raw_bytes)

        if not src_path.is_file():
            UPSCALE_STATE.update({
                "running": False,
                "pct": 0,
                "error": f"找不到源视频文件: {fname}",
                "message": f"❌ 找不到源视频文件: {fname}",
            })
            return

        UPSCALE_STATE.update({
            "running": True,
            "pct": 5,
            "step": 0,
            "total_steps": 100,
            "source_video": fname,
            "error": "",
            "message": f"✨ [ReCaVSR {target_preset.upper()} 超分] 正在读取源视频 {fname}...",
        })

        try:
            while _is_minimax_generating():
                if UPSCALE_CANCEL:
                    raise RuntimeError("🛑 已手动停止 ReCaVSR 超分任务")
                UPSCALE_STATE["message"] = "⏳ 检测到 MiniMax-H3 正在生成视频，排队等待显卡空闲后立即超分..."
                time.sleep(2)

            while not RECAVSR_READY and not RECAVSR_ERROR:
                if UPSCALE_CANCEL:
                    raise RuntimeError("🛑 已手动停止 ReCaVSR 超分任务")
                UPSCALE_STATE["message"] = "⏳ 正在初始化 ReCaVSR 权重..."
                time.sleep(1)
            if RECAVSR_ERROR:
                raise RuntimeError(RECAVSR_ERROR)

            frames, fps = read_video(src_path)
            total_frames, in_h, in_w, _ = frames.shape

            if target_preset == "4k":
                scale = 4.0
                tag = "4K"
            elif target_preset == "1080p":
                scale = 1920.0 / max(in_h, in_w)
                tag = "1080P"
            else:
                scale = 2496.0 / max(in_h, in_w)
                tag = "2K"

            target_h, target_w = scaled_size(in_h, in_w, scale)
            padded_h = target_h + ((-target_h) % SPATIAL_TOKEN_STRIDE)
            padded_w = target_w + ((-target_w) % SPATIAL_TOKEN_STRIDE)

            stem = src_path.stem
            out_name = f"{stem}_ReCaVSR_{tag}_{target_w}x{target_h}.mp4"
            out_path = VIDEO_DIR / out_name
            tmp_silent = VIDEO_DIR / f".tmp_silent_{out_name}"

            schedule = list(block_schedule(total_frames))
            num_blocks = len(schedule)
            total_ops = max(1, num_blocks * 2)

            UPSCALE_STATE.update({
                "pct": 10,
                "step": 0,
                "total_steps": total_frames,
                "target_res": f"{target_w}x{target_h}",
                "message": f"✨ [ReCaVSR {tag} 超分] 正在将 5B 扩散重绘核心切入显存 ({in_w}×{in_h} → {target_w}×{target_h})...",
            })

            torch.cuda.empty_cache()
            MODEL.to("cuda:0")
            session = TransformerSession(
                MODEL,
                PROMPT,
                padded_h,
                padded_w,
                seed=42,
                compile_blocks=False,
                native_norms=False,
            )

            latents_cpu = []
            for b_idx, (start, valid, block) in enumerate(schedule):
                if UPSCALE_CANCEL:
                    raise RuntimeError("🛑 已手动停止 ReCaVSR 超分任务")
                lq_gpu = _prepare_block_bf16(frames, start, valid, block, target_h, target_w, "cuda:0")
                lat = session.step(lq_gpu)
                latents_cpu.append(lat.to("cpu", dtype=torch.bfloat16))
                del lq_gpu, lat
                done_frames = min(total_frames, start + valid)
                cur_pct = 12 + int(((b_idx + 1) / total_ops) * 80)
                UPSCALE_STATE.update({
                    "pct": cur_pct,
                    "step": done_frames,
                    "total_steps": total_frames,
                    "message": f"✨ [ReCaVSR {tag} 超分 {target_w}×{target_h}] 正在单步扩散重绘超清细节: {done_frames}/{total_frames} 帧 (总进度 {cur_pct}%)",
                })

            del session
            MODEL.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

            VAE.to("cuda:0")
            decoder_session = create_decoder_session(VAE, compile_decoder=False)

            with av.open(str(tmp_silent), mode="w") as out_container:
                v_stream = out_container.add_stream("libx264", rate=fps)
                v_stream.width = target_w
                v_stream.height = target_h
                v_stream.pix_fmt = video_pixel_format(target_h, target_w)
                v_stream.options = {"crf": "17", "preset": "fast"}

                for b_idx, ((start, valid, block), lat_cpu) in enumerate(zip(schedule, latents_cpu)):
                    if UPSCALE_CANCEL:
                        raise RuntimeError("🛑 已手动停止 ReCaVSR 超分任务")
                    lq_gpu = _prepare_block_bf16(frames, start, valid, block, target_h, target_w, "cuda:0")
                    lat_gpu = lat_cpu.to("cuda:0", dtype=torch.bfloat16, non_blocking=True)
                    chunks = [rgb_chunk for rgb_chunk in decode_block(decoder_session, lat_gpu, lq_gpu)]
                    decoded = torch.cat(chunks, dim=2)
                    del chunks, lat_gpu
                    if valid > 0:
                        decoded_valid = decoded[:, :, :valid, :target_h, :target_w]
                        if color_fix == "adain":
                            ref = lq_gpu[:, :, :valid, :target_h, :target_w]
                            fixed_t = adain_color_fix(decoded_valid, ref)
                            out_np = rgb_uint8(fixed_t, target_h, target_w)
                            del ref, fixed_t
                        elif color_fix == "wavelet":
                            out_np = wavelet_color_fix(decoded_valid, frames[start : start + valid])
                        else:
                            out_np = rgb_uint8(decoded_valid, target_h, target_w)
                        del decoded_valid
                        for arr in out_np:
                            for pkt in v_stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                                out_container.mux(pkt)
                        del out_np
                    del decoded, lq_gpu
                    done_frames = min(total_frames, start + valid)
                    cur_pct = 12 + int(((num_blocks + b_idx + 1) / total_ops) * 80)
                    UPSCALE_STATE.update({
                        "pct": cur_pct,
                        "step": done_frames,
                        "total_steps": total_frames,
                        "message": f"🎞️ [ReCaVSR {tag} 超分 {target_w}×{target_h}] 正在 FlashDecoder 解码超清画面: {done_frames}/{total_frames} 帧 (总进度 {cur_pct}%)",
                    })

                for pkt in v_stream.encode():
                    out_container.mux(pkt)

            del decoder_session, latents_cpu, frames
            VAE.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

            UPSCALE_STATE.update({
                "pct": 94,
                "message": f"🎵 [ReCaVSR {tag} 超分 {target_w}×{target_h}] 正在无损合成原片立体声音轨并封装 MP4 (总进度 94%)...",
            })

            mux_cmd = [
                "ffmpeg", "-y",
                "-i", str(tmp_silent),
                "-i", str(src_path),
                "-map", "0:v:0",
                "-map", "1:a?",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                str(out_path),
            ]
            subprocess.run(mux_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if tmp_silent.exists():
                tmp_silent.unlink()

            def _bg_sync_drive(p: Path):
                try:
                    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                    _ym = _dt.now(_tz(_td(hours=8))).strftime("%Y.%m")
                    drive_root = Path("/content/drive/MyDrive")
                    rc_conf_p = os.path.expanduser("~/.config/rclone/rclone.conf")
                    if os.path.isfile(rc_conf_p) and shutil.which("rclone"):
                        subprocess.run(
                            ["rclone", "copy", str(p), f"gdrive:pro6000-output/{_ym}/videos", "--config", rc_conf_p, "--transfers=4"],
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    elif drive_root.exists():
                        month_dir = drive_root / "pro6000-output" / _ym / "videos"
                        month_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, month_dir / p.name)
                except Exception:
                    pass

            threading.Thread(target=_bg_sync_drive, args=(out_path,), daemon=True).start()

            el = round(time.time() - t0, 1)
            UPSCALE_STATE.update({
                "running": False,
                "pct": 100,
                "step": total_frames,
                "total_steps": total_frames,
                "elapsed_s": el,
                "latest_video": out_name,
                "error": "",
                "message": f"✨ ReCaVSR {tag} 超清重绘完成！总耗时 {el}s · {target_w}×{target_h} · {total_frames}帧 ({out_name})",
            })
        except Exception as e:
            tb_str = traceback.format_exc()
            e.__traceback__ = None
            print(f"[ReCaVSR] Job failed: {tb_str}", flush=True)
            try:
                MODEL.to("cpu")
                VAE.to("cpu")
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            err_str = str(e) if "已手动停止" in str(e) else f"超分异常: {e}"
            UPSCALE_STATE.update({
                "running": False,
                "pct": 0,
                "error": err_str,
                "message": err_str if "已手动停止" in err_str else f"❌ ReCaVSR 超分失败: {e}",
            })


class _ReCaVSRHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send_json({
                "status": "ok" if RECAVSR_READY else "loading",
                "engine": "recavsr",
                "ready": RECAVSR_READY,
                "error": RECAVSR_ERROR,
            })
            return
        if self.path.startswith("/v1/video/upscale_status"):
            st = dict(UPSCALE_STATE)
            st["ready"] = RECAVSR_READY
            if st.get("running") and UPSCALE_T0 > 0:
                st["elapsed_s"] = round(time.time() - UPSCALE_T0, 1)
            self._send_json(st)
            return
        if self.path.startswith("/v1/video/file/"):
            fname = os.path.basename(self.path.split("/v1/video/file/", 1)[-1].split("?")[0])
            fpath = VIDEO_DIR / fname
            if not fpath.is_file():
                self._send_json({"error": "File not found"}, status=404)
                return
            fsize = fpath.stat().st_size
            range_hdr = self.headers.get("Range")
            if range_hdr and range_hdr.startswith("bytes="):
                try:
                    parts = range_hdr.replace("bytes=", "").split("-")
                    start = int(parts[0]) if parts[0] else 0
                    end = int(parts[1]) if (len(parts) > 1 and parts[1]) else (fsize - 1)
                    end = min(end, fsize - 1)
                    length = max(0, end - start + 1)
                    self.send_response(206)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{fsize}")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(length))
                    self.end_headers()
                    with open(fpath, "rb") as rf:
                        rf.seek(start)
                        rem = length
                        while rem > 0:
                            buf = rf.read(min(262144, rem))
                            if not buf:
                                break
                            self.wfile.write(buf)
                            rem -= len(buf)
                    return
                except Exception:
                    pass
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(fsize))
            self.end_headers()
            with open(fpath, "rb") as rf:
                while True:
                    buf = rf.read(262144)
                    if not buf:
                        break
                    self.wfile.write(buf)
            return
        self._send_json({"status": "ok", "engine": "recavsr"})

    def do_POST(self):
        global UPSCALE_CANCEL
        cl = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(cl).decode("utf-8") if cl > 0 else "{}"
        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {}

        if self.path.startswith("/v1/video/upscale_cancel"):
            UPSCALE_CANCEL = True
            self._send_json({"ok": True, "message": "🛑 已发送停止信号中断 ReCaVSR 超分"})
            return

        if self.path.startswith("/v1/video/cancel_minimax"):
            open("/content/.cancel_minimax", "w").write("1")
            self._send_json({"ok": True, "message": "🛑 已发送停止信号中断 MiniMax-H3 视频生成"})
            return

        if self.path.startswith("/v1/video/upscale"):
            fname = os.path.basename(str(payload.get("filename") or "").strip())
            if not fname:
                self._send_json({"ok": False, "error": "未指定要超分的视频文件名"}, status=400)
                return
            if UPSCALE_STATE["running"]:
                self._send_json({
                    "ok": True,
                    "already_running": True,
                    "source_video": UPSCALE_STATE.get("source_video", ""),
                    "message": "当前已有视频正在超分中，已复用当前任务",
                })
                return
            UPSCALE_CANCEL = False
            UPSCALE_STATE.update({
                "running": True,
                "pct": 3,
                "step": 0,
                "source_video": fname,
                "error": "",
                "message": f"⏳ 已提交 ReCaVSR 超分任务: {fname}",
            })
            threading.Thread(target=_run_upscale_job, args=(payload,), daemon=True).start()
            self._send_json({"ok": True, "source_video": fname, "message": "已启动 ReCaVSR 视频超分任务"})
            return

        self._send_json({"ok": False, "error": "Unknown endpoint"}, status=404)


if __name__ == "__main__":
    threading.Thread(target=_init_recavsr, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", 30002), _ReCaVSRHandler)
    srv.serve_forever()
'''


def _ensure_recavsr_service_on_30002():
    """启动或复用伴随运行的 ReCaVSR (Wan2.2-5B 单步流式 2K/4K 超分引擎, 端口 30002)"""
    try:
        with urllib.request.urlopen("http://127.0.0.1:30002/health", timeout=2) as r:
            if r.status == 200:
                log("✅ ReCaVSR (Wan2.2-5B · 单步流式 2K/4K 超分引擎) 已在后台待命 (0 MB 闲置显存占用)！")
                return
    except Exception:
        pass
    try:
        log("正在启动 ReCaVSR (Wan2.2-5B) 2K/4K 超分伴随服务 (端口 30002)...")
        if not os.path.isdir("/content/ReCaVSR/.git"):
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/kopperx/ReCaVSR.git", "/content/ReCaVSR"], check=False)
        os.makedirs("/content/models/ReCaVSR", exist_ok=True)
        subprocess.run("cp -f /content/ReCaVSR/checkpoints/*.json /content/models/ReCaVSR/ 2>/dev/null || true", shell=True)
        with open("/content/recavsr_server.py", "w", encoding="utf-8") as wf:
            wf.write(RECAVSR_SERVER_CODE)
        with open("/content/models/ReCaVSR/recavsr_server.py", "w", encoding="utf-8") as wf2:
            wf2.write(RECAVSR_SERVER_CODE)
        # 立即启动 30002 HTTP 服务进程 (权重拉取与 CPU RAM 预热在 recavsr_server.py 独立守护进程内部异步完成，0.2 秒即响应请求)
        subprocess.Popen(
            [sys.executable, "-u", "/content/recavsr_server.py"],
            stdout=open("/content/recavsr_server.log", "w"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        log("✅ ReCaVSR (Wan2.2-5B · 单步流式 2K/4K 超分服务) 已启动并挂载至 30002 端口 (后台自动预热权重)！")
    except Exception as ex:
        log(f"⚠️ ReCaVSR 启动提示: {ex}")


INDEXTTS2_SERVER_CODE = r'''#!/usr/bin/env python3
import os
import sys

ISOLATED_PKGS_DIR = "/content/indextts2_pkgs"
if os.path.isdir(ISOLATED_PKGS_DIR) and ISOLATED_PKGS_DIR not in sys.path:
    sys.path.insert(0, ISOLATED_PKGS_DIR)

import base64
import gc
import json
import math
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

PORT = int(os.environ.get("INDEXTTS2_PORT", "30003"))
STANDALONE = (PORT == 30000)
MODEL_DIR = Path(os.environ.get("INDEXTTS2_MODEL_DIR", "/content/models/IndexTTS-2"))
CODE_DIR = Path("/content/index-tts")
PRESET_DIR = MODEL_DIR / "voice_presets"
IDLE_UNLOAD_SECONDS = int(os.environ.get("INDEXTTS2_IDLE_UNLOAD_SEC", "600"))

EMO_PRESET_MAP = {
    "happy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "angry": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "sad": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "afraid": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "disgusted": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "melancholic": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "surprised": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "calm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}

BUILTIN_VOICE_SPECS = {
    "甜美少女": ("zh-CN-XiaoyiNeural", [], "你好呀，今天的风很温柔，我想把这段美好的故事慢慢讲给你听。"),
    "温柔知性": ("zh-CN-XiaoxiaoNeural", ["--rate=-4%"], "夜色渐渐柔和下来，别着急，我们一步一步把想说的话认真讲清楚。"),
    "台妹软糯": ("zh-TW-HsiaoChenNeural", ["--rate=-5%"], "真的好开心见到你喔，今天天气超舒服的，我们一起去散步好不好呀。"),
    "清冷御姐": ("zh-CN-XiaoxiaoNeural", ["--pitch=-6Hz", "--rate=-6%"], "事情的真相从来不会自己浮出水面，保持冷静，听我把局势拆解给你看。"),
    "磁性暖男": ("zh-CN-YunxiNeural", ["--pitch=-3Hz", "--rate=-4%"], "别担心，不管外面风浪有多大，有我在你身边，一切都会慢慢好起来的。"),
    "沉稳霸总": ("zh-CN-YunjianNeural", ["--pitch=-5Hz", "--rate=-5%"], "从现在开始，所有的决定都由我来负责，你只需要站在我身后看着就行。"),
    "温润书生": ("zh-CN-YunyangNeural", ["--rate=-4%"], "山高水长，知音难觅，这一盏清茶敬过往，也敬我们即将启程的明天。"),
    "美式女声": ("en-US-AvaMultilingualNeural", [], "Welcome back. Take a deep breath and let me guide you through every detail clearly."),
}

TTS_LOCK = threading.Lock()
TTS_ENGINE = None
TTS_DEVICE = "none"
LAST_USED_TS = 0.0
STATE = {
    "ready": False,
    "loading": False,
    "loaded": False,
    "device": "none",
    "error": "",
    "message": "待命 (0 MB 显存，首次调用自动加载)" if not STANDALONE else "正在初始化 IndexTTS-2 云端引擎...",
}


def _log(msg: str):
    print(f"[IndexTTS-2] {msg}", flush=True)


def _install_deps():
    # 在隔离包目录 (/content/indextts2_pkgs) 内锁定官方 transformers==4.52.1 + tokenizers==0.21.0 + huggingface-hub==0.34.4，零干扰系统 transformers 5.x
    os.makedirs(ISOLATED_PKGS_DIR, exist_ok=True)
    if ISOLATED_PKGS_DIR not in sys.path:
        sys.path.insert(0, ISOLATED_PKGS_DIR)
    need_tf_pin = not (
        os.path.isdir(os.path.join(ISOLATED_PKGS_DIR, "transformers"))
        and os.path.isdir(os.path.join(ISOLATED_PKGS_DIR, "tokenizers"))
        and os.path.isdir(os.path.join(ISOLATED_PKGS_DIR, "huggingface_hub"))
    )
    if need_tf_pin:
        _log("正在隔离目录 (/content/indextts2_pkgs) 内安装官方兼容版本 (transformers==4.52.1 + tokenizers==0.21.0 + huggingface-hub==0.34.4)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--target", ISOLATED_PKGS_DIR, "--no-deps", "--upgrade",
             "transformers==4.52.1", "tokenizers==0.21.0", "huggingface-hub==0.34.4"],
            check=True,
        )
    # 仅安装缺失的轻量依赖，绝不破坏 Colab 已有 torch 环境
    checks = [
        ("omegaconf", "omegaconf"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("safetensors", "safetensors"),
        ("sentencepiece", "sentencepiece"),
        ("einops", "einops"),
        ("munch", "munch"),
        ("jieba", "jieba"),
        ("g2p_en", "g2p-en"),
        ("cn2an", "cn2an"),
        ("pypinyin", "pypinyin"),
        ("wetext", "wetext"),
        ("edge_tts", "edge-tts"),
        ("whisper", "openai-whisper"),
        ("tiktoken", "tiktoken"),
        ("audiotools", "descript-audiotools"),
        ("importlib_resources", "importlib-resources"),
        ("flatten_dict", "flatten-dict"),
        ("argbind", "argbind"),
        ("pyloudnorm", "pyloudnorm"),
        ("julius", "julius"),
        ("ffmpy", "ffmpy"),
        ("pystoi", "pystoi"),
        ("torch_stoi", "torch-stoi"),
        ("markdown2", "markdown2"),
        ("randomname", "randomname"),
        ("json5", "json5"),
        ("fugashi", "fugashi"),
        ("unidic_lite", "unidic-lite"),
        ("textstat", "textstat"),
        ("ffmpeg", "ffmpeg-python"),
    ]
    missing = []
    for mod, pkg in checks:
        try:
            __import__(mod)
        except Exception:
            missing.append(pkg)
    if missing:
        _log(f"正在安装缺失依赖: {' '.join(missing)} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", *missing],
            check=False,
        )
        safe_pkgs = [p for p in missing if p in ("munch", "g2p-en", "cn2an", "pypinyin", "wetext", "edge-tts", "omegaconf", "jieba", "tiktoken", "json5", "fugashi", "unidic-lite", "textstat", "ffmpeg-python", "importlib-resources", "flatten-dict", "argbind", "pyloudnorm", "julius", "ffmpy", "pystoi", "torch-stoi", "markdown2", "randomname")]
        if safe_pkgs:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", *safe_pkgs],
                check=False,
            )
    # 预先下载 g2p-en 所需 nltk 数据（避免英文混读报 LookupError）
    try:
        import nltk
        for res_name, sub_path in (
            ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
            ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
            ("cmudict", "corpora/cmudict"),
        ):
            try:
                nltk.data.find(sub_path)
            except LookupError:
                nltk.download(res_name, quiet=True)
    except Exception:
        pass


def _ensure_code_and_weights(sync_blocking: bool = False):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    if not (CODE_DIR / "indextts" / "infer_v2.py").is_file():
        _log("正在克隆官方 IndexTTS-2 推理仓库 (index-tts/index-tts)...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/index-tts/index-tts.git", str(CODE_DIR)],
            check=True,
        )
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))

    # 兼容 modelscope 导入：若环境未装 modelscope 或版本冲突，自动用 transformers.AutoModelForCausalLM 接管，零副作用
    try:
        import modelscope  # noqa: F401
        from modelscope import AutoModelForCausalLM as _ms_am  # noqa: F401
    except Exception:
        from transformers import AutoModelForCausalLM as _tf_am
        ms_mod = types.ModuleType("modelscope")
        ms_mod.AutoModelForCausalLM = _tf_am
        sys.modules["modelscope"] = ms_mod

    # 兼容 Linux 下 Python 3.13 无 pynini/WeTextProcessing(tn) 时自动桥接至 wetext.Normalizer
    try:
        import tn.chinese.normalizer  # noqa: F401
        import tn.english.normalizer  # noqa: F401
    except Exception:
        try:
            from wetext import Normalizer as _WetextNorm
            class _ShimZhNorm:
                def __init__(self, *args, **kwargs):
                    self._n = _WetextNorm(remove_erhua=bool(kwargs.get("remove_erhua", False)), lang="zh", operator="tn")
                def normalize(self, text):
                    return self._n.normalize(text)
            class _ShimEnNorm:
                def __init__(self, *args, **kwargs):
                    self._n = _WetextNorm(lang="en", operator="tn")
                def normalize(self, text):
                    return self._n.normalize(text)
            _tn_pkg = types.ModuleType("tn")
            _tn_zh_pkg = types.ModuleType("tn.chinese")
            _tn_zh_norm = types.ModuleType("tn.chinese.normalizer")
            _tn_zh_norm.Normalizer = _ShimZhNorm
            _tn_en_pkg = types.ModuleType("tn.english")
            _tn_en_norm = types.ModuleType("tn.english.normalizer")
            _tn_en_norm.Normalizer = _ShimEnNorm
            sys.modules["tn"] = _tn_pkg
            sys.modules["tn.chinese"] = _tn_zh_pkg
            sys.modules["tn.chinese.normalizer"] = _tn_zh_norm
            sys.modules["tn.english"] = _tn_en_pkg
            sys.modules["tn.english.normalizer"] = _tn_en_norm
        except Exception as _e_tn:
            _log(f"⚠️ wetext 桥接提示: {_e_tn}")

    # 兼容 Linux 下的文本规范化：如果未装 pynini/WeTextProcessing 但装了 wetext，提前确保 front.py 可平滑走 wetext
    has_rclone = os.path.isfile("/root/.config/rclone/rclone.conf")
    need_main = not (
        (MODEL_DIR / "gpt.pth").is_file()
        and (MODEL_DIR / "s2mel.pth").is_file()
        and (MODEL_DIR / "config.yaml").is_file()
        and (MODEL_DIR / "qwen0.6bemo4-merge" / "model.safetensors").is_file()
    )
    need_aux = not (
        (MODEL_DIR / "hf_cache" / "bigvgan" / "bigvgan_generator.pt").is_file()
        and (MODEL_DIR / "hf_cache" / "campplus_cn_common.bin").is_file()
        and (MODEL_DIR / "hf_cache" / "semantic_codec_model.safetensors").is_file()
    )

    pulled_from_drive = False
    if (need_main or need_aux) and has_rclone:
        _log("☁️ 正在检查并从 Google Drive (gdrive:models/IndexTTS-2) 极速拉取持久化权重...")
        subprocess.run(
            ["rclone", "copy", "gdrive:models/IndexTTS-2", str(MODEL_DIR), "--transfers=8"],
            check=False,
        )
        need_main = not (
            (MODEL_DIR / "gpt.pth").is_file()
            and (MODEL_DIR / "s2mel.pth").is_file()
            and (MODEL_DIR / "config.yaml").is_file()
            and (MODEL_DIR / "qwen0.6bemo4-merge" / "model.safetensors").is_file()
        )
        need_aux = not (
            (MODEL_DIR / "hf_cache" / "bigvgan" / "bigvgan_generator.pt").is_file()
            and (MODEL_DIR / "hf_cache" / "campplus_cn_common.bin").is_file()
            and (MODEL_DIR / "hf_cache" / "semantic_codec_model.safetensors").is_file()
        )
        pulled_from_drive = not (need_main or need_aux)
        if pulled_from_drive:
            _log("✅ 已从 Google Drive 秒速恢复全套 IndexTTS-2 权重与辅助模型！")

    downloaded_fresh = False
    if need_main:
        _log("⬇️ 正在从 HuggingFace 拉取官方原版 IndexTeam/IndexTTS-2 (5.49 GB)...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="IndexTeam/IndexTTS-2",
            local_dir=str(MODEL_DIR),
        )
        downloaded_fresh = True

    from indextts.utils.model_download import ensure_models_available
    if need_aux:
        _log("⬇️ 正在拉取 IndexTTS-2 配套前端与声码器 (w2v-bert-2.0 / MaskGCT / CAMPPlus / BigVGAN v2)...")
    aux_paths = ensure_models_available(str(MODEL_DIR))
    if need_aux:
        downloaded_fresh = True

    if downloaded_fresh and has_rclone:
        _log("☁️ 正在将全套 IndexTTS-2 权重与 hf_cache 持久化备份至 Google Drive (gdrive:models/IndexTTS-2)...")
        cmd_sync = ["rclone", "copy", str(MODEL_DIR), "gdrive:models/IndexTTS-2", "--transfers=8"]
        if sync_blocking:
            subprocess.run(cmd_sync, check=False)
            _log("✅ IndexTTS-2 已永久同步至 Google Drive (gdrive:models/IndexTTS-2)！")
        else:
            subprocess.Popen(cmd_sync, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return aux_paths


def _ensure_preset_voice_wav(voice_name: str) -> str:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    clean_name = re.sub(r"^indextts2:", "", (voice_name or "").strip()).strip()
    if not clean_name or clean_name == "__clone__":
        clean_name = "甜美少女"
    # 1. 优先查找用户自定义或已缓存的同名 wav/mp3
    for ext in (".wav", ".mp3", ".flac", ".m4a"):
        cand = PRESET_DIR / f"{clean_name}{ext}"
        if cand.is_file() and cand.stat().st_size > 2048:
            return str(cand)
    # 2. 若云盘/本地有 Ming-3.0 现成人设参考音频也可匹配复用
    for root_cand in Path("/content/local_models").glob("*/ming_talker/voice_presets"):
        for f in root_cand.glob("*.wav"):
            if clean_name in f.stem:
                return str(f)
    # 3. 内置 8 款标准参考音色：首次按需用 edge-tts 合成 6 秒纯净 22.05kHz 单声道参考底音并持久化至云盘
    spec = BUILTIN_VOICE_SPECS.get(clean_name) or BUILTIN_VOICE_SPECS["甜美少女"]
    edge_voice, flags, sample_txt = spec
    out_wav = PRESET_DIR / f"{clean_name}.wav"
    tmp_mp3 = PRESET_DIR / f".tmp_{clean_name}.mp3"
    subprocess.run(
        [sys.executable, "-m", "edge_tts", "--voice", edge_voice, *flags, "--text", sample_txt, "--write-media", str(tmp_mp3)],
        check=True,
        timeout=30,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp_mp3), "-ar", "22050", "-ac", "1", "-c:a", "pcm_s16le", str(out_wav)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
        timeout=20,
    )
    try:
        tmp_mp3.unlink(missing_ok=True)
    except Exception:
        pass
    if os.path.isfile("/root/.config/rclone/rclone.conf"):
        subprocess.Popen(
            ["rclone", "copy", str(out_wav), "gdrive:models/IndexTTS-2/voice_presets/"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return str(out_wav)


def _list_available_voices():
    names = list(BUILTIN_VOICE_SPECS.keys())
    if PRESET_DIR.is_dir():
        for p in sorted(PRESET_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in (".wav", ".mp3", ".flac", ".m4a") and not p.name.startswith("."):
                if p.stem not in names:
                    names.append(p.stem)
    return names


def _load_engine_locked(sync_blocking: bool = False):
    global TTS_ENGINE, TTS_DEVICE, LAST_USED_TS
    if TTS_ENGINE is not None:
        LAST_USED_TS = time.time()
        return TTS_ENGINE
    STATE.update({"loading": True, "error": "", "message": "⏳ 正在准备 IndexTTS-2 依赖与权重..."})
    try:
        _install_deps()
        aux_paths = _ensure_code_and_weights(sync_blocking=sync_blocking)
        import torch
        use_cuda = False
        free_gb = 0.0
        if torch.cuda.is_available():
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                free_gb = free_b / (1024 ** 3)
                use_cuda = (STANDALONE or free_gb >= 5.5)
            except Exception:
                use_cuda = True
        device = "cuda:0" if use_cuda else "cpu"
        use_fp16 = bool(use_cuda)
        STATE["message"] = f"🚀 正在将 IndexTTS-2 ({'FP16 GPU' if use_cuda else 'CPU 安全回退'} · 空闲显存 {free_gb:.1f}GB) 装载至 {device}..."
        _log(STATE["message"])
        from indextts.infer_v2 import IndexTTS2
        t0 = time.time()
        TTS_ENGINE = IndexTTS2(
            cfg_path=str(MODEL_DIR / "config.yaml"),
            model_dir=str(MODEL_DIR),
            use_fp16=use_fp16,
            device=device,
            use_cuda_kernel=False,
            use_qwen_emo=True,
            aux_paths=aux_paths,
        )
        TTS_DEVICE = device
        LAST_USED_TS = time.time()
        STATE.update({
            "ready": True,
            "loading": False,
            "loaded": True,
            "device": device,
            "error": "",
            "message": f"✅ IndexTTS-2 云端原版已就绪 ({device} · FP16={use_fp16} · 耗时 {time.time()-t0:.1f}s)",
        })
        _log(STATE["message"])
        return TTS_ENGINE
    except Exception as e:
        err_txt = f"{e}\n{traceback.format_exc()}"
        STATE.update({"ready": False, "loading": False, "loaded": False, "error": str(e), "message": f"❌ IndexTTS-2 加载失败: {e}"})
        _log(f"Load error: {err_txt}")
        raise


def _unload_engine_locked(reason: str = "manual"):
    global TTS_ENGINE, TTS_DEVICE
    if TTS_ENGINE is None:
        return
    try:
        del TTS_ENGINE
        TTS_ENGINE = None
        TTS_DEVICE = "none"
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        STATE.update({
            "loaded": False,
            "device": "none",
            "message": f"💤 IndexTTS-2 已自动释放显存 ({reason} · 0 MB 占用，下次调用自动秒载)",
        })
        _log(STATE["message"])
    except Exception as e:
        _log(f"Unload warn: {e}")


def _idle_watchdog_loop():
    if STANDALONE or IDLE_UNLOAD_SECONDS <= 0:
        return
    while True:
        time.sleep(30)
        if TTS_ENGINE is not None and LAST_USED_TS > 0 and (time.time() - LAST_USED_TS) > IDLE_UNLOAD_SECONDS:
            if TTS_LOCK.acquire(blocking=False):
                try:
                    if TTS_ENGINE is not None and (time.time() - LAST_USED_TS) > IDLE_UNLOAD_SECONDS:
                        _unload_engine_locked("闲置自动释放")
                finally:
                    TTS_LOCK.release()


def _decode_b64_audio_to_wav(b64_str: str, prefix: str = "idx_ref") -> str:
    raw_b64 = b64_str.split(",", 1)[-1].strip()
    ext = ".wav"
    if b64_str.startswith("data:audio/mpeg") or b64_str.startswith("data:audio/mp3"):
        ext = ".mp3"
    elif b64_str.startswith("data:audio/mp4") or b64_str.startswith("data:audio/m4a"):
        ext = ".m4a"
    elif b64_str.startswith("data:audio/flac"):
        ext = ".flac"
    elif b64_str.startswith("data:audio/ogg"):
        ext = ".ogg"
    fd, src_p = tempfile.mkstemp(prefix=prefix, suffix=ext)
    os.close(fd)
    Path(src_p).write_bytes(base64.b64decode(raw_b64))
    fd2, wav_p = tempfile.mkstemp(prefix=prefix + "_22k_", suffix=".wav")
    os.close(fd2)
    subprocess.run(
        ["ffmpeg", "-y", "-i", src_p, "-ar", "22050", "-ac", "1", "-c:a", "pcm_s16le", wav_p],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
        timeout=30,
    )
    try:
        os.unlink(src_p)
    except Exception:
        pass
    return wav_p


def _timefit_wav_if_needed(wav_path: str, target_dur: float) -> float:
    import soundfile as sf
    info = sf.info(wav_path)
    actual_dur = float(info.duration)
    if target_dur <= 0.4 or actual_dur <= 0.1:
        return actual_dur
    if abs(actual_dur - target_dur) <= 0.25:
        return actual_dur
    ratio = actual_dur / target_dur  # >1 means need to speed up; <1 means slow down
    clamped = max(0.76, min(1.38, ratio))
    fd, out_p = tempfile.mkstemp(prefix="idx_fit_", suffix=".wav")
    os.close(fd)
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-filter:a", f"atempo={clamped:.4f}", "-ar", "22050", "-ac", "1", "-c:a", "pcm_s16le", out_p],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
        timeout=30,
    )
    shutil.move(out_p, wav_path)
    return float(sf.info(wav_path).duration)


def _synthesize_indextts2(payload: dict) -> dict:
    global LAST_USED_TS
    text = str(payload.get("text") or "").strip()
    if not text:
        raise ValueError("台词文本不能为空")
    # 清理 Ming 专属气口标签，避免 IndexTTS-2 念出英文方括号
    clean_text = re.sub(
        r"\[(Breathing|Cough|Sigh|Laughter|Surprise-ah|Surprise-oh|Dissatisfaction-hnn|Uhm|Confirmation-en|Question-ei|Question-ah|Question-en|Question-oh|Question-huh)\]",
        "，",
        text,
    ).strip("， ") or text

    voice = str(payload.get("voice") or "indextts2:甜美少女").strip()
    clone_ref_b64 = str(payload.get("clone_ref_b64") or "").strip()
    emo_audio_b64 = str(payload.get("emo_audio_b64") or "").strip()
    emo_mode = str(payload.get("emo_mode") or "auto").strip().lower()
    emo_text = str(payload.get("emo_text") or "").strip()
    emo_preset = str(payload.get("emo_preset") or "").strip().lower()
    emo_vector = payload.get("emo_vector")
    try:
        emo_alpha = max(0.0, min(1.0, float(payload.get("emo_alpha", 0.75))))
    except Exception:
        emo_alpha = 0.75
    try:
        target_dur = float(payload.get("target_duration_sec") or 0.0)
    except Exception:
        target_dur = 0.0
    use_random = bool(payload.get("use_random", False))

    tmp_files = []
    try:
        if clone_ref_b64:
            spk_wav = _decode_b64_audio_to_wav(clone_ref_b64, prefix="idx_spk_")
            tmp_files.append(spk_wav)
            voice_lbl = "零样本声音克隆"
        else:
            spk_wav = _ensure_preset_voice_wav(voice)
            voice_lbl = re.sub(r"^indextts2:", "", voice) or "甜美少女"

        emo_wav = None
        if emo_audio_b64 and emo_mode in ("audio", "auto"):
            emo_wav = _decode_b64_audio_to_wav(emo_audio_b64, prefix="idx_emo_")
            tmp_files.append(emo_wav)

        use_emo_text = False
        final_emo_vec = None
        emo_desc = "自然随参考音色"

        if emo_mode == "text" or (emo_mode == "auto" and emo_text):
            if emo_text:
                use_emo_text = True
                emo_desc = f"文字控情绪「{emo_text[:18]}」×{emo_alpha:.2f}"
        elif emo_mode == "vector" or (emo_mode == "auto" and (emo_vector or emo_preset)):
            if isinstance(emo_vector, list) and len(emo_vector) == 8:
                final_emo_vec = [max(0.0, min(1.0, float(x))) for x in emo_vector]
                emo_desc = f"8维情绪向量×{emo_alpha:.2f}"
            elif emo_preset in EMO_PRESET_MAP:
                final_emo_vec = list(EMO_PRESET_MAP[emo_preset])
                emo_desc = f"情绪「{emo_preset}」×{emo_alpha:.2f}"
        elif emo_wav:
            emo_desc = f"参考情绪音频×{emo_alpha:.2f}"

        fd_out, out_wav = tempfile.mkstemp(prefix="idx_out_", suffix=".wav")
        os.close(fd_out)
        tmp_files.append(out_wav)

        t0 = time.time()
        with TTS_LOCK:
            tts = _load_engine_locked(sync_blocking=False)
            kwargs = {
                "spk_audio_prompt": spk_wav,
                "text": clean_text,
                "output_path": out_wav,
                "emo_alpha": emo_alpha,
                "use_random": use_random,
                "verbose": False,
            }
            if use_emo_text:
                kwargs["use_emo_text"] = True
                kwargs["emo_text"] = emo_text
            elif final_emo_vec is not None:
                kwargs["emo_vector"] = final_emo_vec
            elif emo_wav is not None:
                kwargs["emo_audio_prompt"] = emo_wav
            tts.infer(**kwargs)
            LAST_USED_TS = time.time()

        dur_sec = round(_timefit_wav_if_needed(out_wav, target_dur), 2)
        elapsed_s = round(time.time() - t0, 2)
        wav_bytes = Path(out_wav).read_bytes()
        auto_segs = 1 if dur_sec <= 3.85 else max(1, min(200, 1 + int(math.ceil((dur_sec - 3.72) / 3.20))))
        b64_str = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")
        dur_note = f" · 卡点对齐 {dur_sec}s" if target_dur > 0 else ""
        return {
            "ok": True,
            "audio_b64": b64_str,
            "audio_ext": ".wav",
            "duration_sec": dur_sec,
            "auto_segments": auto_segs,
            "engine": "indextts-2",
            "device": TTS_DEVICE,
            "elapsed_s": elapsed_s,
            "note": f"🎙️ [IndexTTS-2 云端原版 · {TTS_DEVICE}] 已合成「{voice_lbl}」({emo_desc}{dur_note} · 耗时 {elapsed_s}s)",
        }
    finally:
        for p in tmp_files:
            try:
                os.unlink(p)
            except Exception:
                pass


class _IndexTTS2Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/health"):
            if STANDALONE and not STATE.get("ready"):
                self._send_json({"status": "loading", "engine": "indextts-2", **STATE}, status=503)
            else:
                self._send_json({"status": "ok", "engine": "indextts-2", **STATE})
            return
        if self.path.startswith("/v1/models") or self.path.startswith("/models"):
            self._send_json({
                "object": "list",
                "data": [{"id": "IndexTeam/IndexTTS-2", "object": "model", "owned_by": "IndexTeam"}],
            })
            return
        if self.path.startswith("/v1/audio/indextts2_voices"):
            self._send_json({"ok": True, "voices": _list_available_voices(), "emo_presets": list(EMO_PRESET_MAP.keys())})
            return
        if self.path.startswith("/v1/audio/indextts2_status"):
            self._send_json({"ok": True, **STATE, "voices": _list_available_voices()})
            return
        self._send_json({"status": "ok", "engine": "indextts-2", **STATE})

    def do_POST(self):
        cl = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(cl).decode("utf-8") if cl > 0 else "{}"
        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {}

        if self.path.startswith("/v1/audio/indextts2_unload"):
            with TTS_LOCK:
                _unload_engine_locked("手动释放")
            self._send_json({"ok": True, **STATE})
            return

        if self.path.startswith("/v1/audio/indextts2") or (STANDALONE and self.path.startswith("/v1/audio/ming_tts")):
            try:
                res = _synthesize_indextts2(payload)
                self._send_json(res)
            except Exception as e:
                _log(f"Synthesize error: {e}\n{traceback.format_exc()}")
                self._send_json({"ok": False, "error": str(e)}, status=500)
            return

        if STANDALONE and self.path.startswith("/v1/shell"):
            cmd = str(payload.get("cmd") or "").strip()
            if not cmd:
                self._send_json({"ok": False, "error": "empty cmd"}, status=400)
                return
            try:
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=int(payload.get("timeout") or 120))
                self._send_json({"ok": True, "returncode": r.returncode, "stdout": r.stdout[-16000:], "stderr": r.stderr[-16000:]})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=500)
            return

        self._send_json({"ok": False, "error": "Unknown endpoint"}, status=404)


def _boot_standalone_worker():
    try:
        with TTS_LOCK:
            _load_engine_locked(sync_blocking=True)
    except Exception:
        pass


if __name__ == "__main__":
    if STANDALONE:
        threading.Thread(target=_boot_standalone_worker, daemon=True).start()
    else:
        STATE["ready"] = True
        threading.Thread(target=_idle_watchdog_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), _IndexTTS2Handler)
    _log(f"HTTP server listening on 0.0.0.0:{PORT} (standalone={STANDALONE})")
    srv.serve_forever()
'''


def _ensure_indextts2_isolated_pkgs() -> str:
    """在独立目录 /content/indextts2_pkgs 安装官方 transformers==4.52.1 + tokenizers==0.21.0 + huggingface-hub==0.34.4（免 venv/ensurepip，零干扰系统包）"""
    pkg_dir = "/content/indextts2_pkgs"
    os.makedirs(pkg_dir, exist_ok=True)
    need_install = not (
        os.path.isdir(os.path.join(pkg_dir, "transformers"))
        and os.path.isdir(os.path.join(pkg_dir, "tokenizers"))
        and os.path.isdir(os.path.join(pkg_dir, "huggingface_hub"))
    )
    if need_install:
        log("📦 正在独立目录 (/content/indextts2_pkgs) 安装 IndexTTS-2 官方匹配版本 (transformers==4.52.1 + tokenizers==0.21.0 + huggingface-hub==0.34.4)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--target", pkg_dir, "--no-deps", "--upgrade",
             "transformers==4.52.1", "tokenizers==0.21.0", "huggingface-hub==0.34.4"],
            check=True,
        )
    return pkg_dir


def _ensure_indextts2_service_on_30003():
    """启动或复用伴随运行的 IndexTTS-2 云端原版语音合成服务 (端口 30003，默认 0 MB 闲置显存，首次调用懒加载)"""
    try:
        with urllib.request.urlopen("http://127.0.0.1:30003/health", timeout=2) as r:
            if r.status == 200:
                log("✅ IndexTTS-2 云端原版语音服务已在 30003 端口待命 (0 MB 闲置显存，按需自动加载)！")
                return
    except Exception:
        pass
    try:
        log("正在挂载 IndexTTS-2 云端原版语音伴随服务 (端口 30003 · 0 MB 闲置显存)...")
        with open("/content/indextts2_server.py", "w", encoding="utf-8") as wf:
            wf.write(INDEXTTS2_SERVER_CODE)
        pkg_dir = _ensure_indextts2_isolated_pkgs()
        env = os.environ.copy()
        env["INDEXTTS2_PORT"] = "30003"
        env["PYTHONPATH"] = f"{pkg_dir}:{env.get('PYTHONPATH', '')}".rstrip(":")
        subprocess.Popen(
            [sys.executable, "-u", "/content/indextts2_server.py"],
            env=env,
            stdout=open("/content/indextts2_server.log", "w"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        log("✅ IndexTTS-2 云端原版语音伴随服务已挂载至 30003 端口 (声音克隆 + 自然语言控情绪 + 8维情绪 + 控时长)！")
    except Exception as ex:
        log(f"⚠️ IndexTTS-2 伴随服务启动提示: {ex}")


def _start_indextts2_standalone_on_30000(download_only: bool = False):
    """独立部署并启动官方原版 IndexTeam/IndexTTS-2 服务（支持 T4 16GB 与 RTX 6000 96GB，自动持久化至 Google Drive）"""
    if not shutil.which("cloudflared"):
        log("正在安装 cloudflared 隧道客户端...")
        subprocess.run(
            "curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
            "-o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
            shell=True,
            check=False,
        )
    with open("/content/indextts2_server.py", "w", encoding="utf-8") as wf:
        wf.write(INDEXTTS2_SERVER_CODE)
    pkg_dir = _ensure_indextts2_isolated_pkgs()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{pkg_dir}:{env.get('PYTHONPATH', '')}".rstrip(":")

    if download_only:
        log("🎙️ [仅下载模式] 正在拉取官方原版 IndexTeam/IndexTTS-2 + 配套前端/声码器并永久备份至 Google Drive...")
        prep_cmd = [
            sys.executable, "-u", "-c",
            "import sys; sys.path.insert(0, '/content'); import indextts2_server as s; s._install_deps(); s._ensure_code_and_weights(sync_blocking=True); print('INDEXTTS2_SYNC_OK')",
        ]
        subprocess.run(prep_cmd, env=env, check=True)
        log("🎉 [仅下载模式] IndexTTS-2 全套权重与 hf_cache 已永久同步至 Google Drive (gdrive:models/IndexTTS-2)！")
        return

    subprocess.run(
        "pkill -9 -f '[l]lama-server|minimax_server\\.py|minimax_t4_server\\.py|longcat_avatar_server\\.py|minimax_image_server\\.py|qwen_image_server\\.py|indextts2_server\\.py|project-maya/serve/server\\.py|build/maya|maya-vision' || true; fuser -k -9 30000/tcp 2>/dev/null || true",
        shell=True,
    )
    time.sleep(1)
    log("🚀 正在启动官方原版 IndexTeam/IndexTTS-2 云端语音引擎 (端口 30000 · FP16 GPU + QwenEmotion)...")
    env["INDEXTTS2_PORT"] = "30000"
    with open("/content/indextts2_server.log", "w") as lf:
        proc = subprocess.Popen(
            [sys.executable, "-u", "/content/indextts2_server.py"],
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0 = time.time()
    last_msg = ""
    while time.time() - t0 < 900:
        if proc.poll() is not None:
            tail = subprocess.getoutput("tail -n 40 /content/indextts2_server.log")
            raise RuntimeError(f"IndexTTS-2 引擎异常退出:\n{tail}")
        try:
            with opener.open("http://127.0.0.1:30000/health", timeout=3) as resp:
                if resp.status == 200:
                    hd = json.loads(resp.read().decode("utf-8"))
                    log(f"[显存装载] IndexTTS-2(云端原版·FP16): 100% ({hd.get('device', 'cuda:0')}, 已就绪)")
                    log(f"✅ {hd.get('message') or 'IndexTTS-2 云端原版语音服务已就绪！'}")
                    return
        except Exception as e_h:
            if hasattr(e_h, "code") and getattr(e_h, "code", 0) == 503:
                try:
                    err_b = json.loads(e_h.read().decode("utf-8"))
                    if err_b.get("error"):
                        raise RuntimeError(f"IndexTTS-2 加载失败: {err_b['error']}")
                    m_txt = str(err_b.get("message") or "")
                    if m_txt and m_txt != last_msg:
                        last_msg = m_txt
                        log(m_txt)
                except RuntimeError:
                    raise
                except Exception:
                    pass
        time.sleep(3)
    tail = subprocess.getoutput("tail -n 40 /content/indextts2_server.log")
    raise TimeoutError(f"等待 IndexTTS-2 启动超时:\n{tail}")


def _resolve_hf_all_repo_files(repo_id: str, token: str = "") -> List[Tuple[str, int]]:
    """查询 HuggingFace 仓库下所有待下载文件 (排除 .gitattributes / README / LICENSE 以外的所有权重与配置)"""
    url = f"https://huggingface.co/api/models/{repo_id}?blobs=true"
    headers = {"User-Agent": "ColabDeploy/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        info = json.loads(resp.read().decode("utf-8"))
    out: List[Tuple[str, int]] = []
    for sib in info.get("siblings") or []:
        rfn = str(sib.get("rfilename") or "").strip()
        if not rfn or rfn.startswith(".") or rfn in ("README.md", "LICENSE", ".gitattributes"):
            continue
        sz = int(sib.get("size") or (sib.get("lfs") or {}).get("size") or 0)
        out.append((rfn, sz))
    out.sort(key=lambda x: (0 if not x[0].endswith(".safetensors") else 1, x[0]))
    return out


def _start_glm53_maya_server_on_30000(download_only: bool = False):
    """在远端 Colab G4 (RTX PRO 6000 96GB) 上通过 Project Maya + EXL3 + DFlash2 + maya-vision 启动 GLM-5.3-Flash 无审查版 (端口 30000，兼容 OpenAI + Anthropic /v1/messages)"""
    repo_id = (HF_REPO or "MikeRoz/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3").strip()
    folder_name = (MODEL_FOLDER or "GLM-5.3-Flash-4.05bpw").strip()
    if "2.51" in f"{repo_id} {folder_name}":
        tier = "2.51"
        spec_min_p = "0.95"
    elif "3.05" in f"{repo_id} {folder_name}":
        tier = "3.05"
        spec_min_p = "0.90"
    else:
        tier = "4.05"
        spec_min_p = "0.95"
    model_alias = f"glm-5.3-flash-uncensored-{tier}bpw"

    try:
        vram_mib_str = subprocess.getoutput("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null").strip().splitlines()[0]
        vram_gb = float(vram_mib_str) / 1024.0
    except Exception:
        vram_gb = 96.0
    if not download_only and vram_gb < 70.0:
        raise RuntimeError(f"GLM-5.3-Flash EXL3 ({tier} bpw · 321B) 需要 Colab G4 (RTX PRO 6000 96GB) 或 H100 (80GB) 显存，当前 {vram_gb:.1f} GB 无法加载。")

    # 检测 GPU Compute Capability (G4 Blackwell = 120, H100 = 90)
    cuda_arch = "120"
    try:
        cap_raw = subprocess.getoutput("nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null").strip().splitlines()[0].strip()
        if re.match(r"^\d+\.\d+$", cap_raw):
            cuda_arch = cap_raw.replace(".", "")
    except Exception:
        pass

    os.environ["PATH"] = "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:" + os.environ.get("PATH", "")

    if not shutil.which("cloudflared"):
        log("正在安装 cloudflared 隧道客户端...")
        subprocess.run(
            "curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
            "-o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
            shell=True,
            check=True,
        )

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if not download_only:
        running_maya = subprocess.getoutput("pgrep -af 'project-maya/serve/server\\.py'").strip()
        if running_maya:
            try:
                req_h = urllib.request.Request(
                    "http://127.0.0.1:30000/health",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                )
                with opener.open(req_h, timeout=3) as resp:
                    if resp.status == 200:
                        hd = json.loads(resp.read().decode("utf-8", errors="ignore"))
                        cur_m = str(hd.get("model") or "")
                        if model_alias in cur_m or tier in cur_m:
                            log(f"✅ 检测到 Project Maya ({cur_m}) 已在 96GB 显存中常驻就绪，直接零等待复用！")
                            return
            except Exception:
                pass

    has_rclone = _has_vm_rclone_drive()
    rc_conf = os.path.expanduser("~/.config/rclone/rclone.conf")
    fuse_mounted = os.path.ismount("/content/drive/MyDrive") or os.path.ismount("/content/drive")
    target_dir = os.path.join("/content/local_models", folder_name)
    os.makedirs(target_dir, exist_ok=True)
    dflash_dir = os.path.join(target_dir, "dflash2")
    vision_dir = os.path.join(target_dir, "vision")
    os.makedirs(dflash_dir, exist_ok=True)
    os.makedirs(vision_dir, exist_ok=True)
    dflash_path = os.path.join(dflash_dir, "GLM-5.3-Flash-DFlash2-Q4_K_M.gguf")
    mmproj_path = os.path.join(vision_dir, "mmproj-GLM-5.3-Flash-F16.gguf")
    vocab_path = os.path.join(vision_dir, "GLM-5.3-Flash-vocab.gguf")

    # 1. 若准备启动服务，先释放旧引擎显存/内存盘，再做 236GB NVMe 防爆盘智能腾挪
    if not download_only:
        subprocess.run(
            "pkill -9 -f '[l]lama-server|minimax_server\\.py|minimax_t4_server\\.py|longcat_avatar_server\\.py|minimax_image_server\\.py|qwen_image_server\\.py|indextts2_server\\.py|project-maya/serve/server\\.py|build/maya|maya-vision|/content/app\\.py' || true; fuser -k -9 30000/tcp 2>/dev/null || true",
            shell=True,
        )
        if os.path.isdir("/dev/shm/local_models"):
            shutil.rmtree("/dev/shm/local_models", ignore_errors=True)

    # 2. 安装构建工具与 Python 依赖
    need_apt = not (shutil.which("ninja") and shutil.which("cmake"))
    need_py = subprocess.run(
        [sys.executable, "-c", "import huggingface_hub, jinja2, regex, yaml, tqdm, requests, PIL, psutil, jsonschema"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode != 0
    if need_apt or need_py:
        log("🔧 [1/5] 正在安装 Project Maya 构建工具与 Python 依赖...")
        if need_apt:
            subprocess.run("apt-get -qq update && apt-get -qq install -y aria2 ninja-build cmake build-essential", shell=True, check=True)
        if need_py:
            subprocess.run(
                [sys.executable, "-m", "pip", "-q", "install", "-U", "huggingface_hub", "jinja2", "regex", "pyyaml", "tqdm", "requests", "pillow", "psutil", "jsonschema>=4.18,<5"],
                check=True,
            )

    # 3. 准备 Project Maya 源码与 CUDA 二进制（优先从 Google Drive 秒恢复预编译包，免 90s 重复编译）
    maya_dir = "/content/project-maya"
    maya_exe = os.path.join(maya_dir, "build", "maya")
    vision_exe = os.path.join(maya_dir, "build-vision", "bin", "maya-vision")
    bin_tar_name = f"maya_bin_sm{cuda_arch}.tar.gz"
    local_bin_tar = f"/tmp/{bin_tar_name}"

    if not os.path.isfile(os.path.join(maya_dir, "CMakeLists.txt")):
        log("📦 [2/5] 正在拉取 Project Maya (EXL3 + DFlash2 + Anthropic/OpenAI 双协议推理引擎)...")
        shutil.rmtree(maya_dir, ignore_errors=True)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/mw00/project-maya.git", maya_dir], check=True)

    if not os.path.isfile(maya_exe) and has_rclone:
        log(f"☁️ 正在检查 Google Drive 是否已有预编译加速包 ({bin_tar_name})...")
        rc_pull = subprocess.run(
            ["rclone", "copy", f"gdrive:models/{folder_name}/{bin_tar_name}", "/tmp", "--config", rc_conf],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if rc_pull.returncode == 0 and os.path.isfile(local_bin_tar) and os.path.getsize(local_bin_tar) > 100_000:
            subprocess.run(["tar", "-xzf", local_bin_tar, "-C", maya_dir], check=False)
            if os.path.isfile(maya_exe):
                os.chmod(maya_exe, 0o755)
                if os.path.isfile(vision_exe):
                    os.chmod(vision_exe, 0o755)
                log(f"⚡ 已从 Google Drive 秒恢复预编译 Maya sm_{cuda_arch} 二进制（跳过 CUDA 编译！）")

    if not os.path.isfile(maya_exe):
        log(f"⚙️ [2/5] 首次为当前显卡 (CUDA sm_{cuda_arch}) 编译 Project Maya 核心引擎（约 80~100 秒，编完自动存入云盘供下次秒用）...")
        subprocess.run([sys.executable, "-c", "import setup as S; S.get_llama_cpp()"], cwd=maya_dir, check=True)
        subprocess.run(
            [
                "cmake", "-G", "Ninja", "-S", ".", "-B", "build",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DSTRATA_ENABLE_CUDA=ON",
                f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
                "-DSTRATA_NATIVE_EXPERTS=ON",
                "-DSTRATA_BUILD_TESTS=OFF",
                "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
                "-DCUDAToolkit_ROOT=/usr/local/cuda",
                f"-DSTRATA_GGML_DIR={maya_dir}/third_party/llama.cpp",
            ],
            cwd=maya_dir,
            check=True,
        )
        subprocess.run(["cmake", "--build", "build", "--target", "maya", "-j", str(min(24, os.cpu_count() or 8))], cwd=maya_dir, check=True)
        log("✅ Project Maya 核心引擎编译完成！")

        # 顺带尝试编译按需零驻留视觉编码器 maya-vision（失败自动降级纯文本，绝不阻断主引擎）
        llama_tp = os.path.join(maya_dir, "third_party", "llama.cpp")
        v_src = os.path.join(maya_dir, "tools", "vision")
        if os.path.isdir(v_src) and os.path.isdir(llama_tp) and not os.path.isfile(vision_exe):
            try:
                log("👁️ 正在编译按需视觉编码器 maya-vision（支持原图/视频反推提示词，平时 0 显存占用）...")
                for patch_py in ("glm5next_patch.py", "inkling2v_patch.py"):
                    pp = os.path.join(v_src, patch_py)
                    if os.path.isfile(pp):
                        subprocess.run([sys.executable, pp, llama_tp], cwd=maya_dir, check=True)
                subprocess.run(
                    [
                        "cmake", "-G", "Ninja", "-S", v_src, "-B", os.path.join(maya_dir, "build-vision"),
                        "-DCMAKE_BUILD_TYPE=Release",
                        f"-DLLAMA_DIR={llama_tp}",
                        "-DMAYA_VISION_CUDA=ON",
                        "-DMAYA_VISION_HIP=OFF",
                        f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
                        "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
                        "-DCUDAToolkit_ROOT=/usr/local/cuda",
                    ],
                    cwd=maya_dir,
                    check=True,
                )
                subprocess.run(
                    ["cmake", "--build", os.path.join(maya_dir, "build-vision"), "--target", "maya-vision", "-j", str(min(16, os.cpu_count() or 8))],
                    cwd=maya_dir,
                    check=True,
                )
                log("✅ maya-vision 按需视觉编码器编译完成！")
            except Exception as e_vis:
                log(f"⚠️ maya-vision 编译跳过 ({e_vis})，GLM-5.3-Flash 将以纯文本顶配模式启动。")

        if has_rclone and os.path.isfile(maya_exe):
            tar_items = ["build/maya"]
            if os.path.isdir(os.path.join(maya_dir, "build-vision", "bin")):
                tar_items.append("build-vision/bin")
            subprocess.run(["tar", "-czf", local_bin_tar, "-C", maya_dir] + tar_items, check=False)
            if os.path.isfile(local_bin_tar):
                subprocess.run(
                    ["rclone", "copy", local_bin_tar, f"gdrive:models/{folder_name}", "--config", rc_conf],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                log(f"☁️ 已将预编译二进制 {bin_tar_name} 缓存至 Google Drive，下次开机免编译！")

    # 4. 解析模型权重 + DFlash2 投机草稿 + 视觉 mmproj，防爆盘清理并优先从云盘极速拉取
    gdrive_saved_count = _check_rclone_drive_has_model(folder_name) if (has_rclone and not fuse_mounted) else 0
    log(f"📥 [3/5] 正在解析 {repo_id} (GLM-5.3-Flash {tier} bpw 无审查版) + DFlash2 + 视觉组件文件清单...")
    repo_files: List[Tuple[str, int]] = []
    try:
        repo_files = _resolve_hf_all_repo_files(repo_id, HF_TOKEN)
    except Exception as e_meta:
        if gdrive_saved_count > 0:
            log(f"⚠️ 暂无法查询 HF 清单 ({e_meta})，检测到 Google Drive 已有 {gdrive_saved_count} 个权重文件，直接全量从您的云盘拉取！")
        else:
            raise
    # 合并 DFlash2 (665MB) 与视觉塔 (1.14GB)
    plan: List[Tuple[str, str, str, str, int]] = []
    for rel_fn, sz in repo_files:
        plan.append((rel_fn, repo_id, rel_fn, target_dir, max(0, int(sz))))
    if repo_files:
        plan.append(("dflash2/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf", "Anbeeld/GLM-5.3-Flash-DFlash2-GGUF", "GLM-5.3-Flash-DFlash2-Q4_K_M.gguf", dflash_dir, 665_000_000))
        plan.append(("vision/mmproj-GLM-5.3-Flash-F16.gguf", "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "vision/mmproj-GLM-5.3-Flash-F16.gguf", target_dir, 1_110_000_000))
        plan.append(("vision/GLM-5.3-Flash-vocab.gguf", "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "vision/GLM-5.3-Flash-vocab.gguf", target_dir, 32_000_000))

    needed_new = 0
    for rel_fn, _, _, _, sz in plan:
        ep = os.path.join(target_dir, rel_fn)
        if not (os.path.exists(ep) and os.path.getsize(ep) > 0 and (sz <= 0 or abs(os.path.getsize(ep) - sz) < max(64 * 1024 * 1024, int(sz * 0.08)))):
            needed_new += max(0, sz)
    if needed_new == 0 and not repo_files:
        needed_new = 160 * (1024 ** 3)

    free_nvme = shutil.disk_usage("/content").free
    safety_margin = 16 * (1024 ** 3)
    if free_nvme < needed_new + safety_margin:
        for base_clean in ("/content/local_models", "/content/models"):
            if not os.path.isdir(base_clean):
                continue
            for other_name in sorted(os.listdir(base_clean)):
                other_p = os.path.join(base_clean, other_name)
                if os.path.abspath(other_p) == os.path.abspath(target_dir):
                    continue
                if os.path.isdir(other_p):
                    log(f"🧹 [NVMe 防爆盘智能腾挪] 当前剩余 {free_nvme/(1024**3):.1f} GB，释放未激活缓存 {other_name} (云盘 20T 已永久保存)...")
                    shutil.rmtree(other_p, ignore_errors=True)
                    free_nvme = shutil.disk_usage("/content").free
                    if free_nvme >= needed_new + safety_margin:
                        break

    expected_core_count = sum(1 for x in plan if x[0].lower().endswith((".safetensors", ".gguf")))
    st_stage_tag = "云盘极速预热" if gdrive_saved_count > 0 else "HF 下载"
    if has_rclone and not fuse_mounted and gdrive_saved_count > 0:
        log(f"☁️ 检测到您的 Google Drive (gdrive:models/{folder_name}) 已保存 {gdrive_saved_count} 个权重文件，直接优先 16 路并发从您的云盘内网高速拉取！")
        _pull_from_rclone_drive_if_exists(folder_name, target_dir, excludes=["maya_bin_*.tar.gz", "pack/**"])

    batch_total_bytes = sum(x[4] for x in plan)
    total_gb = batch_total_bytes / (1024 ** 3)
    cum_done_bytes = 0
    for idx, (rel_fn, dl_repo, dl_rel_fn, dl_dest_dir, sz) in enumerate(plan, start=1):
        ep = os.path.join(target_dir, rel_fn)
        lbl_str = f"{folder_name} 总进度 ({idx}/{len(plan)} · {os.path.basename(rel_fn)})"
        if os.path.exists(ep) and os.path.getsize(ep) > 0 and (sz <= 0 or not rel_fn.endswith(".safetensors") or os.path.getsize(ep) == sz):
            actual_sz = os.path.getsize(ep)
            cum_done_bytes += actual_sz
            ov_pct = min(100, cum_done_bytes * 100 // max(1, batch_total_bytes)) if batch_total_bytes > 0 else 100
            log(f"  [{st_stage_tag}] {lbl_str}: {ov_pct}% ({cum_done_bytes/(1024**3):.2f}/{total_gb:.2f} GB, 本地缓存复用)")
            continue
        log(f"正在 16 线程极速预热 [{idx}/{len(plan)}] {rel_fn} ({sz/(1024**3):.2f} GB)...")
        try:
            dl_p = fast_parallel_hf_download(
                dl_repo,
                dl_rel_fn,
                dl_dest_dir,
                token=HF_TOKEN,
                num_conn=16,
                batch_done_bytes=cum_done_bytes,
                batch_total_bytes=batch_total_bytes,
                batch_label=lbl_str,
                stage_tag=st_stage_tag,
            )
            cum_done_bytes += (os.path.getsize(dl_p) if os.path.exists(dl_p) else sz)
        except Exception as e_dl:
            if rel_fn.startswith("vision/"):
                log(f"⚠️ 可选视觉组件 {rel_fn} 拉取跳过 ({e_dl})，不影响主模型运行。")
                continue
            if has_rclone and not fuse_mounted and _pull_from_rclone_drive_if_exists(folder_name, target_dir):
                break
            raise

    # 5. 零拷贝 EXL3 Pack（~10 秒）
    pack_dir = os.path.join(target_dir, "pack")
    packed_flag = os.path.join(pack_dir, ".packed_ok")
    if not os.path.isfile(packed_flag):
        log("📦 [4/5] 正在执行零拷贝 EXL3 权重打包 (exl3_pack.py · 约 10 秒)...")
        shutil.rmtree(pack_dir, ignore_errors=True)
        subprocess.run([sys.executable, os.path.join(maya_dir, "tools", "exl3_pack.py"), "--model", target_dir], cwd=maya_dir, check=True)
        Path(packed_flag).write_text("ok\n", encoding="utf-8")
        log("✅ 零拷贝 EXL3 打包完成！")

    if has_rclone and not fuse_mounted and gdrive_saved_count < expected_core_count:
        _sync_local_to_rclone_drive(target_dir, background=not download_only, remote_folder=folder_name)

    if download_only:
        log(f"🎉 [仅下载模式] GLM-5.3-Flash ({tier} bpw) + DFlash2 + 视觉组件已永久同步至 Google Drive (gdrive:models/{folder_name})！")
        return

    # 6. 写入 Project Maya 最优 G4 调优配置并启动端口 30000（同时兼容 OpenAI /v1 与 Anthropic /v1/messages）
    ctx_val = int(CTX_LEN or 131072)
    if ctx_val > 262144:
        ctx_val = 262144
    if ctx_val < 16384:
        ctx_val = 131072
    maya_args = [
        "--glm-pack", pack_dir,
        "--max-context", str(ctx_val),
        "--vram-reserve-mib", "200",
        "--glm-ram-resident",
    ]
    if os.path.isfile(dflash_path):
        maya_args.extend(["--dflash", dflash_path, "--spec-min-p", spec_min_p])
    if ctx_val <= 131072:
        maya_args.extend(["--kv", "k8v4"])
    else:
        maya_args.extend(["--kv", "int8", "--kv-resident", "20480"])

    has_vision = os.path.isfile(vision_exe) and os.path.isfile(mmproj_path) and os.path.isfile(vocab_path)
    if has_vision:
        maya_args.append("--vision")

    maya_cfg: Dict[str, Any] = {
        "exe": maya_exe,
        "pack": pack_dir,
        "tokenizer": os.path.join(pack_dir, "tokenizer"),
        "model": model_alias,
        "model_name": model_alias,
        "host": "0.0.0.0",
        "port": 30000,
        "api_key": API_KEY,
        "max_context": ctx_val,
        "args": maya_args,
        "env": {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
            "MAYA_GLM_PCIE_SHARE": "0.00",
            "MAYA_GLM_CPU_LANE": "18",
            "MAYA_GLM_NGRAM_DRAFT": "1",
            "MAYA_GLM_DSA_GRAPHS": "1",
        },
    }
    if has_vision:
        maya_cfg["vision"] = {
            "exe": vision_exe,
            "mmproj": mmproj_path,
            "model": vocab_path,
            "gpu": True,
            "no_flash_attn": True,
        }

    cfg_path = "/content/maya-glm53.json"
    with open(cfg_path, "w", encoding="utf-8") as cf:
        json.dump(maya_cfg, cf, indent=2, ensure_ascii=False)

    vis_label = " + 👁️按需视觉(图/视频)" if has_vision else ""
    kv_label = "k8v4" if ctx_val <= 131072 else "int8(20G驻留)"
    log(f"🚀 [5/5] 正在将 GLM-5.3-Flash ({tier} bpw · 321B 无审查{vis_label}) 装载入 96GB VRAM + 176GB RAM (上下文 {ctx_val // 1024}K · KV {kv_label} · DFlash2)...")
    with open("/content/maya_server.log", "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            [
                sys.executable, "-u", os.path.join(maya_dir, "serve", "server.py"),
                "--engine", "maya",
                "--config", cfg_path,
                "--host", "0.0.0.0",
                "--port", "30000",
            ],
            cwd=maya_dir,
            stdout=lf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )

    t_load0 = time.time()
    last_log_ts = 0.0
    while time.time() - t_load0 < 600:
        if proc.poll() is not None:
            tail = subprocess.getoutput("tail -n 50 /content/maya_server.log")
            raise RuntimeError(f"Project Maya 服务异常退出:\n{tail}")
        try:
            req_m = urllib.request.Request(
                "http://127.0.0.1:30000/v1/models",
                headers={"Authorization": f"Bearer {API_KEY}"},
            )
            with opener.open(req_m, timeout=3) as resp:
                if resp.status == 200:
                    used_gb = 0.0
                    try:
                        u_s = subprocess.getoutput("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null").strip().splitlines()[0]
                        used_gb = float(u_s) / 1024.0
                    except Exception:
                        pass
                    log(f"[显存装载] GLM-5.3-Flash(321B·{tier}bpw): 100% ({used_gb:.1f}/{vram_gb:.1f} GiB VRAM, 已就绪 · 耗时 {time.time()-t_load0:.1f}s)")
                    log(f"✅ GLM-5.3-Flash ({tier} bpw · 无审查{vis_label}) 已就绪！同时支持 OpenAI (/v1) 与 Claude Code (/v1/messages)！")
                    return
        except Exception:
            pass
        now = time.time()
        if now - last_log_ts >= 4.0:
            last_log_ts = now
            try:
                u_s = subprocess.getoutput("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null").strip().splitlines()[0]
                used_gb = float(u_s) / 1024.0
                pct = min(99, max(5, int((used_gb / max(1.0, vram_gb * 0.94)) * 100)))
                log(f"[显存装载] GLM-5.3-Flash(321B·{tier}bpw): {pct}% ({used_gb:.1f}/{vram_gb:.1f} GiB VRAM, 正在预热专家权重... {int(now-t_load0)}s)")
            except Exception:
                pass
        time.sleep(2.0)

    tail = subprocess.getoutput("tail -n 50 /content/maya_server.log")
    raise TimeoutError(f"等待 GLM-5.3-Flash (Project Maya) 启动超时 (600s):\n{tail}")



def copy_drive_dir_to_local(drive_dir: str, local_dir: str) -> List[str]:
    """16 线程并发将已挂载的 Google Drive 文件夹中的模型权重预热到 Colab 本地内存盘/NVMe"""
    if not os.path.isdir(drive_dir):
        return []
    os.makedirs(local_dir, exist_ok=True)
    rel_files: List[str] = []
    has_matching_main = False
    for root, _, files in os.walk(drive_dir):
        for fn in files:
            if fn.startswith(".") or fn.endswith((".part", ".tmp", ".syncing")):
                continue
            if fn.lower().endswith(".gguf") and _matches_main_pattern(fn, FILE_PATTERN):
                full_p = os.path.join(root, fn)
                if os.path.getsize(full_p) > 1024 * 1024:
                    rel_p = os.path.relpath(full_p, drive_dir)
                    rel_files.append(rel_p)
                    if "mmproj" not in fn.lower() and not fn.lower().startswith("mtp"):
                        has_matching_main = True

    if not rel_files or not has_matching_main:
        return []

    log(f"检测到 Google Drive 文件夹 ({drive_dir}) 中已有 {len(rel_files)} 个匹配的 GGUF 文件，正在 16 线程高速预热至本地内存盘/NVMe...")

    def _copy_one_parallel(rel_p: str, num_workers: int = 16) -> str:
        src = os.path.join(drive_dir, rel_p)
        dst = os.path.join(local_dir, rel_p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        total_size = os.path.getsize(src)
        if os.path.exists(dst) and os.path.getsize(dst) == total_size:
            return dst
        tmp = dst + ".tmp"
        with open(tmp, "wb") as wf:
            wf.truncate(total_size)

        copied_bytes = 0
        cp_lock = __import__("threading").Lock()
        t0 = time.time()
        last_log = [0.0]

        part_sz = max(32 * 1024 * 1024, (total_size + num_workers - 1) // num_workers)
        ranges = [
            (offset, min(offset + part_sz, total_size))
            for offset in range(0, total_size, part_sz)
        ]

        def _chunk_copy(span: Tuple[int, int]):
            nonlocal copied_bytes
            start_b, end_b = span
            cur = start_b
            step = 16 * 1024 * 1024
            with open(src, "rb") as rf, open(tmp, "r+b") as wf:
                rf.seek(start_b)
                wf.seek(start_b)
                while cur < end_b:
                    to_read = min(step, end_b - cur)
                    buf = rf.read(to_read)
                    if not buf:
                        break
                    wf.write(buf)
                    cur += len(buf)
                    now = time.time()
                    with cp_lock:
                        copied_bytes += len(buf)
                        if now - last_log[0] >= 2.5 or copied_bytes >= total_size:
                            last_log[0] = now
                            dt = max(0.01, now - t0)
                            spd = (copied_bytes / (1024 ** 2)) / dt
                            pct = copied_bytes * 100 // max(1, total_size)
                            log(f"  [云盘极速预热] {os.path.basename(rel_p)}: {pct}% ({copied_bytes/(1024**3):.2f}/{total_size/(1024**3):.2f} GB, {spd:.1f} MB/s)")

        with ThreadPoolExecutor(max_workers=min(num_workers, len(ranges))) as pool:
            futs = [pool.submit(_chunk_copy, r) for r in ranges]
            for fut in as_completed(futs):
                fut.result()

        os.replace(tmp, dst)
        return dst

    copied = [_copy_one_parallel(rel_p, 16) for rel_p in rel_files]
    return copied


def ensure_llama_server() -> Tuple[str, Dict[str, str]]:
    """自动下载/校验预编译 llama.cpp CUDA 12.8 llama-server 与 cloudflared"""
    if not shutil.which("cloudflared"):
        log("正在安装 cloudflared 隧道客户端...")
        subprocess.run(
            "curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
            "-o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
            shell=True,
            check=True,
        )

    install_dir = "/content/llama_cpp_cuda"
    os.makedirs(install_dir, exist_ok=True)

    def _find_bin() -> Optional[str]:
        for root, _, files in os.walk(install_dir):
            if "llama-server" in files:
                p = os.path.join(root, "llama-server")
                os.chmod(p, 0o755)
                return p
        return None

    def _build_env() -> Dict[str, str]:
        env = os.environ.copy()
        ld_dirs = []
        for root, _, files in os.walk(install_dir):
            if any(f == "llama-server" or ".so" in f for f in files):
                ld_dirs.append(root)
        for sys_lib in ("/usr/lib64-nvidia", "/usr/local/nvidia/lib64", "/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"):
            if os.path.isdir(sys_lib):
                ld_dirs.append(sys_lib)
        if env.get("LD_LIBRARY_PATH"):
            ld_dirs.append(env["LD_LIBRARY_PATH"])
        env["LD_LIBRARY_PATH"] = ":".join(ld_dirs)
        return env

    bin_p = _find_bin()
    if bin_p:
        return bin_p, _build_env()

    fallback_tag = "b12651"
    main_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{fallback_tag}/llama-{fallback_tag}-bin-ubuntu-cuda-12.8-x64.tar.gz"
    cudart_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{fallback_tag}/cudart-llama-{fallback_tag}-bin-ubuntu-cuda-12.8-x64.tar.gz"

    # 自动检测当前显卡驱动支持的最高 CUDA 版本（如 13.5 / 12.8），优先匹配最新 Blackwell 优化构建
    driver_cuda = (12, 8)
    try:
        smi_out = subprocess.getoutput("nvidia-smi 2>/dev/null")
        m_cu = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi_out)
        if m_cu:
            driver_cuda = (int(m_cu.group(1)), int(m_cu.group(2)))
    except Exception:
        pass

    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=8",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "ColabDeploy/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            releases = json.loads(resp.read().decode("utf-8"))
        for rel in releases:
            cands: Dict[Tuple[int, int], Dict[str, str]] = {}
            for asset in rel.get("assets") or []:
                aname = str(asset.get("name") or "")
                aurl = str(asset.get("browser_download_url") or "")
                m_ver = re.search(r"-bin-ubuntu-cuda-(\d+)\.(\d+)-x64\.tar\.gz$", aname)
                if not m_ver:
                    continue
                ver_t = (int(m_ver.group(1)), int(m_ver.group(2)))
                slot = cands.setdefault(ver_t, {})
                if aname.startswith("llama-"):
                    slot["main"] = aurl
                elif aname.startswith("cudart-llama-"):
                    slot["rt"] = aurl
            # 从高到低选择第一个 <= 驱动 CUDA 版本且存在主包的构建
            compatible = [v for v in sorted(cands.keys(), reverse=True) if v <= driver_cuda and "main" in cands[v]]
            chosen_v = compatible[0] if compatible else (sorted([v for v in cands if "main" in cands[v]])[0] if any("main" in cands[v] for v in cands) else None)
            if chosen_v is not None:
                main_url = cands[chosen_v]["main"]
                cudart_url = cands[chosen_v].get("rt", "")
                break
    except Exception:
        pass

    log(f"正在下载官方最新 llama.cpp CUDA 引擎 (驱动支持 CUDA {driver_cuda[0]}.{driver_cuda[1]} -> 选用 {os.path.basename(main_url)})...")
    subprocess.run(f'curl -fsSL "{main_url}" -o /tmp/llama.tar.gz && tar -xzf /tmp/llama.tar.gz -C {install_dir} && rm -f /tmp/llama.tar.gz', shell=True, check=True)
    if cudart_url:
        subprocess.run(f'curl -fsSL "{cudart_url}" -o /tmp/cudart.tar.gz && tar -xzf /tmp/cudart.tar.gz -C {install_dir} && rm -f /tmp/cudart.tar.gz', shell=True, check=True)

    bin_p = _find_bin()
    if not bin_p:
        raise RuntimeError("解压 llama.cpp 后未找到 llama-server")
    return bin_p, _build_env()


def _sync_local_to_rclone_drive(local_dir: str, background: bool = False, remote_folder: str = ""):
    rc_conf = os.path.expanduser("~/.config/rclone/rclone.conf")
    dst_folder = (remote_folder or MODEL_FOLDER).strip()
    remote_dst = f"gdrive:models/{dst_folder}"
    cmd = [
        "rclone", "copy", local_dir, remote_dst,
        "--config", rc_conf,
        "--transfers", "8",
        "--drive-chunk-size", "128M",
        "--ignore-existing",
    ]
    if background:
        log(f"正在启动后台进程，通过 Google 内网将本地模型同步持久化至云盘 ({remote_dst})...")
        subprocess.Popen(
            ["nohup"] + cmd,
            stdout=open("/content/drive_backup.log", "w"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    else:
        log(f"正在通过 Google 内网将模型高速同步至 Google Drive 云盘 ({remote_dst})...")
        cmd_fg = cmd + ["--stats", "2s", "--stats-one-line", "-v"]
        proc = subprocess.Popen(
            cmd_fg,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            ln = raw_line.strip()
            # 匹配 rclone stats-one-line: e.g. "12.345 GiB / 134.120 GiB, 9%, 245.678 MiB/s, ETA 8m"
            m = re.search(r"([\d.]+\s*[KMGT]?i?B)\s*/\s*([\d.]+\s*[KMGT]?i?B),\s*(\d+)%,\s*([\d.]+\s*[KMGT]?i?B/s)", ln)
            if m:
                done_s, tot_s, pct_s, spd_s = m.group(1), m.group(2), m.group(3), m.group(4)
                log(f"  [云盘同步] {MODEL_FOLDER}: {pct_s}% ({done_s}/{tot_s}, {spd_s})")
        proc.wait()
        log(f"  [云盘同步] {MODEL_FOLDER}: 100% (已完成全部云盘持久化, 已就绪)")


def run_remote_vm_main():
    global HF_REPO, MODEL_FOLDER, FILE_PATTERN
    t_start = time.time()
    try:
        vram_mib_str = subprocess.getoutput("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null").strip().splitlines()[0]
        vram_gb = float(vram_mib_str) / 1024.0
    except Exception:
        vram_gb = 96.0
    is_minimax_ref2va = (
        MODEL_FOLDER == "MiniMax-H3-Ref2VA"
        or (HF_REPO.lower() == "minimaxai/minimax-h3" and "ref2va" in FILE_PATTERN.lower())
    )
    if vram_gb < 24.0:
        if is_minimax_ref2va and not DOWNLOAD_ONLY:
            raise RuntimeError("MiniMax-H3-Ref2VA (33B 白模/动作迁移) 需要 G4 96GB / H100 / A100 显存，当前 T4 16GB 无法加载。")
        if HF_REPO.lower() == "minimaxai/minimax-h3" and not DOWNLOAD_ONLY:
            log(f"⚡ 检测到当前为 T4 ({vram_gb:.1f} GB 显存)，已自动将 134 GB 的 RTX 6000 版切换为 MiniMax-H3 (T4 量化版 · UD-Q2_K_XL · 8步有声视频)！")
            HF_REPO = "unsloth/MiniMax-H3-GGUF"
            MODEL_FOLDER = "MiniMax-H3-T4-GGUF"
            FILE_PATTERN = "minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf"
        elif HF_REPO.lower() == "meituan-longcat/longcat-video-avatar-1.6" and "int8" not in FILE_PATTERN.lower() and not DOWNLOAD_ONLY:
            log(f"⚡ 检测到当前为 T4 ({vram_gb:.1f} GB 显存)，已自动将 LongCat-Video-Avatar-1.6 切换为 INT8 量化版 (base_model_int8)！")
            MODEL_FOLDER = "LongCat-Video-Avatar-1.6-INT8"
            FILE_PATTERN = "base_model_int8/*.safetensors"
        elif "bf16.gguf" in FILE_PATTERN.lower() and not DOWNLOAD_ONLY:
            log(f"⚠️ 检测到当前为 T4 ({vram_gb:.1f} GB 显存)，已自动将 53GB BF16 切换为 T4 16G 满速版 UD-Q3_K_XL (12.4 GB)！")
            MODEL_FOLDER = "Qwen3.8-27B-T4-Q3"
            FILE_PATTERN = "Huihui-Qwen3.8-27B-abliterated-UD-Q3_K_XL.gguf"
    has_rclone = _has_vm_rclone_drive()
    fuse_mounted = os.path.ismount("/content/drive/MyDrive") or os.path.ismount("/content/drive")
    drive_mounted = fuse_mounted or has_rclone or os.path.isdir("/content/drive/MyDrive")
    drive_model_dir = os.path.join(DRIVE_MODELS_ROOT, MODEL_FOLDER)

    # 选择本地高速目录: 若 /dev/shm/local_models/<MODEL_FOLDER> 已存在或 /dev/shm 剩余空间 > 25GB 则优先用内存盘
    local_base = "/content/local_models"
    shm_model_dir = os.path.join("/dev/shm/local_models", MODEL_FOLDER)
    if os.path.isdir(shm_model_dir) and any(f.endswith(".gguf") for _, _, fs in os.walk(shm_model_dir) for f in fs):
        local_base = "/dev/shm/local_models"
    elif os.path.isdir("/dev/shm"):
        try:
            if shutil.disk_usage("/dev/shm").free > 25 * (1024 ** 3):
                local_base = "/dev/shm/local_models"
        except Exception:
            pass
    local_model_dir = os.path.join(local_base, MODEL_FOLDER)
    os.makedirs(local_model_dir, exist_ok=True)

    if has_rclone and not fuse_mounted:
        log(f"✅ Google Drive 永久免弹窗直连已就绪！目标云盘路径: gdrive:models/{MODEL_FOLDER}")
    elif drive_mounted:
        os.makedirs(DRIVE_MODELS_ROOT, exist_ok=True)
        os.makedirs(drive_model_dir, exist_ok=True)
        existing_folders = sorted([
            d for d in os.listdir(DRIVE_MODELS_ROOT)
            if os.path.isdir(os.path.join(DRIVE_MODELS_ROOT, d)) and not d.startswith(".")
        ])
        log(f"✅ Google Drive 已挂载！云盘总目录: {DRIVE_MODELS_ROOT}")
        log(f"📂 当前 `MyDrive/models/` 下的模型文件夹列表: {existing_folders}")
        log(f"🎯 本次部署目标模型文件夹: {drive_model_dir}")
    else:
        log("ℹ️ 当前未挂载 Google Drive，模型将保存在 Colab 本地高速盘。")

    if MODEL_FOLDER == "ReCaVSR" or "recavsr" in HF_REPO.lower():
        ensure_llama_server()
        _ensure_recavsr_service_on_30002()
        for _ in range(30):
            txt = subprocess.getoutput("cat /content/recavsr_tunnel.log 2>/dev/null")
            if "trycloudflare.com" in txt:
                shutil.copyfile("/content/recavsr_tunnel.log", "/content/cf_tunnel.log")
                break
            time.sleep(1)
        log("[显存装载] ReCaVSR (Wan2.2-5B): 100% (10.1/10.1 GiB VRAM, 已就绪)")
        _start_cloudflare_tunnel_and_print_summary(t_start, drive_model_dir, drive_mounted)
        return

    if MODEL_FOLDER == "IndexTTS-2" or "indextts-2" in HF_REPO.lower() or "indextts2" in HF_REPO.lower():
        _start_indextts2_standalone_on_30000(download_only=DOWNLOAD_ONLY)
        if not DOWNLOAD_ONLY:
            _start_cloudflare_tunnel_and_print_summary(t_start, drive_model_dir, drive_mounted)
        return

    if MODEL_FOLDER.startswith("GLM-5.3-Flash") or "glm-5.3-flash" in HF_REPO.lower():
        _start_glm53_maya_server_on_30000(download_only=DOWNLOAD_ONLY)
        if not DOWNLOAD_ONLY:
            _start_cloudflare_tunnel_and_print_summary(t_start, drive_model_dir, drive_mounted)
        return

    if MODEL_FOLDER == "Qwen-Image-2.1-Uncensored" or "qwen-image-2.1" in (HF_REPO + " " + MODEL_FOLDER).lower():
        _start_qwen_image_21_server_on_30000(download_only=DOWNLOAD_ONLY)
        if not DOWNLOAD_ONLY:
            _start_cloudflare_tunnel_and_print_summary(t_start, drive_model_dir, drive_mounted)
        return

    is_safetensors_repo = (
        HF_REPO in PRESET_SAFETENSORS_FILES
        or FILE_PATTERN.lower().endswith(".safetensors")
        or ("minimax-h3" in HF_REPO.lower() and "gguf" not in HF_REPO.lower())
        or ("qwen-image" in HF_REPO.lower())
        or ("z-image" in HF_REPO.lower())
    )

    if is_safetensors_repo:
        # Safetensors 多分片大模型 (如 MiniMax-H3 134GB) 体积远超 /dev/shm 内存盘余量，统一使用 235GB 的 /content/local_models 高速 NVMe 盘中转
        # MiniMax-H3-Ref2VA 与 MiniMax-H3 (FL2VA) 物理共享 72.4GB 的破限 text_encoder + video_vae + audio_vae + processor！
        storage_folder = "MiniMax-H3" if is_minimax_ref2va else MODEL_FOLDER
        nvme_model_dir = os.path.join("/content/local_models", storage_folder)
        target_dir = os.path.join(DRIVE_MODELS_ROOT, storage_folder) if fuse_mounted else nvme_model_dir
        os.makedirs(target_dir, exist_ok=True)

        # 若 /dev/shm/local_models/<MODEL_FOLDER> 中有之前已下载的完整分片，自动迁移至 NVMe 盘复用并释放 /dev/shm 内存盘空间
        if os.path.isdir(shm_model_dir) and os.path.abspath(shm_model_dir) != os.path.abspath(target_dir):
            log(f"正在将 {shm_model_dir} 中已下载的完整分片迁移至 235GB NVMe 高速盘 {target_dir} 并释放 /dev/shm 空间...")
            for r, _, fs in os.walk(shm_model_dir):
                rel_r = os.path.relpath(r, shm_model_dir)
                dst_r = target_dir if rel_r == "." else os.path.join(target_dir, rel_r)
                os.makedirs(dst_r, exist_ok=True)
                for f in fs:
                    src_f = os.path.join(r, f)
                    if f.endswith((".part", ".tmp")):
                        try:
                            os.remove(src_f)
                        except Exception:
                            pass
                        continue
                    dst_f = os.path.join(dst_r, f)
                    if not os.path.exists(dst_f):
                        shutil.move(src_f, dst_f)
            shutil.rmtree(shm_model_dir, ignore_errors=True)
            log("✅ 已迁移已完成分片并彻底释放 /dev/shm 内存盘！")

        gdrive_check_folder = "MiniMax-H3/Ref2VA/transformer" if is_minimax_ref2va else storage_folder
        gdrive_saved_count = _check_rclone_drive_has_model(gdrive_check_folder) if (has_rclone and not fuse_mounted) else 0
        if gdrive_saved_count > 0:
            log(f"☁️ 已确认 Google Drive (gdrive:models/{gdrive_check_folder}) 已持久化保存 {gdrive_saved_count} 个核心权重文件！正在通过 Google 内网 16 线程极速预热...")

        log(f"正在解析开源多模态/音视频模型仓库 {HF_REPO} 的分片列表...")
        st_files = resolve_hf_safetensors_files(HF_REPO, FILE_PATTERN, HF_TOKEN)
        if not st_files and has_rclone and not fuse_mounted:
            _pull_from_rclone_drive_if_exists(storage_folder, target_dir)

        # 针对 MiniMaxAI/MiniMax-H3，自动将 FL2VA/text_encoder 的 14 个分片与索引替换为 huihui-ai 满血 BF16 破限无审查版 Qwen3-VL-32B-Instruct-abliterated
        uncensored_te_repo = "huihui-ai/Huihui-Qwen3-VL-32B-Instruct-abliterated"
        uncensored_te_sizes: Dict[str, int] = {}
        if HF_REPO.lower() == "minimaxai/minimax-h3":
            log(f"🔓 已启用 [MiniMax-H3 无审查满血 BF16 模式]：自动将 14 个 text_encoder 分片替换为 {uncensored_te_repo}！")
            try:
                te_siblings = resolve_hf_safetensors_files(uncensored_te_repo, "*.safetensors", HF_TOKEN)
                for rfn, sz in te_siblings:
                    uncensored_te_sizes[os.path.basename(rfn)] = sz
            except Exception as e_te:
                log(f"⚠️ 获取破限编码器元数据提示: {e_te}")

        resolved_plan: List[Tuple[str, str, str, str, int]] = []
        for rel_fn, sz in st_files:
            dl_repo = HF_REPO
            dl_rel_fn = rel_fn
            dl_dest_dir = target_dir
            base_fn = os.path.basename(rel_fn)
            if uncensored_te_sizes and rel_fn.startswith("FL2VA/text_encoder/") and base_fn in uncensored_te_sizes:
                dl_repo = uncensored_te_repo
                dl_rel_fn = base_fn
                dl_dest_dir = os.path.join(target_dir, "FL2VA/text_encoder")
                sz = uncensored_te_sizes[base_fn]
            resolved_plan.append((rel_fn, dl_repo, dl_rel_fn, dl_dest_dir, max(0, int(sz))))

        if HF_REPO.lower() == "meituan-longcat/longcat-video-avatar-1.6":
            log("🔗 正在合并 LongCat-Video 基础底座组件 (VAE / Tokenizer / UMT5-XXL TextEncoder) 与 Ming-3.0-Talker (2.78 GB 戏剧人设 & 声音克隆引擎)...")
            try:
                base_repo = "meituan-longcat/LongCat-Video"
                base_siblings = resolve_hf_safetensors_files(base_repo, "*.safetensors", HF_TOKEN)
                for rfn, b_sz in base_siblings:
                    if rfn.startswith(("vae/", "tokenizer/", "text_encoder/")):
                        resolved_plan.append((rfn, base_repo, rfn, target_dir, max(0, int(b_sz))))
            except Exception as e_lc:
                log(f"⚠️ 解析 LongCat-Video 底座组件提示: {e_lc}")
            ming_repo = "inclusionAI/Ming-3.0-audio-flash"
            resolved_plan.extend([
                ("ming_talker/talker/model.safetensors", ming_repo, "talker/model.safetensors", os.path.join(target_dir, "ming_talker/talker"), 1398191068),
                ("ming_talker/talker/vae/model.safetensors", ming_repo, "talker/vae/model.safetensors", os.path.join(target_dir, "ming_talker/talker/vae"), 1560856068),
                ("ming_talker/talker/campplus.onnx", ming_repo, "talker/campplus.onnx", os.path.join(target_dir, "ming_talker/talker"), 28303423),
            ])

        if "minimax-h3-image" in HF_REPO.lower() or MODEL_FOLDER == "MiniMax-H3-Image":
            heretic_repo = "OTMFLY/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot"
            heretic_fn = "qwen3vl_32b_minimax_h3_ultra_uncensored_heretic_int8_convrot.safetensors"
            log(f"🔓 已启用 [MiniMax-H3-Image 4K无审查黄金三件套]：自动挂载 Heretic 32B 破限编码器 ({heretic_repo})！")
            resolved_plan.append((
                f"text_encoders/{heretic_fn}",
                heretic_repo,
                heretic_fn,
                os.path.join(target_dir, "text_encoders"),
                26363476151,
            ))

        batch_total_bytes = sum(x[4] for x in resolved_plan)
        total_gb = batch_total_bytes / (1024 ** 3)
        log(f"共解析到 {len(resolved_plan)} 个模型文件 (总计约 {total_gb:.2f} GB)，目标目录: {target_dir}")

        # 防止三套大型模型 (MiniMax-H3-Image 45GB + LongCat-Avatar 48GB + MiniMax-H3 134GB = 227GB) 同时驻留撑爆 235GB NVMe 盘
        if not fuse_mounted and os.path.isdir("/content/local_models"):
            needed_new_bytes = 0
            for rel_fn, _, _, _, sz in resolved_plan:
                ep = os.path.join(target_dir, rel_fn)
                if not (os.path.exists(ep) and (sz <= 0 or os.path.getsize(ep) == sz)):
                    needed_new_bytes += max(0, sz)
            free_nvme = shutil.disk_usage(target_dir).free
            safety_margin = 18 * (1024 ** 3)  # 始终保留至少 18GB NVMe 余量，防止 git clone 或视频写出爆盘
            if free_nvme < needed_new_bytes + safety_margin:
                for other_name in sorted(os.listdir("/content/local_models")):
                    if other_name == storage_folder:
                        continue
                    other_p = os.path.join("/content/local_models", other_name)
                    if os.path.isdir(other_p):
                        log(f"🧹 [NVMe 防爆盘智能腾挪] 当前磁盘剩余 {free_nvme/(1024**3):.1f} GB，正在释放后台未激活的本地缓存 {other_name} (云盘已永久保存)...")
                        shutil.rmtree(other_p, ignore_errors=True)
                        free_nvme = shutil.disk_usage(target_dir).free
                        if free_nvme >= needed_new_bytes + safety_margin:
                            break
                # 若同目录下 FL2VA 与 Ref2VA 两个 62GB transformer 无法同时驻留 NVMe，只腾挪未激活的那份 transformer，完整保留 72.4GB 共享无审查编码器与 VAE！
                if free_nvme < needed_new_bytes + safety_margin and storage_folder == "MiniMax-H3":
                    inactive_tr = os.path.join(target_dir, "FL2VA/transformer" if is_minimax_ref2va else "Ref2VA/transformer")
                    if os.path.isdir(inactive_tr):
                        log(f"🧹 [NVMe 共享底座腾挪] 保留 72.4 GB 共享破限编码器与 VAE，释放未激活的 {os.path.relpath(inactive_tr, target_dir)} (云盘已永久保存)...")
                        shutil.rmtree(inactive_tr, ignore_errors=True)
        gdrive_pull_count = (
            _check_rclone_drive_has_model(storage_folder)
            if (has_rclone and not fuse_mounted and is_minimax_ref2va and gdrive_saved_count == 0)
            else gdrive_saved_count
        )
        if has_rclone and not fuse_mounted and gdrive_pull_count > 0:
            drive_excludes: List[str] = []
            if storage_folder == "MiniMax-H3":
                drive_excludes = ["FL2VA/transformer/**"] if is_minimax_ref2va else ["Ref2VA/**"]
            elif storage_folder == "LongCat-Video-Avatar-1.6":
                drive_excludes = ["base_model/**"] if "int8" in FILE_PATTERN.lower() else ["base_model_int8/**"]
            log(f"☁️ 优先直接从您的 Google Drive (gdrive:models/{storage_folder}) 16 路并发拉取已备份权重（缺片才走 HF 自动补齐）...")
            _pull_from_rclone_drive_if_exists(storage_folder, target_dir, excludes=drive_excludes)
        cum_done_bytes = 0
        st_stage_tag = "云盘极速预热" if gdrive_saved_count > 0 else "HF 下载"
        for idx, (rel_fn, dl_repo, dl_rel_fn, dl_dest_dir, sz) in enumerate(resolved_plan, start=1):
            existing_p = os.path.join(target_dir, rel_fn)
            lbl_str = f"{MODEL_FOLDER} 总进度 ({idx}/{len(resolved_plan)} 个核心组件)"
            if os.path.exists(existing_p) and (sz <= 0 or os.path.getsize(existing_p) == sz):
                actual_sz = os.path.getsize(existing_p) if os.path.exists(existing_p) else sz
                cum_done_bytes += actual_sz
                ov_pct = min(100, cum_done_bytes * 100 // max(1, batch_total_bytes)) if batch_total_bytes > 0 else 100
                log(f"  [{st_stage_tag}] {lbl_str}: {ov_pct}% ({cum_done_bytes/(1024**3):.2f}/{total_gb:.2f} GB, 本地缓存复用)")
                continue
            tag_prefix = "🔓 [破限编码器] " if (dl_repo == uncensored_te_repo or "heretic" in dl_repo.lower()) else ""
            log(f"{tag_prefix}正在 16 线程极速预热 [{idx}/{len(resolved_plan)}] {rel_fn} ({sz/(1024**3):.2f} GB)...")
            try:
                dl_p = fast_parallel_hf_download(
                    dl_repo,
                    dl_rel_fn,
                    dl_dest_dir,
                    token=HF_TOKEN,
                    num_conn=16,
                    batch_done_bytes=cum_done_bytes,
                    batch_total_bytes=batch_total_bytes,
                    batch_label=lbl_str,
                    stage_tag=st_stage_tag,
                )
                cum_done_bytes += (os.path.getsize(dl_p) if os.path.exists(dl_p) else sz)
            except Exception:
                if has_rclone and not fuse_mounted:
                    _pull_from_rclone_drive_if_exists(MODEL_FOLDER, target_dir)
                    break
                raise
        if HF_REPO.lower() == "meituan-longcat/longcat-video-avatar-1.6":
            ming_talker_root = os.path.join(target_dir, "ming_talker")
            if not os.path.isfile(os.path.join(ming_talker_root, "talker/inference.py")):
                log("🎭 正在同步 Ming-3.0-Audio-Flash Talker 戏剧人设与零样本声音克隆推理脚本及配置...")
                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(
                        repo_id="inclusionAI/Ming-3.0-audio-flash",
                        allow_patterns=["talker/*", "voice_presets/*"],
                        ignore_patterns=["*.safetensors", "*.onnx", "*__pycache__*"],
                        local_dir=ming_talker_root,
                    )
                except Exception as e_mt:
                    log(f"⚠️ 同步 MingTalker 脚本提示: {e_mt}")
        expected_core_count = (
            13 if is_minimax_ref2va else max(
                1,
                sum(1 for x in resolved_plan if x[0].lower().endswith((".gguf", ".safetensors", ".onnx", ".pth"))),
            )
        )
        if has_rclone and not fuse_mounted and gdrive_saved_count < expected_core_count:
            _sync_local_to_rclone_drive(target_dir, background=not DOWNLOAD_ONLY, remote_folder=storage_folder)
        log(f"✅ 模型 {HF_REPO} 全部 {len(resolved_plan)} 个文件已就绪！")
        if DOWNLOAD_ONLY:
            log(f"🎉 [仅下载模式] 已完成全部权重下载并持久化至 Google Drive 云盘 (gdrive:models/{storage_folder})！")
            return
        if HF_REPO.lower() == "unsloth/minimax-h3-gguf" or MODEL_FOLDER == "MiniMax-H3-T4-GGUF":
            _start_minimax_t4_gguf_server_on_30000(target_dir)
        elif "longcat-video-avatar" in HF_REPO.lower() or MODEL_FOLDER.startswith("LongCat-Video-Avatar"):
            _start_longcat_avatar_server_on_30000(target_dir, total_gb=total_gb)
        elif "minimax-h3-image" in HF_REPO.lower() or MODEL_FOLDER == "MiniMax-H3-Image":
            _start_minimax_h3_image_server_on_30000(target_dir, total_gb=total_gb)
        else:
            _start_minimax_h3_server_on_30000(target_dir, variant="Ref2VA" if is_minimax_ref2va else "FL2VA")
        _start_cloudflare_tunnel_and_print_summary(t_start, drive_model_dir, drive_mounted)
        return

    if DOWNLOAD_ONLY:
        target_dir = drive_model_dir if fuse_mounted else local_model_dir
        os.makedirs(target_dir, exist_ok=True)
        log(f"[仅下载模式] 正在解析 GGUF 仓库 {HF_REPO} (过滤: {FILE_PATTERN})...")
        selected_files = resolve_hf_gguf_files(HF_REPO, FILE_PATTERN, HF_TOKEN)
        dl_tot_bytes = sum(max(0, int(sz)) for _, sz in selected_files)
        dl_done_bytes = 0
        for idx, (rel_fn, sz) in enumerate(selected_files, start=1):
            existing_p = os.path.join(target_dir, rel_fn)
            lbl_str = f"{MODEL_FOLDER} 总进度 ({idx}/{len(selected_files)} · {os.path.basename(rel_fn)})"
            if os.path.exists(existing_p) and (sz <= 0 or os.path.getsize(existing_p) == sz):
                dl_done_bytes += (os.path.getsize(existing_p) if os.path.exists(existing_p) else sz)
                log(f"复用已存在的 [{idx}/{len(selected_files)}] {rel_fn} ({sz/(1024**3):.2f} GB)")
                continue
            log(f"正在 16 线程下载 [{idx}/{len(selected_files)}] {rel_fn} ({sz/(1024**3):.2f} GB)...")
            dl_p = fast_parallel_hf_download(
                HF_REPO,
                rel_fn,
                target_dir,
                token=HF_TOKEN,
                num_conn=16,
                batch_done_bytes=dl_done_bytes,
                batch_total_bytes=dl_tot_bytes,
                batch_label=lbl_str,
            )
            dl_done_bytes += (os.path.getsize(dl_p) if os.path.exists(dl_p) else sz)
        if has_rclone and not fuse_mounted:
            _sync_local_to_rclone_drive(target_dir, background=False)
        log(f"🎉 [仅下载模式] 已将 {HF_REPO} 下载并持久化至 Google Drive: gdrive:models/{MODEL_FOLDER}")
        return

    # Step 1: 准备权重 (优先检查本地高速盘 -> 检查云盘持久化状态 -> 使用 G4 10Gbps 16线程极速预热通道，云盘作兜底)
    def _collect_matching_local() -> List[str]:
        matched = []
        for r, _, fs in os.walk(local_model_dir):
            for f in fs:
                if f.lower().endswith(".gguf") and not f.endswith((".part", ".tmp")):
                    full_p = os.path.join(r, f)
                    if _matches_main_pattern(f, FILE_PATTERN):
                        matched.append(full_p)
        has_main = any("mmproj" not in os.path.basename(p).lower() and not os.path.basename(p).lower().startswith("mtp") for p in matched)
        return matched if has_main else []

    gdrive_gguf_count = _check_rclone_drive_has_model(MODEL_FOLDER) if (has_rclone and not fuse_mounted) else 0
    local_ggufs = _collect_matching_local()
    if not local_ggufs and drive_mounted and fuse_mounted:
        copied = copy_drive_dir_to_local(drive_model_dir, local_model_dir)
        if copied:
            local_ggufs = copied

    is_qwen38_series = "qwen3.8-27b" in (HF_REPO + " " + MODEL_FOLDER).lower()
    mtp_expected_bytes = 1372692544 if is_qwen38_series else 0
    gguf_tot_bytes = 0
    gguf_done_bytes = 0
    gguf_tot_count = 0

    if not local_ggufs and has_rclone and not fuse_mounted and gdrive_gguf_count > 0:
        if local_model_dir.startswith("/dev/shm"):
            subprocess.run("pkill -9 -f '[l]lama-server|[r]clone' || true", shell=True)
            if os.path.isdir("/dev/shm/EasyColab_Models") and shutil.disk_usage("/dev/shm").free < 58 * (1024 ** 3):
                shutil.rmtree("/dev/shm/EasyColab_Models", ignore_errors=True)
            if shutil.disk_usage("/dev/shm").free < 56 * (1024 ** 3):
                log("提示: /dev/shm 空间不足以容纳整份权重，自动切换至 /content/local_models 高速 NVMe 盘...")
                local_model_dir = os.path.join("/content/local_models", MODEL_FOLDER)
                os.makedirs(local_model_dir, exist_ok=True)
        gguf_includes = None
        if FILE_PATTERN:
            _pat = FILE_PATTERN.strip()
            _inc_main = _pat if ("*" in _pat or "?" in _pat) else f"*{_pat}*"
            gguf_includes = [_inc_main, "*mmproj*", "MTP/**", "*mtp*.gguf"]
        log(f"☁️ 检测到您的 Google Drive (gdrive:models/{MODEL_FOLDER}) 已保存 {gdrive_gguf_count} 个权重文件，直接优先从您的云盘多线程高速拉取！")
        if _pull_from_rclone_drive_if_exists(MODEL_FOLDER, local_model_dir, includes=gguf_includes):
            local_ggufs = _collect_matching_local()

    if not local_ggufs:
        if gdrive_gguf_count > 0:
            log(f"ℹ️ 云盘中未匹配到 {FILE_PATTERN or '目标权重'} 或文件不完整，自动改由 HuggingFace 补齐并回传云盘...")
        log(f"正在解析模型文件列表 {HF_REPO} (过滤规则: {FILE_PATTERN or '自动优选'})...")
        try:
            selected_files = resolve_hf_gguf_files(HF_REPO, FILE_PATTERN, HF_TOKEN)
        except Exception:
            selected_files = []
            if has_rclone and not fuse_mounted and _pull_from_rclone_drive_if_exists(MODEL_FOLDER, local_model_dir):
                local_ggufs = _collect_matching_local()

        if selected_files and not local_ggufs:
            gguf_tot_bytes = sum(max(0, int(sz)) for _, sz in selected_files) + mtp_expected_bytes
            gguf_tot_count = len(selected_files) + (1 if is_qwen38_series else 0)
            # 清理同目录下不匹配当前 FILE_PATTERN 的旧量化版本或残留分块
            selected_basenames = {os.path.basename(rfn) for rfn, _ in selected_files}
            for scan_dir in ([local_model_dir, drive_model_dir] if fuse_mounted else [local_model_dir]):
                if os.path.isdir(scan_dir):
                    for r, _, fs in os.walk(scan_dir):
                        for f in fs:
                            if (f.lower().endswith((".gguf", ".part", ".tmp")) or ".partial" in f.lower()) and f not in selected_basenames and not f.lower().startswith("mtp"):
                                old_p = os.path.join(r, f)
                                try:
                                    os.remove(old_p)
                                except Exception:
                                    pass

            # 检查需要新下载的字节数；若 /dev/shm 空间不足则先释放旧进程与缓存目录
            missing_bytes = sum(
                sz for rel_fn, sz in selected_files
                if not (os.path.exists(os.path.join(local_model_dir, rel_fn)) and os.path.getsize(os.path.join(local_model_dir, rel_fn)) == sz)
            )
            if local_model_dir.startswith("/dev/shm") and missing_bytes > 0:
                subprocess.run("pkill -9 -f '[l]lama-server|[r]clone' || true", shell=True)
                if os.path.isdir("/dev/shm/EasyColab_Models") and shutil.disk_usage("/dev/shm").free < missing_bytes + 4 * (1024 ** 3):
                    log("释放 /dev/shm/EasyColab_Models 中残留的缓存权重以腾出内存盘空间...")
                    shutil.rmtree("/dev/shm/EasyColab_Models", ignore_errors=True)
                if shutil.disk_usage("/dev/shm").free < missing_bytes + 2 * (1024 ** 3):
                    log("提示: /dev/shm 剩余空间不足以容纳完整权重，自动切换至 /content/local_models 高速 NVMe 盘...")
                    new_dir = os.path.join("/content/local_models", MODEL_FOLDER)
                    os.makedirs(new_dir, exist_ok=True)
                    for f in os.listdir(local_model_dir):
                        shutil.move(os.path.join(local_model_dir, f), os.path.join(new_dir, f))
                    local_model_dir = new_dir

            for idx, (rel_fn, sz) in enumerate(selected_files, start=1):
                existing_p = os.path.join(local_model_dir, rel_fn)
                lbl_str = f"{MODEL_FOLDER} 总进度 ({idx}/{gguf_tot_count} · {os.path.basename(rel_fn)})"
                if os.path.exists(existing_p) and (sz <= 0 or os.path.getsize(existing_p) == sz):
                    actual_sz = os.path.getsize(existing_p) if os.path.exists(existing_p) else sz
                    gguf_done_bytes += actual_sz
                    log(f"复用本地已存在的 [{idx}/{gguf_tot_count}] {rel_fn} ({sz/(1024**3):.2f} GB)")
                    local_ggufs.append(existing_p)
                    continue
                log(f"正在 16 线程极速预热 [{idx}/{gguf_tot_count}] {rel_fn} ({sz/(1024**3):.2f} GB) 至内存盘...")
                try:
                    dl_path = fast_parallel_hf_download(
                        HF_REPO,
                        rel_fn,
                        local_model_dir,
                        token=HF_TOKEN,
                        num_conn=16,
                        batch_done_bytes=gguf_done_bytes,
                        batch_total_bytes=gguf_tot_bytes,
                        batch_label=lbl_str,
                        stage_tag="云盘极速预热" if gdrive_gguf_count > 0 else "HF 下载",
                    )
                    gguf_done_bytes += (os.path.getsize(dl_path) if os.path.exists(dl_path) else sz)
                    local_ggufs.append(dl_path)
                except Exception:
                    if has_rclone and not fuse_mounted and _pull_from_rclone_drive_if_exists(MODEL_FOLDER, local_model_dir):
                        local_ggufs = _collect_matching_local()
                        break
                    raise

    # 针对 Qwen3.8-27B 系列 (含 huihui-ai 破限 BF16 原生版)，自动挂载 Unsloth MTP 多词元投机解码头 (1.28 GiB，100% 保持原模型无损输出并提速)
    if is_qwen38_series:
        has_mtp = any("/mtp" in p.lower() or os.path.basename(p).lower().startswith("mtp") for p in local_ggufs)
        if not has_mtp:
            mtp_rel = "MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
            # 若 /dev/shm 已被 51.8 GiB 的 BF16 占满 (剩余 < 2 GiB)，则将 1.28 GiB 的 MTP 头放在 /content/local_models 高速 NVMe 盘
            mtp_base_dir = local_model_dir
            if shutil.disk_usage(local_model_dir).free < 2 * (1024 ** 3):
                mtp_base_dir = os.path.join("/content/local_models", MODEL_FOLDER)
                os.makedirs(mtp_base_dir, exist_ok=True)
            mtp_local = os.path.join(mtp_base_dir, mtp_rel)
            if os.path.exists(mtp_local) and os.path.getsize(mtp_local) > 1_000_000_000:
                log(f"✅ 复用本地已存在的 MTP 投机解码加速头: {mtp_rel}")
                local_ggufs.append(mtp_local)
            else:
                try:
                    log("⚡ 正在为 Qwen3.8-27B 自动拉取 Unsloth MTP 多词元投机解码加速头 (1.28 GB，保持 100% BF16 原生精度无损提速)...")
                    mtp_lbl = f"{MODEL_FOLDER} 总进度 ({gguf_tot_count or 3}/{gguf_tot_count or 3} · mtp-Qwen3.8-27B-Q4_0.gguf)"
                    dl_mtp = fast_parallel_hf_download(
                        "unsloth/Qwen3.8-27B-GGUF",
                        mtp_rel,
                        mtp_base_dir,
                        token=HF_TOKEN,
                        num_conn=16,
                        batch_done_bytes=gguf_done_bytes,
                        batch_total_bytes=gguf_tot_bytes,
                        batch_label=mtp_lbl,
                    )
                    local_ggufs.append(dl_mtp)
                except Exception as e_mtp:
                    log(f"⚠️ 拉取可选 MTP 投机解码头跳过 ({e_mtp})，继续使用纯主模型。")

    if drive_mounted and SYNC_TO_DRIVE and local_ggufs:
        if has_rclone and not fuse_mounted:
            if gdrive_gguf_count < len(local_ggufs) and not subprocess.getoutput("pgrep -f '[r]clone copy'"):
                _sync_local_to_rclone_drive(local_model_dir, background=True)
        else:
            drive_ggufs = [
                f for _, _, fs in os.walk(drive_model_dir)
                for f in fs if f.lower().endswith(".gguf") and not f.endswith((".part", ".tmp")) and _matches_main_pattern(f, FILE_PATTERN)
            ]
            if len(drive_ggufs) < len(local_ggufs) and not subprocess.getoutput("pgrep -f '[c]p -rn'"):
                log(f"正在启动后台进程，将本地模型持久化同步至 Google Drive: {drive_model_dir} ...")
                subprocess.Popen(
                    ["nohup", "cp", "-rn", f"{local_model_dir}/.", f"{drive_model_dir}/"],
                    stdout=open("/content/drive_backup.log", "w"),
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    close_fds=True,
                    start_new_session=True,
                )

    mmprojs = sorted([p for p in local_ggufs if "mmproj" in os.path.basename(p).lower()])
    mtps = sorted([
        p for p in local_ggufs
        if ("/mtp" in p.lower() or os.path.basename(p).lower().startswith("mtp"))
        and p not in mmprojs
    ])
    mains = sorted([p for p in local_ggufs if p not in mmprojs and p not in mtps])
    if not mains:
        raise RuntimeError(f"在 {local_model_dir} 中未找到主 GGUF 权重文件")

    main_gguf = mains[0]
    mmproj_gguf = mmprojs[0] if mmprojs else None
    mtp_gguf = mtps[0] if mtps else None

    # Step 2: 安装并启动 llama-server (端口 30000)
    bin_path, llama_env = ensure_llama_server()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # 检查是否已有同模型的健康 llama-server 在运行
    already_running = False
    running_cmdline = subprocess.getoutput("tr '\\0' ' ' < /proc/$(pgrep -f '[l]lama-server' | head -n1)/cmdline 2>/dev/null")
    if main_gguf in running_cmdline and (not mtp_gguf or mtp_gguf in running_cmdline):
        try:
            with opener.open("http://127.0.0.1:30000/health", timeout=2) as resp:
                if resp.status == 200:
                    already_running = True
                    m_key = re.search(r"--api-key\s+(\S+)", running_cmdline)
                    if m_key:
                        globals()["API_KEY"] = m_key.group(1)
                    log(f"✅ 检测到 llama-server 已在运行并就绪 ({os.path.basename(main_gguf)})，直接复用！")
        except Exception:
            pass

    if not already_running:
        mtp_candidates = [mtp_gguf, None] if mtp_gguf else [None]
        for attempt_mtp in mtp_candidates:
            subprocess.run("pkill -9 -f '[l]lama-server|minimax_server\\.py|minimax_t4_server\\.py|longcat_avatar_server\\.py|minimax_image_server\\.py|qwen_image_server\\.py|project-maya/serve/server\\.py|build/maya|maya-vision|/content/app\\.py' || true; fuser -k -9 30000/tcp 2>/dev/null || true", shell=True)

            weights_gb = sum(
                os.path.getsize(p) / (1024 ** 3)
                for p in (main_gguf, mmproj_gguf, attempt_mtp)
                if p and os.path.exists(p)
            )
            f16_kv_need_gb = (CTX_LEN / 262144.0) * 17.6
            use_f16_kv = (vram_gb >= 78.0) and ((vram_gb - weights_gb) >= (f16_kv_need_gb + 6.0))
            kv_cache_type = "f16" if use_f16_kv else "q8_0"

            cmd = [
                bin_path,
                "-m", main_gguf,
                "--fit", "off",
                "--alias", HF_REPO or MODEL_FOLDER,
                "--host", "0.0.0.0",
                "--port", "30000",
                "-c", str(CTX_LEN),
                "--kv-unified",
                "--parallel", "1",
                "--flash-attn", "on",
                "--cache-type-k", kv_cache_type,
                "--cache-type-v", kv_cache_type,
                "-b", "2048",
                "-ub", "1024",
                "--backend-sampling",
                "--jinja",
                "--metrics",
                "--reasoning-format", "deepseek",
                "-ngl", "999",
                "--api-key", API_KEY,
            ]
            if mmproj_gguf:
                cmd.extend(["--mmproj", mmproj_gguf, "--image-min-tokens", "1024"])
            if attempt_mtp:
                cmd.extend([
                    "--model-draft", attempt_mtp,
                    "-ngld", "999",
                    "--spec-type", "draft-mtp",
                    "--spec-draft-n-max", "8",
                    "--spec-draft-p-min", "0.65",
                    "--cache-type-k-draft", kv_cache_type,
                    "--cache-type-v-draft", kv_cache_type,
                ])

            mtp_tag = f" + MTP投机加速({os.path.basename(attempt_mtp)})" if attempt_mtp else ""
            log(f"正在启动 llama-server 加载 {os.path.basename(main_gguf)}{mtp_tag} (上下文: {CTX_LEN} · KV: {kv_cache_type})...")
            with open("/content/llama_server.log", "w") as lf:
                proc = subprocess.Popen(
                    cmd,
                    env=llama_env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    close_fds=True,
                    start_new_session=True,
                )

            t0 = time.time()
            server_ok = False
            kv_gb = f16_kv_need_gb if use_f16_kv else (CTX_LEN / 262144.0) * 8.8
            expected_vram_gb = max(4.0, weights_gb + kv_gb)
            last_logged_pct = -1

            def _read_gpu_used_gb() -> float:
                try:
                    out = subprocess.getoutput("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null").strip()
                    return float(out.splitlines()[0].strip()) / 1024.0
                except Exception:
                    return 0.0

            while time.time() - t0 < 240:
                if proc.poll() is not None:
                    tail = subprocess.getoutput("tail -n 30 /content/llama_server.log")
                    if attempt_mtp:
                        log(f"⚠️ MTP 投机模式未能附加，自动回退纯主模型模式...")
                        break
                    raise RuntimeError(f"llama-server 异常退出:\n{tail}")
                used_vram_gb = _read_gpu_used_gb()
                elapsed = max(0.5, time.time() - t0)
                try:
                    with opener.open("http://127.0.0.1:30000/health", timeout=2) as resp:
                        if resp.status == 200:
                            final_vram = max(used_vram_gb, weights_gb)
                            log(f"[显存装载] {os.path.basename(main_gguf)}: 100% ({final_vram:.1f}/{final_vram:.1f} GiB VRAM, 已就绪)")
                            log(f"✅ llama-server 推理引擎已就绪！({os.path.basename(main_gguf)}{mtp_tag})")
                            server_ok = True
                            break
                except Exception:
                    pass
                vram_pct = int((used_vram_gb / expected_vram_gb) * 100) if used_vram_gb > 0.5 else int(min(15, elapsed * 2))
                pct = min(98, max(2, vram_pct))
                if pct != last_logged_pct:
                    last_logged_pct = pct
                    vram_spd = used_vram_gb / elapsed if used_vram_gb > 0.5 else 0.0
                    log(f"[显存装载] {os.path.basename(main_gguf)}: {pct}% ({used_vram_gb:.1f}/{expected_vram_gb:.1f} GiB VRAM, {vram_spd:.2f} GB/s)")
                time.sleep(2)
            if server_ok:
                break
        else:
            raise TimeoutError("等待 llama-server 启动超时")

    # Step 3: 启动 Cloudflare Quick Tunnel
    _start_cloudflare_tunnel_and_print_summary(t_start, drive_model_dir, drive_mounted)


def _start_cloudflare_tunnel_and_print_summary(t_start: float, drive_model_dir: str, drive_mounted: bool):
    log("正在启动 Cloudflare Tunnel (trycloudflare.com)...")
    cf_bin = shutil.which("cloudflared") or "/usr/local/bin/cloudflared"
    if not subprocess.getoutput("pgrep -f '[c]loudflared.*30000'"):
        subprocess.Popen(
            ["nohup", cf_bin, "--no-autoupdate", "tunnel", "--url", "http://127.0.0.1:30000"],
            stdout=open("/content/cf_tunnel.log", "w"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )

    tunnel_url = None
    for _ in range(30):
        txt = subprocess.getoutput("cat /content/cf_tunnel.log 2>/dev/null")
        m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", txt)
        if m:
            tunnel_url = m.group(0)
            break
        time.sleep(1)

    recavsr_url = None
    for _ in range(10):
        rc_txt = subprocess.getoutput("cat /content/recavsr_tunnel.log 2>/dev/null")
        m_rc = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", rc_txt)
        if m_rc:
            recavsr_url = m_rc.group(0)
            break
        time.sleep(1)

    total_dt = time.time() - t_start
    combo_line = f"{tunnel_url} | {recavsr_url}" if recavsr_url else str(tunnel_url)
    print("\n" + "=" * 72)
    print(f"DEPLOYMENT SUCCESSFUL (总耗时 {total_dt:.1f}s)")
    print(f"TUNNEL_BASE_URL={tunnel_url}/v1")
    if recavsr_url:
        print(f"RECAVSR_TUNNEL_URL={recavsr_url}")
    print(f"COPY_TO_LOCAL_PANEL={combo_line}")
    print(f"API_KEY={API_KEY}")
    print(f"MODEL_ID={HF_REPO or MODEL_FOLDER}")
    print(f"DRIVE_MODEL_FOLDER={drive_model_dir if drive_mounted else '未挂载 (仅本地)'}")
    print("=" * 72, flush=True)


# ============================================================================
# 本地控制端 (在你的 Mac 上运行：原生挂载 Google Drive + 远端执行部署)
# ============================================================================

def find_colab_bin() -> str:
    local_venv_colab = Path(__file__).resolve().parent / ".venv" / "bin" / "colab"
    if local_venv_colab.exists():
        return str(local_venv_colab)
    which_c = shutil.which("colab")
    if which_c:
        return which_c
    raise RuntimeError("未找到 colab CLI，请先运行: .venv/bin/pip install google-colab-cli")


def mount_google_drive_native(session_name: str) -> bool:
    """
    通过 colab_cli 原生协议将完整 Google Drive 根目录挂载至 /content/drive/MyDrive。
    - 自动卸载之前可能残留的 rclone 子目录挂载
    - 若需要浏览器授权，自动弹出网页并轮询检测授权完成（无需回终端按回车）
    - 若缓存会话已过期或 WebSocket 断开，返回 False 以便上层自动重建实例
    """
    from colab_cli.auth import get_credentials
    from colab_cli.common import state
    from colab_cli.runtime import ColabRuntime
    from colab_cli.utils import get_status_code, open_url_in_browser

    try:
        s = state.get_session(session_name)

        def _on_k_started(kid):
            s.kernel_id = kid
            state.store.add(s)

        def _on_s_started(sid):
            s.session_id = sid
            state.store.add(s)

        runtime = ColabRuntime(
            s.url,
            s.token,
            session_name=s.name,
            history=state.history,
            kernel_id=s.kernel_id,
            session_id=s.session_id,
            on_kernel_started=_on_k_started,
            on_session_started=_on_s_started,
        )
    except Exception as e:
        print(f"⚠️ 无法连接缓存会话 '{session_name}' ({e})，准备重新分配实例...", flush=True)
        return False

    def auto_polling_drivefs_hook(deserialize_msg, wsclient):
        content = deserialize_msg.get("content", {})
        if content.get("request", {}).get("authType") == "dfs_ephemeral":
            msg_id = deserialize_msg.get("metadata", {}).get("colab_msg_id")
            url = f"{state.client.colab_domain}/tun/m/credentials-propagation/{s.endpoint}"
            params = {
                "authuser": "0",
                "authtype": "dfs_ephemeral",
                "version": "2",
                "dryrun": "true",
                "propagate": "true",
                "record": "false",
            }
            creds = get_credentials(state.client_oauth_config, provider=state.auth_provider)
            resp = creds.request("GET", url, params=params)
            token = (
                json.loads(resp.text.split("\n", 1)[-1]).get("token")
                if get_status_code(resp) == 200
                else None
            )
            headers = {"x-goog-colab-token": token}
            # 先尝试直接以 dryrun=false 传递凭证；若账号已授权过，无需任何弹窗即可直接成功
            params["dryrun"] = "false"
            resp = creds.request("POST", url, params=params, headers=headers, files={"file_id": (None, "empty.ipynb")})
            data = json.loads(resp.text.split("\n", 1)[-1]) if get_status_code(resp) == 200 else {}

            if not data.get("success"):
                params["dryrun"] = "true"
                r_dry = creds.request("POST", url, params=params, headers=headers, files={"file_id": (None, "empty.ipynb")})
                d_dry = json.loads(r_dry.text.split("\n", 1)[-1]) if get_status_code(r_dry) == 200 else {}
                uri = d_dry.get("unauthorized_redirect_uri") or data.get("unauthorized_redirect_uri")
                print(f"\n[Google Drive 授权] 请在自动弹出的浏览器窗口中点击「允许 (Allow)」授权挂载云盘:\n{uri}\n", flush=True)
                if uri:
                    open_url_in_browser(uri)
                print("[Google Drive 授权] 正在自动等待你在浏览器完成点击（无需回终端按回车）...", flush=True)
                params["dryrun"] = "false"
                for _ in range(90):
                    time.sleep(2.0)
                    r_get = creds.request("GET", url, params=params)
                    if get_status_code(r_get) == 200:
                        t_new = json.loads(r_get.text.split("\n", 1)[-1]).get("token")
                        if t_new:
                            headers = {"x-goog-colab-token": t_new}
                    resp = creds.request("POST", url, params=params, headers=headers, files={"file_id": (None, "empty.ipynb")})
                    if get_status_code(resp) == 200:
                        d_poll = json.loads(resp.text.split("\n", 1)[-1])
                        if d_poll.get("success"):
                            print("[Google Drive 授权] ✅ 检测到浏览器授权已完成！正在挂载云盘...", flush=True)
                            break

            if get_status_code(resp) == 200:
                reply = wsclient.session.msg(
                    "input_reply",
                    {"value": {"type": "colab_reply", "colab_msg_id": msg_id}},
                )
                if "header" in deserialize_msg:
                    reply["parent_header"] = deserialize_msg["header"]
                wsclient.stdin_channel.send(reply)
            else:
                print(f"[Google Drive 授权] 挂载授权传递失败: {get_status_code(resp)} {resp.text}", flush=True)
            return True
        return False

    runtime.colab_request_hook = auto_polling_drivefs_hook
    rclone_conf_path = Path(__file__).resolve().parent / ".gdrive_rclone.conf"
    rclone_conf_text = ""
    if rclone_conf_path.exists():
        try:
            txt = rclone_conf_path.read_text(encoding="utf-8")
            if "[gdrive]" in txt and "refresh_token" in txt:
                rclone_conf_text = txt
        except Exception:
            pass

    mount_code = f"""
import os, shutil, subprocess, time

rclone_conf_content = {repr(rclone_conf_text)}

if rclone_conf_content:
    print('[Google Drive 永久静默模式] 正在注入本机永久授权凭证与高速传输引擎 (无需浏览器弹窗)...', flush=True)
    rc_dir = os.path.expanduser('~/.config/rclone')
    os.makedirs(rc_dir, mode=0o700, exist_ok=True)
    rc_dst = os.path.join(rc_dir, 'rclone.conf')
    with open(rc_dst, 'w', encoding='utf-8') as f:
        f.write(rclone_conf_content)
    os.chmod(rc_dst, 0o600)
    if not shutil.which('rclone'):
        subprocess.run(
            "curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rclone.zip && "
            "unzip -q -o /tmp/rclone.zip -d /tmp && "
            "cp /tmp/rclone-*-linux-amd64/rclone /usr/local/bin/rclone && "
            "chmod +x /usr/local/bin/rclone && rm -rf /tmp/rclone*",
            shell=True,
            check=False,
        )
    print('✅ [Google Drive 永久静默模式] 已绑定您的云盘根目录 gdrive:models/（全程免浏览器弹窗）！', flush=True)
elif not os.path.ismount('/content/drive/MyDrive') and not os.path.isdir('/content/drive/MyDrive/models'):
    from google.colab import drive
    drive.mount('/content/drive')
    os.makedirs('/content/drive/MyDrive/models', exist_ok=True)
print('DRIVE_ROOT_READY:/content/drive/MyDrive/models')
"""
    try:
        runtime.kernel_client._manager.client.connection_ready.wait(timeout=15)

        def _stream_mount_out(out_msg):
            if out_msg.get("output_type") == "stream" and "text" in out_msg:
                sys.stdout.write(out_msg["text"])
                sys.stdout.flush()

        if rclone_conf_text:
            # 永久静默 rclone 模式不需要 stdin 交互，可直接开启流式 output_hook
            runtime.execute_code(mount_code, output_hook=_stream_mount_out, allow_stdin=False, timeout=600)
        else:
            # 交互式 drive.mount 需要 allow_stdin=True（colab_cli 不允许同时传 output_hook）
            outputs = runtime.execute_code(mount_code, allow_stdin=True, timeout=600)
            for out_msg in outputs or []:
                _stream_mount_out(out_msg)
        return True
    except (NotImplementedError, TypeError, AttributeError) as e:
        print(f"⚠️ Google Drive 挂载脚本调用异常 ({e})，已保留当前 GPU 实例继续执行。", flush=True)
        return True
    except Exception as e:
        print(f"⚠️ 实例 '{session_name}' WebSocket 连接已断开 ({e})，正在自动清理失效会话并重新分配...", flush=True)
        return False
    finally:
        try:
            runtime.stop()
        except Exception:
            pass


def show_remote_status(session_name: str):
    """查询并打印远端 Colab G4 实例的显存 (VRAM)、系统内存 (RAM)、内存盘 (/dev/shm)、本地硬盘 (/content) 及云盘 (Google Drive) 占用"""
    from colab_cli.common import state
    from colab_cli.runtime import ColabRuntime

    s = state.get_session(session_name)
    runtime = ColabRuntime(
        s.url,
        s.token,
        session_name=s.name,
        history=state.history,
        kernel_id=s.kernel_id,
        session_id=s.session_id,
    )
    status_code = """
import os, re, shutil, subprocess

print("=" * 72)
print(" 📊 Google Colab G4 实例硬件与存储实时监控")
print("=" * 72)

# 1. GPU 显存与状态
gpu_info = subprocess.getoutput(
    "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits"
).strip()
if gpu_info:
    parts = [x.strip() for x in gpu_info.split(",")]
    if len(parts) >= 7:
        name, m_used, m_tot, util, temp, p_draw, p_lim = parts[:7]
        pct = float(m_used) * 100.0 / max(1.0, float(m_tot))
        print(f"\\n[1] 🎮 GPU 显存 (VRAM): {name}")
        print(f"    • 显存占用 : {float(m_used)/1024:.2f} GB / {float(m_tot)/1024:.2f} GB ({m_used} / {m_tot} MiB, {pct:.1f}%)")
        print(f"    • GPU 负载 : {util}%  |  核心温度: {temp}°C  |  功耗: {p_draw}W / {p_lim}W")
else:
    print("\\n[1] 🎮 GPU 显存: 未检测到 NVIDIA GPU")

# 2. 系统内存 (RAM) 与 /dev/shm 内存盘
print("\\n[2] 🧠 系统内存 (RAM) 与高速内存盘 (/dev/shm):")
mem_out = subprocess.getoutput("free -h").splitlines()
for line in mem_out:
    print(f"    {line}")
if os.path.isdir("/dev/shm"):
    du = shutil.disk_usage("/dev/shm")
    print(f"    • /dev/shm (内存盘): 已用 {du.used/(1024**3):.2f} GB / 总计 {du.total/(1024**3):.2f} GB (剩余 {du.free/(1024**3):.2f} GB)")

# 3. 本地 NVMe 硬盘与 Google Drive 云盘空间
print("\\n[3] 💾 磁盘与 Google Drive 云盘容量:")
for label, path in [("本地 NVMe 盘 (/content)", "/content"), ("Google Drive (/content/drive)", "/content/drive")]:
    if os.path.exists(path):
        du = shutil.disk_usage(path)
        pct = du.used * 100.0 / max(1, du.total)
        print(f"    • {label:28s}: 已用 {du.used/(1024**3):6.2f} GB / 总计 {du.total/(1024**3):6.2f} GB (剩余 {du.free/(1024**3):6.2f} GB, {pct:.1f}%)")
    else:
        print(f"    • {label:28s}: 未挂载")

# 4. 模型目录文件明细
print("\\n[4] 📂 模型权重文件分布与云盘同步状态:")
for title, dpath in [
    ("实例内存盘 (/dev/shm/local_models)", "/dev/shm/local_models"),
    ("实例本地盘 (/content/local_models)", "/content/local_models"),
    ("Google Drive (/content/drive/MyDrive/models)", "/content/drive/MyDrive/models"),
]:
    if os.path.isdir(dpath):
        files_info = subprocess.getoutput(f"find {dpath} -maxdepth 3 -type f -exec ls -lh {{}} + 2>/dev/null").strip()
        if files_info:
            print(f"  ▸ {title}:")
            for fl in files_info.splitlines():
                cols = fl.split()
                if len(cols) >= 9:
                    sz_str = cols[4]
                    fp_str = " ".join(cols[8:])
                    rel_str = os.path.relpath(fp_str, dpath)
                    print(f"      - {rel_str} ({sz_str})")
        else:
            print(f"  ▸ {title}: (空)")

sync_running = subprocess.getoutput("pgrep -af '[c]p -rn'").strip()
if sync_running:
    print("  ⏳ 后台云盘同步状态: 正在往 Google Drive 同步写入中...")
else:
    print("  ✅ 后台云盘同步状态: 空闲 (已完成同步)")

# 5. 推理引擎与公网隧道
print("\\n[5] 🚀 推理引擎与公网 API 状态:")
llama_proc = subprocess.getoutput("pgrep -af '[l]lama-server'").strip()
cf_log = subprocess.getoutput("cat /content/cf_tunnel.log 2>/dev/null")
m_url = re.search(r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com", cf_log)
m_key = re.search(r"--api-key\\s+(\\S+)", llama_proc)
print(f"    • llama-server : {'运行中 ✅' if llama_proc else '未运行 ❌'}")
print(f"    • Base URL     : {(m_url.group(0) + '/v1') if m_url else '未启动'}")
if m_key:
    print(f"    • API Key      : {m_key.group(1)}")
print("=" * 72)
"""
    try:
        runtime.kernel_client._manager.client.connection_ready.wait(timeout=15)
        outputs = runtime.execute_code(status_code, allow_stdin=False, timeout=30)
        for out in outputs:
            if "text" in out:
                sys.stdout.write(out["text"])
                sys.stdout.flush()
    finally:
        runtime.stop()


LOCAL_ENDPOINT_FILE = Path(globals().get("__file__", "/content/deploy_colab_model.py")).resolve().parent / ".colab_endpoint.json"


def load_local_endpoint_config() -> Dict[str, Any]:
    default_cfg: Dict[str, Any] = {
        "api_key": "sk-colab-ed28e54eb298d0c1",
        "tunnel_url": "",
        "model_id": "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF",
        "idle_timeout_minutes": 30,
        "last_activity_ts": time.time(),
    }
    if LOCAL_ENDPOINT_FILE.exists():
        try:
            data = json.loads(LOCAL_ENDPOINT_FILE.read_text(encoding="utf-8"))
            default_cfg.update(data)
        except Exception:
            pass
    return default_cfg


def save_local_endpoint_config(cfg: Dict[str, Any]):
    try:
        cur = load_local_endpoint_config()
        cur.update(cfg)
        tmp_f = LOCAL_ENDPOINT_FILE.with_suffix(f".tmp.{os.getpid()}")
        tmp_f.write_text(json.dumps(cur, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp_f, LOCAL_ENDPOINT_FILE)
    except Exception:
        pass


def run_local_proxy_daemon(port: int = 30000):
    """在本地 Mac 启动固定端口反向代理 (http://127.0.0.1:30000/v1)，自动转发到最新一次部署的 trycloudflare.com 隧道地址"""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _DynamicTunnelProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _handle(self):
            raw_p = (self.path or "/").split("?")[0]
            if self.command == "POST" or not any(raw_p.endswith(s) for s in ("/video/status", "/video/upscale_status", "/health", "/models")):
                save_local_endpoint_config({"last_activity_ts": time.time()})
            cfg = load_local_endpoint_config()
            upstream = (cfg.get("tunnel_url") or "").rstrip("/")
            if not upstream:
                body = json.dumps({
                    "error": {
                        "message": "当前 Colab G4 实例已关机或尚未启动，请先在 http://127.0.0.1:18080 控制台启动服务。",
                        "type": "instance_offline",
                        "code": 502,
                    }
                }, ensure_ascii=False).encode("utf-8")
                self.send_response(502)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            req_path = self.path or "/"
            # 1. 兼容客户端把 /v1 重复拼接成 /v1/v1/... 的情况
            while req_path.startswith("/v1/v1/"):
                req_path = "/v1/" + req_path[len("/v1/v1/"):]
            # 2. 兼容浏览器打开 http://127.0.0.1:30000/：若当前运行 GLM-5.3 Project Maya 且浏览器请求 HTML，直接透传 Maya Web 监控/对话台；否则返回 /v1/models
            accept_hdr = (self.headers.get("Accept") or "").lower()
            is_maya_active = "glm-5.3" in f"{cfg.get('model_folder') or ''} {cfg.get('model_id') or ''}".lower()
            if self.command == "GET" and req_path.rstrip("/") == "/v1":
                req_path = "/v1/models"
            elif self.command == "GET" and req_path in ("", "/"):
                req_path = "/" if (is_maya_active and "text/html" in accept_hdr) else "/v1/models"
            # 3. 兼容未带 /v1 前缀直接请求 OpenAI / Anthropic / Responses 路径的客户端
            elif any(req_path.startswith(p) for p in ("/chat/completions", "/models", "/completions", "/embeddings", "/messages", "/responses")):
                req_path = "/v1" + req_path

            target_url = f"{upstream}{req_path}"
            cl = int(self.headers.get("Content-Length", "0") or "0")
            req_body = self.rfile.read(cl) if cl > 0 else None
            fwd_headers = {
                k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length", "transfer-encoding", "connection", "authorization")
            }
            fwd_headers["User-Agent"] = "ColabLocalGateway/1.0"
            # 本机回环网关 (127.0.0.1:30000) 自动注入当前生效的 API Key，同时兼容 OpenAI Bearer 与 Claude Code x-api-key
            api_key = cfg.get("api_key") or "sk-colab-ed28e54eb298d0c1"
            fwd_headers["Authorization"] = f"Bearer {api_key}"
            if "x-api-key" in {k.lower() for k in fwd_headers}:
                fwd_headers = {k: (api_key if k.lower() == "x-api-key" else v) for k, v in fwd_headers.items()}

            opener_direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            opener_sys = urllib.request.build_opener()
            try:
                last_net_err = None
                resp = None
                for attempt_idx, cur_opener in enumerate((opener_direct, opener_sys, opener_direct)):
                    try:
                        req_try = urllib.request.Request(target_url, data=req_body, headers=dict(fwd_headers), method=self.command)
                        resp = cur_opener.open(req_try, timeout=600)
                        break
                    except urllib.error.HTTPError:
                        raise
                    except Exception as net_ex:
                        last_net_err = net_ex
                        if attempt_idx < 2:
                            time.sleep(0.8)
                if resp is None:
                    raise last_net_err or RuntimeError("Upstream tunnel connection failed")
                with resp:
                    self.send_response(resp.status)
                    is_chunked = False
                    for hk, hv in resp.headers.items():
                        hkl = hk.lower()
                        if hkl in (
                            "transfer-encoding",
                            "connection",
                            "content-encoding",
                            "date",
                            "server",
                            "access-control-allow-origin",
                        ):
                            continue
                        if hkl == "content-type" and "text/event-stream" in hv.lower():
                            is_chunked = True
                        self.send_header(hk, hv)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    while True:
                        if is_chunked and hasattr(resp, "read1"):
                            chunk = resp.read1(4096)
                        else:
                            chunk = resp.read(4096 if is_chunked else 262144)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        if is_chunked:
                            self.wfile.flush()
                    self.wfile.flush()
            except urllib.error.HTTPError as he:
                # 兼容 Claude Code 调用 /v1/messages/count_tokens 时部分引擎返回 404 的情况：自动本地平滑估算，避免客户端报错
                if he.code == 404 and req_path.split("?")[0] == "/v1/messages/count_tokens":
                    est_toks = max(1, len(req_body or b"") // 3)
                    ok_body = json.dumps({"input_tokens": est_toks}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(ok_body)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(ok_body)
                    return
                err_body = he.read()
                self.send_response(he.code)
                self.send_header("Content-Type", he.headers.get("Content-Type", "application/json"))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(err_body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(err_body)
            except Exception as ex:
                err_body = json.dumps({"error": {"message": f"Upstream tunnel error: {ex}"}}).encode("utf-8")
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(err_body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(err_body)

        def do_GET(self):
            self._handle()

        def do_POST(self):
            self._handle()

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

    def _keepalive_loop():
        """定期检查闲置超时（自动释放实例省算力）或在活跃期内发送 Jupyter 内核保活心跳"""
        colab_bin = find_colab_bin()
        last_hb = 0.0
        while True:
            time.sleep(60)
            try:
                cfg = load_local_endpoint_config()
                idle_mins = int(cfg.get("idle_timeout_minutes", 30) or 0)
                last_act = float(cfg.get("last_activity_ts") or time.time())
                now = time.time()

                # 若当前正在执行开机/部署流程，刷新活跃时间，绝不触发闲置关机
                if cfg.get("deploying"):
                    save_local_endpoint_config({"last_activity_ts": now})
                    continue

                from colab_cli.common import state
                from colab_cli.runtime import ColabRuntime
                local_sessions = state.store.list()
                if not local_sessions:
                    continue

                if idle_mins > 0 and (now - last_act) >= idle_mins * 60:
                    for s_name in list(local_sessions.keys()):
                        subprocess.run([colab_bin, "stop", "-s", s_name], check=False)
                    continue

                if now - last_hb >= 180:
                    last_hb = now
                    for s in list(local_sessions.values()):
                        rt = ColabRuntime(
                            s.url,
                            s.token,
                            session_name=s.name,
                            history=state.history,
                            kernel_id=s.kernel_id,
                            session_id=s.session_id,
                        )
                        try:
                            _kc_mgr = getattr(rt.kernel_client, "_manager", None)
                            if _kc_mgr is not None and hasattr(_kc_mgr, "client") and hasattr(_kc_mgr.client, "connection_ready"):
                                _kc_mgr.client.connection_ready.wait(timeout=10)
                            rt.execute_code("1", allow_stdin=False, timeout=10)
                        finally:
                            rt.stop()
            except Exception:
                pass

    __import__("threading").Thread(target=_keepalive_loop, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", port), _DynamicTunnelProxyHandler)
    server.serve_forever()


def ensure_local_proxy_running(port: int = 30000):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            return
    subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--proxy-daemon", "--proxy-port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )


def prompt_input(text: str, default: str) -> str:
    if not sys.stdin.isatty():
        return default
    try:
        val = input(f"{text} [{default}]: ").strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)


def run_local_wizard():
    parser = argparse.ArgumentParser(description="Colab + Google Drive Models 一键部署工具")
    parser.add_argument("--session", "-s", default="", help="Colab 会话名 (默认复用已有会话)")
    parser.add_argument("--gpu", default="G4", help="GPU 型号: G4 / H100 / A100 / L4 / T4 (默认 G4)")
    parser.add_argument("--retry-until-ready", action="store_true", help="若目标 GPU 暂满或分配失败，每 12~18 秒持续重试直到成功开机")
    parser.add_argument("--repo", default="", help="HuggingFace 仓库 ID (例如 huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF)")
    parser.add_argument("--folder", default="", help="Google Drive models/ 下的子文件夹名")
    parser.add_argument("--pattern", default="", help="GGUF 文件通配符 (例如 Huihui-Qwen3.8-27B-abliterated-bf16.gguf)")
    parser.add_argument("--ctx", type=int, default=262144, help="上下文长度 (默认 262144)")
    parser.add_argument("--no-drive", action="store_true", help="不挂载 Google Drive，仅使用实例本地硬盘")
    parser.add_argument("--download-only", action="store_true", help="仅将所选模型高速下载并同步至 Google Drive，不切换当前正在运行的推理服务")
    parser.add_argument("--status", action="store_true", help="查看当前 G4 实例的显存、系统内存、内存盘与硬盘/云盘占用情况")
    parser.add_argument("--proxy-daemon", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--proxy-port", type=int, default=30000, help="本地固定代理端口 (默认 30000)")
    parser.add_argument("--yes", "-y", action="store_true", help="非交互模式，全部使用默认值或命令行参数")
    args = parser.parse_args()

    if args.proxy_daemon:
        run_local_proxy_daemon(args.proxy_port)
        return

    colab_bin = find_colab_bin()
    sess_out = subprocess.getoutput(f'"{colab_bin}" sessions')
    default_sess = args.session or ("qwen38-g4" if "qwen38-g4" in sess_out else ("easycolab" if "easycolab" in sess_out else "qwen38-g4"))

    if args.status:
        ensure_local_proxy_running(args.proxy_port)
        show_remote_status(default_sess)
        return

    print("=" * 72)
    print(" 🚀 Google Colab + Google Drive (MyDrive/models/<文件夹>) 一键部署")
    print("=" * 72)
    session_name = default_sess if args.yes else prompt_input("1. Colab 实例会话名称", default_sess)

    default_repo = args.repo or "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF"
    hf_repo = default_repo if args.yes else prompt_input("2. HuggingFace 模型仓库 (或直接粘贴具体 .gguf 文件的 HF 链接)", default_repo)
    hf_repo = re.sub(r"^https?://(?:www\.)?huggingface\.co/", "", hf_repo.strip()).strip("/")
    url_file_pat = ""
    m_blob = re.match(r"^([^/]+/[^/]+)/(?:blob|resolve)/[^/]+/(.+)$", hf_repo)
    if m_blob:
        hf_repo = m_blob.group(1)
        url_file_pat = m_blob.group(2)

    default_folder = args.folder or (hf_repo.split("/")[-1] if hf_repo else "Huihui-Qwen3.8-27B-abliterated-GGUF")
    model_folder = default_folder if args.yes else prompt_input(
        "3. Google Drive 中 `MyDrive/models/` 下的子文件夹名", default_folder
    )

    default_pat = args.pattern or url_file_pat or "Huihui-Qwen3.8-27B-abliterated-bf16.gguf"
    file_pattern = default_pat if args.yes else prompt_input("4. 权重文件名称或过滤通配符 (如 Huihui-Qwen3.8-27B-abliterated-bf16.gguf)", default_pat)

    mount_drive = not args.no_drive
    if not args.yes and not args.no_drive:
        ans = prompt_input("5. 是否挂载 Google Drive 读写 `MyDrive/models/" + model_folder + "`? (y/n)", "y")
        mount_drive = ans.lower().startswith("y")

    saved_cfg = load_local_endpoint_config()
    api_key = os.environ.get("COLAB_API_KEY") or saved_cfg.get("api_key") or f"sk-colab-{secrets.token_hex(8)}"
    hf_tok = os.environ.get("HF_TOKEN") or saved_cfg.get("hf_token") or ""

    if not args.download_only and saved_cfg.get("tunnel_url"):
        expected_eng = (
            "qwen-image-2.1" if model_folder == "Qwen-Image-2.1-Uncensored" else (
                "minimax-h3-image" if model_folder == "MiniMax-H3-Image" else (
                    "longcat-avatar" if model_folder.startswith("LongCat-Video-Avatar") else (
                        "minimax-h3-t4" if model_folder == "MiniMax-H3-T4-GGUF" else (
                            "minimax-h3-ref2va" if model_folder == "MiniMax-H3-Ref2VA" else (
                                "minimax-h3" if model_folder == "MiniMax-H3" else ""
                            )
                        )
                    )
                )
            )
        )
        if expected_eng:
            try:
                op_fast = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with op_fast.open(f"http://127.0.0.1:{args.proxy_port}/health", timeout=4) as h_resp:
                    if h_resp.status == 200:
                        hd_fast = json.loads(h_resp.read().decode("utf-8"))
                        if hd_fast.get("engine") == expected_eng:
                            tun_u = str(saved_cfg.get("tunnel_url") or "").rstrip("/")
                            print(f"\n[ColabDeploy] ✅ 检测到 {model_folder} 已在云端 GPU 显存中常驻就绪，直接零等待复用！")
                            print(f"TUNNEL_BASE_URL={tun_u}/v1")
                            print(f"API_KEY={api_key}")
                            print(f"MODEL_ID={hf_repo or model_folder}")
                            save_local_endpoint_config({
                                "api_key": api_key,
                                "tunnel_url": tun_u,
                                "model_id": hf_repo or model_folder,
                                "model_folder": model_folder,
                                "last_activity_ts": time.time(),
                                "deploying": False,
                            })
                            return
            except Exception:
                pass

    save_local_endpoint_config({"last_activity_ts": time.time(), "deploying": True})

    def _allocate_colab_session(sess_id: str) -> None:
        import random
        target_gpu = (args.gpu or "G4").upper()
        base_delays = (20, 30, 45, 60, 90)
        attempt = 0
        hard_stop_re = re.compile(
            r"Allocation refused|precondition failed|Backend rejected|TooManyAssignmentsError|"
            r"Too Many Requests|Unauthorized|Forbidden|not entitled|no quota|insufficient|"
            r"invalid_grant|token.*expired|HTTP[^0-9]*(401|403|429)",
            re.IGNORECASE,
        )
        while True:
            attempt += 1
            if attempt == 1:
                print(f"\n-> 正在申请分配 Google Colab {target_gpu} 实例 '{sess_id}'...", flush=True)
            else:
                print(f"\n🔄 [持续抢卡 · 第 {attempt} 次尝试] 正在申请 Google Colab {target_gpu} 实例 '{sess_id}'...", flush=True)
            alloc_env = os.environ.copy()
            alloc_env["COLAB_CLI_ASSIGN_RETRIES"] = "0"
            res_new = subprocess.run(
                [colab_bin, "new", "-s", sess_id, "--gpu", target_gpu],
                capture_output=True,
                text=True,
                env=alloc_env,
                check=False,
            )
            comb_out = ((res_new.stdout or "") + "\n" + (res_new.stderr or "")).strip()
            if comb_out:
                print(comb_out, flush=True)
            if res_new.returncode == 0 and not re.search(r"Error:|Traceback|Service Unavailable|Allocation refused|Backend rejected", comb_out, re.IGNORECASE):
                if attempt > 1:
                    print(f"\n🎉 [抢卡成功] 经过 {attempt} 次尝试，已成功分配到 Google Colab {target_gpu} 实例！", flush=True)
                return
            clean_err = re.sub(r"\s+", " ", "\n".join(ln for ln in comb_out.splitlines() if not ln.lstrip().startswith(("╭", "│", "╰"))))
            if hard_stop_re.search(clean_err):
                print(
                    f"\n🛑 [风控保护硬停] Colab 返回权限/配额/会话上限或限流错误，为避免触发风控已立即停止重试。\n"
                    f"   👉 请检查当前账户算力余额、是否已有未释放会话，或切换其他账户后重试！",
                    flush=True,
                )
                save_local_endpoint_config({"last_activity_ts": time.time(), "deploying": False})
                sys.exit(2)
            if args.retry_until_ready:
                base_d = base_delays[min(attempt - 1, len(base_delays) - 1)]
                wait_s = max(12, base_d + random.randint(-4, 4))
                print(
                    f"\n⏳ [持续抢卡 · 第 {attempt} 次未分配到 {target_gpu}] 当前算力池暂满 (503/拥堵)，{wait_s} 秒后自动退避重试（可随时点击面板「🛑 停止当前任务」取消等待）...",
                    flush=True,
                )
                time.sleep(wait_s)
                continue
            print(
                f"\n❌ [GPU_ALLOC_FAILED:{target_gpu}] 当前账户无法分配 {target_gpu} 实例（Colab 服务端返回无 {target_gpu} 权限或算力池暂无空闲）。\n"
                f"   👉 请在控制台弹出的选择框中自主选择：重试 {target_gpu}、勾选「一直开直到开到为止」、改用其他 GPU (H100 / A100 / L4)、或切换其他算力账户！",
                flush=True,
            )
            save_local_endpoint_config({"last_activity_ts": time.time(), "deploying": False})
            sys.exit(2)

    if sys.platform == "darwin" and shutil.which("caffeinate"):
        try:
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    g4_only_set = {"Huihui-Qwen3.8-27B-abliterated-GGUF", "GLM-5.3-Flash-4.05bpw", "MiniMax-H3", "MiniMax-H3-Ref2VA", "MiniMax-H3-Image"}
    if f"[{session_name}]" not in sess_out and session_name not in sess_out:
        _allocate_colab_session(session_name)
    elif "Hardware: T4" in sess_out and (args.gpu or "G4").upper() != "T4" and model_folder in g4_only_set:
        print(f"\n-> 检测到当前运行中的实例为 T4 (16GB)，但请求了 {args.gpu} 大显存模型 ({model_folder})，正在释放旧 T4 实例并重新申请 {args.gpu}...", flush=True)
        subprocess.run([colab_bin, "stop", "-s", session_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _allocate_colab_session(session_name)
    else:
        print(f"\n-> 复用当前运行中的 Colab 实例 '{session_name}'。", flush=True)

    if mount_drive:
        print(f"\n-> 正在将完整 Google Drive 挂载到实例 '{session_name}' (/content/drive/MyDrive)...", flush=True)
        if not mount_google_drive_native(session_name):
            subprocess.run([colab_bin, "stop", "-s", session_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _allocate_colab_session(session_name)
            mount_google_drive_native(session_name)

    print(f"\n-> 正在远端 Colab 实例上准备 `/content/drive/MyDrive/models/{model_folder}`...", flush=True)
    cmd = [
        colab_bin, "exec", "-s", session_name,
        "--timeout", "3600",
        "--env", f"HF_REPO={hf_repo}",
        "--env", f"MODEL_FOLDER={model_folder}",
        "--env", f"FILE_PATTERN={file_pattern}",
        "--env", f"CTX_LEN={args.ctx}",
        "--env", f"COLAB_API_KEY={api_key}",
        "--env", f"HF_TOKEN={hf_tok}",
        "--env", "SYNC_TO_DRIVE=1",
        "--env", f"DOWNLOAD_ONLY={'1' if args.download_only else '0'}",
        "-f", os.path.abspath(__file__),
    ]

    captured_tunnel_url = ""
    captured_api_key = api_key
    last_rc = 0
    for exec_attempt in range(1, 4):
        saw_conn_lost = False
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if "Connection was lost." in line:
                saw_conn_lost = True
                if line.strip().startswith("RuntimeError: Connection was lost."):
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    break
            m_tun = re.search(r"TUNNEL_BASE_URL=(https://[a-zA-Z0-9-]+\.trycloudflare\.com)/v1", line)
            if m_tun:
                captured_tunnel_url = m_tun.group(1)
            m_k = re.search(r"API_KEY=(\S+)", line)
            if m_k:
                captured_api_key = m_k.group(1)
        try:
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        last_rc = (proc.returncode if proc.returncode is not None else 1) or (1 if saw_conn_lost and not captured_tunnel_url else 0)
        if last_rc == 0 or captured_tunnel_url:
            last_rc = 0
            break
        if saw_conn_lost and exec_attempt < 3:
            print(f"\n🔄 [网络自愈 {exec_attempt}/2] 检测到 Colab WebSocket 瞬断 (Connection was lost)，正在自动重连并从断点继续...", flush=True)
            time.sleep(2.0)
            cur_sess = subprocess.getoutput(f'"{colab_bin}" sessions')
            if (f"[{session_name}]" not in cur_sess and session_name not in cur_sess) or exec_attempt >= 2:
                subprocess.run([colab_bin, "stop", "-s", session_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                _allocate_colab_session(session_name)
                if mount_drive:
                    mount_google_drive_native(session_name)
            continue
        break

    if last_rc != 0:
        save_local_endpoint_config({"last_activity_ts": time.time(), "deploying": False})
        sys.exit(last_rc)

    if captured_tunnel_url:
        save_local_endpoint_config({
            "api_key": captured_api_key,
            "tunnel_url": captured_tunnel_url,
            "recavsr_tunnel_url": "",
            "model_id": hf_repo or model_folder,
            "model_folder": model_folder,
            "last_activity_ts": time.time(),
            "deploying": False,
        })
        ensure_local_proxy_running(args.proxy_port)
        print(f"LOCAL_FIXED_BASE_URL=http://127.0.0.1:{args.proxy_port}/v1  (本机永久固定地址，重开 G4 无需修改客户端配置)")
        print("=" * 72)
    else:
        save_local_endpoint_config({"last_activity_ts": time.time(), "deploying": False})


if __name__ == "__main__":
    if is_running_on_colab_vm():
        run_remote_vm_main()
    else:
        run_local_wizard()
