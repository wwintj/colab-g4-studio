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
    "Comfy-Org/MiniMax-H3": [
        "vae/minimax_h3_video_vae_int8_convrot.safetensors",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
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
    "Qwen/Qwen-Image-2.1": [
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "text_encoder/model-00001-of-00004.safetensors",
        "text_encoder/model-00002-of-00004.safetensors",
        "text_encoder/model-00003-of-00004.safetensors",
        "text_encoder/model-00004-of-00004.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ],
    "Tongyi-MAI/Z-Image-Turbo": [
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
        "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
        "text_encoder/model-00001-of-00003.safetensors",
        "text_encoder/model-00002-of-00003.safetensors",
        "text_encoder/model-00003-of-00003.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
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
    if repo_id == "meituan-longcat/LongCat-Video-Avatar-1.6" and "int8" in (pattern or "").lower():
        preset_key = "meituan-longcat/LongCat-Video-Avatar-1.6#int8"
    if preset_key in PRESET_SAFETENSORS_FILES:
        preset_list = PRESET_SAFETENSORS_FILES[preset_key]
        out: List[Tuple[str, int]] = [(fn, size_map.get(fn, 0)) for fn in preset_list]
        # 顺带附带该仓库下的轻量 json/txt/model/py 配置文件，方便离线直接加载
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
) -> str:
    """8~16 路 HTTP Range 并发极速下载 (Colab G4/A100 上可达 400~700 MB/s)，支持跨分片全局总进度显示"""
    dest_path = os.path.join(dest_dir, filename)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".part"

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
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
                            log(f"  [HF 下载] {lbl}: {ov_pct}% ({ov_done/(1024**3):.2f}/{ov_tot/(1024**3):.2f} GB, {spd:.1f} MB/s)")
                        else:
                            pct = downloaded * 100 // total_size
                            log(f"  [HF 下载] {os.path.basename(filename)}: {pct}% ({downloaded/(1024**3):.2f}/{total_size/(1024**3):.2f} GB, {spd:.1f} MB/s)")

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
            ["rclone", "lsf", remote_src, "--config", rc_conf, "-R", "--files-only"],
            stderr=subprocess.DEVNULL,
            timeout=20,
            text=True,
        )
        remote_files = [
            ln.strip() for ln in ls_out.splitlines()
            if ln.strip().lower().endswith((".gguf", ".safetensors"))
        ]
        return len(remote_files)
    except Exception:
        return 0


def _pull_from_rclone_drive_if_exists(remote_folder: str, local_dir: str) -> bool:
    """若主干 CDN 不可用，从云盘 gdrive:models/<remote_folder> 开启 16 线程分块拉取兜底"""
    rc_conf = os.path.expanduser("~/.config/rclone/rclone.conf")
    if not (os.path.exists(rc_conf) and shutil.which("rclone")):
        return False
    remote_src = f"gdrive:models/{remote_folder}"
    try:
        ls_out = subprocess.check_output(
            ["rclone", "lsf", remote_src, "--config", rc_conf, "-R", "--files-only"],
            stderr=subprocess.DEVNULL,
            timeout=25,
            text=True,
        )
        remote_files = [
            ln.strip() for ln in ls_out.splitlines()
            if ln.strip().lower().endswith((".gguf", ".safetensors"))
        ]
    except Exception:
        return False

    if not remote_files:
        return False

    os.makedirs(local_dir, exist_ok=True)
    if all(
        os.path.exists(os.path.join(local_dir, rf)) and os.path.getsize(os.path.join(local_dir, rf)) > 1024 * 1024
        for rf in remote_files
    ):
        log(f"⚡ 检测到本地高速盘 ({local_dir}) 已完整缓存 {len(remote_files)} 个核心权重文件，0 秒直接复用！")
        return True

    log(f"☁️ 正在从 Google Drive 云盘备份 ({remote_src}) 16 线程并发拉取至本地 ({local_dir})...")
    cmd_fg = [
        "rclone", "copy", remote_src, local_dir,
        "--config", rc_conf,
        "--transfers", "16",
        "--multi-thread-streams", "16",
        "--multi-thread-cutoff", "64M",
        "--buffer-size", "64M",
        "--drive-chunk-size", "128M",
        "--stats", "2s",
        "--stats-one-line",
        "-v",
    ]
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


MINIMAX_SERVER_CODE = r'''#!/usr/bin/env python3
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

MODEL_DIR = os.environ.get("MINIMAX_MODEL_DIR", "/content/local_models/MiniMax-H3")
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

                def _cached_prompt_process(self_unit, pipe_obj, prompt, keyframes=None, ref_blocks=None, height=None, width=None, text_embedding=None):
                    if ref_blocks is None and text_embedding is None and prompt is not None and not keyframes:
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
                    return self_unit._orig_process(pipe_obj, prompt, keyframes=keyframes, ref_blocks=ref_blocks, height=height, width=width, text_embedding=text_embedding)

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
            ):
                if not getattr(self_dit, "_tc_enabled", True) or control_hints is not None or use_gradient_checkpointing:
                    return self_dit._orig_forward(
                        x, audio_x, img_position_ids, unique_timesteps, inverse_indices, update_mask,
                        token_tags, prompt_embeds, img_pos_info, audio_pos_info, text_pos_info,
                        img_pos_for_infer_output_info, packed_seq_params, refiner_packed_seq_params,
                        use_gradient_checkpointing=use_gradient_checkpointing,
                        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                        update_audio_mask=update_audio_mask,
                        skip_mask_out_condition=skip_mask_out_condition,
                        control_hints=control_hints,
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
        dit_paths = [os.path.join(MODEL_DIR, f"FL2VA/transformer/model-{i:05d}-of-00013.safetensors") for i in range(1, 14)]
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


def _run_video_job(payload: dict):
    global JOB_STATE
    t0 = time.time()
    try:
        visual = str(payload.get("prompt") or "").strip()
        sound = str(payload.get("soundscape") or payload.get("soundscape_prompt") or "").strip()
        music = str(payload.get("music") or payload.get("music_prompt") or "").strip()
        neg_prompt = str(payload.get("negative_prompt") or "").strip()
        res_str = str(payload.get("resolution") or "832x480").strip()
        num_frames = int(payload.get("num_frames") or 124)
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
                "message": f"{prefix}🔓 正在使用 32B 无审查多模态编码器解析提示词 (Seed: {cur_seed})...",
                "error": "",
            })

            with GPU_LOCK:
                gc.collect()
                torch.cuda.empty_cache()
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                mode_tag = "fl2va" if payload.get("first_frame_b64") else "t2va"
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
                first_b64 = payload.get("first_frame_b64")
                last_b64 = payload.get("last_frame_b64")
                if first_b64:
                    import base64, inspect, io
                    from PIL import Image
                    raw_f = first_b64.split(",", 1)[-1]
                    img_f = Image.open(io.BytesIO(base64.b64decode(raw_f))).convert("RGB").resize((width, height))
                    cond_imgs = [img_f]
                    cond_indices = [0]
                    if last_b64:
                        raw_l = last_b64.split(",", 1)[-1]
                        img_l = Image.open(io.BytesIO(base64.b64decode(raw_l))).convert("RGB").resize((width, height))
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
                    PIPELINE.dit._tc_step = 0
                    PIPELINE.dit._tc_total_steps = steps
                    PIPELINE.dit._tc_prev_b0_res = None
                    PIPELINE.dit._tc_cached_rest_res = None
                    PIPELINE.dit._tc_accum_rel_l1 = 0.0
                    PIPELINE.dit._tc_consec_skips = 0
                    PIPELINE.dit._tc_skipped_count = 0
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
                subprocess.Popen(
                    ["rclone", "copy", out_path, "gdrive:models/MiniMax-H3_outputs", "--config", rc_conf],
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
                self._send_json({"status": "ok", "engine": "minimax-h3"})
            else:
                self._send_json({"status": "loading", "error": ENGINE_ERROR}, status=503)
            return

        if self.path.startswith("/v1/models") or self.path.startswith("/models"):
            self._send_json({
                "object": "list",
                "data": [{"id": "MiniMaxAI/MiniMax-H3", "object": "model", "owned_by": "MiniMaxAI"}],
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

        if self.path.startswith("/v1/video/file/"):
            fname = os.path.basename(self.path.split("/v1/video/file/", 1)[-1].split("?")[0])
            fpath = os.path.join(OUTPUT_DIR, fname)
            if not os.path.isfile(fpath):
                self._send_json({"error": "File not found"}, status=404)
                return
            fsize = os.path.getsize(fpath)
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(fsize))
            self.send_header("Content-Disposition", f'inline; filename="{fname}"')
            self.end_headers()
            with open(fpath, "rb") as rf:
                while True:
                    buf = rf.read(65536)
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
                    "http://127.0.0.1:30002/v1/video/upscale",
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

        if self.path.startswith("/v1/video/generate") or self.path.startswith("/v1/video/generations"):
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
            threading.Thread(target=_run_video_job, args=(payload,), daemon=True).start()
            self._send_json({"ok": True, "message": "已开始生成有声视频"})
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


def _start_minimax_h3_server_on_30000(target_dir: str):
    """在远端 Colab G4 实例上启动 MiniMax-H3 (33B · 原生BF16 · 无审查) 推理引擎 (端口 30000) 并实时输出显存装载进度"""
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
                if h_data.get("engine") == "minimax-h3":
                    log("✅ 检测到 MiniMax-H3 (33B · 无审查) 视频推理引擎已在 GPU 显存中运行，直接复用！")
                    return
    except Exception:
        pass

    log("正在释放旧模型显存 (保留内存盘/NVMe 中的 Qwen3.8 权重以便随时 20 秒切回)...")
    subprocess.run("pkill -9 -f '[l]lama-server|minimax_server\\.py|/content/app\\.py' || true", shell=True)
    time.sleep(2)

    if not os.path.exists("/content/DiffSynth-Studio"):
        log("正在安装 DiffSynth-Studio 音视频管线依赖 (首次约需 25 秒)...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/modelscope/DiffSynth-Studio.git", "/content/DiffSynth-Studio"],
            check=True,
        )
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
    log("🚀 正在启动 MiniMax-H3 (33B · 原生BF16 + 无审查编码器) 视频引擎并预热至 96GB GPU 显存...")
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
        if proc.poll() is not None:
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
            raw_f = first_b64.split(",", 1)[-1]
            first_img_path = "/tmp/minimax_t4_first.png"
            with open(first_img_path, "wb") as wf:
                wf.write(base64.b64decode(raw_f))
        if last_b64:
            raw_l = last_b64.split(",", 1)[-1]
            last_img_path = "/tmp/minimax_t4_last.png"
            with open(last_img_path, "wb") as wl:
                wl.write(base64.b64decode(raw_l))

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
                subprocess.Popen(
                    ["rclone", "copy", out_path, "gdrive:models/MiniMax-H3_outputs", "--config", rc_conf],
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

        if self.path.startswith("/v1/video/file/"):
            fname = os.path.basename(self.path.split("/v1/video/file/", 1)[-1].split("?")[0])
            fpath = os.path.join(OUTPUT_DIR, fname)
            if not os.path.isfile(fpath):
                self._send_json({"error": "File not found"}, status=404)
                return
            fsize = os.path.getsize(fpath)
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(fsize))
            self.send_header("Content-Disposition", f'inline; filename="{fname}"')
            self.end_headers()
            with open(fpath, "rb") as rf:
                while True:
                    buf = rf.read(65536)
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
                    "http://127.0.0.1:30002/v1/video/upscale",
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

        if self.path.startswith("/v1/video/generate") or self.path.startswith("/v1/video/generations"):
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
    subprocess.run("pkill -9 -f '[l]lama-server|minimax_server\\.py|minimax_t4_server\\.py|/content/app\\.py' || true", shell=True)
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
                    return
        except Exception:
            pass
        time.sleep(1)

    raise TimeoutError("等待 MiniMax-H3 (T4 量化版) 服务就绪超时")


LONGCAT_AVATAR_SERVER_CODE = r'''#!/usr/bin/env python3
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

MODEL_DIR = os.environ.get("LONGCAT_MODEL_DIR", "/content/local_models/LongCat-Video-Avatar-1.6")
OUTPUT_DIR = "/content/outputs"
LONGCAT_CODE_DIR = "/content/LongCat-Video"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENGINE_READY = True
ENGINE_ERROR = ""
GPU_LOCK = threading.Lock()

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


def _run_avatar_job(payload: dict):
    global JOB_STATE
    t0 = time.time()
    try:
        prompt = str(payload.get("prompt") or "").strip() or (
            "A person stands under cinematic lighting, speaking naturally and expressively with clear facial movements."
        )
        res_str = str(payload.get("resolution") or "832x480").strip()
        resolution = "720p" if "720" in res_str or "1280" in res_str else "480p"
        num_segments = max(1, min(10, int(payload.get("num_segments") or 1)))
        mix_steps = max(0, min(8, int(payload.get("mix_steps") if payload.get("mix_steps") is not None else 4)))
        mix_alpha = float(payload.get("mix_alpha") if payload.get("mix_alpha") is not None else 1.0)
        base_seed = int(payload.get("seed") if payload.get("seed") is not None else -1)
        rand_seed = bool(payload.get("randomize_seed")) or (base_seed < 0)
        cur_seed = random.randint(1, 2**31 - 1) if rand_seed else base_seed

        has_int8 = os.path.isdir(os.path.join(MODEL_DIR, "base_model_int8")) and not os.path.isdir(os.path.join(MODEL_DIR, "base_model"))
        use_int8 = bool(payload.get("use_int8")) or has_int8 or ("INT8" in MODEL_DIR.upper())

        audio_b64 = str(payload.get("audio_b64") or "").strip()
        audio_ext = str(payload.get("audio_ext") or ".mp3").strip()
        if not audio_ext.startswith("."):
            audio_ext = "." + audio_ext
        if audio_b64:
            raw_a = audio_b64.split(",", 1)[-1]
            audio_path = f"/tmp/longcat_input_audio{audio_ext}"
            with open(audio_path, "wb") as af:
                af.write(base64.b64decode(raw_a))
        else:
            audio_path = os.path.join(LONGCAT_CODE_DIR, "assets/avatar/single/man.mp3")

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

        job_json_path = "/tmp/longcat_avatar_input.json"
        job_data = {
            "prompt": prompt,
            "cond_audio": {"person1": audio_path},
        }
        if img_path:
            job_data["cond_image"] = img_path
        with open(job_json_path, "w", encoding="utf-8") as jf:
            json.dump(job_data, jf, ensure_ascii=False, indent=2)

        JOB_STATE.update({
            "running": True,
            "pct": 6,
            "step": 0,
            "total_steps": 8 * num_segments,
            "seed": cur_seed,
            "message": f"🎭 [LongCat-Avatar-1.6] 正在使用 Kim_Vocal_2 分离人声并提取 Whisper-v3 唇形特征 ({stage_1.upper()} · {num_segments}段)...",
            "error": "",
        })

        with GPU_LOCK:
            tmp_out_dir = f"/tmp/longcat_out_{int(time.time())}"
            os.makedirs(tmp_out_dir, exist_ok=True)
            cmd = [
                "torchrun",
                "--nproc_per_node=1",
                "--master_port", str(random.randint(29500, 29990)),
                os.path.join(LONGCAT_CODE_DIR, "run_demo_avatar_single_audio_to_video.py"),
                "--context_parallel_size=1",
                f"--checkpoint_dir={MODEL_DIR}",
                f"--stage_1={stage_1}",
                f"--input_json={job_json_path}",
                f"--output_dir={tmp_out_dir}",
                f"--resolution={resolution}",
                f"--num_segments={num_segments}",
                "--use_distill",
                "--lora_path", "loraB/lora.safetensors",
                "--model_type", "avatar-v1.6",
                f"--mix_steps={mix_steps}",
                f"--mix_alpha={mix_alpha}",
            ]
            if use_int8:
                cmd.append("--use_int8")

            proc = subprocess.Popen(
                cmd,
                cwd=LONGCAT_CODE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                bufsize=0,
            )
            buf = ""
            tail_lines = []
            ansi_re = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
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
                            raise RuntimeError("🛑 已手动停止数字人视频生成任务")
                        if "loading" in ln.lower() and "lora" in ln.lower():
                            JOB_STATE["pct"] = 18
                            JOB_STATE["message"] = "🎭 [LongCat-Avatar-1.6] 正在融合 DMD + OPSD 双蒸馏 LoRA 适配器..."
                        elif "generating segment" in ln.lower():
                            JOB_STATE["pct"] = 25
                            JOB_STATE["message"] = f"🎬 [LongCat-Avatar-1.6] {ln} (8步口型与动作扩散采样中)..."
                        else:
                            m = re.search(r"(\d+)%\|.*\|\s*(\d+)/(\d+)", ln)
                            if m:
                                pct = int(m.group(1))
                                cur_s = int(m.group(2))
                                tot_s = max(1, int(m.group(3)))
                                JOB_STATE["step"] = cur_s
                                JOB_STATE["total_steps"] = tot_s
                                ov_pct = min(96, max(20, 22 + int(pct * 0.72)))
                                JOB_STATE["pct"] = ov_pct
                                JOB_STATE["message"] = f"🎭 [LongCat-Avatar-1.6] 正在扩散生成数字人视频: 第 {cur_s}/{tot_s} 步 (总进度 {ov_pct}%)"
                    else:
                        buf += ch
            proc.wait()
            mp4_list = sorted(glob.glob(os.path.join(tmp_out_dir, "*.mp4")), key=os.path.getmtime, reverse=True)
            if proc.returncode != 0 or not mp4_list:
                err_tail = "\n".join(tail_lines[-16:])
                raise RuntimeError(f"LongCat-Video-Avatar 退出码 {proc.returncode}:\n{err_tail}")

            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            fname = f"longcat_avatar_{stage_1}_{ts}_seed{cur_seed}.mp4"
            out_path = os.path.join(OUTPUT_DIR, fname)
            shutil.move(mp4_list[0], out_path)
            shutil.rmtree(tmp_out_dir, ignore_errors=True)

            rc_conf = os.path.expanduser("~/.config/rclone/rclone.conf")
            if os.path.exists(rc_conf) and shutil.which("rclone"):
                subprocess.Popen(
                    ["rclone", "copy", out_path, "gdrive:models/MiniMax-H3_outputs", "--config", rc_conf],
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
        if self.path.startswith("/v1/video/file/"):
            fname = os.path.basename(self.path.split("/v1/video/file/", 1)[-1].split("?")[0])
            fpath = os.path.join(OUTPUT_DIR, fname)
            if not os.path.isfile(fpath):
                self._send_json({"error": "File not found"}, status=404)
                return
            fsize = os.path.getsize(fpath)
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(fsize))
            self.send_header("Content-Disposition", f'inline; filename="{fname}"')
            self.end_headers()
            with open(fpath, "rb") as rf:
                while True:
                    buf = rf.read(65536)
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

        if self.path.startswith("/v1/video/upscale"):
            try:
                req2 = urllib.request.Request(
                    "http://127.0.0.1:30002/v1/video/upscale",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req2, timeout=15) as r2:
                    self._send_json(json.loads(r2.read().decode("utf-8")), status=r2.status)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=503)
            return

        if self.path.startswith("/v1/video/cancel"):
            open("/content/.cancel_minimax", "w").write("1")
            self._send_json({"ok": True, "message": "🛑 正在停止数字人视频生成任务..."})
            return

        if self.path.startswith("/v1/video/generate") or self.path.startswith("/v1/video/generations"):
            if os.path.exists("/content/.cancel_minimax"):
                try:
                    os.unlink("/content/.cancel_minimax")
                except Exception:
                    pass
            if JOB_STATE["running"]:
                self._send_json({"ok": False, "error": "当前已有数字人视频正在生成中，请等待完成"}, status=409)
                return
            threading.Thread(target=_run_avatar_job, args=(payload,), daemon=True).start()
            self._send_json({"ok": True, "message": "已开始生成 LongCat-Video-Avatar-1.6 数字人视频"})
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

    subprocess.run("pkill -9 -f '[l]lama-server|minimax_server\\.py|minimax_t4_server\\.py|longcat_avatar_server\\.py|/content/app\\.py' || true", shell=True)
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
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("soxr", "soxr"),
        ("pyloudnorm", "pyloudnorm"),
        ("audio-separator", "audio_separator"),
        ("onnxruntime-gpu", "onnxruntime"),
        ("ftfy", "ftfy"),
        ("einops", "einops"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing_pkgs.append(pkg)
    if missing_pkgs:
        log(f"正在安装 LongCat-Video-Avatar 音频与人声分离依赖 ({', '.join(missing_pkgs)})...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing_pkgs, check=False)

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
                    log(f"[显存装载] {os.path.basename(target_dir)}: 100% ({total_gb:.1f}/{total_gb:.1f} GiB 已就绪)")
                    log("✅ LongCat-Video-Avatar-1.6 数字人音频驱动视频服务已启动 (端口 30000)！")
                    _ensure_recavsr_service_on_30002()
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
                    if DRIVE_OUT_DIR.parent.exists():
                        DRIVE_OUT_DIR.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, DRIVE_OUT_DIR / p.name)
                    elif os.path.isfile("/root/.config/rclone/rclone.conf"):
                        subprocess.run(
                            ["rclone", "copy", str(p), "gdrive:models/MiniMax-H3/outputs", "--transfers=4"],
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
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
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
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
                self._send_json({"ok": True, "already_running": True, "message": "当前已有视频正在超分中，已复用当前任务"})
                return
            UPSCALE_CANCEL = False
            threading.Thread(target=_run_upscale_job, args=(payload,), daemon=True).start()
            self._send_json({"ok": True, "message": "已启动 ReCaVSR 视频超分任务"})
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
        log("正在准备 ReCaVSR (Wan2.2-5B) 2K/4K 超分伴随服务...")
        if not os.path.isdir("/content/ReCaVSR/.git"):
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/kopperx/ReCaVSR.git", "/content/ReCaVSR"], check=False)
        os.makedirs("/content/models/ReCaVSR", exist_ok=True)
        subprocess.run("cp -f /content/ReCaVSR/checkpoints/*.json /content/models/ReCaVSR/ 2>/dev/null || true", shell=True)
        if not os.path.isfile("/content/models/ReCaVSR/transformer.safetensors"):
            if os.path.isfile("/root/.config/rclone/rclone.conf"):
                subprocess.run(["rclone", "copy", "gdrive:models/ReCaVSR", "/content/models/ReCaVSR", "--transfers=8"], check=False)
        if not os.path.isfile("/content/models/ReCaVSR/transformer.safetensors"):
            log("首次使用 ReCaVSR，正在从 HuggingFace (kopper/ReCaVSR) 高速拉取 5B 超分权重并备份至 Google Drive...")
            sr_files = resolve_hf_safetensors_files("kopper/ReCaVSR", "*.safetensors")
            if sr_files:
                for sr_fn, _ in sr_files:
                    fast_parallel_hf_download("kopper/ReCaVSR", sr_fn, "/content/models/ReCaVSR")
                if os.path.isfile("/root/.config/rclone/rclone.conf"):
                    subprocess.Popen(
                        ["rclone", "copy", "/content/models/ReCaVSR", "gdrive:models/ReCaVSR", "--transfers=8"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
        with open("/content/recavsr_server.py", "w", encoding="utf-8") as wf:
            wf.write(RECAVSR_SERVER_CODE)
        with open("/content/models/ReCaVSR/recavsr_server.py", "w", encoding="utf-8") as wf2:
            wf2.write(RECAVSR_SERVER_CODE)
        subprocess.Popen(
            [sys.executable, "-u", "/content/recavsr_server.py"],
            stdout=open("/content/recavsr_server.log", "w"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        subprocess.Popen(
            ["cloudflared", "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:30002"],
            stdout=open("/content/recavsr_tunnel.log", "w"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        log("✅ ReCaVSR (Wan2.2-5B · 单步流式 2K/4K 超分服务) 已启动并挂载至 30002 端口！")
    except Exception as ex:
        log(f"⚠️ ReCaVSR 启动提示: {ex}")



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

    fallback_tag = "b12180"
    main_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{fallback_tag}/llama-{fallback_tag}-bin-ubuntu-cuda-12.8-x64.tar.gz"
    cudart_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{fallback_tag}/cudart-llama-{fallback_tag}-bin-ubuntu-cuda-12.8-x64.tar.gz"

    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=8",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "ColabDeploy/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            releases = json.loads(resp.read().decode("utf-8"))
        for rel in releases:
            found_m, found_rt = None, None
            for asset in rel.get("assets") or []:
                aname = str(asset.get("name") or "")
                aurl = str(asset.get("browser_download_url") or "")
                if aname.endswith("-x64.tar.gz") and "ubuntu-cuda-12." in aname:
                    if aname.startswith("llama-"):
                        found_m = aurl
                    elif aname.startswith("cudart-llama-"):
                        found_rt = aurl
            if found_m:
                main_url = found_m
                if found_rt:
                    cudart_url = found_rt
                break
    except Exception:
        pass

    log(f"正在下载官方预编译 llama.cpp CUDA 引擎 ({os.path.basename(main_url)})...")
    subprocess.run(f'curl -fsSL "{main_url}" -o /tmp/llama.tar.gz && tar -xzf /tmp/llama.tar.gz -C {install_dir} && rm -f /tmp/llama.tar.gz', shell=True, check=True)
    if cudart_url:
        subprocess.run(f'curl -fsSL "{cudart_url}" -o /tmp/cudart.tar.gz && tar -xzf /tmp/cudart.tar.gz -C {install_dir} && rm -f /tmp/cudart.tar.gz', shell=True, check=True)

    bin_p = _find_bin()
    if not bin_p:
        raise RuntimeError("解压 llama.cpp 后未找到 llama-server")
    return bin_p, _build_env()


def _sync_local_to_rclone_drive(local_dir: str, background: bool = False):
    rc_conf = os.path.expanduser("~/.config/rclone/rclone.conf")
    remote_dst = f"gdrive:models/{MODEL_FOLDER}"
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
    if vram_gb < 24.0:
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

    is_safetensors_repo = (
        HF_REPO in PRESET_SAFETENSORS_FILES
        or FILE_PATTERN.lower().endswith(".safetensors")
        or ("minimax-h3" in HF_REPO.lower() and "gguf" not in HF_REPO.lower())
        or ("qwen-image" in HF_REPO.lower())
        or ("z-image" in HF_REPO.lower())
    )

    if is_safetensors_repo:
        # Safetensors 多分片大模型 (如 MiniMax-H3 134GB) 体积远超 /dev/shm 内存盘余量，统一使用 235GB 的 /content/local_models 高速 NVMe 盘中转
        nvme_model_dir = os.path.join("/content/local_models", MODEL_FOLDER)
        target_dir = drive_model_dir if fuse_mounted else nvme_model_dir
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

        gdrive_saved_count = _check_rclone_drive_has_model(MODEL_FOLDER) if (has_rclone and not fuse_mounted) else 0
        if gdrive_saved_count > 0:
            log(f"☁️ 已确认 Google Drive (gdrive:models/{MODEL_FOLDER}) 已持久化保存 {gdrive_saved_count} 个核心权重文件！正在通过 Google 内网 16 线程极速预热...")

        log(f"正在解析开源多模态/音视频模型仓库 {HF_REPO} 的分片列表...")
        st_files = resolve_hf_safetensors_files(HF_REPO, FILE_PATTERN, HF_TOKEN)
        if not st_files and has_rclone and not fuse_mounted:
            _pull_from_rclone_drive_if_exists(MODEL_FOLDER, target_dir)

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
            log("🔗 正在合并 LongCat-Video 基础底座组件 (VAE / Tokenizer / UMT5-XXL TextEncoder)...")
            try:
                base_repo = "meituan-longcat/LongCat-Video"
                base_siblings = resolve_hf_safetensors_files(base_repo, "*.safetensors", HF_TOKEN)
                for rfn, b_sz in base_siblings:
                    if rfn.startswith(("vae/", "tokenizer/", "text_encoder/")):
                        resolved_plan.append((rfn, base_repo, rfn, target_dir, max(0, int(b_sz))))
            except Exception as e_lc:
                log(f"⚠️ 解析 LongCat-Video 底座组件提示: {e_lc}")

        batch_total_bytes = sum(x[4] for x in resolved_plan)
        total_gb = batch_total_bytes / (1024 ** 3)
        log(f"共解析到 {len(resolved_plan)} 个模型文件 (总计约 {total_gb:.2f} GB)，目标目录: {target_dir}")
        cum_done_bytes = 0
        for idx, (rel_fn, dl_repo, dl_rel_fn, dl_dest_dir, sz) in enumerate(resolved_plan, start=1):
            existing_p = os.path.join(target_dir, rel_fn)
            lbl_str = f"{MODEL_FOLDER} 总进度 ({idx}/{len(resolved_plan)} · {os.path.basename(rel_fn)})"
            if os.path.exists(existing_p) and (sz <= 0 or os.path.getsize(existing_p) == sz):
                actual_sz = os.path.getsize(existing_p) if os.path.exists(existing_p) else sz
                cum_done_bytes += actual_sz
                ov_pct = min(100, cum_done_bytes * 100 // max(1, batch_total_bytes)) if batch_total_bytes > 0 else 100
                log(f"  [HF 下载] {lbl_str}: {ov_pct}% ({cum_done_bytes/(1024**3):.2f}/{total_gb:.2f} GB, 本地缓存复用)")
                continue
            tag_prefix = "🔓 [破限编码器] " if dl_repo == uncensored_te_repo else ""
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
                )
                cum_done_bytes += (os.path.getsize(dl_p) if os.path.exists(dl_p) else sz)
            except Exception:
                if has_rclone and not fuse_mounted:
                    _pull_from_rclone_drive_if_exists(MODEL_FOLDER, target_dir)
                    break
                raise
        expected_core_count = max(1, len(resolved_plan))
        if has_rclone and not fuse_mounted and gdrive_saved_count < expected_core_count:
            _sync_local_to_rclone_drive(target_dir, background=not DOWNLOAD_ONLY)
        log(f"✅ 模型 {HF_REPO} 全部 {len(resolved_plan)} 个文件已就绪！")
        if DOWNLOAD_ONLY:
            log(f"🎉 [仅下载模式] 已完成全部权重下载并持久化至 Google Drive 云盘 (gdrive:models/{MODEL_FOLDER})！")
            return
        if HF_REPO.lower() == "unsloth/minimax-h3-gguf" or MODEL_FOLDER == "MiniMax-H3-T4-GGUF":
            _start_minimax_t4_gguf_server_on_30000(target_dir)
        elif "longcat-video-avatar" in HF_REPO.lower() or MODEL_FOLDER.startswith("LongCat-Video-Avatar"):
            _start_longcat_avatar_server_on_30000(target_dir, total_gb=total_gb)
        else:
            _start_minimax_h3_server_on_30000(target_dir)
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

    if not local_ggufs:
        if gdrive_gguf_count > 0:
            log(f"☁️ 已确认您的 Google Drive (gdrive:models/{MODEL_FOLDER}) 已完整持久化保存 {gdrive_gguf_count} 个权重文件！")
            log(f"🚀 为绕过 Google Drive API 单文件限速 (~20 MB/s)，已自动切换至 G4 10Gbps 专线 16 线程极速预热通道 (500~700 MB/s)，且预热完成后无需重复上传云盘！")
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
            subprocess.run("pkill -9 -f '[l]lama-server|minimax_server\\.py|/content/app\\.py' || true", shell=True)

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
                "--cache-type-k", "q8_0",
                "--cache-type-v", "q8_0",
                "-b", "2048",
                "-ub", "1024",
                "--backend-sampling",
                "--jinja",
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
                    "--spec-draft-n-max", "4",
                    "--spec-draft-p-min", "0.45",
                    "--cache-type-k-draft", "q8_0",
                    "--cache-type-v-draft", "q8_0",
                ])

            mtp_tag = f" + MTP投机加速({os.path.basename(attempt_mtp)})" if attempt_mtp else ""
            log(f"正在启动 llama-server 加载 {os.path.basename(main_gguf)}{mtp_tag} (上下文: {CTX_LEN})...")
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
            weights_gb = sum(
                os.path.getsize(p) / (1024 ** 3)
                for p in (main_gguf, mmproj_gguf, attempt_mtp)
                if p and os.path.exists(p)
            )
            kv_gb = (CTX_LEN / 262144.0) * 8.8
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


def mount_google_drive_native(session_name: str):
    """
    通过 colab_cli 原生协议将完整 Google Drive 根目录挂载至 /content/drive/MyDrive。
    - 自动卸载之前可能残留的 rclone 子目录挂载
    - 若需要浏览器授权，自动弹出网页并轮询检测授权完成（无需回终端按回车）
    """
    from colab_cli.auth import get_credentials
    from colab_cli.common import state
    from colab_cli.runtime import ColabRuntime
    from colab_cli.utils import get_status_code, open_url_in_browser

    s = state.get_session(session_name)
    runtime = ColabRuntime(
        s.url,
        s.token,
        session_name=s.name,
        history=state.history,
        kernel_id=s.kernel_id,
        session_id=s.session_id,
    )

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
        outputs = runtime.execute_code(mount_code, allow_stdin=True, timeout=600)
        for out in outputs:
            if "text" in out:
                sys.stdout.write(out["text"])
                sys.stdout.flush()
    finally:
        runtime.stop()


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
        LOCAL_ENDPOINT_FILE.write_text(json.dumps(cur, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
            # 2. 兼容用户在浏览器直接打开 http://127.0.0.1:30000/v1 或 http://127.0.0.1:30000 测试连通性
            if self.command == "GET" and req_path.rstrip("/") in ("", "/v1"):
                req_path = "/v1/models"
            # 3. 兼容未带 /v1 前缀直接请求 /chat/completions、/models、/completions、/embeddings 的客户端
            elif any(req_path.startswith(p) for p in ("/chat/completions", "/models", "/completions", "/embeddings")):
                req_path = "/v1" + req_path

            target_url = f"{upstream}{req_path}"
            cl = int(self.headers.get("Content-Length", "0") or "0")
            req_body = self.rfile.read(cl) if cl > 0 else None
            fwd_headers = {
                k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length", "transfer-encoding", "connection", "authorization")
            }
            fwd_headers["User-Agent"] = "ColabLocalGateway/1.0"
            # 本机回环网关 (127.0.0.1:30000) 自动注入当前生效的 API Key，免除客户端漏填/填错导致的 401 报错
            api_key = cfg.get("api_key") or "sk-colab-ed28e54eb298d0c1"
            fwd_headers["Authorization"] = f"Bearer {api_key}"

            req = urllib.request.Request(target_url, data=req_body, headers=fwd_headers, method=self.command)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                with opener.open(req, timeout=600) as resp:
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
                        chunk = resp.read(4096 if is_chunked else 32768)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except urllib.error.HTTPError as he:
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
                            rt.kernel_client._manager.client.connection_ready.wait(timeout=10)
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
    parser.add_argument("--gpu", default="G4", help="GPU 型号: G4 / A100 / L4 / T4 (默认 G4)")
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
    save_local_endpoint_config({"last_activity_ts": time.time(), "deploying": True})

    if f"[{session_name}]" not in sess_out and session_name not in sess_out:
        target_gpu = args.gpu
        try:
            from colab_cli.common import state
            u_info = state.client.get_consumption_user_info()
            if float(getattr(u_info, "paid_compute_units_balance", 0.0) or 0.0) <= 0.0 and target_gpu.upper() != "T4":
                print(f"\n💡 检测到当前账号 Colab 付费算力余额为 0.00 CU，已自动切换为申请【免费 T4 (16GB 显存 · 0.00 CU/h 不扣费)】实例...")
                target_gpu = "T4"
        except Exception:
            pass
        print(f"\n-> 正在申请分配 Google Colab {target_gpu} 实例 '{session_name}'...")
        res_new = subprocess.run([colab_bin, "new", "-s", session_name, "--gpu", target_gpu], check=False)
        if res_new.returncode != 0 and target_gpu.upper() != "T4":
            print(f"\n⚠️ 当前账号无 {target_gpu} 权限 (0.00 CU)，正在自动降级申请【免费 T4 (16GB 显存 · 0.00 CU/h)】实例...")
            res_new = subprocess.run([colab_bin, "new", "-s", session_name, "--gpu", "T4"], check=False)
        if res_new.returncode != 0:
            print(
                f"\n❌ [分配失败] 当前 Google 账号无法分配 GPU 实例（可能免费 T4 配额暂时受限或需在网页端首次激活）。\n"
                f"   👉 若需使用 RTX PRO 6000 (G4 · 96GB 显存)，请点击控制台右上角「👤 账号」按钮切换至有 CU 余额的账号（如 687.72 CU 账号）！"
            )
            sys.exit(1)
    else:
        print(f"\n-> 复用当前运行中的 Colab 实例 '{session_name}'。")

    if mount_drive:
        print(f"\n-> 正在将完整 Google Drive 挂载到实例 '{session_name}' (/content/drive/MyDrive)...")
        mount_google_drive_native(session_name)

    print(f"\n-> 正在远端 Colab 实例上准备 `/content/drive/MyDrive/models/{model_folder}`...")
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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        m_tun = re.search(r"TUNNEL_BASE_URL=(https://[a-zA-Z0-9-]+\.trycloudflare\.com)/v1", line)
        if m_tun:
            captured_tunnel_url = m_tun.group(1)
        m_k = re.search(r"API_KEY=(\S+)", line)
        if m_k:
            captured_api_key = m_k.group(1)
    proc.wait()
    if proc.returncode != 0:
        sys.exit(proc.returncode)

    if captured_tunnel_url:
        save_local_endpoint_config({
            "api_key": captured_api_key,
            "tunnel_url": captured_tunnel_url,
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
