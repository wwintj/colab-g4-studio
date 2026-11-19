#!/usr/bin/env python3
"""
Google Colab G4 本地可视化网页控制台 (http://127.0.0.1:18080)
提供免记命令的图形化界面：
  - 一键启动 / 切换模型 / 挂载 Google Drive (MyDrive/models/<文件夹>)
  - 一键关停 G4 实例省算力 (colab stop)
  - 显存 (VRAM)、系统内存 (RAM)、内存盘 (/dev/shm)、NVMe 硬盘、Google Drive 实时进度条监控
  - 一键复制永久固定 API 地址 (http://127.0.0.1:30000/v1) 与在线快速对话测试
"""

import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
DEPLOY_SCRIPT = BASE_DIR / "deploy_colab_model.py"
COLAB_BIN = BASE_DIR / ".venv" / "bin" / "colab"
PYTHON_BIN = BASE_DIR / ".venv" / "bin" / "python"
ENDPOINT_FILE = BASE_DIR / ".colab_endpoint.json"
CUSTOM_MODELS_FILE = BASE_DIR / ".custom_hf_models.json"
ACCOUNTS_FILE = BASE_DIR / ".colab_accounts.json"
LOCAL_MOVIES_DIR = Path.home() / "Movies" / "EasyColab"
ELAPSED_META_FILE = LOCAL_MOVIES_DIR / ".elapsed_meta.json"
PANEL_PORT = int(os.environ.get("COLAB_PANEL_PORT", "18080"))
_HF_TRENDING_CACHE: Dict[str, Any] = {}

try:
    import colab_cli.auth as _c_auth
    _c_auth.PUBLIC_SCOPES[:] = [
        "openid",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/colaboratory",
    ]
    def _non_blocking_remote_flow(_cfg):
        raise RuntimeError("OAuth 凭证已过期或需要重新绑定")
    _c_auth._run_remote_flow = _non_blocking_remote_flow
except Exception:
    pass


def _format_short_elapsed(sec: float) -> str:
    s = max(1, int(round(sec)))
    if s >= 60:
        return f"{s // 60}m{s % 60}s"
    return f"{s}s"


def _format_long_elapsed(sec: float) -> str:
    if sec >= 60:
        m = int(sec // 60)
        r = round(sec - m * 60, 1)
        return f"生成耗时 {m}分{r:.1f}秒"
    return f"生成耗时 {sec:.1f}秒"


def _infer_elapsed_sec(fname: str, mtime: float, meta_map: Dict[str, float]) -> float:
    if fname in meta_map and meta_map[fname] > 0:
        return float(meta_map[fname])
    if "_ReCaVSR_" in fname:
        return 0.0
    # 从文件名 minimax_h3_t2va_YYYYMMDD_HHMMSS_seed... 解析 UTC 启动时间戳并与文件落盘 mtime 对比
    m = re.search(r"_(\d{8})_(\d{6})_", fname)
    if m:
        try:
            import calendar
            st = time.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            start_utc = float(calendar.timegm(st))
            diff = mtime - start_utc
            if 5.0 <= diff <= 7200.0:
                return round(diff, 1)
        except Exception:
            pass
    return 0.0


def _abbreviate_filename(fname: str) -> str:
    m_sr = re.search(r"_ReCaVSR_([^_]+)_", fname)
    sr_tag = f"[{m_sr.group(1)}] " if m_sr else ""
    if len(fname) <= 22:
        return sr_tag + fname
    base = re.sub(r"_ReCaVSR_.*$", ".mp4", fname)
    return f"{sr_tag}{base[:8]}...{base[-8:]}"


def _list_local_easycolab_videos() -> List[Dict[str, Any]]:
    LOCAL_MOVIES_DIR.mkdir(parents=True, exist_ok=True)
    meta_map: Dict[str, float] = {}
    # 兼容读取原版 EasyColab macOS 偏好设置中的 mediaElapsedSecondsMap
    for plist_path in (
        Path.home() / "Library/Preferences/app.easycolab.studio.macos.plist",
        Path.home() / "Library/Containers/app.easycolab.studio.macos/Data/Library/Preferences/app.easycolab.studio.macos.plist",
    ):
        if plist_path.exists():
            try:
                import plistlib
                pl = plistlib.loads(plist_path.read_bytes())
                raw_m = pl.get("mediaElapsedSecondsMap")
                if isinstance(raw_m, dict):
                    for k, v in raw_m.items():
                        try:
                            meta_map[str(k)] = float(v)
                        except Exception:
                            pass
            except Exception:
                pass
    if ELAPSED_META_FILE.exists():
        try:
            custom_m = json.loads(ELAPSED_META_FILE.read_text(encoding="utf-8"))
            if isinstance(custom_m, dict):
                for k, v in custom_m.items():
                    try:
                        meta_map[str(k)] = float(v)
                    except Exception:
                        pass
        except Exception:
            pass

    items: List[Dict[str, Any]] = []
    for p in LOCAL_MOVIES_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in (".mp4", ".png", ".jpg", ".jpeg", ".webp") and not p.name.startswith("."):
            try:
                st = p.stat()
                ctime = float(getattr(st, "st_birthtime", st.st_mtime) or st.st_mtime)
                mtime = float(st.st_mtime)
                sz_mb = round(st.st_size / (1024 * 1024), 1)
            except Exception:
                continue
            lt = time.localtime(ctime)
            date_str = f"{lt.tm_year}年{lt.tm_mon}月{lt.tm_mday}日 {lt.tm_hour}:{lt.tm_min:02d}"
            el_sec = _infer_elapsed_sec(p.name, ctime, meta_map)
            items.append({
                "name": p.name,
                "short_name": _abbreviate_filename(p.name),
                "path": str(p),
                "mtime": ctime,
                "date_str": date_str,
                "size_mb": sz_mb,
                "elapsed_sec": el_sec,
                "elapsed_short": _format_short_elapsed(el_sec) if el_sec > 0 else f"{sz_mb}MB",
                "elapsed_long": _format_long_elapsed(el_sec) if el_sec > 0 else f"文件大小 {sz_mb} MB",
            })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items


_VIDEO_DL_LOCK = threading.Lock()
_VIDEO_DL_STATE: Dict[str, Any] = {
    "active": False,
    "fname": "",
    "done_mb": 0.0,
    "tot_mb": 0.0,
    "pct": 0,
}
_PENDING_VIDEO_JOB: Dict[str, Any] = {
    "active": False,
    "payload": None,
    "error": "",
    "started_at": 0.0,
}
_LAST_VIDEO_LOG_MSG: List[str] = [""]
_LOGGED_COMPLETED_VIDEOS: set = set()
_LAST_LOGGED_UPSCALE_ERR: List[str] = [""]
_VIDEO_STEP_TIMING: Dict[str, float] = {
    "first_step_ts": 0.0,
    "first_step_num": 0.0,
}
_UPSCALE_TIMING: Dict[str, Any] = {
    "start_ts": 0.0,
    "compute_start_ts": 0.0,
    "preset": "2k",
}


PANEL_LOGS_FILE = BASE_DIR / ".colab_panel_logs.json"


def _compact_progress_log_lines(raw_lines: List[str]) -> List[str]:
    out: List[str] = []
    for ln in raw_lines:
        s = str(ln or "").rstrip("\n")
        if not s:
            continue
        if out:
            prev = out[-1]
            if "正在扩散采样生成有声视频:" in s and "正在扩散采样生成有声视频:" in prev:
                out[-1] = s
                continue
            if "采样完成，正在合成音视频画面:" in s and "采样完成，正在合成音视频画面:" in prev:
                out[-1] = s
                continue
            if "[视频超分]" in s and "[视频超分]" in prev:
                out[-1] = s
                continue
            if "[显存装载]" in s and "[显存装载]" in prev:
                out[-1] = s
                continue
            m_dl = re.search(r"\[(视频回传|图片回传)\]\s+([^:]+):", s)
            if m_dl and f"[{m_dl.group(1)}] {m_dl.group(2)}:" in prev:
                out[-1] = s
                continue
        out.append(s)
    return out


def _save_panel_logs_snapshot():
    try:
        snap = {
            "running": False,
            "logs": _compact_progress_log_lines(list(_DEPLOY_STATE.get("logs") or []))[-400:],
            "exit_code": _DEPLOY_STATE.get("exit_code"),
        }
        PANEL_LOGS_FILE.write_text(json.dumps(snap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def _append_panel_log(msg_text: str):
    clean = str(msg_text or "").strip()
    if not clean or clean == _LAST_VIDEO_LOG_MSG[0]:
        return
    _LAST_VIDEO_LOG_MSG[0] = clean
    now_ts = time.time()
    if "[提交视频生成任务]" in clean or "[提交 4K 写真/换装任务]" in clean or "正在使用 32B 无审查多模态编码器解析提示词" in clean or "正在使用 Heretic 32B 编码指令" in clean:
        _VIDEO_STEP_TIMING["first_step_ts"] = 0.0
        _VIDEO_STEP_TIMING["first_step_num"] = 0.0
    elif "正在扩散采样生成有声视频:" in clean or "正在 6 步极速扩散采样:" in clean:
        m_st = re.search(r"第\s*(\d+)/(\d+)\s*步", clean)
        if m_st and _VIDEO_STEP_TIMING["first_step_ts"] <= 0.0:
            _VIDEO_STEP_TIMING["first_step_ts"] = now_ts
            _VIDEO_STEP_TIMING["first_step_num"] = float(int(m_st.group(1)))

    ts = time.strftime("%H:%M:%S", time.localtime(now_ts))
    line = f"[{ts}] {clean}"
    with _DEPLOY_LOCK:
        logs = _DEPLOY_STATE["logs"]
        replaced = False
        if logs:
            m_dl = re.match(r"^\[(视频回传|图片回传)\]\s+([^:]+):", clean)
            for idx in range(len(logs) - 1, max(-1, len(logs) - 7), -1):
                cand_ln = logs[idx]
                if m_dl and f"[{m_dl.group(1)}] {m_dl.group(2)}:" in cand_ln:
                    logs[idx] = line
                    replaced = True
                    break
                if "正在扩散采样生成有声视频:" in clean and "正在扩散采样生成有声视频:" in cand_ln:
                    logs[idx] = line
                    replaced = True
                    break
                if "采样完成，正在合成音视频画面:" in clean and "采样完成，正在合成音视频画面:" in cand_ln:
                    logs[idx] = line
                    replaced = True
                    break
                if "[视频超分]" in clean and "[视频超分]" in cand_ln:
                    logs[idx] = line
                    replaced = True
                    break
                if "[显存装载]" in clean and "[显存装载]" in cand_ln:
                    logs[idx] = line
                    replaced = True
                    break
        if not replaced:
            logs.append(line)
            if len(logs) > 400:
                _DEPLOY_STATE["logs"] = logs[-400:]
        _save_panel_logs_snapshot()


def _parse_log_hms_to_epoch(hms_str: str, now_ts: float) -> float:
    try:
        parts = [int(x) for x in hms_str.split(":")]
        if len(parts) != 3:
            return now_ts
        lt = time.localtime(now_ts)
        cand = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, parts[0], parts[1], parts[2], lt.tm_wday, lt.tm_yday, lt.tm_isdst))
        if cand > now_ts + 120:
            cand -= 86400.0
        return cand
    except Exception:
        return now_ts


def _compute_video_timing_from_logs(pct_val: int) -> Dict[str, int]:
    now_ts = time.time()
    with _DEPLOY_LOCK:
        lines = list(_DEPLOY_STATE.get("logs") or [])
    start_idx = -1
    start_ts = 0.0
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i]
        if "[提交视频生成任务]" in ln or "[提交 4K 写真/换装任务]" in ln:
            m_t = re.match(r"^\[(\d{2}:\d{2}:\d{2})\]", ln)
            start_ts = _parse_log_hms_to_epoch(m_t.group(1), now_ts) if m_t else now_ts
            start_idx = i
            break
    if start_idx < 0:
        for i in range(len(lines) - 1, -1, -1):
            ln = lines[i]
            if "[视频生成]" in ln or "[图片生成]" in ln:
                m_t = re.match(r"^\[(\d{2}:\d{2}:\d{2})\]", ln)
                start_ts = _parse_log_hms_to_epoch(m_t.group(1), now_ts) if m_t else now_ts
                start_idx = i
            else:
                if start_idx >= 0:
                    break
    if start_ts <= 0:
        start_ts = now_ts

    elapsed_sec = max(1, int(round(now_ts - start_ts)))
    first_step_ts = float(_VIDEO_STEP_TIMING.get("first_step_ts") or 0.0)
    first_step_num = int(_VIDEO_STEP_TIMING.get("first_step_num") or 0)
    prompt_enc_ts = 0.0
    latest_step_ts = 0.0
    latest_step_num = 0
    total_steps = 0
    scan_from = max(0, start_idx)
    for ln in lines[scan_from:]:
        if "正在使用 32B 无审查多模态编码器解析提示词" in ln or "正在使用 Heretic 32B 编码指令" in ln:
            m_p = re.match(r"^\[(\d{2}:\d{2}:\d{2})\]", ln)
            if m_p:
                prompt_enc_ts = _parse_log_hms_to_epoch(m_p.group(1), now_ts)
        elif "正在扩散采样生成有声视频" in ln or "正在扩散生成数字人视频" in ln or "正在 6 步极速扩散采样" in ln:
            m_tot = re.search(r"^\[(\d{2}:\d{2}:\d{2})\].*总计\s*(\d+)/(\d+)\s*步", ln)
            m_seg_s = re.search(r"^\[(\d{2}:\d{2}:\d{2})\].*第\s*(\d+)/(\d+)\s*段.*第\s*(\d+)/(\d+)\s*步", ln)
            m_s = re.search(r"^\[(\d{2}:\d{2}:\d{2})\].*第\s*(\d+)/(\d+)\s*步", ln)
            if m_tot:
                ts_val = _parse_log_hms_to_epoch(m_tot.group(1), now_ts)
                c_s = int(m_tot.group(2))
                t_s = int(m_tot.group(3))
            elif m_seg_s:
                ts_val = _parse_log_hms_to_epoch(m_seg_s.group(1), now_ts)
                seg_i, seg_n = int(m_seg_s.group(2)), max(1, int(m_seg_s.group(3)))
                stp_i, stp_n = int(m_seg_s.group(4)), max(1, int(m_seg_s.group(5)))
                c_s = (seg_i - 1) * stp_n + stp_i
                t_s = seg_n * stp_n
            elif m_s:
                ts_val = _parse_log_hms_to_epoch(m_s.group(1), now_ts)
                c_s = int(m_s.group(2))
                t_s = int(m_s.group(3))
            else:
                ts_val = 0.0
                c_s = 0
                t_s = 0
            if ts_val > 0 and t_s > 0:
                if first_step_ts <= 0.0:
                    first_step_ts = ts_val
                    first_step_num = c_s
                latest_step_ts = ts_val
                latest_step_num = c_s
                total_steps = t_s

    if first_step_ts >= latest_step_ts and prompt_enc_ts > 0.0 and latest_step_ts > prompt_enc_ts + 2.0:
        first_step_ts = prompt_enc_ts + 2.5
        first_step_num = 0

    eta_sec = 0
    if total_steps > 0 and latest_step_num > first_step_num and latest_step_ts > first_step_ts:
        sec_per_step = (latest_step_ts - first_step_ts) / max(1, latest_step_num - first_step_num)
        in_step_elapsed = max(0.0, min(sec_per_step * 0.95, now_ts - latest_step_ts))
        rem_steps = max(0, total_steps - latest_step_num)
        rem_diff_sec = max(0.0, rem_steps * sec_per_step - in_step_elapsed)
        eta_sec = max(5, int(round(rem_diff_sec + 45.0)))
    elif pct_val >= 88:
        eta_sec = max(3, int(round((100 - min(99, pct_val)) * 4.5)))
    elif pct_val > 5 and elapsed_sec > 10:
        eta_sec = max(10, int(round(elapsed_sec * (100 - pct_val) / max(1, pct_val))))

    return {"elapsed_sec": elapsed_sec, "eta_sec": eta_sec}


def _download_remote_video_to_local(fname: str, elapsed_s: float = 0.0) -> bool:
    if not fname:
        return False
    safe_name = os.path.basename(fname)
    LOCAL_MOVIES_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = LOCAL_MOVIES_DIR / safe_name
    if dest_path.exists() and dest_path.stat().st_size > 1024:
        return True
    if not _VIDEO_DL_LOCK.acquire(blocking=False):
        return False
    try:
        if dest_path.exists() and dest_path.stat().st_size > 1024:
            return True
        _VIDEO_DL_STATE.update({
            "active": True,
            "fname": safe_name,
            "done_mb": 0.0,
            "tot_mb": 0.0,
            "pct": 0,
            "start_ts": time.time(),
        })
        cfg = _load_endpoint_cfg()
        upstream = (cfg.get("tunnel_url") or "").rstrip("/")
        fallback_url = (
            f"{upstream}/v1/video/file/{urllib.parse.quote(safe_name)}"
            if upstream
            else f"http://127.0.0.1:30000/v1/video/file/{urllib.parse.quote(safe_name)}"
        )
        url = fallback_url
        auth_hdr = {
            "Authorization": f"Bearer {cfg.get('api_key', 'sk-colab-ed28e54eb298d0c1')}",
            "User-Agent": "ColabLocalGateway/1.0",
        }
        # 优先使用 Google Colab 原生直连网关 (prod.r.colab.dev/files/content/outputs/...)，绕开 trycloudflare 免费隧道跨境限速 (实测 2.6+ MB/s，4.9MB 仅需 1.8 秒)
        try:
            from colab_cli.common import state
            sess_map = state.store.list()
            if sess_map:
                sname = list(sess_map.keys())[0]
                s_obj = state.get_session(sname)
                if s_obj and getattr(s_obj, "url", None) and getattr(s_obj, "token", None):
                    url = f"{s_obj.url.rstrip('/')}/files/content/outputs/{urllib.parse.quote(safe_name)}?colab-runtime-proxy-token={urllib.parse.quote(s_obj.token)}"
                    auth_hdr = {
                        "X-Colab-Runtime-Proxy-Token": s_obj.token,
                        "User-Agent": "ColabLocalGateway/1.0",
                    }
        except Exception:
            pass

        tmp_path = LOCAL_MOVIES_DIR / (safe_name + ".part")
        is_img_file = safe_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        dl_tag = "图片回传" if is_img_file else "视频回传"
        done_tag = "图片已落盘" if is_img_file else "视频已落盘"
        try:
            # 1. 探测远端是否支持 HTTP 206 Range 并发分片拉取 (优先走 Google 原生 prod.r.colab.dev 专线，失败自动回退隧道)
            tot_bytes = 0
            supports_range = False
            try:
                probe_req = urllib.request.Request(url, headers={**auth_hdr, "Range": "bytes=0-0"}, method="GET")
                probe_resp = urllib.request.urlopen(probe_req, timeout=12)
            except Exception:
                url = fallback_url
                auth_hdr = {
                    "Authorization": f"Bearer {cfg.get('api_key', 'sk-colab-ed28e54eb298d0c1')}",
                    "User-Agent": "ColabLocalGateway/1.0",
                }
                probe_req = urllib.request.Request(url, headers={**auth_hdr, "Range": "bytes=0-0"}, method="GET")
                probe_resp = urllib.request.urlopen(probe_req, timeout=20)
            try:
                if probe_resp.status == 206:
                    cr = probe_resp.headers.get("Content-Range", "")
                    if "/" in cr:
                        tot_bytes = int(cr.split("/")[-1].strip() or "0")
                    if tot_bytes > 0:
                        supports_range = True
                        probe_resp.close()
                else:
                    tot_bytes = int(probe_resp.headers.get("Content-Length", "0") or "0")
            except Exception:
                pass

            tot_mb = tot_bytes / (1024 * 1024) if tot_bytes > 0 else 12.0
            _VIDEO_DL_STATE["tot_mb"] = tot_mb

            if supports_range and tot_bytes > 262144:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with open(tmp_path, "wb") as wf:
                    wf.truncate(tot_bytes)
                chunk_sz = 524288  # 512 KB per slice
                ranges = []
                off = 0
                while off < tot_bytes:
                    end_b = min(tot_bytes - 1, off + chunk_sz - 1)
                    ranges.append((off, end_b))
                    off = end_b + 1

                prog_lock = threading.Lock()
                state_box = {"done": 0, "last_log": 0.0}
                fd = os.open(str(tmp_path), os.O_RDWR)
                try:
                    def _fetch_slice(rng):
                        s_b, e_b = rng
                        for attempt in range(3):
                            try:
                                r_req = urllib.request.Request(
                                    url,
                                    headers={**auth_hdr, "Range": f"bytes={s_b}-{e_b}"},
                                    method="GET",
                                )
                                with urllib.request.urlopen(r_req, timeout=45) as r_s:
                                    pos = s_b
                                    while True:
                                        buf = r_s.read(131072)
                                        if not buf:
                                            break
                                        os.pwrite(fd, buf, pos)
                                        pos += len(buf)
                                        with prog_lock:
                                            state_box["done"] += len(buf)
                                            d_mb = state_box["done"] / (1024 * 1024)
                                            pct = min(99, int(state_box["done"] * 100 // max(1, tot_bytes)))
                                            _VIDEO_DL_STATE["done_mb"] = d_mb
                                            _VIDEO_DL_STATE["pct"] = pct
                                            now = time.time()
                                            if now - state_box["last_log"] >= 1.2:
                                                state_box["last_log"] = now
                                                _append_panel_log(f"[{dl_tag}] {safe_name}: {pct}% ({d_mb:.1f}/{tot_mb:.1f} MB · 16线程极速回传)")
                                return True
                            except Exception:
                                if attempt == 2:
                                    raise
                                time.sleep(0.3)
                        return False

                    n_workers = min(16, max(2, len(ranges)))
                    with ThreadPoolExecutor(max_workers=n_workers) as pool:
                        futs = [pool.submit(_fetch_slice, rg) for rg in ranges]
                        for f in as_completed(futs):
                            f.result()
                finally:
                    os.close(fd)
            else:
                # 2. 若远端旧进程尚未启用 206 Range，且文件较大 (>5MB)，同时启动 Google Drive rclone 8线程极速通道竞速
                rclone_won = threading.Event()
                tmp_rc_path = LOCAL_MOVIES_DIR / (safe_name + ".rclone.part")
                rclone_bin = BASE_DIR / ".venv" / "bin" / "rclone"
                rclone_conf = BASE_DIR / ".gdrive_rclone.conf"
                if tot_bytes > 5 * 1024 * 1024 and rclone_bin.exists() and rclone_conf.exists():
                    def _race_rclone():
                        time.sleep(1.2)
                        if rclone_won.is_set():
                            return
                        out_sub = "LongCat-Avatar_outputs" if "longcat" in safe_name.lower() else "MiniMax-H3_outputs"
                        try:
                            rc_res = subprocess.run(
                                [
                                    str(rclone_bin), "copyto",
                                    f"gdrive:models/{out_sub}/{safe_name}",
                                    str(tmp_rc_path),
                                    "--config", str(rclone_conf),
                                    "--multi-thread-streams", "8",
                                    "--multi-thread-cutoff", "1M",
                                ],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=60,
                            )
                            if rc_res.returncode == 0 and tmp_rc_path.exists() and tmp_rc_path.stat().st_size >= max(1024, tot_bytes):
                                rclone_won.set()
                        except Exception:
                            pass
                    threading.Thread(target=_race_rclone, daemon=True).start()

                with probe_resp as resp, open(tmp_path, "wb") as wf:
                    done_bytes = 0
                    last_log_t = 0.0
                    while True:
                        if rclone_won.is_set() and tmp_rc_path.exists():
                            break
                        chunk = resp.read(262144)
                        if not chunk:
                            break
                        wf.write(chunk)
                        done_bytes += len(chunk)
                        done_mb = done_bytes / (1024 * 1024)
                        pct = min(99, int(done_bytes * 100 // max(1, tot_bytes))) if tot_bytes > 0 else 50
                        _VIDEO_DL_STATE["done_mb"] = done_mb
                        _VIDEO_DL_STATE["pct"] = pct
                        now = time.time()
                        if now - last_log_t >= 1.5:
                            last_log_t = now
                            _append_panel_log(f"[{dl_tag}] {safe_name}: {pct}% ({done_mb:.1f}/{tot_mb:.1f} MB, 保存至 ~/Movies/EasyColab)")
                if rclone_won.is_set() and tmp_rc_path.exists():
                    os.replace(tmp_rc_path, tmp_path)
                else:
                    rclone_won.set()
                    if tmp_rc_path.exists():
                        try:
                            tmp_rc_path.unlink()
                        except Exception:
                            pass
            os.replace(tmp_path, dest_path)
            if elapsed_s and elapsed_s > 0:
                meta_map: Dict[str, float] = {}
                if ELAPSED_META_FILE.exists():
                    try:
                        meta_map = json.loads(ELAPSED_META_FILE.read_text(encoding="utf-8"))
                    except Exception:
                        meta_map = {}
                meta_map[safe_name] = float(elapsed_s)
                ELAPSED_META_FILE.write_text(json.dumps(meta_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            _append_panel_log(f"✅ [{done_tag}] {safe_name} 已完整保存至 {LOCAL_MOVIES_DIR}/{safe_name}")
            return True
        except Exception as ex:
            _append_panel_log(f"⚠️ [{dl_tag}重试] {safe_name}: {ex}")
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            return False
    finally:
        _VIDEO_DL_STATE["active"] = False
        _VIDEO_DL_LOCK.release()

# 后台部署任务状态与实时日志缓冲
_DEPLOY_LOCK = threading.Lock()
_DEPLOY_STATE: Dict[str, Any] = {
    "running": False,
    "logs": [],
    "exit_code": None,
    "started_at": None,
}
if PANEL_LOGS_FILE.exists():
    try:
        _init_logs_obj = json.loads(PANEL_LOGS_FILE.read_text(encoding="utf-8"))
        if isinstance(_init_logs_obj.get("logs"), list):
            _raw_init = _compact_progress_log_lines(_init_logs_obj["logs"])[-400:]
            if any("RuntimeError" in x or "Traceback" in x or "[自动切换引擎]" in x for x in _raw_init):
                _DEPLOY_STATE["logs"] = [f"[{time.strftime('%H:%M:%S')}] ✅ 控制台已就绪，请选择模型并点击「🚀 下载/加载并启动」或「▶️ 一键开机」。"]
                _DEPLOY_STATE["exit_code"] = None
            else:
                _DEPLOY_STATE["logs"] = _raw_init
                if "exit_code" in _init_logs_obj:
                    _DEPLOY_STATE["exit_code"] = _init_logs_obj["exit_code"]
        _save_panel_logs_snapshot()
    except Exception:
        pass
_LAST_METRICS_CACHE: Dict[str, Any] = {
    "timestamp": 0.0,
    "data": None,
}
_LAST_CU_CACHE: Dict[str, Any] = {
    "timestamp": 0.0,
    "data": None,
}
_LAST_GDRIVE_CACHE: Dict[str, Any] = {
    "timestamp": 0.0,
    "data": None,
}


def _load_endpoint_cfg() -> Dict[str, Any]:
    if ENDPOINT_FILE.exists():
        try:
            data = json.loads(ENDPOINT_FILE.read_text(encoding="utf-8"))
            if "idle_timeout_minutes" not in data:
                data["idle_timeout_minutes"] = 30
            if "last_activity_ts" not in data:
                data["last_activity_ts"] = time.time()
            return data
        except Exception:
            pass
    return {
        "api_key": "sk-colab-ed28e54eb298d0c1",
        "tunnel_url": "",
        "model_id": "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF",
        "idle_timeout_minutes": 30,
        "last_activity_ts": time.time(),
    }


def _get_recavsr_base_url() -> str:
    # 统一走本机永久固定代理网关 127.0.0.1:30000 -> 云端 30000 端口内部转发至 127.0.0.1:30002，彻底避免旧会话残留的 recavsr_tunnel_url 域名过期导致 Errno 8 DNS 报错
    return "http://127.0.0.1:30000"


def _load_custom_models() -> List[Dict[str, Any]]:
    if CUSTOM_MODELS_FILE.exists():
        try:
            items = json.loads(CUSTOM_MODELS_FILE.read_text(encoding="utf-8"))
            if isinstance(items, list):
                return items
        except Exception:
            pass
    return []


def _save_custom_models(items: List[Dict[str, Any]]):
    CUSTOM_MODELS_FILE.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _estimate_hf_item_sizing(item: Dict[str, Any], is_gguf: bool, fallback_gb: float = 0.0) -> Dict[str, Any]:
    rid = str(item.get("id") or item.get("modelId") or "")
    st = item.get("safetensors") if isinstance(item.get("safetensors"), dict) else {}
    gg = item.get("gguf") if isinstance(item.get("gguf"), dict) else {}
    total_params = int(st.get("total") or gg.get("total") or 0)
    if total_params <= 0:
        m = re.search(r"(?:^|[-_/])(\d+(?:\.\d+)?)\s*([bBmMtT])(?:[-_/\b]|$)", rid)
        if m:
            val = float(m.group(1))
            unit = m.group(2).upper()
            total_params = int(val * (1e12 if unit == "T" else (1e9 if unit == "B" else 1e6)))
    if total_params >= 1e12:
        param_label = f"{total_params / 1e12:.1f}T"
    elif total_params >= 1e9:
        b = total_params / 1e9
        param_label = f"{b:.0f}B" if b >= 20 else f"{b:.1f}B".replace(".0B", "B")
    elif total_params >= 1e6:
        param_label = f"{total_params / 1e6:.0f}M"
    else:
        param_label = ""

    params_b = total_params / 1e9 if total_params > 0 else 0.0
    if is_gguf:
        est_vram_gb = round(params_b * 0.65 + 1.5, 1) if params_b > 0 else round(fallback_gb * 1.1, 1)
    else:
        est_vram_gb = round(params_b * 2.05 + 1.5, 1) if params_b > 0 else round(fallback_gb * 1.12, 1)

    if est_vram_gb <= 0:
        fit_label = "点详情探测体积"
        fit_color = "#94a3b8"
    elif est_vram_gb <= 24:
        fit_label = f"RTX PRO 6000(96G)/L4 轻松跑 (~{int(est_vram_gb)}G)"
        fit_color = "#10b981"
    elif est_vram_gb <= 88:
        fit_label = f"RTX PRO 6000(96G) 满血可跑 (~{int(est_vram_gb)}G)"
        fit_color = "#3b82f6"
    else:
        fit_label = f"需量化或 RTX PRO 6000 内存热备 (~{int(est_vram_gb)}G)"
        fit_color = "#f59e0b"
    return {"param_label": param_label, "est_vram_gb": est_vram_gb, "fit_label": fit_label, "fit_color": fit_color}


def _fetch_hf_trending(category: str = "all", query: str = "", force: bool = False) -> List[Dict[str, Any]]:
    cat = (category or "all").strip().lower()
    sq = (query or "").strip()
    cache_k = f"{cat}:{sq.lower()}"
    now = time.time()
    if not force and cache_k in _HF_TRENDING_CACHE and (now - _HF_TRENDING_CACHE[cache_k][0] < 120):
        return _HF_TRENDING_CACHE[cache_k][1]

    cfg = _load_endpoint_cfg()
    tok = cfg.get("hf_token", "") or os.environ.get("HF_TOKEN", "")
    headers = {"User-Agent": "ColabPanel/1.0", "Accept": "application/json"}
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    expand_qs = "&expand[]=safetensors&expand[]=gguf&expand[]=likes&expand[]=downloads&expand[]=pipeline_tag&expand[]=tags&expand[]=gated"
    search_qs = f"&search={urllib.parse.quote(sq)}" if sq else ""
    if cat == "gguf":
        qpath = f"/api/models?sort=trendingScore&direction=-1&filter=gguf&limit=24{search_qs}{expand_qs}"
    elif cat == "llm_vlm":
        qpath = f"/api/models?sort=trendingScore&direction=-1&pipeline_tag=text-generation&limit=24{search_qs}{expand_qs}"
    elif cat == "image_video":
        qpath = f"/api/models?sort=trendingScore&direction=-1&pipeline_tag=text-to-image&limit=24{search_qs}{expand_qs}"
    else:
        qpath = f"/api/models?sort=trendingScore&direction=-1&limit=24{search_qs}{expand_qs}"

    raw_items: List[Dict[str, Any]] = []
    for host in ("https://huggingface.co", "https://hf-mirror.com"):
        try:
            req = urllib.request.Request(f"{host}{qpath}", headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, list):
                    raw_items = data
                    break
        except Exception:
            continue

    results = []
    for item in raw_items:
        rid = str(item.get("id") or item.get("modelId") or "").strip()
        if not rid or "/" not in rid:
            continue
        tags = [str(t).lower() for t in (item.get("tags") or []) if isinstance(t, str)]
        ptag = str(item.get("pipeline_tag") or "").strip().lower()
        is_gguf = ("gguf" in rid.lower()) or ("gguf" in tags) or bool(item.get("gguf"))
        sizing = _estimate_hf_item_sizing(item, is_gguf)
        suggested_pat = "*Q4_K_M*.gguf" if is_gguf else "*.safetensors"
        results.append({
            "repo_id": rid,
            "name": rid.split("/")[-1],
            "folder": rid.split("/")[-1],
            "is_gguf": is_gguf,
            "pipeline_tag": "GGUF" if is_gguf else (ptag or "safetensors"),
            "likes": int(item.get("likes") or 0),
            "suggested_pattern": suggested_pat,
            **sizing,
        })
        if len(results) >= 14:
            break

    if results:
        _HF_TRENDING_CACHE[cache_k] = (now, results)
    return results


def _fetch_hf_repo_detail(repo_id: str) -> Dict[str, Any]:
    s = re.sub(r"^https?://(?:www\.)?(?:huggingface\.co|hf\.co|hf-mirror\.com)/", "", (repo_id or "").strip()).strip("/")
    parts = [p for p in s.split("/") if p]
    clean_id = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else s
    if not clean_id or "/" not in clean_id:
        return {"ok": False, "error": "请输入有效的 HuggingFace 仓库名 (如 Owner/Repo)"}

    cfg = _load_endpoint_cfg()
    tok = cfg.get("hf_token", "") or os.environ.get("HF_TOKEN", "")
    headers = {"User-Agent": "ColabPanel/1.0", "Accept": "application/json"}
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    meta = None
    for host in ("https://huggingface.co", "https://hf-mirror.com"):
        try:
            req = urllib.request.Request(f"{host}/api/models/{clean_id}?blobs=true", headers=headers)
            with urllib.request.urlopen(req, timeout=12) as resp:
                meta = json.loads(resp.read().decode("utf-8"))
                break
        except Exception:
            continue

    if not meta:
        return {"ok": False, "error": f"无法读取仓库 {clean_id}，请检查名称或填写 HF Token"}

    files = []
    total_gb = 0.0
    for sib in meta.get("siblings") or []:
        rfn = str(sib.get("rfilename") or "").strip()
        if not rfn or not rfn.lower().endswith((".gguf", ".safetensors")):
            continue
        lfs = sib.get("lfs") if isinstance(sib.get("lfs"), dict) else {}
        sz_b = float(sib.get("size") or lfs.get("size") or 0.0)
        sz_gb = round(sz_b / (1024**3), 2)
        total_gb += sz_gb
        files.append({"filename": rfn, "size_gb": sz_gb})

    is_gguf = any(f["filename"].lower().endswith(".gguf") for f in files)
    sizing = _estimate_hf_item_sizing(meta, is_gguf, fallback_gb=total_gb)
    return {
        "ok": True,
        "repo_id": clean_id,
        "name": clean_id.split("/")[-1],
        "is_gguf": is_gguf,
        "total_gb": round(total_gb, 2),
        "files": files,
        **sizing,
    }


def _query_cu_usage(force: bool = False) -> Dict[str, Any]:
    """查询 Google Colab 算力余额 (CU)、实时扣费速率 (CU/h) 与免费 T4 可用状态"""
    now = time.time()
    if not force and _LAST_CU_CACHE["data"] and (now - _LAST_CU_CACHE["timestamp"] < 20):
        return _LAST_CU_CACHE["data"]
    try:
        from urllib.parse import urljoin
        from colab_cli.client import TUN_ENDPOINT
        from colab_cli.common import state

        url = urljoin(state.client.colab_domain, f"{TUN_ENDPOINT}/ccu-info")
        resp = state.client.session.request(
            "GET",
            url,
            headers={"Accept": "application/json", "X-Colab-Client-Agent": "colab-cli"},
            params={"authuser": "0"},
            timeout=10,
        )
        raw_txt = state.client._strip_xssi_prefix(resp.text)
        raw_obj = json.loads(raw_txt) if raw_txt else {}
        el_gpus = [str(g).upper() for g in (raw_obj.get("eligibleGpus") or [])]
        data = {
            "paid_balance": round(float(raw_obj.get("currentBalance") or 0.0), 2),
            "burn_rate_hourly": round(float(raw_obj.get("consumptionRateHourly") or 0.0), 2),
            "active_assignments": int(raw_obj.get("assignmentsCount") or 0),
            "eligible_gpus": el_gpus,
            "t4_available": ("T4" in el_gpus) or (len(el_gpus) > 0),
        }
        _LAST_CU_CACHE["data"] = data
        _LAST_CU_CACHE["timestamp"] = now
        return data
    except Exception:
        return _LAST_CU_CACHE["data"] or {
            "paid_balance": None,
            "burn_rate_hourly": 0.0,
            "active_assignments": 0,
            "eligible_gpus": ["T4"],
            "t4_available": True,
        }


_ACCOUNT_INFO_CACHE: Dict[str, Any] = {"timestamp": 0.0, "data": None}
_PENDING_REMOTE_OAUTH: Dict[str, Any] = {"flow": None}


def _load_saved_accounts() -> Dict[str, Dict[str, Any]]:
    if ACCOUNTS_FILE.exists():
        try:
            obj = json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def _save_saved_accounts(acc_map: Dict[str, Dict[str, Any]]):
    ACCOUNTS_FILE.write_text(json.dumps(acc_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        os.chmod(ACCOUNTS_FILE, 0o600)
    except Exception:
        pass


def _reset_colab_cli_state():
    try:
        from colab_cli.common import state
        state._client = None
        state._sessions = None
    except Exception:
        pass
    _LAST_CU_CACHE["data"] = None
    _LAST_CU_CACHE["timestamp"] = 0.0
    _LAST_METRICS_CACHE["data"] = None
    _LAST_METRICS_CACHE["timestamp"] = 0.0
    _ACCOUNT_INFO_CACHE["data"] = None
    _ACCOUNT_INFO_CACHE["timestamp"] = 0.0


def _get_accounts_status(force: bool = False, cu_balance: Any = None, eligible_gpus: Any = None) -> Dict[str, Any]:
    now = time.time()
    if not force and _ACCOUNT_INFO_CACHE["data"] and (now - _ACCOUNT_INFO_CACHE["timestamp"] < 45):
        cached = dict(_ACCOUNT_INFO_CACHE["data"])
        if cu_balance is not None and cached.get("active_email"):
            for it in cached.get("accounts", []):
                if it.get("email") == cached["active_email"]:
                    it["cu_balance"] = cu_balance
                    if eligible_gpus is not None:
                        it["eligible_gpus"] = eligible_gpus
                        it["g4_eligible"] = (float(cu_balance or 0.0) > 0.0) or ("G4" in eligible_gpus)
        return cached

    acc_map = _load_saved_accounts()
    active_email = ""
    active_name = ""
    active_pic = ""
    try:
        import keyring
        from colab_cli.auth import KEYRING_SERVICE_NAME, KEYRING_USERNAME
        from colab_cli.common import state

        token_json = keyring.get_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME)
        if token_json:
            resp = state.client.session.get("https://www.googleapis.com/oauth2/v2/userinfo", timeout=6)
            if resp.status_code == 200:
                uinfo = resp.json()
                active_email = str(uinfo.get("email") or "").strip()
                active_name = str(uinfo.get("name") or active_email).strip()
                active_pic = str(uinfo.get("picture") or "").strip()
                if active_email:
                    prev = acc_map.get(active_email, {})
                    rclone_p = BASE_DIR / ".gdrive_rclone.conf"
                    rclone_txt = prev.get("rclone_conf", "")
                    if not rclone_txt and rclone_p.exists() and len(acc_map) == 0:
                        rclone_txt = rclone_p.read_text(encoding="utf-8")
                    eff_bal = cu_balance if cu_balance is not None else prev.get("cu_balance")
                    eff_gpus = eligible_gpus if eligible_gpus is not None else (prev.get("eligible_gpus") or [])
                    acc_map[active_email] = {
                        "email": active_email,
                        "name": active_name,
                        "picture": active_pic,
                        "token_json": token_json,
                        "rclone_conf": rclone_txt,
                        "cu_balance": eff_bal,
                        "eligible_gpus": eff_gpus,
                        " updated_at": time.strftime("%Y-%m-%d %H:%M"),
                        "updated_at": time.strftime("%Y-%m-%d %H:%M"),
                    }
                    _save_saved_accounts(acc_map)
    except Exception:
        pass

    public_list = []
    for em, info in acc_map.items():
        bal_v = info.get("cu_balance")
        el_g = [str(g).upper() for g in (info.get("eligible_gpus") or [])]
        g4_ok = (bal_v is not None and float(bal_v or 0.0) > 0.0) or ("G4" in el_g)
        public_list.append({
            "email": em,
            "name": info.get("name") or em,
            "picture": info.get("picture") or "",
            "cu_balance": bal_v,
            "eligible_gpus": el_g,
            "g4_eligible": g4_ok,
            "has_drive_conf": bool(info.get("rclone_conf")),
            "updated_at": info.get("updated_at") or "",
            "is_active": (em == active_email),
        })
    public_list.sort(key=lambda x: (0 if x["is_active"] else 1, x["email"]))
    res = {
        "active_email": active_email,
        "active_name": active_name,
        "accounts": public_list,
    }
    _ACCOUNT_INFO_CACHE["data"] = res
    _ACCOUNT_INFO_CACHE["timestamp"] = now
    return res


def _refresh_all_saved_accounts_cu() -> Dict[str, Any]:
    """并发向 Google Colab 服务端强制刷新所有已绑定账号的实时 CU 余额与 G4 可用权限"""
    from concurrent.futures import ThreadPoolExecutor
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    import requests

    acc_map = _load_saved_accounts()

    def _probe_one(em_and_info):
        em, info = em_and_info
        tok_str = info.get("token_json") or ""
        if not tok_str:
            return em, None, []
        try:
            tok_obj = json.loads(tok_str)
            creds = Credentials.from_authorized_user_info(tok_obj)
            if creds.refresh_token:
                creds.refresh(Request())
                info["token_json"] = creds.to_json()
            r = requests.get(
                "https://colab.research.google.com/tun/m/ccu-info?authuser=0",
                headers={
                    "Authorization": f"Bearer {creds.token}",
                    "Accept": "application/json",
                    "X-Colab-Client-Agent": "colab-cli",
                },
                timeout=8,
            )
            if r.status_code == 200:
                txt = r.text.split("\n", 1)[-1] if r.text.startswith(")]}") else r.text
                obj = json.loads(txt)
                bal = round(float(obj.get("currentBalance") or 0.0), 2)
                el_gpus = [str(g).upper() for g in (obj.get("eligibleGpus") or [])]
                return em, bal, el_gpus
        except Exception:
            pass
        return em, info.get("cu_balance"), info.get("eligible_gpus") or []

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_probe_one, list(acc_map.items())))

    for em, bal, el_gpus in results:
        if em in acc_map:
            acc_map[em]["cu_balance"] = bal
            acc_map[em]["eligible_gpus"] = el_gpus
            acc_map[em]["updated_at"] = time.strftime("%Y-%m-%d %H:%M")

    _save_saved_accounts(acc_map)
    _LAST_CU_CACHE["data"] = None
    _LAST_CU_CACHE["timestamp"] = 0.0
    _ACCOUNT_INFO_CACHE["data"] = None
    _ACCOUNT_INFO_CACHE["timestamp"] = 0.0
    cu_now = _query_cu_usage(force=True)
    return _get_accounts_status(
        force=True,
        cu_balance=(cu_now or {}).get("paid_balance"),
        eligible_gpus=(cu_now or {}).get("eligible_gpus"),
    )



_GDRIVE_REFRESHING = False


def _refresh_gdrive_cache_worker():
    global _GDRIVE_REFRESHING
    try:
        rclone_bin = BASE_DIR / ".venv" / "bin" / "rclone"
        rclone_conf = BASE_DIR / ".gdrive_rclone.conf"
        res: Dict[str, Any] = {"bound": True, "quota": None, "drive_folders": []}
        try:
            about_out = subprocess.check_output(
                [str(rclone_bin), "about", "gdrive:", "--json", "--config", str(rclone_conf)],
                stderr=subprocess.DEVNULL,
                timeout=25,
                text=True,
            )
            ab = json.loads(about_out)
            tot_b = float(ab.get("total") or 0.0)
            used_b = float(ab.get("used") or 0.0)
            free_b = float(ab.get("free") or max(0.0, tot_b - used_b))
            if tot_b > 0:
                res["quota"] = {
                    "used_gb": round(used_b / (1024**3), 2),
                    "total_gb": round(tot_b / (1024**3), 2),
                    "free_gb": round(free_b / (1024**3), 2),
                    "pct": round(used_b * 100.0 / tot_b, 1),
                }
        except Exception:
            if _LAST_GDRIVE_CACHE["data"] and _LAST_GDRIVE_CACHE["data"].get("quota"):
                res["quota"] = _LAST_GDRIVE_CACHE["data"]["quota"]

        try:
            ls_out = subprocess.check_output(
                [str(rclone_bin), "lsjson", "gdrive:models", "-R", "--config", str(rclone_conf)],
                stderr=subprocess.DEVNULL,
                timeout=25,
                text=True,
            )
            items = json.loads(ls_out)
            folders_map: Dict[str, List[Dict[str, Any]]] = {}
            for it in items:
                p = str(it.get("Path") or "")
                if not p or p.startswith("."):
                    continue
                parts = p.split("/")
                top = parts[0]
                if top.endswith("_outputs"):
                    continue
                if it.get("IsDir"):
                    if len(parts) == 1:
                        folders_map.setdefault(top, [])
                else:
                    if p.lower().endswith((".gguf", ".safetensors", ".pth")):
                        rel = "/".join(parts[1:]) if len(parts) > 1 else p
                        sz_gb = round(float(it.get("Size") or 0.0) / (1024**3), 2)
                        folders_map.setdefault(top, []).append({"name": rel, "size_gb": sz_gb})
            for fname in sorted(folders_map.keys()):
                flist = sorted(folders_map[fname], key=lambda x: x["name"])
                tot_gb = round(sum(x["size_gb"] for x in flist), 2)
                res["drive_folders"].append({"folder": fname, "files": flist, "total_gb": tot_gb})
        except Exception:
            if _LAST_GDRIVE_CACHE["data"] and _LAST_GDRIVE_CACHE["data"].get("drive_folders"):
                res["drive_folders"] = _LAST_GDRIVE_CACHE["data"]["drive_folders"]

        _LAST_GDRIVE_CACHE["data"] = res
        _LAST_GDRIVE_CACHE["timestamp"] = time.time()
    finally:
        _GDRIVE_REFRESHING = False


def _query_local_gdrive_info(force: bool = False) -> Dict[str, Any]:
    """利用本机永久授权的 .gdrive_rclone.conf 异步查询 Google Drive 真实容量与 MyDrive/models 目录（即使 G4 关机也能实时查看）"""
    global _GDRIVE_REFRESHING
    rclone_bin = BASE_DIR / ".venv" / "bin" / "rclone"
    rclone_conf = BASE_DIR / ".gdrive_rclone.conf"
    if not rclone_bin.exists() or not rclone_conf.exists():
        return {"bound": False}

    now = time.time()
    if (force or not _LAST_GDRIVE_CACHE["data"] or (now - _LAST_GDRIVE_CACHE["timestamp"] >= 60)) and not _GDRIVE_REFRESHING:
        _GDRIVE_REFRESHING = True
        threading.Thread(target=_refresh_gdrive_cache_worker, daemon=True).start()

    return _LAST_GDRIVE_CACHE["data"] or {"bound": True, "quota": None, "drive_folders": []}


def _list_active_sessions() -> List[Dict[str, str]]:
    try:
        from colab_cli.common import state
        from colab_cli.state import SessionState
        state._sessions = None
        local_map, assignments = state.sync_sessions()
        name_by_ep = {s.endpoint: s.name for s in local_map.values()}
        sessions: List[Dict[str, str]] = []
        for a in assignments:
            name = name_by_ep.get(a.endpoint)
            if not name:
                name = "qwen38-g4"
                s_adopted = SessionState(
                    name=name,
                    token=a.runtime_proxy_info.token,
                    url=a.runtime_proxy_info.url,
                    endpoint=a.endpoint,
                    token_expires_at=a.runtime_proxy_info.expires_at(),
                    variant=a.variant.name,
                    accelerator=a.accelerator.value,
                    machine_shape=a.machine_shape.name,
                )
                state.store.add(s_adopted)
            sessions.append({
                "name": name,
                "endpoint": a.endpoint,
                "gpu": a.accelerator.value.upper(),
                "raw": f"[{name}] {a.endpoint} | Hardware: {a.accelerator.value.upper()}",
            })
        return sessions
    except Exception:
        return []


def _query_remote_metrics(session_name: str) -> Dict[str, Any]:
    """在远端 Colab 实例中执行轻量采集脚本，返回 JSON 结构化硬件与目录数据"""
    from colab_cli.common import state
    from colab_cli.runtime import ColabRuntime

    s = state.get_session(session_name)
    runtime = ColabRuntime(
        s.url,
        s.token,
        session_name=s.name,
        history=state.history,
        kernel_id=s.kernel_id,
        session_id=None,
    )
    probe_code = """
import json, os, re, shutil, subprocess

res = {
    "gpu": None,
    "ram": None,
    "shm": None,
    "disk_content": None,
    "disk_drive": None,
    "drive_mounted": os.path.isdir("/content/drive/MyDrive"),
    "drive_folders": [],
    "local_files": [],
    "drive_files": [],
    "sync_running": False,
    "llama_running": False,
    "tunnel_url": "",
    "api_key": "",
}

gpu_raw = subprocess.getoutput(
    "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null"
).strip()
if gpu_raw and "," in gpu_raw:
    p = [x.strip() for x in gpu_raw.split(",")]
    if len(p) >= 7:
        used_mb = float(p[1])
        tot_mb = max(1.0, float(p[2]))
        res["gpu"] = {
            "name": p[0],
            "used_gb": round(used_mb / 1024.0, 2),
            "total_gb": round(tot_mb / 1024.0, 2),
            "pct": round(used_mb * 100.0 / tot_mb, 1),
            "util": p[3],
            "temp": p[4],
            "power": f"{p[5]}W / {p[6]}W",
        }

mem_raw = subprocess.getoutput("free -b 2>/dev/null").splitlines()
for line in mem_raw:
    if line.lower().startswith("mem:"):
        cols = line.split()
        if len(cols) >= 7:
            tot_b = float(cols[1])
            avail_b = float(cols[6])
            used_b = max(0.0, tot_b - avail_b)
            res["ram"] = {
                "used_gb": round(used_b / (1024**3), 2),
                "total_gb": round(tot_b / (1024**3), 2),
                "pct": round(used_b * 100.0 / max(1.0, tot_b), 1),
            }

for key, path in [("shm", "/dev/shm"), ("disk_content", "/content"), ("disk_drive", "/content/drive/MyDrive")]:
    if os.path.exists(path):
        try:
            du = shutil.disk_usage(path)
            res[key] = {
                "used_gb": round(du.used / (1024**3), 2),
                "total_gb": round(du.total / (1024**3), 2),
                "free_gb": round(du.free / (1024**3), 2),
                "pct": round(du.used * 100.0 / max(1, du.total), 1),
            }
        except Exception:
            pass

models_root = "/content/drive/MyDrive/models"
if os.path.isdir(models_root):
    for d in sorted(os.listdir(models_root)):
        dp = os.path.join(models_root, d)
        if os.path.isdir(dp) and not d.startswith("."):
            weights = []
            for r, _, fs in os.walk(dp):
                for f in sorted(fs):
                    if f.lower().endswith((".gguf", ".safetensors")):
                        fp = os.path.join(r, f)
                        try:
                            sz_gb = round(os.path.getsize(fp) / (1024**3), 2)
                        except Exception:
                            sz_gb = 0.0
                        weights.append({"name": os.path.relpath(fp, dp), "size_gb": sz_gb})
            tot_gb = round(sum(x["size_gb"] for x in weights), 2)
            res["drive_folders"].append({"folder": d, "files": weights, "total_gb": tot_gb})

for key, root_p in [("local_files", "/dev/shm/local_models"), ("drive_files", "/content/drive/MyDrive/models")]:
    if os.path.isdir(root_p):
        items = []
        for r, _, fs in os.walk(root_p):
            for f in sorted(fs):
                if f.lower().endswith((".gguf", ".safetensors")):
                    fp = os.path.join(r, f)
                    try:
                        sz_gb = round(os.path.getsize(fp) / (1024**3), 2)
                    except Exception:
                        sz_gb = 0.0
                    items.append({"path": os.path.relpath(fp, root_p), "size_gb": sz_gb})
        res[key] = items

res["sync_running"] = bool(subprocess.getoutput("pgrep -af '[c]p -rn|[r]clone copy'").strip())
llama_proc = subprocess.getoutput("pgrep -af '[l]lama-server|minimax_server\\.py|minimax_t4_server\\.py|longcat_avatar_server\\.py|minimax_image_server\\.py'").strip()
res["llama_running"] = bool(llama_proc)
m_k = re.search(r"--api-key\\s+(\\S+)", llama_proc)
if m_k:
    res["api_key"] = m_k.group(1)
cf_log = subprocess.getoutput("cat /content/cf_tunnel.log 2>/dev/null")
m_u = re.search(r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com", cf_log)
if m_u:
    res["tunnel_url"] = m_u.group(0)
rc_log = subprocess.getoutput("cat /content/recavsr_tunnel.url /content/recavsr_tunnel.log 2>/dev/null")
m_rc = re.search(r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com", rc_log)
if m_rc:
    res["recavsr_tunnel_url"] = m_rc.group(0)
try:
    res["uptime_sec"] = int(float(open("/proc/uptime").read().split()[0]))
except Exception:
    res["uptime_sec"] = 0
try:
    mtimes = [
        float(os.path.getmtime(p))
        for p in ("/content/llama_server.log", "/content/minimax_server.log", "/content/recavsr_server.log")
        if os.path.exists(p)
    ]
    if mtimes:
        res["llama_log_mtime"] = max(mtimes)
except Exception:
    pass

print("PANEL_JSON_START:" + json.dumps(res, ensure_ascii=False) + ":PANEL_JSON_END")
"""
    try:
        runtime.kernel_client._manager.client.connection_ready.wait(timeout=15)
        outputs = runtime.execute_code(probe_code, allow_stdin=False, timeout=25)
        full_text = "".join(o.get("text", "") for o in outputs)
        m = re.search(r"PANEL_JSON_START:(\{.*\}):PANEL_JSON_END", full_text, flags=re.DOTALL)
        if m:
            data = json.loads(m.group(1))
            # 如果远端有最新 tunnel_url 或推理活动时间戳，自动同步到本地 .colab_endpoint.json
            cfg = _load_endpoint_cfg()
            changed = False
            if data.get("tunnel_url") and cfg.get("tunnel_url") != data["tunnel_url"]:
                cfg["tunnel_url"] = data["tunnel_url"]
                changed = True
            if data.get("recavsr_tunnel_url") and cfg.get("recavsr_tunnel_url") != data["recavsr_tunnel_url"]:
                cfg["recavsr_tunnel_url"] = data["recavsr_tunnel_url"]
                changed = True
            if data.get("api_key") and cfg.get("api_key") != data["api_key"]:
                cfg["api_key"] = data["api_key"]
                changed = True
            if data.get("llama_log_mtime"):
                prev_ts = float(cfg.get("last_activity_ts", 0.0) or 0.0)
                if float(data["llama_log_mtime"]) > prev_ts:
                    cfg["last_activity_ts"] = float(data["llama_log_mtime"])
                    changed = True
            if changed:
                ENDPOINT_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return data
        raise RuntimeError(f"未解析到监控数据: {full_text[:200]}")
    finally:
        runtime.stop()


def _start_deploy_background(repo: str, folder: str, pattern: str, ctx: int, gpu: str, download_only: bool = False):
    g4_only_folders = {
        "Huihui-Qwen3.8-27B-abliterated-GGUF",
        "MiniMax-H3",
        "MiniMax-H3-Image",
    }
    auto_switched_note = ""
    if not download_only:
        try:
            cu_check = _query_cu_usage(force=True)
            bal_val = cu_check.get("paid_balance")
            el_gpus = [str(g).upper() for g in (cu_check.get("eligible_gpus") or [])]
            has_g4_entitlement = (bal_val is not None and float(bal_val) > 0.0) or ("G4" in el_gpus)
            if not has_g4_entitlement and bal_val is not None and float(bal_val) <= 0.0:
                gpu = "T4"
                if folder in ("MiniMax-H3", "MiniMax-H3-Turbo"):
                    old_folder = folder
                    repo = "unsloth/MiniMax-H3-GGUF"
                    folder = "MiniMax-H3-T4-GGUF"
                    pattern = "minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf"
                    auto_switched_note = f"⚡ [T4 智能适配] 当前为 T4 (16GB) 账号，已自动将 {old_folder} 切换为 T4 专享量化版 {folder} (UD-Q2_K_XL · 8步有声视频)"
                elif folder == "LongCat-Video-Avatar-1.6":
                    repo = "meituan-longcat/LongCat-Video-Avatar-1.6"
                    folder = "LongCat-Video-Avatar-1.6-INT8"
                    pattern = "base_model_int8/*.safetensors"
                    auto_switched_note = "⚡ [T4 智能适配] 当前为 T4 (16GB) 账号，已自动将 LongCat-Video-Avatar-1.6 切换为 INT8 量化版 (LongCat-Video-Avatar-1.6-INT8)"
                elif folder in g4_only_folders or "bf16.gguf" in (pattern or "").lower():
                    old_folder = folder
                    repo = "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF"
                    folder = "Qwen3.8-27B-T4-Q3"
                    pattern = "Huihui-Qwen3.8-27B-abliterated-UD-Q3_K_XL.gguf"
                    auto_switched_note = f"⚡ [T4 防爆盘保护] 当前为 T4 (16GB) 账号，已自动将启动模型从 {old_folder} 切换为 T4 适配模型 {folder} (12.4 GB)"
        except Exception:
            pass

    with _DEPLOY_LOCK:
        if _DEPLOY_STATE["running"]:
            return False, "已有部署/下载任务正在运行中"
        _DEPLOY_STATE["running"] = True
        action_label = "仅下载模型至 Google Drive 云盘" if download_only else "一键部署与加载模型"
        init_logs = []
        if auto_switched_note:
            init_logs.append(f"[{time.strftime('%H:%M:%S')}] {auto_switched_note}")
        init_logs.append(f"[{time.strftime('%H:%M:%S')}] 🚀 开始执行 [{action_label}] ({repo} · {folder})...")
        _DEPLOY_STATE["logs"] = init_logs
        _DEPLOY_STATE["exit_code"] = None
        _DEPLOY_STATE["started_at"] = time.time()
        _save_panel_logs_snapshot()

    cfg = _load_endpoint_cfg()
    cfg["last_activity_ts"] = time.time()
    cfg["deploying"] = True
    if not download_only:
        cfg["model_id"] = repo
        cfg["model_folder"] = folder
    ENDPOINT_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _worker():
        cmd = [
            str(PYTHON_BIN),
            str(DEPLOY_SCRIPT),
            "--gpu", gpu or "G4",
            "--repo", repo,
            "--folder", folder,
            "--pattern", pattern,
            "--ctx", str(ctx),
            "-y",
        ]
        if download_only:
            cmd.append("--download-only")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                rln = line.rstrip("\n")
                with _DEPLOY_LOCK:
                    logs_ref = _DEPLOY_STATE["logs"]
                    if logs_ref and "[显存装载]" in rln and "[显存装载]" in logs_ref[-1]:
                        logs_ref[-1] = rln
                    else:
                        logs_ref.append(rln)
                        if len(logs_ref) > 500:
                            _DEPLOY_STATE["logs"] = logs_ref[-500:]
            proc.wait()
            with _DEPLOY_LOCK:
                _DEPLOY_STATE["exit_code"] = proc.returncode
                _DEPLOY_STATE["running"] = False
                _LAST_METRICS_CACHE["timestamp"] = 0.0
                _save_panel_logs_snapshot()
        except Exception as e:
            with _DEPLOY_LOCK:
                _DEPLOY_STATE["logs"].append(f"[错误] {e}")
                _DEPLOY_STATE["exit_code"] = 1
                _DEPLOY_STATE["running"] = False
                _save_panel_logs_snapshot()
        finally:
            c2 = _load_endpoint_cfg()
            c2["last_activity_ts"] = time.time()
            c2["deploying"] = False
            ENDPOINT_FILE.write_text(json.dumps(c2, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            _query_local_gdrive_info(force=True)

    threading.Thread(target=_worker, daemon=True).start()
    return True, "已启动后台部署任务"


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NVIDIA RTX PRO 6000 Blackwell Server</title>
<style>
  :root {
    --bg: #0f172a;
    --card: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #3b82f6;
    --green: #10b981;
    --amber: #f59e0b;
    --red: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC", sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    max-width: 1200px;
    margin: 0 auto;
    line-height: 1.5;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--card);
    border: 1px solid var(--border);
    padding: 18px 24px;
    border-radius: 14px;
    margin-bottom: 20px;
    flex-wrap: wrap;
    gap: 12px;
  }
  .title-area h1 { font-size: 20px; font-weight: 700; display: flex; align-items: center; gap: 10px; }
  .title-area p { font-size: 13px; color: var(--muted); margin-top: 4px; }
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
  }
  .badge.online { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
  .badge.offline { background: rgba(148,163,184,0.15); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.3); }
  .btn-group { display: flex; gap: 10px; flex-wrap: wrap; }
  button {
    cursor: pointer;
    border: none;
    border-radius: 8px;
    padding: 9px 16px;
    font-size: 13px;
    font-weight: 600;
    transition: all 0.15s;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 6px;
  }
  .dur-pill, .batch-pill {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    width: 100%;
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-primary { background: var(--accent); color: white; }
  .btn-primary:hover:not(:disabled) { background: #2563eb; }
  .btn-success { background: var(--green); color: white; }
  .btn-success:hover:not(:disabled) { background: #059669; }
  .btn-danger { background: var(--red); color: white; }
  .btn-danger:hover:not(:disabled) { background: #dc2626; }
  .btn-outline { background: transparent; color: var(--text); border: 1px solid var(--border); }
  .btn-outline:hover:not(:disabled) { background: rgba(255,255,255,0.06); }

  .grid-top {
    display: grid;
    grid-template-columns: 1.2fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }
  .grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 20px;
  }
  .grid-2 {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
    gap: 16px;
    margin-bottom: 20px;
  }
  @media (max-width: 900px) {
    .grid-top, .grid-3 { grid-template-columns: 1fr; }
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
  }
  .card h2 {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .metric-val { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
  .metric-sub { font-size: 12px; color: var(--muted); margin-bottom: 10px; }
  .progress-bar {
    width: 100%;
    height: 8px;
    background: #0f172a;
    border-radius: 999px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 999px;
    transition: width 0.3s ease;
  }
  .progress-fill.green { background: var(--green); }
  .progress-fill.amber { background: var(--amber); }

  .kv-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 9px 12px;
    background: #0f172a;
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 10px;
    font-size: 13px;
  }
  .kv-label { color: var(--muted); font-size: 12px; min-width: 125px; }
  .kv-val {
    font-family: "SF Mono", Menlo, monospace;
    color: #93c5fd;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
  }
  .form-group { margin-bottom: 12px; }
  .form-group label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
  .form-group input, .form-group select {
    width: 100%;
    padding: 9px 12px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: #0f172a;
    color: var(--text);
    font-size: 13px;
    font-family: inherit;
  }
  .folder-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #0f172a;
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 12px;
    margin: 4px 6px 4px 0;
    cursor: pointer;
  }
  .folder-tag:hover { border-color: var(--accent); color: #93c5fd; }
  pre.log-box {
    background: #090d16;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    font-family: "SF Mono", Menlo, monospace;
    font-size: 12px;
    color: #cbd5e1;
    max-height: 280px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
  }
  .chat-out {
    background: #0f172a;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    font-size: 13px;
    min-height: 76px;
    max-height: 200px;
    overflow-y: auto;
    margin-top: 10px;
    white-space: pre-wrap;
  }
  .cu-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px 16px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 6px;
  }
  .cu-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2px;
    flex-wrap: nowrap;
    gap: 8px;
  }
  .cu-title {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 14px;
    font-weight: 700;
    white-space: nowrap;
  }
  .cu-metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    text-align: center;
    padding: 2px 0 0 0;
  }
  .cu-col {
    padding: 0 6px;
    border-right: 1px solid var(--border);
    min-width: 0;
  }
  .cu-col:last-child {
    border-right: none;
  }
  .cu-col-label {
    font-size: 11.5px;
    color: var(--muted);
    margin-bottom: 2px;
    white-space: nowrap;
  }
  .cu-col-val {
    font-size: 18px;
    font-weight: 700;
    line-height: 1.2;
    white-space: nowrap;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Mono", sans-serif;
  }
  .cu-controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    flex-wrap: nowrap;
    padding-top: 8px;
    margin-top: 6px;
    border-top: 1px solid var(--border);
  }
  .cu-select {
    background: #0f172a;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 4px 8px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
  }
  .btn-release-all {
    justify-content: center;
    background: rgba(255,255,255,0.07);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 4px 10px;
    border-radius: 7px;
    font-size: 12px;
    font-weight: 600;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .btn-release-all:hover:not(:disabled) {
    background: rgba(239,68,68,0.18);
    border-color: rgba(239,68,68,0.45);
    color: #fca5a5;
  }
</style>
</head>
<body>

<header style="padding:14px 20px;">
  <div class="title-area">
    <h1 style="font-size:18px;white-space:nowrap;">
      🚀 NVIDIA RTX PRO 6000 Blackwell
      <span id="sessionBadge" class="badge offline">检测中...</span>
    </h1>
  </div>
  <div class="btn-group" style="align-items:center;gap:8px;">
    <button class="btn-outline" onclick="openAccountModal()" id="btnAccountSwitch" title="切换或添加 Google Colab Pro 账号" style="padding:7px 12px;border-color:rgba(59,130,246,0.5);color:#93c5fd;">👤 <span id="headerAccountLabel">账号检测中...</span> ▾</button>
    <button class="btn-outline" onclick="refreshStatus(true)" id="btnRefresh" style="padding:7px 12px;">🔄 刷新</button>
    <button class="btn-success" onclick="startDeploy()" id="btnQuickStart" style="padding:7px 14px;">▶️ 一键开机</button>
    <button class="btn-danger" onclick="stopColab()" id="btnStop" style="padding:7px 14px;">🛑 关机</button>
  </div>
</header>

<div class="grid-top">
  <div class="cu-card">
    <div class="cu-header">
      <div class="cu-title">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 14a1.5 1.5 0 1 1 1.5-1.5A1.5 1.5 0 0 1 12 16zm1-4.5V7"/>
          <path d="M12 12L15.5 8.5"/>
        </svg>
        <span>算力额度与自动休眠</span>
      </div>
      <span id="idleCountdownText" style="font-size:12px;color:var(--muted);">已接入自动保活与闲置断电守护</span>
    </div>

    <div class="cu-metrics">
      <div class="cu-col">
        <div class="cu-col-label">预计余额</div>
        <div class="cu-col-val" id="cuBalance">-- CU</div>
        <div id="cuRemainHours" style="font-size:11px;color:var(--muted);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">可跑约 -- 小时</div>
      </div>
      <div class="cu-col">
        <div class="cu-col-label">扣费速率</div>
        <div class="cu-col-val" id="cuBurnRate" style="color:var(--muted)">0.00 CU/h</div>
        <div style="font-size:11px;color:var(--muted);margin-top:2px;white-space:nowrap;">有效期 90 天</div>
      </div>
      <div class="cu-col">
        <div class="cu-col-label">本次运行</div>
        <div class="cu-col-val" id="cuUptime" style="color:var(--muted)">00:00:00</div>
        <div style="font-size:11px;margin-top:2px;white-space:nowrap;">
          <span id="cuResetDayLabel" onclick="setBillingResetDay()" style="color:var(--muted);cursor:pointer;" title="点击修改每月重置日">⚙️ 设重置日</span>
        </div>
      </div>
      <div class="cu-col">
        <div class="cu-col-label">本次消耗</div>
        <div class="cu-col-val" id="cuSessionUsed" style="color:var(--muted)">0.00 CU</div>
        <div id="cuSessionUsedSub" style="font-size:11px;color:var(--muted);margin-top:2px;white-space:nowrap;">累计: 0.000 CU</div>
      </div>
    </div>
  </div>

  <div class="card" style="display:flex;flex-direction:column;justify-content:center;padding:10px 16px;border-radius:12px;">
    <div>
      <h2 style="margin-bottom:4px;font-size:14px;">
        <span>☁️ Google Drive 云盘</span>
        <span id="syncBadge" style="font-size:12px;color:var(--green)">--</span>
      </h2>
      <div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:nowrap;margin-bottom:5px;" title="云盘模型目录：/content/drive/MyDrive/models/">
        <div style="display:flex;align-items:baseline;gap:8px;white-space:nowrap;">
          <span class="metric-val" id="driveVal" style="font-size:18px;margin-bottom:0;">-- / -- GB</span>
          <span class="metric-sub" id="driveSub" style="margin-bottom:0;font-size:11.5px;">剩余: -- GB</span>
        </div>
        <code style="font-size:11px;color:var(--muted);white-space:nowrap;">MyDrive/models/</code>
      </div>
      <div class="progress-bar" style="height:5px;margin-top:0;"><div class="progress-fill amber" id="driveBar" style="width:0%"></div></div>
    </div>
    <div class="cu-controls">
      <div style="display:flex;align-items:center;gap:6px;white-space:nowrap;min-width:0;">
        <span style="font-size:12px;color:var(--muted);">闲置自动释放<small id="idleElapsedHint" style="color:#64748b;margin-left:4px;"></small></span>
        <select id="idleSelect" class="cu-select" onchange="updateIdleConfig()">
          <option value="15">15 分钟</option>
          <option value="30" selected>30 分钟</option>
          <option value="60">60 分钟</option>
          <option value="120">120 分钟</option>
          <option value="0">从不</option>
        </select>
      </div>
      <button class="btn-release-all" onclick="stopColab()" id="btnReleaseAll">
        ⏻ 释放实例
      </button>
    </div>
  </div>
</div>

<div class="grid-3">
  <div class="card" style="display:flex;flex-direction:column;justify-content:space-between;">
    <div>
      <h2><span>🎮 GPU 显存 (VRAM)</span><span id="gpuTemp" style="font-size:12px;color:var(--muted)">--</span></h2>
      <div class="metric-val" id="gpuVal">-- / -- GB</div>
      <div class="metric-sub" id="gpuSub" style="white-space:pre-line;line-height:1.45;">等待连接 RTX PRO 6000 实例...</div>
    </div>
    <div class="progress-bar"><div class="progress-fill green" id="gpuBar" style="width:0%"></div></div>
  </div>

  <div class="card" style="display:flex;flex-direction:column;justify-content:space-between;">
    <div>
      <h2><span>🧠 内存盘 (/dev/shm)</span><span id="ramInfo" style="font-size:12px;color:var(--muted)">RAM: --</span></h2>
      <div class="metric-val" id="shmVal">-- / -- GB</div>
      <div class="metric-sub" id="shmSub" style="white-space:pre-line;line-height:1.45;">高速内存盘（存放当前运行权重）</div>
    </div>
    <div class="progress-bar"><div class="progress-fill" id="shmBar" style="width:0%"></div></div>
  </div>

  <div class="card" style="display:flex;flex-direction:column;justify-content:space-between;">
    <div>
      <h2><span>💾 本地 NVMe 盘 (/content)</span></h2>
      <div class="metric-val" id="diskVal">-- / -- GB</div>
      <div class="metric-sub" id="diskSub" style="white-space:pre-line;line-height:1.45;">剩余可用: -- GB</div>
    </div>
    <div class="progress-bar"><div class="progress-fill" id="diskBar" style="width:0%"></div></div>
  </div>
</div>

<div class="grid-2">
  <div class="card">
    <h2>
      <span>🔗 OpenAI 兼容 API 接入参数（客户端直接填）</span>
      <span id="llamaBadge" class="badge offline" style="font-size:11px">未就绪</span>
    </h2>

    <div class="kv-row">
      <span class="kv-label">本机永久固定 URL</span>
      <span class="kv-val" id="fixedUrl">http://127.0.0.1:30000/v1</span>
      <button class="btn-outline" style="padding:4px 10px;font-size:12px" onclick="copyText('fixedUrl')">复制</button>
    </div>

    <div class="kv-row">
      <span class="kv-label">公网临时 Tunnel URL</span>
      <span class="kv-val" id="tunnelUrl">--</span>
      <button class="btn-outline" style="padding:4px 10px;font-size:12px" onclick="copyText('tunnelUrl')">复制</button>
    </div>

    <div class="kv-row">
      <span class="kv-label">永久固定 API Key</span>
      <span class="kv-val" id="apiKey">sk-colab-ed28e54eb298d0c1</span>
      <button class="btn-outline" style="padding:4px 10px;font-size:12px" onclick="copyText('apiKey')">复制</button>
    </div>

    <div class="kv-row">
      <span class="kv-label">Model ID (模型名)</span>
      <span class="kv-val" id="modelId">huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF</span>
      <button class="btn-outline" style="padding:4px 10px;font-size:12px" onclick="copyText('modelId')">复制</button>
    </div>

    <div style="margin-top: 14px; border-top: 1px solid var(--border); padding-top: 12px;">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap;">
        <label style="font-size:12.5px;font-weight:700;color:#f8fafc;">💬 输入你想生成的视频想法 (由 Qwen3.8-27B 无审查自动扩写四项提示词)：</label>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <button class="btn-primary" style="padding:5px 12px;font-size:12px;white-space:nowrap;" onclick="sendTestChat(true)" id="btnExpand4">✨ 生成四项提示词并填入下方</button>
          <button class="btn-outline" style="padding:5px 10px;font-size:12px;border-color:#3b82f6;color:#93c5fd;white-space:nowrap;" onclick="copyChatToVideoPrompt()">📋 重新填入下方四框</button>
          <button class="btn-outline" style="padding:5px 10px;font-size:12px;white-space:nowrap;" onclick="sendTestChat(false)" id="btnChat">💬 普通对话</button>
        </div>
      </div>
      <textarea id="chatPrompt" rows="2" placeholder="直接用中文写你想生成什么样的视频，例如：赛博朋克雨夜东京街头，穿透明雨衣的少女走向镜头..." style="width:100%;padding:9px 12px;border-radius:8px;border:1px solid var(--border);background:#0f172a;color:#f8fafc;font-size:13px;line-height:1.5;resize:vertical;">赛博朋克雨夜东京街头，霓虹灯倒映在积水中，一个穿透明雨衣的女孩打着伞缓缓走向镜头，电影级光影。</textarea>
      <div class="chat-out" id="chatOutput" style="min-height:84px;max-height:190px;font-size:12px;line-height:1.5;">输入视频想法后点击「✨ 生成四项提示词并填入下方」，Qwen3.8 会自动生成【主体画面 / 环境音效 / 背景音乐 / 反向排除项】并直接填进下方四个框！</div>
    </div>
  </div>

  <div class="card">
    <h2 style="margin-bottom:10px;white-space:nowrap;">
      <span>📦 模型管理</span>
      <div style="display:inline-flex;gap:6px;flex-shrink:0;">
        <button type="button" id="tabPresetG4" onclick="switchPresetTierTab('g4')" style="padding:4px 12px;font-size:11.5px;font-weight:700;border-radius:999px;border:1px solid #3b82f6;background:rgba(59,130,246,0.22);color:#93c5fd;cursor:pointer;white-space:nowrap;">🔥 RTX 6000 (96GB)</button>
        <button type="button" id="tabPresetT4" onclick="switchPresetTierTab('t4')" style="padding:4px 12px;font-size:11.5px;font-weight:700;border-radius:999px;border:1px solid var(--border);background:#0f172a;color:#94a3b8;cursor:pointer;white-space:nowrap;">⚡ T4 (16GB)</button>
      </div>
    </h2>

    <!-- G4 96GB 满血模型专区 -->
    <div style="display:grid;grid-template-columns:repeat(2, 1fr);gap:8px;margin-bottom:12px;" id="presetModelGrid">
      <div class="folder-tag preset-card" id="preset_qwen_bf16" onclick="selectPresetModel('preset_qwen_bf16', 'huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF', 'Huihui-Qwen3.8-27B-abliterated-GGUF', 'Huihui-Qwen3.8-27B-abliterated-bf16.gguf')" style="margin:0;padding:8px 10px;border-color:#3b82f6;background:rgba(59,130,246,0.14);justify-content:space-between;">
        <div>
          <div style="font-weight:700;color:#f8fafc;font-size:13px;">🧠 Qwen3.8-27B (无审查)</div>
          <div style="color:var(--muted);font-size:11px;">BF16 原生 + MTP加速 · 53.1 GB</div>
        </div>
        <span class="preset-check" style="color:#3b82f6;font-weight:700;">✔</span>
      </div>

      <div class="folder-tag preset-card" id="preset_minimax_h3" onclick="selectPresetModel('preset_minimax_h3', 'MiniMaxAI/MiniMax-H3', 'MiniMax-H3', 'FL2VA/*.safetensors')" style="margin:0;padding:8px 10px;justify-content:space-between;">
        <div>
          <div style="font-weight:700;color:#f8fafc;font-size:13px;">🎬 MiniMax-H3 (33B · 无审查)</div>
          <div style="color:var(--muted);font-size:11px;">BF16 原生 + 破限编码器 · 134 GB</div>
        </div>
        <span class="preset-check" style="color:#3b82f6;font-weight:700;display:none;">✔</span>
      </div>

      <div class="folder-tag preset-card" id="preset_minimax_image" onclick="selectPresetModel('preset_minimax_image', 'Comfy-Org/MiniMax-H3-Image', 'MiniMax-H3-Image', '*.safetensors')" style="margin:0;padding:8px 10px;justify-content:space-between;">
        <div>
          <div style="font-weight:700;color:#f8fafc;font-size:13px;">📸 MiniMax-H3-Image (4K无审查)</div>
          <div style="color:var(--muted);font-size:11px;">INT8 + Heretic破限编码器 · 45.2 GB (4K生图/免Mask换装)</div>
        </div>
        <span class="preset-check" style="color:#3b82f6;font-weight:700;display:none;">✔</span>
      </div>

      <div class="folder-tag preset-card" id="preset_longcat_avatar" onclick="selectPresetModel('preset_longcat_avatar', 'meituan-longcat/LongCat-Video-Avatar-1.6', 'LongCat-Video-Avatar-1.6', 'base_model/*.safetensors')" style="margin:0;padding:8px 10px;justify-content:space-between;">
        <div>
          <div style="font-weight:700;color:#f8fafc;font-size:13px;">🎭 LongCat-Video-Avatar-1.6</div>
          <div style="color:var(--muted);font-size:11px;">BF16 + 双LoRA + 底座 · 58.8 GB (数字人视频)</div>
        </div>
        <span class="preset-check" style="color:#3b82f6;font-weight:700;display:none;">✔</span>
      </div>
    </div>

    <!-- T4 16GB 模型区 -->
    <div style="display:none;grid-template-columns:repeat(2, 1fr);gap:8px;margin-bottom:12px;" id="presetModelGridT4">
      <div class="folder-tag preset-card" id="preset_t4_minimax_h3" onclick="selectPresetModel('preset_t4_minimax_h3', 'unsloth/MiniMax-H3-GGUF', 'MiniMax-H3-T4-GGUF', 'minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf')" style="margin:0;padding:8px 10px;justify-content:space-between;">
        <div>
          <div style="font-weight:700;color:#34d399;font-size:13px;">🎬 MiniMax-H3 (T4 量化版)</div>
          <div style="color:var(--muted);font-size:11px;">UD-Q2_K_XL 蒸馏 · 7.5 GB (8步有声)</div>
        </div>
        <span class="preset-check" style="color:#10b981;font-weight:700;display:none;">✔</span>
      </div>

      <div class="folder-tag preset-card" id="preset_t4_longcat_int8" onclick="selectPresetModel('preset_t4_longcat_int8', 'meituan-longcat/LongCat-Video-Avatar-1.6', 'LongCat-Video-Avatar-1.6-INT8', 'base_model_int8/*.safetensors')" style="margin:0;padding:8px 10px;justify-content:space-between;">
        <div>
          <div style="font-weight:700;color:#34d399;font-size:13px;">🎭 LongCat-Avatar-1.6 (INT8版)</div>
          <div style="color:var(--muted);font-size:11px;">INT8 量化主干 + 双LoRA · 14.8 GB (数字人)</div>
        </div>
        <span class="preset-check" style="color:#10b981;font-weight:700;display:none;">✔</span>
      </div>

      <div class="folder-tag preset-card" id="preset_t4_recavsr" onclick="selectPresetModel('preset_t4_recavsr', 'Decorator/ReCaVSR', 'ReCaVSR', '*.pth')" style="margin:0;padding:8px 10px;justify-content:space-between;">
        <div>
          <div style="font-weight:700;color:#34d399;font-size:13px;">✨ ReCaVSR (2K/4K 视频超分)</div>
          <div style="color:var(--muted);font-size:11px;">Wan2.2-5B 单步超分 · 10.1 GB (云盘已存)</div>
        </div>
        <span class="preset-check" style="color:#10b981;font-weight:700;display:none;">✔</span>
      </div>

      <div class="folder-tag preset-card" id="preset_t4_qwen38_q3" onclick="selectPresetModel('preset_t4_qwen38_q3', 'huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF', 'Qwen3.8-27B-T4-Q3', 'Huihui-Qwen3.8-27B-abliterated-UD-Q3_K_XL.gguf')" style="margin:0;padding:8px 10px;justify-content:space-between;">
        <div>
          <div style="font-weight:700;color:#34d399;font-size:13px;">🧠 Qwen3.8-27B 无审查 (T4版)</div>
          <div style="color:var(--muted);font-size:11px;">UD-Q3_K_XL · 12.4 GB (16G显存满速)</div>
        </div>
        <span class="preset-check" style="color:#10b981;font-weight:700;display:none;">✔</span>
      </div>

      <div class="folder-tag preset-card" id="preset_t4_qwen38_mtp" onclick="selectPresetModel('preset_t4_qwen38_mtp', 'huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF', 'Qwen3.8-27B-T4-MTP', 'Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_S-mtp.gguf')" style="margin:0;padding:8px 10px;justify-content:space-between;">
        <div>
          <div style="font-weight:700;color:#34d399;font-size:13px;">⚡ Qwen3.8-27B 无审查 (MTP版)</div>
          <div style="color:var(--muted);font-size:11px;">IQ3_S + 原生MTP投机 · 11.3 GB</div>
        </div>
        <span class="preset-check" style="color:#10b981;font-weight:700;display:none;">✔</span>
      </div>

      <div class="folder-tag preset-card" id="preset_t4_qwen14b" onclick="selectPresetModel('preset_t4_qwen14b', 'bartowski/huihui-ai_Qwen3-14B-abliterated-GGUF', 'Qwen3-14B-T4-Q6', 'huihui-ai_Qwen3-14B-abliterated-Q6_K.gguf')" style="margin:0;padding:8px 10px;justify-content:space-between;">
        <div>
          <div style="font-weight:700;color:#34d399;font-size:13px;">💎 Qwen3-14B 无审查 (Q6_K)</div>
          <div style="color:var(--muted);font-size:11px;">准无损精度 · 11.3 GB (~50 tok/s)</div>
        </div>
        <span class="preset-check" style="color:#10b981;font-weight:700;display:none;">✔</span>
      </div>

      <div class="folder-tag preset-card" id="preset_t4_qwenvl8b" onclick="selectPresetModel('preset_t4_qwenvl8b', 'mradermacher/Qwen3-VL-8B-Instruct-abliterated-v2.0-GGUF', 'Qwen3-VL-8B-T4-Q8', '*Q8_0.gguf')" style="margin:0;padding:8px 10px;justify-content:space-between;">
        <div>
          <div style="font-weight:700;color:#34d399;font-size:13px;">👁️ Qwen3-VL-8B 视觉无审查</div>
          <div style="color:var(--muted);font-size:11px;">满血 Q8_0 + 视觉 · 8.8 GB (~75 tok/s)</div>
        </div>
        <span class="preset-check" style="color:#10b981;font-weight:700;display:none;">✔</span>
      </div>
    </div>

    <div id="customPresetGrid" style="display:grid;grid-template-columns:repeat(2, 1fr);gap:8px;margin-bottom:8px;"></div>

    <div onclick="openCustomHFModal()" style="display:flex;align-items:center;justify-content:space-between;padding:10px 14px;margin-bottom:12px;border-radius:10px;border:1.5px dashed rgba(59,130,246,0.55);background:rgba(59,130,246,0.08);cursor:pointer;transition:all 0.15s;" onmouseover="this.style.background='rgba(59,130,246,0.15)'" onmouseout="this.style.background='rgba(59,130,246,0.08)'">
      <div style="display:flex;align-items:center;gap:10px;color:#60a5fa;font-weight:700;font-size:13.5px;">
        <span style="display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;border-radius:999px;background:#3b82f6;color:white;font-size:15px;font-weight:800;line-height:1;">+</span>
        <span>添加自定义 HuggingFace 模型...</span>
      </div>
      <span style="font-family:'SF Mono',Menlo,monospace;font-size:12px;color:var(--muted);">GGUF / Safetensors</span>
    </div>

    <div style="margin-bottom:10px;">
      <label style="font-size:12px;color:var(--muted);display:block;margin-bottom:4px;">☁️ 当前您的 Google Drive `MyDrive/models/` 下已保存的模型文件夹（点击直接选中加载）：</label>
      <div id="driveFoldersList" style="min-height:28px;font-size:12px;color:var(--muted);">正在读取云盘目录...</div>
    </div>

    <div class="form-group">
      <label>1. HuggingFace 仓库 ID（或直接粘贴任意 HF 模型链接）</label>
      <input id="inpRepo" type="text" value="huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF">
    </div>

    <div style="display:grid;grid-template-columns: 1fr 1.2fr 0.7fr;gap:8px;">
      <div class="form-group">
        <label>2. 云盘 `MyDrive/models/` 子文件夹</label>
        <input id="inpFolder" type="text" value="Huihui-Qwen3.8-27B-abliterated-GGUF" oninput="syncDriveFolderHighlight(this.value.trim())">
      </div>
      <div class="form-group">
        <label>3. 权重文件名或通配符</label>
        <input id="inpPattern" type="text" value="Huihui-Qwen3.8-27B-abliterated-bf16.gguf">
      </div>
      <div class="form-group">
        <label>4. 上下文</label>
        <input id="inpCtx" type="number" value="262144">
      </div>
    </div>

    <div style="display:grid;grid-template-columns: 1fr 1.25fr;gap:10px;margin-top:4px;">
      <button class="btn-outline" style="width:100%;justify-content:center;padding:10px;border-color:rgba(16,185,129,0.5);color:#6ee7b7;" onclick="startDeploy(true)" id="btnDownloadOnly" title="该模型已保存在您的 Google Drive 云盘中；点击仅校验或补全云盘缺失文件，不加载进显存">
        ✅ 云盘已存妥 (校验/补全文件)
      </button>
      <button class="btn-primary" style="width:100%;justify-content:center;padding:10px;" onclick="startDeploy(false)" id="btnDeployForm" title="直接从您的 Google Drive 云盘极速预热并装载至 GPU 显存启动运行（无需从 HF 重新下载）">
        🚀 从云盘加载至显存并启动 (Qwen3.8-27B)
      </button>
    </div>
  </div>
</div>

<!-- 单独一行：EasyColab 宽屏全高布局 · MiniMax-H3 (33B · 原生BF16 · 无审查) 多模态有声视频创作工作台 -->
<div class="card" id="minimaxVideoBox" style="margin-bottom:16px;border:1px solid rgba(59,130,246,0.35);background:linear-gradient(180deg, rgba(30,41,59,0.98) 0%, rgba(15,23,42,0.98) 100%);">
  <div style="display:flex;flex-direction:column;gap:12px;margin-bottom:16px;padding-bottom:14px;border-bottom:1px solid var(--border);">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
        <span id="videoStudioTitle" style="font-size:16.5px;font-weight:900;color:#f8fafc;">🎬 MiniMax-H3 & LongCat-Avatar 全流程影视与视觉创作中心</span>
        <span id="videoStudioBadge" style="font-size:11.5px;background:rgba(16,185,129,0.16);color:#34d399;padding:3px 10px;border-radius:999px;font-weight:700;">①生图定妆 ➔ ②换脸换装融景 ➔ ③④⑤单镜视频 ➔ ⑥全剧自动导演</span>
      </div>
    </div>

    <!-- 6 大功能按创作流水线序列一字排开 (6列等宽醒目导航卡：01生图 -> 02换脸换装融景 -> 03文生视频 -> 04首尾帧视频 -> 05数字人说话 -> 06全自动短剧导演) -->
    <div style="display:grid;grid-template-columns:repeat(6, minmax(0, 1fr));gap:8px;width:100%;">
      <!-- 01: 4K写真生图 -->
      <button type="button" id="tabBtnImage" onclick="switchMiniMaxMode('image')" style="text-align:left;padding:9px 11px;border-radius:10px;border:1.5px solid #475569;background:rgba(15,23,42,0.85);cursor:pointer;display:flex;flex-direction:column;gap:3px;transition:all 0.18s;">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:4px;">
          <span style="font-size:10px;font-weight:900;padding:1px 6px;border-radius:4px;background:rgba(244,114,182,0.22);color:#f472b6;">01 · 图像创作</span>
          <span style="font-size:14px;">📸</span>
        </div>
        <div style="font-size:13px;font-weight:800;color:#f8fafc;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">4K 写真生图</div>
        <div style="font-size:10.5px;color:#94a3b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">H3-Image · 文本直出大图</div>
      </button>

      <!-- 02: 智能换脸·换装·场景融合 -->
      <button type="button" id="tabBtnFusion" onclick="switchMiniMaxMode('fusion')" style="text-align:left;padding:9px 11px;border-radius:10px;border:1.5px solid #475569;background:rgba(15,23,42,0.85);cursor:pointer;display:flex;flex-direction:column;gap:3px;transition:all 0.18s;">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:4px;">
          <span style="font-size:10px;font-weight:900;padding:1px 6px;border-radius:4px;background:rgba(192,132,252,0.22);color:#c084fc;">02 · 角色合成</span>
          <span style="font-size:14px;">🪄</span>
        </div>
        <div style="font-size:13px;font-weight:800;color:#f8fafc;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">换脸·换装·融景</div>
        <div style="font-size:10.5px;color:#94a3b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">人脸+服装+实景三合一</div>
      </button>

      <!-- 03: 文生音视频 (T2VA) -->
      <button type="button" id="tabBtnT2va" onclick="switchMiniMaxMode('t2va')" style="text-align:left;padding:9px 11px;border-radius:10px;border:1.5px solid #3b82f6;background:linear-gradient(135deg,rgba(37,99,235,0.35),rgba(15,23,42,0.95));box-shadow:0 6px 16px -4px rgba(59,130,246,0.45);cursor:pointer;display:flex;flex-direction:column;gap:3px;transition:all 0.18s;">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:4px;">
          <span style="font-size:10px;font-weight:900;padding:1px 6px;border-radius:4px;background:rgba(59,130,246,0.28);color:#93c5fd;">03 · 纯文生视频</span>
          <span style="font-size:14px;">✨</span>
        </div>
        <div style="font-size:13px;font-weight:800;color:#ffffff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">文生有声视频</div>
        <div style="font-size:10.5px;color:#bfdbfe;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">T2VA · 提示词直出音视频</div>
      </button>

      <!-- 04: 首尾帧生音视频 (FL2VA) -->
      <button type="button" id="tabBtnFl2va" onclick="switchMiniMaxMode('fl2va')" style="text-align:left;padding:9px 11px;border-radius:10px;border:1.5px solid #475569;background:rgba(15,23,42,0.85);cursor:pointer;display:flex;flex-direction:column;gap:3px;transition:all 0.18s;">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:4px;">
          <span style="font-size:10px;font-weight:900;padding:1px 6px;border-radius:4px;background:rgba(56,189,248,0.22);color:#38bdf8;">04 · 图生视频</span>
          <span style="font-size:14px;">🖼️</span>
        </div>
        <div style="font-size:13px;font-weight:800;color:#f8fafc;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">首尾帧有声视频</div>
        <div style="font-size:10.5px;color:#94a3b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">FL2VA · 首尾图精准运镜</div>
      </button>

      <!-- 05: 数字人音频驱动 (LongCat-Avatar) -->
      <button type="button" id="tabBtnAvatar" onclick="switchMiniMaxMode('avatar')" style="text-align:left;padding:9px 11px;border-radius:10px;border:1.5px solid #475569;background:rgba(15,23,42,0.85);cursor:pointer;display:flex;flex-direction:column;gap:3px;transition:all 0.18s;">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:4px;">
          <span style="font-size:10px;font-weight:900;padding:1px 6px;border-radius:4px;background:rgba(16,185,129,0.22);color:#34d399;">05 · 口型配音</span>
          <span style="font-size:14px;">🎭</span>
        </div>
        <div style="font-size:13px;font-weight:800;color:#f8fafc;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">数字人配音说话</div>
        <div style="font-size:10.5px;color:#94a3b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">LongCat + Ming3.0 驱动</div>
      </button>

      <!-- 06: 短剧分镜全自动导演台 -->
      <button type="button" id="tabBtnAutoDrama" onclick="switchMiniMaxMode('drama')" style="text-align:left;padding:9px 11px;border-radius:10px;border:1.5px solid #475569;background:rgba(15,23,42,0.85);cursor:pointer;display:flex;flex-direction:column;gap:3px;transition:all 0.18s;">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:4px;">
          <span style="font-size:10px;font-weight:900;padding:1px 6px;border-radius:4px;background:rgba(245,158,11,0.24);color:#fbbf24;">06 · 全剧集总控</span>
          <span style="font-size:14px;">🎬</span>
        </div>
        <div style="font-size:13px;font-weight:800;color:#f8fafc;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">短剧全自动导演</div>
        <div style="font-size:10.5px;color:#94a3b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">多角色锁脸+全剧自动合片</div>
      </button>
    </div>
  </div>

  <!-- 第 6 大功能：🪄 极简傻瓜式「换脸 · 换装 · 场景融合」工坊 (支持多图动态增删、左右双图一键拆分、一次批量合成全部) -->
  <div id="fusionStudioRow" style="display:none;padding:14px 16px;margin-bottom:16px;background:linear-gradient(135deg, rgba(15,23,42,0.98) 0%, rgba(46,16,101,0.85) 100%);border:1.5px solid #c084fc;border-radius:12px;">
    <!-- 第 1 步：选模式 (3 个直观大按钮) -->
    <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px;">
      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
        <span style="font-size:14px;font-weight:900;color:#f3e8ff;">🪄 选择合成模式：</span>
        <button type="button" id="btnFusionModeTryon" onclick="switchFusionSubMode('tryon')" style="padding:6px 14px;font-size:12.5px;font-weight:800;border-radius:8px;border:1.5px solid #c084fc;background:linear-gradient(135deg,#9333ea,#c026d3);color:#ffffff;cursor:pointer;">👗 1. 人脸 + 服装合成</button>
        <button type="button" id="btnFusionModeFaceswap" onclick="switchFusionSubMode('faceswap')" style="padding:6px 14px;font-size:12.5px;font-weight:700;border-radius:8px;border:1px solid #475569;background:#0f172a;color:#cbd5e1;cursor:pointer;">🎭 2. 人脸替换他人照片 (换脸)</button>
        <button type="button" id="btnFusionModeScenefuse" onclick="switchFusionSubMode('scenefuse')" style="padding:6px 14px;font-size:12.5px;font-weight:700;border-radius:8px;border:1px solid #475569;background:#0f172a;color:#cbd5e1;cursor:pointer;">🏞️ 3. 融入场景照片</button>
      </div>
      <button type="button" onclick="toggleFusionAdvancedDrawer()" id="btnToggleFusionAdv" style="padding:4px 10px;font-size:11.5px;font-weight:700;border-radius:6px;border:1px solid #475569;background:rgba(15,23,42,0.8);color:#cbd5e1;cursor:pointer;">⚙️ 高级微调 (默认自动最优) ▾</button>
    </div>

    <!-- 第 2 步：上传图片框 (左侧固定人脸主照；右侧支持 1~N 张自由新增/删除/批量上传/双联图一键拆分) -->
    <div id="fusionSlotsGrid" style="display:grid;grid-template-columns:1fr 1.65fr;gap:12px;margin-bottom:12px;">
      <!-- 框 1：🧑 上传人脸照片 -->
      <div style="padding:10px 12px;border-radius:10px;background:rgba(2,6,23,0.85);border:1px solid #a855f7;display:flex;flex-direction:column;gap:6px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <span id="fusionSlot1Title" style="font-size:13px;font-weight:800;color:#e9d5ff;">🧑 第一步：上传人脸照片</span>
          <button type="button" onclick="clearFusionSlot(0)" style="padding:2px 8px;font-size:11px;border-radius:4px;border:1px solid #475569;background:transparent;color:#94a3b8;cursor:pointer;">✕ 清空</button>
        </div>
        <label style="position:relative;height:165px;border-radius:8px;border:1.5px dashed #7e22ce;background:#0f172a;display:flex;flex-direction:column;align-items:center;justify-content:center;cursor:pointer;overflow:hidden;">
          <img id="fusionSlot1Img" style="display:none;width:100%;height:100%;object-fit:contain;background:#020617;">
          <div id="fusionSlot1Empty" style="text-align:center;color:#94a3b8;font-size:12px;padding:10px;">
            <div style="font-size:28px;margin-bottom:4px;">🧑</div>
            <div style="color:#e9d5ff;font-weight:700;">点击上传【目标人脸照片】</div>
            <div style="font-size:11px;color:#64748b;margin-top:2px;">100% 锁定该人物的五官骨相与脸型</div>
          </div>
          <input type="file" id="inpFusionSlot1" accept="image/*" style="display:none;" onchange="onFusionSlotFileChange(0, this)">
        </label>
      </div>

      <!-- 框 2 (支持动态 1~N 张自由增删)：👗 上传服装图 / 🖼️ 上传被换脸照片列表 -->
      <div id="fusionSlot2Card" style="padding:10px 12px;border-radius:10px;background:rgba(2,6,23,0.85);border:1px solid #ec4899;display:flex;flex-direction:column;gap:6px;">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap;">
          <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
            <span id="fusionSlot2Title" style="font-size:13px;font-weight:800;color:#fbcfe8;">👗 第二步：上传服装/穿搭参考图</span>
            <span id="fusionTargetCountBadge" style="font-size:11px;padding:1px 7px;border-radius:999px;background:rgba(236,72,153,0.2);color:#f9a8d4;font-weight:700;">1 个槽位</span>
          </div>
          <div style="display:flex;align-items:center;gap:5px;flex-wrap:wrap;">
            <select id="fusionMultiActionMode" onchange="updateFusionRunBtnLabel()" style="display:none;padding:2px 6px;font-size:11px;font-weight:700;border-radius:5px;border:1px solid #c084fc;background:#0f172a;color:#e9d5ff;" title="当上传了 2 张以上照片时的合成方式">
              <option value="batch" selected>🗂️ 逐张批量生成 (每张图各出1张成片)</option>
              <option value="combine">🧩 多图合并为1张 (如:上衣+下装合为1张)</option>
            </select>
            <button type="button" onclick="addFusionTargetSlot()" style="padding:2px 8px;font-size:11px;font-weight:800;border-radius:5px;border:1px solid #10b981;background:rgba(16,185,129,0.18);color:#6ee7b7;cursor:pointer;" title="增加一个新的照片槽位（支持 2张、3张或任意多张）">➕ 新增照片槽位</button>
            <label style="padding:2px 8px;font-size:11px;font-weight:700;border-radius:5px;border:1px solid #38bdf8;background:rgba(56,189,248,0.16);color:#7dd3fc;cursor:pointer;" title="在文件选择器中一次按住 Command/Shift 勾选多张照片批量导入">
              📂 批量多选导入
              <input type="file" accept="image/*" multiple style="display:none;" onchange="onFusionMultiFilesSelect(this)">
            </label>
            <button type="button" onclick="clearAllFusionTargets()" style="padding:2px 8px;font-size:11px;border-radius:4px;border:1px solid #475569;background:transparent;color:#94a3b8;cursor:pointer;">✕ 清空全部</button>
          </div>
        </div>
        <!-- 动态照片槽位容器：1张时占满，2~N张时自动并排网格展示，每张均可独立上传、左右拆双图或点 ✕ 删除 -->
        <div id="fusionTargetCardsGrid" style="display:grid;grid-template-columns:1fr;gap:8px;max-height:190px;overflow-y:auto;padding-right:2px;"></div>
      </div>

      <!-- 框 3：🏞️ 上传场景照片 (仅在选“3. 融入场景照片”时自动显示！) -->
      <div id="fusionSlot3Card" style="display:none;padding:10px 12px;border-radius:10px;background:rgba(2,6,23,0.85);border:1px solid #38bdf8;flex-direction:column;gap:6px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <span id="fusionSlot3Title" style="font-size:13px;font-weight:800;color:#bae6fd;">🏞️ 第三步：上传场景照片</span>
          <button type="button" onclick="clearFusionSlot(2)" style="padding:2px 8px;font-size:11px;border-radius:4px;border:1px solid #475569;background:transparent;color:#94a3b8;cursor:pointer;">✕ 清空</button>
        </div>
        <label style="position:relative;height:165px;border-radius:8px;border:1.5px dashed #0284c7;background:#0f172a;display:flex;flex-direction:column;align-items:center;justify-content:center;cursor:pointer;overflow:hidden;">
          <img id="fusionSlot3Img" style="display:none;width:100%;height:100%;object-fit:contain;background:#020617;">
          <div id="fusionSlot3Empty" style="text-align:center;color:#94a3b8;font-size:12px;padding:10px;">
            <div style="font-size:28px;margin-bottom:4px;">🏞️</div>
            <div id="fusionSlot3MainHint" style="color:#bae6fd;font-weight:700;">点击上传【场景环境照片】</div>
            <div id="fusionSlot3SubHint" style="font-size:11px;color:#64748b;margin-top:2px;">自动把人物融入该实景并匹配光线</div>
          </div>
          <input type="file" id="inpFusionSlot3" accept="image/*" style="display:none;" onchange="onFusionSlotFileChange(2, this)">
        </label>
      </div>
    </div>

    <!-- 第 3 步 (仅一行极简操作栏)：一句话补充要求(选填) + 画幅(默认自动跟随原图比例) + 🪄 一键开始合成 + 停止 + 锁入短剧角色 -->
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
      <input type="text" id="fusionPropsActionInput" placeholder="💬 补充要求 (选填，不填即自动合成；也可写如：保留白吊带与高腰牛仔裤坐姿、拿威士忌杯...)" style="flex:1;min-width:230px;padding:8px 12px;font-size:12.5px;border-radius:8px;border:1px solid #475569;background:#0f172a;color:#f8fafc;">
      <select id="fusionResSelect" style="padding:8px 10px;font-size:12.5px;font-weight:700;border-radius:8px;border:1px solid #c084fc;background:#0f172a;color:#f8fafc;">
        <option value="auto" selected>📐 自动匹配参考图比例 (横/竖自适应·推荐)</option>
        <option value="1152x2048">📱 9:16 竖屏写真 (1152×2048)</option>
        <option value="2048x1152">📺 16:9 横屏剧集 (2048×1152)</option>
        <option value="3552x4736">🖼️ 3:4 4K超清竖版 (3552×4736)</option>
      </select>
      <button type="button" id="btnRunFusionStudio" onclick="startFusionStudioGenerate()" style="padding:8px 20px;font-size:13.5px;font-weight:900;border-radius:8px;border:none;background:linear-gradient(135deg,#a855f7,#ec4899,#f59e0b);color:#ffffff;cursor:pointer;box-shadow:0 6px 18px -4px rgba(236,72,153,0.55);white-space:nowrap;">
        🪄 一键开始合成
      </button>
      <button type="button" id="btnStopFusionStudio" onclick="stopFusionBatchGenerate()" style="display:none;padding:8px 14px;font-size:12.5px;font-weight:800;border-radius:8px;border:1px solid #ef4444;background:rgba(239,68,68,0.22);color:#fca5a5;cursor:pointer;white-space:nowrap;">
        🛑 停止合成
      </button>
      <div style="display:flex;align-items:center;gap:4px;background:rgba(15,23,42,0.9);padding:4px 8px;border-radius:8px;border:1px solid rgba(16,185,129,0.4);">
        <select id="selFusionTargetRole" style="padding:3px 5px;font-size:11.5px;font-weight:700;border-radius:5px;border:1px solid #10b981;background:#020617;color:#a7f3d0;">
          <option value="A">角色A</option>
          <option value="B">角色B</option>
          <option value="C">角色C</option>
          <option value="D">角色D</option>
          <option value="E">角色E</option>
        </select>
        <button type="button" onclick="sendCurrentPreviewToDramaRole()" style="padding:4px 8px;font-size:11.5px;font-weight:800;border-radius:5px;border:1px solid #10b981;background:rgba(16,185,129,0.22);color:#6ee7b7;cursor:pointer;" title="把生成好的图直接设为短剧对应角色定妆照">📌设为角色定妆</button>
      </div>
    </div>

    <!-- 角色合成专属实时进度条与状态提示 (绝不跳回 01 标签页) -->
    <div id="fusionProgressWrap" style="display:none;margin-top:10px;">
      <div class="progress-bar" style="height:7px;background:#0f172a;"><div class="progress-fill" id="fusionProgressBar" style="width:8%;background:linear-gradient(90deg,#a855f7,#ec4899,#10b981);"></div></div>
    </div>
    <div id="fusionStudioStatusText" style="margin-top:6px;font-size:12px;font-weight:700;color:#cbd5e1;"></div>

    <!-- 批量合成结果缩略图画廊 (当生成 1~N 张后在此横向展示，点击任意一张即可在下方大屏查看/流转) -->
    <div id="fusionBatchResultsBar" style="display:none;margin-top:10px;padding:8px 10px;border-radius:8px;background:rgba(2,6,23,0.78);border:1px solid rgba(192,132,252,0.4);"></div>

    <!-- 可选展开：⚙️ 高级微调抽屉 (默认 display:none 隐藏，点右上角「⚙️ 高级微调」才展开) -->
    <div id="fusionAdvancedDrawer" style="display:none;margin-top:10px;padding:10px 12px;border-radius:8px;background:rgba(2,6,23,0.85);border:1px dashed #64748b;">
      <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:8px;">
        <div>
          <label style="font-size:10.5px;color:#cbd5e1;display:block;margin-bottom:3px;">🎭 表情与脸型保持</label>
          <select id="fusionExprSelect" style="width:100%;padding:4px 6px;font-size:11px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#f8fafc;">
            <option value="keep_orig" selected>📸 保持人脸原图自然表情与脸型 (推荐)</option>
            <option value="natural_closed">😌 自然平静闭唇 (数字人底图)</option>
            <option value="gentle_smile">😊 温和浅笑 (不露齿)</option>
            <option value="boss_serious">😠 冷峻威严 (大佬气场)</option>
            <option value="sly_smirk">😏 狡黠似笑非笑</option>
            <option value="sad_teary">😢 眼眶微红忧郁</option>
          </select>
        </div>
        <div>
          <label style="font-size:10.5px;color:#cbd5e1;display:block;margin-bottom:3px;">👁️ 视线方向</label>
          <select id="fusionGazeSelect" style="width:100%;padding:4px 6px;font-size:11px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#f8fafc;">
            <option value="keep_target" selected>🖼️ 跟随姿势与构图视线 (推荐)</option>
            <option value="front_camera">👁️ 正面平视镜头</option>
            <option value="side_15_right">↗️ 微侧15°看右侧 (正反打)</option>
            <option value="side_15_left">↖️ 微侧15°看左侧 (正反打)</option>
          </select>
        </div>
        <div>
          <label style="font-size:10.5px;color:#cbd5e1;display:block;margin-bottom:3px;">🧍 体型身材</label>
          <select id="fusionBodySelect" style="width:100%;padding:4px 6px;font-size:11px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#f8fafc;">
            <option value="standard" selected>🖼️ 100% 跟随姿势底图身材 (推荐)</option>
            <option value="boss_stocky">🐻 敦实微胖大佬体型</option>
            <option value="athletic_tall">💪 宽肩健硕体型</option>
            <option value="slim_youth">🌿 清瘦修长少年体型</option>
            <option value="curvy_tall">👗 高挑曼妙女性体型</option>
          </select>
        </div>
        <div>
          <label style="font-size:10.5px;color:#cbd5e1;display:block;margin-bottom:3px;">📐 姿势与景别范围</label>
          <select id="fusionFramingSelect" style="width:100%;padding:4px 6px;font-size:11px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#f8fafc;">
            <option value="keep_target_frame" selected>🖼️ 100% 保持参考图原有姿势与构图 (推荐)</option>
            <option value="waist_medium">🧍 腰部/大腿以上中景</option>
            <option value="bust_half">📸 胸口以上半身 (定妆特写)</option>
            <option value="closeup_face">🔍 面部近景特写</option>
            <option value="full_body">🏞️ 全身融入场景</option>
          </select>
        </div>
        <div>
          <label style="font-size:10.5px;color:#cbd5e1;display:block;margin-bottom:3px;">💡 背景与光影</label>
          <select id="fusionLightingSelect" style="width:100%;padding:4px 6px;font-size:11px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#f8fafc;">
            <option value="keep_target_light" selected>🖼️ 100% 保持参考图背景与光影 (推荐)</option>
            <option value="cinema_85mm_bokeh">🎬 85mm电影浅景深虚化</option>
            <option value="clear_35mm_scene">🏞️ 35mm场景清晰融入</option>
            <option value="dramatic_noir">🕯️ 港风暗调高反差光影</option>
            <option value="soft_natural">☀️ 柔和均匀自然光</option>
          </select>
        </div>
        <div>
          <label style="font-size:10.5px;color:#cbd5e1;display:block;margin-bottom:3px;">💇 发型保留策略</label>
          <select id="fusionHairModeSelect" style="width:100%;padding:4px 6px;font-size:11px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#f8fafc;">
            <option value="keep_face_hair" selected>💇 使用【人脸照】的发型</option>
            <option value="keep_target_hair">👑 保留【被换脸原图】的发型/头饰</option>
          </select>
        </div>
        <div>
          <label style="font-size:10.5px;color:#cbd5e1;display:block;margin-bottom:3px;">👓 眼镜与胡须</label>
          <select id="fusionAccessorySelect" style="width:100%;padding:4px 6px;font-size:11px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#f8fafc;">
            <option value="auto_face" selected>👓 自动跟随人脸原图</option>
            <option value="no_glasses">🚫 强制去除眼镜</option>
            <option value="thin_glasses">👓 戴金丝细框眼镜</option>
            <option value="sunglasses">🕶️ 戴墨镜</option>
            <option value="stubble_beard">🧔 加上青色胡茬</option>
            <option value="clean_shaven">✨ 清爽无胡须</option>
          </select>
        </div>
        <div>
          <label style="font-size:10.5px;color:#cbd5e1;display:block;margin-bottom:3px;">🎲 单张连抽对比</label>
          <select id="fusionBatchSelect" onchange="updateFusionRunBtnLabel()" style="width:100%;padding:4px 6px;font-size:11px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#fde68a;">
            <option value="1" selected>每槽位生成 1 张 (极速)</option>
            <option value="2">每槽位连抽 2 张对比</option>
            <option value="4">每槽位连抽 4 张挑最优</option>
          </select>
        </div>
      </div>
    </div>
  </div>

  <!-- 通用多分镜短剧全自动导演流水线面板 (仅在选中第 5 个标签 drama 时显示 · 不限分镜数与总时长) -->
  <div id="autoDramaDirectorRow" style="display:none;padding:14px 16px;margin-bottom:16px;background:linear-gradient(135deg, rgba(15,23,42,0.98) 0%, rgba(30,27,75,0.92) 100%);border:1.5px solid #f59e0b;border-radius:12px;">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px;">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
        <span id="dramaTitleHeader" style="font-size:14.5px;font-weight:800;color:#fde68a;">🎬 通用分镜短剧全自动工作流引擎</span>
        <span style="font-size:11px;padding:2px 9px;border-radius:999px;background:rgba(245,158,11,0.2);color:#fbbf24;font-weight:700;">支持外部 AI 生成任意时长剧本直接粘贴 · 单镜独立重抽 · 0.1秒卡点合片</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
        <button type="button" id="btnCopyExtAiTop" onclick="copyExternalAiScriptPrompt(this)" style="padding:4px 11px;font-size:12px;font-weight:700;border-radius:6px;border:1px solid #a855f7;background:rgba(168,85,247,0.18);color:#e9d5ff;cursor:pointer;transition:all 0.2s;" title="一键复制发给 ChatGPT / Claude / DeepSeek / Gemini / Kimi 的分镜剧本生成指令">🤖 复制外部 AI 编剧指令</button>
        <button type="button" onclick="toggleDramaWorkflowEditor()" style="padding:4px 11px;font-size:12px;font-weight:700;border-radius:6px;border:1px solid #38bdf8;background:rgba(56,189,248,0.16);color:#7dd3fc;cursor:pointer;">📜 展开/收起剧本与角色工作流工作台 ▾</button>
        <select id="dramaAspectSelect" style="padding:4px 10px;font-size:12px;border-radius:6px;border:1px solid #475569;background:#0f172a;color:#f8fafc;font-weight:600;">
          <option value="16:9" selected>📺 16:9 横屏电影剧集 (推荐)</option>
          <option value="9:16">📱 9:16 竖屏短剧</option>
        </select>
        <label style="display:inline-flex;align-items:center;gap:5px;font-size:11.5px;color:#cbd5e1;cursor:pointer;padding:3px 8px;border-radius:6px;background:rgba(255,255,255,0.05);">
          <input type="checkbox" id="chkDramaResume" checked style="accent-color:#10b981;">
          <span>♻️ 断点续拍 (自动复用已生好的定妆照/分镜)</span>
        </label>
        <label style="display:inline-flex;align-items:center;gap:5px;font-size:11.5px;color:#c4b5fd;cursor:pointer;padding:3px 8px;border-radius:6px;background:rgba(139,92,246,0.12);">
          <input type="checkbox" id="chkDramaAuto2K" style="accent-color:#8b5cf6;">
          <span>✨ 成片后自动 ReCaVSR 2K 超分</span>
        </label>
      </div>
    </div>

    <!-- 默认展开：通用剧本解析器 + 外部AI提示词生成器 + 可视化角色/分镜工作流节点编辑器 -->
    <div id="dramaWorkflowEditorDrawer" style="display:block;margin-bottom:12px;padding:12px;border-radius:10px;background:rgba(2,6,23,0.82);border:1px solid #38bdf8;">
      <!-- 顶部：外部在线 AI (ChatGPT / Claude / DeepSeek / Gemini / Kimi) 编剧指令一键生成器 -->
      <div style="margin-bottom:10px;padding:8px 10px;border-radius:8px;background:rgba(88,28,135,0.22);border:1px dashed #a855f7;display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;">
        <div style="display:flex;align-items:center;gap:6px;flex:1;flex-wrap:wrap;">
          <span style="font-size:12px;font-weight:800;color:#e9d5ff;white-space:nowrap;">🤖 让外部 AI 帮写分镜剧本：</span>
          <input type="text" id="inpAiIdeaTheme" placeholder="输入一句话剧情/题材（如：古装仙侠雪夜竹林定情 / 现代豪门雨夜重逢，留空也可）" style="flex:1;min-width:210px;padding:4px 8px;font-size:11.5px;border-radius:5px;border:1px solid #6b21a8;background:#0f172a;color:#f8fafc;">
          <select id="selAiTargetDur" style="padding:4px 8px;font-size:11.5px;border-radius:5px;border:1px solid #6b21a8;background:#0f172a;color:#f8fafc;font-weight:600;">
            <option value="不限总时长（由剧情节奏自然决定镜头数与秒数）" selected>⏱️ 不限总时长 (按剧情自由决定)</option>
            <option value="约 15 秒（建议 4~6 个分镜）">⏱️ 约 15 秒 (4~6 镜)</option>
            <option value="约 30 秒（建议 8~12 个分镜）">⏱️ 约 30 秒 (8~12 镜)</option>
            <option value="约 45 秒（建议 12~16 个分镜）">⏱️ 约 45 秒 (12~16 镜)</option>
            <option value="约 60 秒 1分钟（建议 16~22 个分镜）">⏱️ 约 60 秒 (16~22 镜)</option>
            <option value="约 90 秒 1分半（建议 24~30 个分镜）">⏱️ 约 90 秒 (24~30 镜)</option>
            <option value="约 120 秒 2分钟（建议 32~40 个分镜）">⏱️ 约 120 秒 (32~40 镜)</option>
          </select>
          <select id="selAiVisualStyle" style="padding:4px 8px;font-size:11.5px;border-radius:5px;border:1px solid #6b21a8;background:#0f172a;color:#f8fafc;">
            <option value="写实真人电影级质感，85mm浅景深，真实细腻皮肤纹理，自然柔和光影" selected>🎥 写实真人电影级</option>
            <option value="唯美中国古装仙侠剧质感，精致古装汉服与发髻配饰，电影级逆光氛围">🏮 中国古装/仙侠剧</option>
            <option value="现代都市高质感影视剧，时尚精致妆造，电影级都市光影与景深">🏙️ 现代都市影视剧</option>
            <option value="新海诚高画质唯美2D动漫电影风格，绚丽光影与细腻人物表情">🌸 唯美2D动漫电影</option>
          </select>
        </div>
        <button type="button" id="btnCopyExtAiDrawer" onclick="copyExternalAiScriptPrompt(this)" style="padding:5px 12px;font-size:11.5px;font-weight:800;border-radius:6px;border:none;background:linear-gradient(135deg,#9333ea,#6366f1);color:#ffffff;cursor:pointer;white-space:nowrap;transition:all 0.2s;">📋 一键复制【发给外部 AI 的编剧指令】</button>
      </div>

      <!-- 中部：🎭 剧组多角色 (A / B / C / D / E...) 定妆照上传 + 5条黄金标准生图词复制 + 专属固定音色绑定 -->
      <div style="margin-bottom:10px;padding:9px 11px;border-radius:8px;background:rgba(15,23,42,0.9);border:1px solid rgba(244,114,182,0.4);">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px;">
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <span style="font-size:12.5px;font-weight:800;color:#fbcfe8;">🎭 剧组角色定妆与音色卡槽 (支持 A / B / C / D / E… 多角色)</span>
            <span style="font-size:11px;color:#94a3b8;">点「📋 定妆生图词」发给 ChatGPT/Gemini 生成标准半身照 → 点「📤 传定妆照」即可 100% 锁死该演员长相与穿搭（跳过云端捏脸）</span>
          </div>
          <div style="display:flex;align-items:center;gap:6px;">
            <button type="button" onclick="copyCharacterGoldenPrompt('', this)" style="padding:3px 9px;font-size:11px;font-weight:700;border-radius:5px;border:1px solid #f472b6;background:rgba(244,114,182,0.15);color:#fbcfe8;cursor:pointer;">📋 复制【通用5条黄金标准·定妆照生图词】</button>
            <button type="button" onclick="addDramaCastRole()" style="padding:3px 10px;font-size:11px;font-weight:800;border-radius:5px;border:1px solid #10b981;background:rgba(16,185,129,0.18);color:#6ee7b7;cursor:pointer;">+ 新增角色 (A/B/C...)</button>
            <button type="button" onclick="clearAllDramaCastRoles()" style="padding:3px 9px;font-size:11px;font-weight:700;border-radius:5px;border:1px solid #ef4444;background:rgba(239,68,68,0.14);color:#fca5a5;cursor:pointer;" title="一键清空当前所有角色卡槽（清空后发给外部 AI 时完全由您的故事自由决定角色）">🗑️ 清空全部角色</button>
          </div>
        </div>
        <div id="dramaCastCardsGrid" style="display:grid;grid-template-columns:repeat(auto-fill, minmax(295px, 1fr));gap:8px;max-height:210px;overflow-y:auto;padding-right:2px;"></div>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1.45fr;gap:12px;">
        <!-- 左栏：既可输入原著故事与台词点上方复制给外部 AI，也可粘贴外部 AI 生成好的分镜剧本直接解析 -->
        <div style="display:flex;flex-direction:column;gap:8px;padding-right:10px;border-right:1px solid rgba(148,163,184,0.2);">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap;">
            <span style="font-size:12.5px;font-weight:800;color:#7dd3fc;">📜 1. 在此输入您的原著故事/台词（点上方复制给AI），或粘贴 AI 生成好的分镜剧本</span>
            <div style="display:flex;align-items:center;gap:5px;">
              <button type="button" onclick="document.getElementById('inpDramaRawScript').value='';document.getElementById('inpAiIdeaTheme').value='';document.getElementById('inpDramaRawScript').focus();" style="padding:2px 7px;font-size:11px;border-radius:4px;border:1px solid rgba(239,68,68,0.45);background:transparent;color:#fca5a5;cursor:pointer;" title="清空文本框以输入您的新故事与台词">🗑️ 清空文本框</button>
              <input type="text" id="inpDramaProjectTitle" placeholder="剧目名称 (自动提取或手填)" style="padding:3px 8px;font-size:11.5px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#f8fafc;width:150px;">
            </div>
          </div>
          <textarea id="inpDramaRawScript" rows="10" placeholder="【用法 A：带您自己的故事与台词去让外部 AI 拆解分镜】&#10;1. 直接在此框（或上方紫色框）写下您的原本故事、人物设定与每一句台词；&#10;2. 点击上方「📋 一键复制【发给外部 AI 的编剧指令】」，系统会把您写的全部故事和台词 100% 打包进剪贴板（绝不篡改您的原意）；&#10;&#10;【用法 B：把外部 AI 生成好的分镜剧本粘回来】&#10;3. 将外部 AI 输出的【剧名】...【分镜】镜头01... 粘贴回此框，点击下方「🔍 智能解析剧本生成工作流节点」即可开拍！" style="width:100%;padding:8px 10px;border-radius:8px;border:1px solid #334155;background:#0f172a;color:#f8fafc;font-size:11.5px;line-height:1.45;resize:vertical;"></textarea>
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;">
            <label style="display:inline-flex;align-items:center;gap:5px;font-size:11px;color:#fcd34d;cursor:pointer;">
              <input type="checkbox" id="chkDramaClearOldAssets" checked style="accent-color:#f59e0b;">
              <span>🧹 解析新剧本时清空上一部剧的缓存素材</span>
            </label>
            <div style="display:flex;gap:6px;">
              <button type="button" onclick="resetDefaultDramaWorkflow()" class="btn-outline" style="padding:4px 9px;font-size:11px;color:#94a3b8;">↺ 恢复内置古装示例</button>
              <button type="button" onclick="parseRawScriptToWorkflow()" class="btn-primary" style="padding:5px 12px;font-size:11.5px;font-weight:700;background:linear-gradient(135deg,#0284c7,#2563eb);">🔍 智能解析剧本生成工作流节点</button>
            </div>
          </div>
        </div>

        <!-- 右栏：可视化图片节点与分镜视频节点表格 (可随时修改提示词/台词/秒数/引擎/单步重抽) -->
        <div style="display:flex;flex-direction:column;gap:8px;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;">
            <div style="display:flex;gap:6px;">
              <button type="button" id="btnWfTabImg" onclick="switchDramaWfTab('img')" style="padding:4px 10px;font-size:11.5px;font-weight:700;border-radius:6px;border:1px solid #f472b6;background:rgba(244,114,182,0.18);color:#fbcfe8;cursor:pointer;">📸 定妆与首尾帧图片节点 (<span id="wfImgCount">15</span>)</button>
              <button type="button" id="btnWfTabVid" onclick="switchDramaWfTab('vid')" style="padding:4px 10px;font-size:11.5px;font-weight:700;border-radius:6px;border:1px solid #475569;background:#0f172a;color:#94a3b8;cursor:pointer;">🎬 视频分镜与台词节点 (<span id="wfVidCount">11</span>)</button>
            </div>
            <div style="display:flex;gap:6px;">
              <button type="button" onclick="addDramaWorkflowNode()" class="btn-outline" style="padding:4px 9px;font-size:11px;border-color:#10b981;color:#6ee7b7;">+ 新增当前节点</button>
              <button type="button" onclick="saveDramaWorkflowEdits()" class="btn-primary" style="padding:4px 12px;font-size:11.5px;font-weight:700;background:linear-gradient(135deg,#10b981,#059669);">💾 保存工作流修改</button>
            </div>
          </div>

          <div id="wfNodesImgContainer" style="max-height:245px;overflow-y:auto;display:flex;flex-direction:column;gap:6px;padding-right:4px;"></div>
          <div id="wfNodesVidContainer" style="max-height:245px;overflow-y:auto;display:none;flex-direction:column;gap:6px;padding-right:4px;"></div>
        </div>
      </div>
    </div>

    <div id="dramaDynamicSummaryBox" style="font-size:12px;color:#cbd5e1;line-height:1.55;margin-bottom:10px;padding:8px 12px;border-radius:8px;background:rgba(2,6,23,0.55);border:1px solid rgba(148,163,184,0.2);">
      <div><b>🤖 全自动四步执行流程（支持粘贴外部在线 AI 生成的任意时长分镜剧本）：</b></div>
      <div>① <b>阶段一 & 二 (MiniMax-H3-Image)</b>：自动生成角色锁脸定妆基准图，并自动挂载参考图生成全部 <b>16:9 (2048×1152) 或 9:16 (1152×2048) 锁脸首尾关键帧</b>；</div>
      <div>② <b>阶段三-A (LongCat-Avatar-1.6 + Ming-3.0)</b>：自动切换数字人引擎，按剧本台词、角色性别与情绪气口生成全部口型同步对白镜头；</div>
      <div>③ <b>阶段三-B (MiniMax-H3 FL2VA)</b>：自动切换首尾帧有声视频引擎，按各镜设定时长生成全部动作与原生立体声环境音镜头；</div>
      <div>④ <b>阶段四 (FFmpeg 自动精剪混音合片)</b>：自动按每一镜配置的秒数精准卡点裁切，并将环境底噪无缝铺入对白背景，输出完整短剧成片至 <code>~/Movies/EasyColab/</code>！</div>
    </div>

    <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px;">
      <div id="dramaStatusText" style="font-size:12.5px;font-weight:700;color:#34d399;">✅ 已加载分镜工作流，可点上方「🤖 复制外部 AI 编剧指令」让任意 AI 写新剧本，或直接点击右侧按钮全自动开拍！</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <button type="button" id="btnRemuxAutoDrama" onclick="remuxAutoDramaOnly()" class="btn-outline" style="padding:7px 12px;font-size:12px;border-color:#38bdf8;color:#7dd3fc;white-space:nowrap;">🎞️ 仅用已有分镜重新合成成片</button>
        <button type="button" id="btnStopAutoDrama" onclick="stopAutoDramaPipeline()" class="btn-danger" style="display:none;padding:7px 14px;font-size:12.5px;font-weight:700;white-space:nowrap;">🛑 暂停自动导演 (已生成素材保留)</button>
        <button type="button" id="btnStartAutoDrama" onclick="startAutoDramaPipeline()" style="padding:8px 18px;font-size:13px;font-weight:800;border-radius:8px;border:none;background:linear-gradient(135deg, #f59e0b, #ea580c);color:#ffffff;cursor:pointer;box-shadow:0 4px 14px rgba(245,158,11,0.35);white-space:nowrap;">🚀 一键全自动拍摄当前工作流</button>
      </div>
    </div>

    <div class="progress-bar" style="height:7px;margin-bottom:10px;background:#0f172a;"><div class="progress-fill" id="dramaProgressBar" style="width:0%;background:linear-gradient(90deg,#f59e0b,#10b981);"></div></div>

    <div id="dramaStepsGrid" style="display:grid;grid-template-columns:repeat(auto-fill, minmax(155px, 1fr));gap:6px;font-size:11px;"></div>
  </div>

  <!-- MiniMax-H3-Image 4K无审查写真与免Mask锁脸换装操作区 (切换至 image 模式自动显示) -->
  <div id="h3ImageControlRow" style="display:none;grid-template-columns:1.1fr 1.1fr;gap:12px;padding:14px;margin-bottom:16px;background:#0f172a;border:1.5px dashed #ec4899;border-radius:12px;">
    <div style="display:flex;flex-direction:column;justify-content:space-between;">
      <div>
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px;">
          <label style="font-size:12.5px;font-weight:700;color:#f472b6;margin:0;">🖼️ 1. 参考图上传 (不传=纯文生4K写真；上传=免涂Mask锁脸换装/多图融合)</label>
          <button type="button" onclick="clearAllH3RefImages()" style="padding:2px 8px;font-size:11px;border-radius:5px;border:1px solid var(--border);background:transparent;color:#94a3b8;cursor:pointer;">✕ 清空参考图 (纯文生4K)</button>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:8px;">
          <div style="padding:8px;border-radius:8px;background:#1e293b;border:1px solid #334155;">
            <div style="font-size:11px;font-weight:700;color:#fbcfe8;margin-bottom:4px;">👩 &lt;Picture 1&gt; 主人物/底图</div>
            <input type="file" id="inpH3ImgRef1" accept="image/*" onchange="handleH3RefImageSelect(0, this)" style="font-size:10.5px;color:var(--muted);width:100%;">
            <img id="h3ImgRef1Preview" style="display:none;margin-top:6px;width:100%;height:88px;object-fit:contain;border-radius:5px;background:#020617;">
          </div>
          <div style="padding:8px;border-radius:8px;background:#1e293b;border:1px solid #334155;">
            <div style="font-size:11px;font-weight:700;color:#cbd5e1;margin-bottom:4px;">👗 &lt;Picture 2&gt; 服饰/道具(可选)</div>
            <input type="file" id="inpH3ImgRef2" accept="image/*" onchange="handleH3RefImageSelect(1, this)" style="font-size:10.5px;color:var(--muted);width:100%;">
            <img id="h3ImgRef2Preview" style="display:none;margin-top:6px;width:100%;height:88px;object-fit:contain;border-radius:5px;background:#020617;">
          </div>
          <div style="padding:8px;border-radius:8px;background:#1e293b;border:1px solid #334155;">
            <div style="font-size:11px;font-weight:700;color:#cbd5e1;margin-bottom:4px;">🏞️ &lt;Picture 3&gt; 新场景(可选)</div>
            <input type="file" id="inpH3ImgRef3" accept="image/*" onchange="handleH3RefImageSelect(2, this)" style="font-size:10.5px;color:var(--muted);width:100%;">
            <img id="h3ImgRef3Preview" style="display:none;margin-top:6px;width:100%;height:88px;object-fit:contain;border-radius:5px;background:#020617;">
          </div>
        </div>
      </div>
      <div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;gap:8px;font-size:11.5px;">
        <span id="h3ImgRefStatus" style="color:#f472b6;font-weight:600;">✨ 当前为「纯文生 4K 超清写真」模式（上传 &lt;Picture 1&gt; 即自动切换为免涂 Mask 锁脸换装）</span>
      </div>
    </div>

    <div style="display:flex;flex-direction:column;justify-content:space-between;">
      <div>
        <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:6px;">
          <label style="font-size:12.5px;font-weight:700;color:#fcd34d;margin:0;">🪄 2. 免Mask锁脸换装生成器 &amp; 4K去磨砂黄金模板</label>
          <select id="h3ImageTemplateSelect" onchange="applyH3ImagePresetTemplate(this.value)" style="padding:3px 8px;border-radius:6px;border:1px solid #ec4899;background:#1e293b;color:#fbcfe8;font-size:11px;font-weight:700;max-width:255px;">
            <option value="">✨ 一键套用 4K 写真 / 锁脸换装模板...</option>
            <optgroup label="👗 免涂 Mask 锁脸换装模板 (需上传 Picture 1)">
              <option value="edit_qipao">👘 锁脸换装：黑金凤凰刺绣真丝旗袍 + 帝王绿翡翠耳坠</option>
              <option value="edit_silk">🍷 锁脸换装：酒红真丝滑落肩头吊带睡袍 + 黑色蕾丝花边</option>
              <option value="edit_lace">🤍 锁脸换装：纯白镂空蕾丝吊带短裙 + 柔光私房质感</option>
              <option value="edit_ol">🖤 锁脸换装：修身黑丝绒高开叉晚礼服 + 璀璨钻石颈链</option>
            </optgroup>
            <optgroup label="📸 纯文生 4K 原生超清写真模板 (无需传图 · 去磨砂真皮)">
              <option value="t2i_courtyard">🏮 4K江南少女：苏州庭院黄昏白玉簪中国少女 (单反真皮)</option>
              <option value="t2i_boudoir">🥂 4K私房写真：午后百叶窗暖光酒红真丝蕾丝写真 (Heretic无审查)</option>
              <option value="t2i_white_lace">🦢 4K纯白蕾丝：晨光落地窗纯白蕾丝吊带少女特写 (通透真皮)</option>
              <option value="t2i_chongqing">🌃 4K雨夜街拍：重庆洪崖洞透明伞回眸姑娘 + 中英双语招牌</option>
            </optgroup>
          </select>
        </div>
        <div style="padding:8px 10px;border-radius:8px;background:#1e293b;border:1px solid #334155;">
          <div style="font-size:11px;color:#94a3b8;margin-bottom:6px;">上传 &lt;Picture 1&gt; 后，只需填写下方任意一项，后台自动合成 <code style="color:#f472b6;">Picture + Subject</code> 三段式像素级锁脸语法：</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
            <div>
              <label style="font-size:11px;color:#fbcfe8;display:block;margin-bottom:2px;">👗 换成什么新衣服/穿搭 (&lt;Subject 2&gt;)</label>
              <input id="inpH3EditOutfit" type="text" oninput="syncH3EditPreviewPrompt()" placeholder="例如: a white lace camisole dress / 黑色真丝旗袍" style="width:100%;padding:5px 8px;border-radius:6px;border:1px solid #475569;background:#0f172a;color:#f8fafc;font-size:12px;">
            </div>
            <div>
              <label style="font-size:11px;color:#93c5fd;display:block;margin-bottom:2px;">🏞️ 换成什么新场景 (留空则100%保留原背景)</label>
              <input id="inpH3EditScene" type="text" oninput="syncH3EditPreviewPrompt()" placeholder="留空保留原背景；或填 sunlit bedroom" style="width:100%;padding:5px 8px;border-radius:6px;border:1px solid #475569;background:#0f172a;color:#f8fafc;font-size:12px;">
            </div>
          </div>
        </div>
      </div>
      <label style="display:flex;align-items:center;gap:6px;margin-top:8px;padding-top:8px;border-top:1px solid var(--border);font-size:11.5px;color:#6ee7b7;cursor:pointer;user-select:none;">
        <input type="checkbox" id="chkCleanMatteWords" checked style="width:13.5px;height:13.5px;accent-color:#10b981;">
        <span>🛡️ 自动开启「去磨砂真皮过滤」（自动剔除 <code style="color:#cbd5e1;">matte skin / fine pores / film grain</code> 等导致砂纸脸的旧词）</span>
      </label>
    </div>
  </div>

  <!-- LongCat-Video-Avatar-1.6 数字人音频驱动图形化操作区 (切换至 avatar 模式自动显示) -->
  <div id="avatarControlRow" style="display:none;grid-template-columns:1.18fr 1fr;gap:12px;padding:14px;margin-bottom:16px;background:#0f172a;border:1.5px dashed #10b981;border-radius:12px;">
    <div style="display:flex;flex-direction:column;justify-content:space-between;">
      <div>
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;">
          <label style="font-size:12.5px;font-weight:700;color:#34d399;margin:0;">🎤 1. 驱动音源：上传录音 (.mp3/.wav) 或 直接输入台词配音</label>
          <button type="button" onclick="useBuiltinAvatarSample()" style="padding:2px 9px;font-size:11.5px;font-weight:700;border-radius:6px;border:1px solid rgba(16,185,129,0.5);background:rgba(16,185,129,0.16);color:#6ee7b7;cursor:pointer;">🎙️ 使用内置示例人声+人像</button>
        </div>
        <input type="file" id="inpAvatarAudio" accept=".wav,.mp3,.m4a,.flac,.ogg,audio/*" onchange="handleAvatarAudioSelect(this)" style="font-size:12px;color:var(--muted);width:100%;">
        <div style="margin-top:8px;padding:8px 10px;border-radius:8px;background:#1e293b;border:1px solid #334155;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:6px;">
            <span style="font-size:11.5px;font-weight:700;color:#fcd34d;">💬 台词配音 (Ming-3.0 戏剧人设/气口/克隆 + 微软极速音色 · 共75+款)</span>
            <div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap;">
              <select id="avatarTtsVoiceSelect" onchange="onAvatarVoiceOrStyleChange()" style="padding:2px 6px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#f8fafc;font-size:11px;font-weight:700;max-width:235px;">
                <optgroup label="🔥 Ming-3.0 戏剧女声 (百灵3.0 GPU · 支持呼吸/笑声/哼声气口)">
                  <option value="ming:妩媚妲己">🔥 妩媚妲己 (酥麻撩人·尾音微扬魅惑女声)</option>
                  <option value="ming:灵梦">💕 灵梦 (撒娇粘人·恋爱女友求陪伴)</option>
                  <option value="ming:灵恬">🍬 灵恬 (蜜糖甜妹·娇羞笃定表白腔)</option>
                  <option value="ming:灵雨汐_中性" selected>☁️ 灵雨汐·温柔 (棉花糖软妹·贴耳自然呼吸感)</option>
                  <option value="ming:灵雨汐_高兴">✨ 灵雨汐·开心 (清甜少女·雀跃欢快)</option>
                  <option value="ming:灵雨汐_生气">💢 灵雨汐·嗔怒 (傲娇女友/闺蜜急促娇嗔)</option>
                  <option value="ming:灵雨汐_悲伤">💧 灵雨汐·委屈 (楚楚可怜·带脆弱呼吸感)</option>
                  <option value="ming:灵雨汐_惊讶">😮 灵雨汐·惊喜 (银铃清脆赞叹少女音)</option>
                  <option value="ming:灵雨汐_恐惧">😨 灵雨汐·惊慌 (压低声音急促呼吸颤音)</option>
                  <option value="ming:灵黛嗔_生气">👑 灵黛嗔·傲娇 (大小姐发脾气·要人哄的娇纵感)</option>
                  <option value="ming:灵黛嗔_高兴">🌻 灵黛嗔·明媚 (午后暖阳带笑意治愈女声)</option>
                  <option value="ming:灵黛嗔_惊讶">🎉 灵黛嗔·惊喜 (中奖般高亢雀跃邻家女孩)</option>
                  <option value="ming:灵书悦_悲伤">📱 灵书悦·吃醋 (查岗怀疑男友·委屈质问女友腔)</option>
                  <option value="ming:灵书悦_厌恶">🙄 灵书悦·绿茶 (看似无辜柔弱·实则敷衍拉扯)</option>
                  <option value="ming:灵书悦_高兴">🍰 灵书悦·吃货 (邻家甜妹吃到美食的幸福感)</option>
                  <option value="ming:灵风信_中性">🌷 灵风信·暖心 (春光般温柔体贴·深夜哄睡女友)</option>
                  <option value="ming:灵惜梦_中性">💎 灵惜梦·御姐 (珠落玉盘·独立干练都市姐姐)</option>
                  <option value="ming:灵惜梦_悲伤">💔 灵惜梦·破碎 (带明显哭腔与脆弱绝望感)</option>
                  <option value="ming:灵采薇_悲伤">🌙 灵采薇·凄美 (字字裹着气音的深情破碎感)</option>
                  <option value="ming:灵心澜_厌恶">👠 灵心澜·女王 (冷艳高贵·大女主压迫感)</option>
                  <option value="ming:灵小柔">🌸 灵小柔 (清亮甜美·百灵官方看板娘)</option>
                  <option value="ming:灵婉">💌 灵婉 (深夜谈心·私密治愈告白少女)</option>
                  <option value="ming:灵小梨">🍐 灵小梨 (江南水乡微风·灵动俏皮文艺女声)</option>
                  <option value="ming:水浒传_潘金莲">🎬 潘金莲 (《水浒传》经典娇媚影视角色原声)</option>
                </optgroup>
                <optgroup label="🕺 Ming-3.0 戏剧男声 & 影视名角 (低音炮/霸总/帝王/动漫)">
                  <option value="ming:灵空">🎧 灵空 (深夜哄睡低音炮男友·贴耳呼吸感 ASMR)</option>
                  <option value="ming:灵宇">🌹 灵宇 (乙女游戏男主·自信含笑撩人学长)</option>
                  <option value="ming:灵朗">🥂 灵朗 (深情霸总·成熟稳重宠溺男主)</option>
                  <option value="ming:灵信远_中性">☀️ 灵信远·暖男 (带浅笑与呼吸感的新婚男友)</option>
                  <option value="ming:灵孤影_厌恶">❄️ 灵孤影·高冷 (冰冷禁欲系霸总·生人勿近)</option>
                  <option value="ming:灵孤影_悲伤">🌧️ 灵孤影·心碎 (浓重鼻音与叹息·深情虐恋男主)</option>
                  <option value="ming:灵孤影_生气">🔥 灵孤影·震怒 (低沉压迫感·暴风雨前夕霸总)</option>
                  <option value="ming:灵思齐_生气">💳 灵思齐·霸总 (给钱不给爱·居高临下冷酷总裁)</option>
                  <option value="ming:灵晓声_中性">♟️ 灵晓声·腹黑 (运筹帷幄·极具侵略性宿敌男主)</option>
                  <option value="ming:灵楷">📻 灵楷 (温润磁性电台主播·百灵官方男主)</option>
                  <option value="ming:甄嬛传_四郎">🎬 甄嬛传·四郎 (经典雍正帝沉稳威严影视原声)</option>
                  <option value="ming:康熙王朝_康熙">🎬 康熙大帝 (《康熙王朝》霸气帝王影视原声)</option>
                  <option value="ming:雍正王朝_雍正">🎬 雍正王朝·雍正 (冷峻威严帝王影视原声)</option>
                  <option value="ming:西游记_唐僧">🎬 西游记·唐僧 (温润慈悲经典影视原声)</option>
                  <option value="ming:水浒传_武松">🎬 水浒传·武松 (豪迈硬汉经典影视原声)</option>
                  <option value="ming:潜伏_余则成">🎬 潜伏·余则成 (沉稳内敛谍战经典原声)</option>
                  <option value="ming:悟空">🐵 悟空 (活灵活现神话评书大师腔)</option>
                  <option value="ming:说唱小哥">🎤 说唱小哥 (节奏感拉满潮流 Rap 青年)</option>
                  <option value="ming:蜡笔小新_野原新之助">🖍️ 蜡笔小新 (经典搞怪呆萌卡通原声)</option>
                  <option value="ming:蜡笔小新_野原美伢">👩 野原美伢 (《蜡笔小新》妈妈经典原声)</option>
                  <option value="ming:小猪佩奇_佩奇">🐷 小猪佩奇 (经典可爱动画角色原声)</option>
                  <option value="ming:US_Stella">🇺🇸 US_Stella (地道美式温暖自然女声)</option>
                  <option value="ming:US_James">🇺🇸 US_James (地道美式磁性纪录片男声)</option>
                </optgroup>
                <optgroup label="🧬 Ming-3.0 零样本声音克隆 (3~10秒样本复刻任意人声)">
                  <option value="ming:__clone__">🧬 零样本声音克隆 (使用下方上传的 3~10 秒声音样本)</option>
                </optgroup>
                <optgroup label="⚡ 微软 Edge-TTS 极速女声 (本机0.5秒直出 · 15款)">
                  <option value="zh-CN-XiaoyiNeural">🎀 晓伊 (俏皮甜美·活泼少女音)</option>
                  <option value="zh-CN-XiaoyiNeural|pitch=+6Hz|rate=-5%">🍬 晓伊·撒娇版 (软糯夹子音·甜度拉满)</option>
                  <option value="zh-CN-XiaoxiaoNeural">🌸 晓晓 (温柔知性·自然女友音)</option>
                  <option value="zh-CN-XiaoxiaoNeural|pitch=-6Hz|rate=-6%">🍷 晓晓·御姐版 (低沉磁性·成熟姐姐音)</option>
                  <option value="zh-TW-HsiaoChenNeural">🧋 晓臻 (台湾甜美软糯·超嗲台妹腔)</option>
                  <option value="zh-TW-HsiaoYuNeural">🍃 晓雨 (台湾温柔知性·治愈台湾腔)</option>
                  <option value="zh-HK-HiuMaanNeural">🇭🇰 晓曼 (香港粤语·甜美港风女声)</option>
                  <option value="zh-HK-HiuGaaiNeural">🇭🇰 晓佳 (香港粤语·知性都市女声)</option>
                  <option value="zh-CN-liaoning-XiaobeiNeural">🥟 晓北 (东北大妞·幽默爽朗东北话)</option>
                  <option value="zh-CN-shaanxi-XiaoniNeural">🍜 晓妮 (陕西妹子·俏皮关中陕北话)</option>
                  <option value="en-US-AvaMultilingualNeural">💎 Ava (轻熟温柔女声·支持中英双语)</option>
                  <option value="en-US-EmmaMultilingualNeural">✨ Emma (明快清亮女声·支持中英双语)</option>
                  <option value="fr-FR-VivienneMultilingualNeural">🌹 Vivienne (法式微醺女声·可说中文)</option>
                  <option value="ja-JP-NanamiNeural">🇯🇵 七海 Nanami (日本动漫萌妹·日语)</option>
                  <option value="ko-KR-SunHiNeural">🇰🇷 善熙 SunHi (韩国温柔韩剧女声·韩语)</option>
                </optgroup>
                <optgroup label="⚡ 微软 Edge-TTS 极速男声 (本机0.5秒直出 · 12款)">
                  <option value="zh-CN-YunxiNeural">🎙️ 云希 (阳光少年·清朗磁性男声)</option>
                  <option value="zh-CN-YunxiNeural|pitch=-8Hz|rate=-5%">🥂 云希·低音炮 (深情耳畔低语·磁性男声)</option>
                  <option value="zh-CN-YunjianNeural">🦁 云健 (霸总沉稳·浑厚爆发力男声)</option>
                  <option value="zh-CN-YunyangNeural">👔 云扬 (儒雅温润·成熟磁性男声)</option>
                  <option value="zh-CN-YunxiaNeural">👦 云夏 (元气正太·清脆少年音)</option>
                  <option value="zh-TW-YunJheNeural">🧋 云哲 (台湾温柔暖男·偶像剧台湾腔)</option>
                  <option value="zh-HK-WanLungNeural">🇭🇰 云龙 (香港粤语·港片磁性男声)</option>
                  <option value="en-US-AndrewMultilingualNeural">🥃 Andrew (欧美深沉磁性男声·可说中文)</option>
                  <option value="en-US-BrianMultilingualNeural">🎸 Brian (随性自然青年男声·可说中文)</option>
                  <option value="fr-FR-RemyMultilingualNeural">🍷 Remy (法式绅士低音男声·可说中文)</option>
                  <option value="ko-KR-HyunsuMultilingualNeural">🇰🇷 贤洙 Hyunsu (韩国欧巴低音·支持中韩)</option>
                  <option value="ja-JP-KeitaNeural">🇯🇵 圭太 Keita (日本爽朗青年男声·日语)</option>
                </optgroup>
              </select>
              <select id="avatarTtsStyleSelect" onchange="onAvatarVoiceOrStyleChange()" title="微调 Edge-TTS 音色的音调高低与语速快慢" style="padding:2px 5px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#cbd5e1;font-size:11px;font-weight:600;">
                <option value="" selected>🎛️ 默认语感</option>
                <option value="pitch=+6Hz|rate=-5%">🍬 升调撒娇 (+6Hz)</option>
                <option value="pitch=-6Hz|rate=-6%">🍷 降调御姐/低音 (-6Hz)</option>
                <option value="pitch=+0Hz|rate=-12%">🌙 深情慢速 (-12%)</option>
                <option value="pitch=+0Hz|rate=+12%">⚡ 轻快语速 (+12%)</option>
              </select>
              <button type="button" id="btnPreviewTts" onclick="previewAvatarTts()" style="padding:2px 8px;font-size:11px;font-weight:700;border-radius:5px;border:1px solid #3b82f6;background:rgba(59,130,246,0.2);color:#93c5fd;cursor:pointer;white-space:nowrap;">🔊 合成并试听</button>
            </div>
          </div>

          <!-- Ming-3.0 副语言气口快捷插入栏 (呼吸/笑声/叹气/傲娇哼~) -->
          <div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap;margin-bottom:6px;padding:4px 6px;border-radius:6px;background:rgba(15,23,42,0.75);border:1px dashed rgba(244,114,182,0.35);">
            <span style="font-size:10.5px;color:#f472b6;font-weight:700;margin-right:2px;">🎭 插入真人气口:</span>
            <button type="button" onclick="insertMingParalinguisticTag('[Breathing]')" title="插入贴耳呼吸/喘息声，数字人嘴型与神态同步联动" style="padding:1px 6px;font-size:10.5px;border-radius:4px;border:1px solid rgba(244,114,182,0.4);background:rgba(244,114,182,0.12);color:#fbcfe8;cursor:pointer;">😮‍💨 呼吸</button>
            <button type="button" onclick="insertMingParalinguisticTag('[Laughter]')" title="插入真实笑出声" style="padding:1px 6px;font-size:10.5px;border-radius:4px;border:1px solid rgba(244,114,182,0.4);background:rgba(244,114,182,0.12);color:#fbcfe8;cursor:pointer;">🤭 笑声</button>
            <button type="button" onclick="insertMingParalinguisticTag('[Sigh]')" title="插入无奈或深情叹气声" style="padding:1px 6px;font-size:10.5px;border-radius:4px;border:1px solid rgba(244,114,182,0.4);background:rgba(244,114,182,0.12);color:#fbcfe8;cursor:pointer;">😔 叹气</button>
            <button type="button" onclick="insertMingParalinguisticTag('[Dissatisfaction-hnn]')" title="插入傲娇不满的『哼~』声" style="padding:1px 6px;font-size:10.5px;border-radius:4px;border:1px solid rgba(244,114,182,0.4);background:rgba(244,114,182,0.12);color:#fbcfe8;cursor:pointer;">😤 傲娇哼~</button>
            <button type="button" onclick="insertMingParalinguisticTag('[Surprise-ah]')" title="插入惊喜『啊~』声" style="padding:1px 6px;font-size:10.5px;border-radius:4px;border:1px solid rgba(244,114,182,0.4);background:rgba(244,114,182,0.12);color:#fbcfe8;cursor:pointer;">😲 惊喜啊~</button>
            <button type="button" onclick="insertMingParalinguisticTag('[Question-ei]')" title="插入好奇疑问『诶？』" style="padding:1px 6px;font-size:10.5px;border-radius:4px;border:1px solid rgba(244,114,182,0.4);background:rgba(244,114,182,0.12);color:#fbcfe8;cursor:pointer;">❓ 诶？</button>
            <button type="button" onclick="insertMingParalinguisticTag('[Uhm]')" title="插入思考犹豫『唔...』" style="padding:1px 6px;font-size:10.5px;border-radius:4px;border:1px solid rgba(244,114,182,0.4);background:rgba(244,114,182,0.12);color:#fbcfe8;cursor:pointer;">🤔 唔...</button>
            <button type="button" onclick="insertMingParalinguisticTag('[Cough]')" title="插入自然轻咳清嗓" style="padding:1px 6px;font-size:10.5px;border-radius:4px;border:1px solid rgba(244,114,182,0.4);background:rgba(244,114,182,0.12);color:#fbcfe8;cursor:pointer;">🤧 轻咳</button>
          </div>

          <textarea id="inpAvatarTtsText" rows="2" oninput="onAvatarTtsInput(this)" placeholder="在此粘贴人物要说的话（支持点击上方按钮穿插 [Sigh] 叹气、[Dissatisfaction-hnn] 傲娇哼~、[Laughter] 笑声等真人气口）" style="width:100%;min-height:52px;padding:6px 10px;border-radius:6px;border:1px solid #334155;background:#0f172a;color:#f8fafc;font-size:12.5px;line-height:1.45;resize:vertical;"></textarea>

          <!-- Ming-3.0 零样本 3~10秒声音克隆槽位 -->
          <div style="margin-top:6px;padding:6px 8px;border-radius:6px;background:rgba(15,23,42,0.85);border:1px solid #334155;display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;">
            <div style="display:flex;align-items:center;gap:6px;flex:1;min-width:210px;">
              <span style="font-size:11px;font-weight:700;color:#c084fc;white-space:nowrap;">🧬 3秒声音克隆(可选):</span>
              <input type="file" id="inpMingCloneAudio" accept=".wav,.mp3,.m4a,.flac,.ogg,audio/*" onchange="handleMingCloneAudioSelect(this)" style="font-size:10.5px;color:var(--muted);max-width:165px;">
              <input type="text" id="inpMingClonePromptText" placeholder="可选：样本音频原话(填上可开启高保真韵律续写)" style="flex:1;min-width:120px;padding:3px 7px;border-radius:5px;border:1px solid #475569;background:#020617;color:#f8fafc;font-size:11px;">
            </div>
            <button type="button" onclick="clearMingCloneAudio()" style="padding:2px 7px;font-size:10.5px;border-radius:4px;border:1px solid var(--border);background:transparent;color:#94a3b8;cursor:pointer;white-space:nowrap;">✕ 清除克隆样本</button>
          </div>
        </div>
        <audio id="avatarAudioPreview" controls style="display:none;margin-top:8px;width:100%;height:34px;"></audio>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-top:10px;padding-top:8px;border-top:1px solid var(--border);font-size:12px;">
        <span id="avatarAudioStatus" style="color:#34d399;font-weight:600;">✅ 可上传音频，或在上方黄色框直接粘贴台词自动配音</span>
        <div style="display:flex;align-items:center;gap:6px;flex-shrink:0;">
          <span style="color:var(--muted);">生成时长/段数:</span>
          <select id="avatarSegmentsSelect" style="padding:3px 8px;border-radius:6px;border:1px solid var(--border);background:#1e293b;color:#f8fafc;font-size:12px;font-weight:700;">
            <option value="0" selected>🔄 自动匹配音频全长 (推荐)</option>
            <option value="1">1段 (~3.7秒 · 8步极速)</option>
            <option value="2">2段 (~7秒)</option>
            <option value="3">3段 (~10秒)</option>
            <option value="5">5段 (~17秒)</option>
            <option value="10">10段 (~33秒)</option>
            <option value="19">19段 (~1分钟)</option>
            <option value="38">38段 (~2分钟)</option>
            <option value="56">56段 (~3分钟)</option>
            <option value="94">94段 (~5分钟)</option>
            <option value="188">188段 (~10分钟 · 超长极限)</option>
          </select>
        </div>
      </div>
    </div>
    <div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <label style="font-size:12.5px;font-weight:700;color:#93c5fd;margin:0;">🧑 2. 人物参考照片 (上传为图+音 AI2V；不传为文+音 AT2V)</label>
        <button type="button" onclick="clearRefFrame('first')" style="padding:2px 8px;font-size:11px;border-radius:5px;border:1px solid var(--border);background:transparent;color:#94a3b8;cursor:pointer;">✕ 清除照片 (切为纯文+音)</button>
      </div>
      <input type="file" id="inpAvatarImage" accept="image/*" onchange="handleFrameFileSelect(this, 'avatarImagePreview')" style="font-size:12px;color:var(--muted);width:100%;">
      <img id="avatarImagePreview" style="display:none;margin-top:8px;width:100%;max-height:165px;object-fit:contain;border-radius:6px;background:#020617;">
      <div id="avatarImageHint" style="font-size:11.5px;color:#94a3b8;margin-top:8px;">💡 上传一张人物正面/半身照，LongCat-Avatar-1.6 会驱动照片中的人物精准对口型说话/唱歌；若不上传，则直接根据下方提示词生成说话人物。</div>
    </div>
  </div>

  <!-- FL2VA 首尾帧图片上传区 (默认隐藏，切换至 FL2VA 显示) -->
  <div id="fl2vaImageRow" style="display:none;grid-template-columns:1fr 1fr;gap:12px;padding:12px;margin-bottom:16px;background:#0f172a;border:1px dashed #3b82f6;border-radius:12px;">
    <div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <label style="font-size:12px;font-weight:600;color:#93c5fd;margin:0;">📌 首帧图片 (First Frame - index 0)</label>
        <button type="button" onclick="clearRefFrame('first')" style="padding:1px 7px;font-size:11px;border-radius:4px;border:1px solid var(--border);background:transparent;color:#94a3b8;cursor:pointer;">✕ 清除首帧</button>
      </div>
      <input type="file" id="inpFirstFrame" accept="image/*" onchange="handleFrameFileSelect(this, 'firstFramePreview')" style="font-size:12px;color:var(--muted);width:100%;">
      <img id="firstFramePreview" style="display:none;margin-top:8px;width:100%;max-height:140px;object-fit:contain;border-radius:6px;background:#020617;">
    </div>
    <div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <label style="font-size:12px;font-weight:600;color:#93c5fd;margin:0;">🏁 尾帧图片 (Last Frame - index -1，可选)</label>
        <button type="button" onclick="clearRefFrame('last')" style="padding:1px 7px;font-size:11px;border-radius:4px;border:1px solid var(--border);background:transparent;color:#94a3b8;cursor:pointer;">✕ 清除尾帧</button>
      </div>
      <input type="file" id="inpLastFrame" accept="image/*" onchange="handleFrameFileSelect(this, 'lastFramePreview')" style="font-size:12px;color:var(--muted);width:100%;">
      <img id="lastFramePreview" style="display:none;margin-top:8px;width:100%;max-height:140px;object-fit:contain;border-radius:6px;background:#020617;">
    </div>
  </div>

  <!-- ==================== 第 1 行 (单独一行 100% 宽度)：☰ 镜头与多模态提示词 (内部四个框也全部各自单独一行) ==================== -->
  <div id="promptSectionCard" style="padding:18px;margin-bottom:16px;background:#0f172a;border:1px solid var(--border);border-radius:14px;">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:14px;">
      <div style="display:flex;align-items:center;gap:8px;font-size:15.5px;font-weight:800;color:#f8fafc;">
        <span style="color:#3b82f6;">☰</span>
        <span id="promptSectionTitle">镜头与多模态提示词</span>
      </div>
      <div id="fourBoxActionBtns" style="display:flex;align-items:center;gap:8px;">
        <button type="button" class="btn-outline" style="padding:4px 12px;font-size:12px;border-color:#3b82f6;color:#93c5fd;" onclick="copyChatToVideoPrompt()">📋 从上方 Qwen3.8 结果填入四框</button>
        <button type="button" class="btn-outline" style="padding:4px 12px;font-size:12px;border-color:rgba(239,68,68,0.45);color:#fca5a5;" onclick="clearAllVideoPrompts()">🗑️ 一键清空四框</button>
      </div>
    </div>

    <!-- 框 1 (单独一行)：主体、构图、光影与风格描述 -->
    <div id="videoPromptBox1Wrap" style="margin-bottom:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <label id="videoPromptBox1Label" style="font-size:12.5px;font-weight:700;color:#94a3b8;margin:0;">1. 主体、构图、光影与风格描述</label>
        <button type="button" onclick="clearSingleVideoPrompt('videoPrompt')" style="padding:2px 8px;font-size:11px;border-radius:5px;border:1px solid var(--border);background:transparent;color:#94a3b8;cursor:pointer;" onmouseover="this.style.color='#fca5a5';this.style.borderColor='rgba(239,68,68,0.45)'" onmouseout="this.style.color='#94a3b8';this.style.borderColor='var(--border)'">✕ 清空</button>
      </div>
      <textarea id="videoPrompt" rows="4" oninput="autoSplitAvatarPromptAndTts(this)" placeholder="描述画面主体、构图、运镜方式、光影与风格..." style="width:100%;min-height:110px;padding:10px 14px;border-radius:10px;border:1px solid var(--border);background:#1e293b;color:#f8fafc;font-size:13.5px;line-height:1.55;resize:vertical;">Masterpiece Makoto Shinkai anime style, 2D cel-shaded animation. A teenage girl in a Japanese sailor school uniform standing at a sunlit railway crossing in late spring. A green commuter train speeds past in front of her, creating a strong gust of wind that blows her long black hair and a flurry of pink cherry blossom petals across the frame. Dramatic lens flare, glowing cumulous clouds in a deep blue sky, vibrant cinematic lighting, smooth 24fps keyframe animation.</textarea>
    </div>

    <!-- 框 2 (单独一行)：环境音效与台词对白（可选） -->
    <div id="videoSoundBoxWrap" style="margin-bottom:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <label style="font-size:12.5px;font-weight:700;color:#94a3b8;margin:0;">2. 环境音效与台词对白（可选）</label>
        <button type="button" onclick="clearSingleVideoPrompt('videoSoundPrompt')" style="padding:2px 8px;font-size:11px;border-radius:5px;border:1px solid var(--border);background:transparent;color:#94a3b8;cursor:pointer;" onmouseover="this.style.color='#fca5a5';this.style.borderColor='rgba(239,68,68,0.45)'" onmouseout="this.style.color='#94a3b8';this.style.borderColor='var(--border)'">✕ 清空</button>
      </div>
      <textarea id="videoSoundPrompt" rows="2" placeholder="例如：雨滴敲击声、脚步踩水声、角色台词对白..." style="width:100%;min-height:68px;padding:10px 14px;border-radius:10px;border:1px solid var(--border);background:#1e293b;color:#f8fafc;font-size:13px;line-height:1.5;resize:vertical;">Metallic clacking of train wheels on railway tracks, rhythmic dinging of the railroad crossing warning bell, warm spring wind rustling through leaves, a soft gasp from the girl.</textarea>
    </div>

    <!-- 框 3 (单独一行)：背景音乐风格（可选） -->
    <div id="videoMusicBoxWrap" style="margin-bottom:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <label style="font-size:12.5px;font-weight:700;color:#94a3b8;margin:0;">3. 背景音乐风格（可选）</label>
        <button type="button" onclick="clearSingleVideoPrompt('videoMusicPrompt')" style="padding:2px 8px;font-size:11px;border-radius:5px;border:1px solid var(--border);background:transparent;color:#94a3b8;cursor:pointer;" onmouseover="this.style.color='#fca5a5';this.style.borderColor='rgba(239,68,68,0.45)'" onmouseout="this.style.color='#94a3b8';this.style.borderColor='var(--border)'">✕ 清空</button>
      </div>
      <textarea id="videoMusicPrompt" rows="2" placeholder="例如：抒情独奏钢琴、赛博朋克合成器电子乐、恢弘管弦乐..." style="width:100%;min-height:68px;padding:10px 14px;border-radius:10px;border:1px solid var(--border);background:#1e293b;color:#f8fafc;font-size:13px;line-height:1.5;resize:vertical;">Emotional and nostalgic solo piano melody joined by warm acoustic guitar and soaring cinematic strings.</textarea>
    </div>

    <!-- 框 4 (单独一行)：反向提示词 Negative Prompt（可选，排除低质或多余元素） -->
    <div id="videoNegBoxWrap">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <div onclick="toggleNegPromptBox()" style="font-size:12.5px;font-weight:700;color:#94a3b8;cursor:pointer;user-select:none;display:flex;align-items:center;gap:4px;">
          <span id="negArrowIcon">⌄</span>
          <span>4. 反向提示词 Negative Prompt（可选，排除低质或多余元素）</span>
        </div>
        <button type="button" onclick="clearSingleVideoPrompt('videoNegPrompt')" style="padding:2px 8px;font-size:11px;border-radius:5px;border:1px solid var(--border);background:transparent;color:#94a3b8;cursor:pointer;" onmouseover="this.style.color='#fca5a5';this.style.borderColor='rgba(239,68,68,0.45)'" onmouseout="this.style.color='#94a3b8';this.style.borderColor='var(--border)'">✕ 清空</button>
      </div>
      <div id="negPromptWrap">
        <label style="font-size:11.5px;color:#64748b;display:block;margin-bottom:5px;">反向排除项</label>
        <textarea id="videoNegPrompt" rows="2" placeholder="例如：low quality, blurry, watermark, text, deformed, bad anatomy, extra limbs, static frame..." style="width:100%;min-height:62px;padding:10px 14px;border-radius:10px;border:1px solid var(--border);background:#1e293b;color:#f8fafc;font-size:13px;line-height:1.5;resize:vertical;">low quality, worst quality, blurry, watermark, text, signature, deformed, bad anatomy, extra limbs, distorted face, jittery motion</textarea>
      </div>
    </div>
  </div>

  <!-- ==================== 第 2 行 (紧凑两排横条布局)：🎛️ 规格与采样参数 + 开始生成按钮 ==================== -->
  <div id="specSectionCard" style="padding:12px 16px;margin-bottom:14px;background:#0f172a;border:1px solid var(--border);border-radius:14px;">
    <!-- 顶部紧凑标题栏 + 实时状态文字同行显示 -->
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid rgba(51,65,85,0.6);">
      <div style="display:flex;align-items:center;gap:6px;font-size:14.5px;font-weight:800;color:#f8fafc;">
        <span style="color:#3b82f6;">🎛️</span>
        <span>规格与采样参数</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
        <div style="font-family:'SF Mono',Menlo,monospace;font-size:11.5px;color:#93c5fd;" id="videoGenStatus">
          状态：等待任务...（自动保存至 <b>~/Movies/EasyColab/</b> 与云盘双备份）
        </div>
        <button type="button" id="btnStopTaskTop" onclick="stopActiveVideoTask()" style="display:none;padding:3px 10px;font-size:11.5px;font-weight:800;border-radius:6px;border:1px solid rgba(239,68,68,0.65);background:rgba(239,68,68,0.18);color:#fca5a5;cursor:pointer;">🛑 停止</button>
      </div>
    </div>

    <!-- 第 1 排横条：分辨率 | 采样步数 | 随机种子 | 批量连抽 (四项一字排开) -->
    <div id="specRow1Grid" style="display:grid;grid-template-columns: 1.3fr 1.15fr 1fr 1.15fr;gap:12px;align-items:center;margin-bottom:10px;">
      <!-- 1. 视频分辨率 -->
      <div style="display:flex;align-items:center;gap:8px;background:#1e293b;padding:5px 10px;border-radius:9px;border:1px solid var(--border);">
        <span style="font-size:12px;color:#94a3b8;white-space:nowrap;font-weight:600;">分辨率</span>
        <select id="videoRes" class="cu-select" style="flex:1;padding:4px 6px;font-size:12.5px;font-weight:700;border:none;background:transparent;text-align:center;text-align-last:center;min-width:0;">
          <option value="832x480" selected>832×480 (16:9 横屏·推荐)</option>
          <option value="480x832">480×832 (9:16 竖屏)</option>
          <option value="1280x720">1280×720 (720P 横屏)</option>
          <option value="720x1280">720×1280 (720P 竖屏)</option>
          <option value="640x640">640×640 (1:1 方形)</option>
        </select>
      </div>

      <!-- 2. 采样步数 -->
      <div id="stepsControlWrap" style="display:flex;align-items:center;gap:8px;background:#1e293b;padding:6px 12px;border-radius:9px;border:1px solid var(--border);">
        <span style="font-size:12px;color:#94a3b8;white-space:nowrap;font-weight:600;">步数</span>
        <input type="range" id="videoSteps" min="10" max="60" step="2" value="30" oninput="document.getElementById('stepsValLabel').innerText = this.value + ' 步'" style="flex:1;accent-color:#3b82f6;cursor:pointer;min-width:60px;">
        <span id="stepsValLabel" style="font-size:12.5px;font-weight:700;color:#f8fafc;white-space:nowrap;min-width:36px;text-align:right;">30 步</span>
      </div>

      <!-- 3. 随机种子 -->
      <div style="display:flex;align-items:center;justify-content:space-between;gap:6px;background:#1e293b;padding:5px 10px;border-radius:9px;border:1px solid var(--border);">
        <span style="font-size:12px;color:#94a3b8;white-space:nowrap;font-weight:600;">种子</span>
        <input id="videoSeed" type="number" value="42" style="width:74px;padding:3px 6px;border-radius:6px;border:1px solid #334155;background:#0f172a;color:#f8fafc;font-size:12.5px;text-align:center;">
        <label style="font-size:12px;color:#f8fafc;display:inline-flex;align-items:center;gap:4px;cursor:pointer;font-weight:600;white-space:nowrap;">
          <input id="videoRandSeed" type="checkbox" checked style="width:14px;height:14px;accent-color:#3b82f6;"> 随机
        </label>
        <input type="hidden" id="videoFps" value="24">
      </div>

      <!-- 4. 批量连抽 -->
      <div id="batchControlWrap" style="display:flex;align-items:center;gap:8px;background:#1e293b;padding:4px 8px;border-radius:9px;border:1px solid var(--border);">
        <span style="font-size:12px;color:#94a3b8;white-space:nowrap;font-weight:600;padding-left:2px;">连抽</span>
        <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:4px;flex:1;" id="batchCountGroup">
          <button type="button" class="batch-pill" data-batch="1" onclick="selectBatchCount(1)" style="padding:4px 0;border-radius:6px;border:none;background:#3b82f6;color:#ffffff;font-size:12px;font-weight:700;cursor:pointer;">1条</button>
          <button type="button" class="batch-pill" data-batch="2" onclick="selectBatchCount(2)" style="padding:4px 0;border-radius:6px;border:none;background:transparent;color:#cbd5e1;font-size:12px;font-weight:600;cursor:pointer;">2条</button>
          <button type="button" class="batch-pill" data-batch="3" onclick="selectBatchCount(3)" style="padding:4px 0;border-radius:6px;border:none;background:transparent;color:#cbd5e1;font-size:12px;font-weight:600;cursor:pointer;">3条</button>
          <button type="button" class="batch-pill" data-batch="4" onclick="selectBatchCount(4)" style="padding:4px 0;border-radius:6px;border:none;background:transparent;color:#cbd5e1;font-size:12px;font-weight:600;cursor:pointer;">4条</button>
        </div>
      </div>
    </div>

    <!-- 第 2 排横条：左侧「视频总帧数滑块 + 4s~20s 快捷按钮 + 自动 2K 超分开关」，右侧双按钮「📸 仅生成 4K 图片」+「🎬 生成有声视频」 -->
    <div id="specRow2ActionGrid" style="display:grid;grid-template-columns: 1fr 430px;gap:12px;align-items:center;">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;background:#1e293b;padding:6px 12px;border-radius:9px;border:1px solid var(--border);flex-wrap:wrap;">
        <div id="framesSliderInner" style="display:flex;align-items:center;gap:10px;flex:1;flex-wrap:wrap;">
          <span style="font-size:12px;color:#94a3b8;white-space:nowrap;font-weight:600;">时长/帧数</span>
          <input type="range" id="videoFrames" min="90" max="481" step="17" value="124" oninput="onFramesSliderChange(this.value)" style="width:115px;accent-color:#3b82f6;cursor:pointer;">
          <span id="framesValLabel" style="font-size:12px;font-weight:700;color:#f8fafc;white-space:nowrap;min-width:102px;">124 帧 · 约 5.2 秒</span>
          <div style="display:grid;grid-template-columns:repeat(7, 1fr);gap:4px;flex:1;min-width:235px;" id="durationPills">
            <button type="button" class="dur-pill" data-frames="90" onclick="selectDurationFrames(90)" style="padding:4px 0;border-radius:6px;border:1px solid var(--border);background:#0f172a;color:#cbd5e1;font-size:11.5px;font-weight:600;cursor:pointer;">4s</button>
            <button type="button" class="dur-pill" data-frames="124" onclick="selectDurationFrames(124)" style="padding:4px 0;border-radius:6px;border:1px solid #3b82f6;background:rgba(59,130,246,0.22);color:#60a5fa;font-size:11.5px;font-weight:700;cursor:pointer;">5s</button>
            <button type="button" class="dur-pill" data-frames="192" onclick="selectDurationFrames(192)" style="padding:4px 0;border-radius:6px;border:1px solid var(--border);background:#0f172a;color:#cbd5e1;font-size:11.5px;font-weight:600;cursor:pointer;">8s</button>
            <button type="button" class="dur-pill" data-frames="243" onclick="selectDurationFrames(243)" style="padding:4px 0;border-radius:6px;border:1px solid var(--border);background:#0f172a;color:#cbd5e1;font-size:11.5px;font-weight:600;cursor:pointer;">10s</button>
            <button type="button" class="dur-pill" data-frames="294" onclick="selectDurationFrames(294)" style="padding:4px 0;border-radius:6px;border:1px solid var(--border);background:#0f172a;color:#cbd5e1;font-size:11.5px;font-weight:600;cursor:pointer;">12s</button>
            <button type="button" class="dur-pill" data-frames="362" onclick="selectDurationFrames(362)" style="padding:4px 0;border-radius:6px;border:1px solid var(--border);background:#0f172a;color:#cbd5e1;font-size:11.5px;font-weight:600;cursor:pointer;">15s</button>
            <button type="button" class="dur-pill" data-frames="481" title="单次 20 秒实验档位 (建议在 832×480 / 480×832 分辨率下使用；若需 30s~60s 满血画质推荐用下方「🔗 尾帧续写」)" onclick="selectDurationFrames(481)" style="padding:4px 0;border-radius:6px;border:1px solid var(--border);background:#0f172a;color:#cbd5e1;font-size:11.5px;font-weight:600;cursor:pointer;">20s</button>
          </div>
        </div>
        <div id="avatarSpecHint" style="display:none;font-size:12px;font-weight:600;color:#34d399;">
          ⚡ LongCat-Avatar-1.6：固定 8 步双蒸馏 (DMD + OPSD) · 声音与口型由上方驱动音频决定 · 视频时长由上方「生成段数」控制
        </div>
        <label id="autoUpscale2kLabel" title="勾选后：生成完 480p 视频后，自动接力调用 ReCaVSR (Wan2.2-5B) 单步流式超分至 2K 超清并保留原生音轨" style="display:inline-flex;align-items:center;gap:5px;padding:3px 9px;border-radius:7px;background:rgba(139,92,246,0.16);border:1px solid rgba(139,92,246,0.42);color:#c4b5fd;font-size:11.5px;font-weight:700;cursor:pointer;white-space:nowrap;user-select:none;">
          <input id="autoUpscale2kChk" type="checkbox" style="width:13.5px;height:13.5px;accent-color:#8b5cf6;cursor:pointer;">
          <span>✨ 生成后自动 2K 超分</span>
        </label>
        <span id="h3ImagePureTag" style="display:none;align-items:center;gap:5px;padding:3px 9px;border-radius:7px;background:rgba(244,114,182,0.15);border:1px solid rgba(244,114,182,0.42);color:#f9a8d4;font-size:11.5px;font-weight:700;white-space:nowrap;">
          🖼️ 纯图片直出 (不生成视频 · 满意后可在右侧点「🎭 送入数字人说话」)
        </span>
      </div>

      <div style="display:flex;gap:8px;width:100%;height:100%;">
        <button type="button" id="btnImageOnlyGen" onclick="startVideoGenerate('image')" title="仅生成原生 4K 高清写真/换装 PNG 图片（约 11~17 秒，不生成视频）" style="flex:1;min-height:38px;display:inline-flex;align-items:center;justify-content:center;padding:8px 10px;font-size:13.5px;font-weight:800;border-radius:9px;border:1px solid rgba(244,114,182,0.55);background:linear-gradient(135deg, #ec4899, #9333ea);color:#ffffff;cursor:pointer;box-shadow:0 6px 16px -4px rgba(236,72,153,0.45);white-space:nowrap;">
          📸 仅生成 4K 图片
        </button>
        <button class="btn-primary" style="flex:1.15;min-height:38px;justify-content:center;padding:8px 12px;font-size:13.5px;font-weight:800;border-radius:9px;box-shadow:0 6px 16px -4px rgba(59,130,246,0.5);white-space:nowrap;" onclick="startVideoGenerate('video')" id="btnVideoGen">
          🎬 生成有声视频
        </button>
      </div>
    </div>

    <div id="videoProgressWrap" style="display:none;margin-top:8px;">
      <div class="progress-bar" style="height:7px;"><div class="progress-fill green" id="videoProgressBar" style="width:5%;"></div></div>
    </div>
  </div>

  <!-- ==================== 第 3 行 (单独一行 100% 宽度，1:1 复刻原版 EasyColab 预览图)：▶️ 生成结果预览 (左大屏) + 🕒 本地历史 (右列表) ==================== -->
  <div style="display:grid;grid-template-columns: minmax(0, 2.8fr) minmax(265px, 1fr);gap:16px;align-items:stretch;width:100%;box-sizing:border-box;">
    <!-- 左卡片：▶️ 生成结果预览 (超大屏 16:9 画面) -->
    <div style="background:#0f172a;border:1px solid var(--border);border-radius:14px;padding:18px;display:flex;flex-direction:column;justify-content:space-between;min-width:0;overflow:hidden;">
      <!-- 顶部标题栏：左「▶️ 生成结果预览」，右「⏱ 生成耗时 8分31.9秒」蓝色胶囊 -->
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:8px;font-size:15.5px;font-weight:800;color:#f8fafc;">
          <span style="display:inline-flex;align-items:center;justify-content:center;width:22px;height:18px;border:1.8px solid #3b82f6;border-radius:5px;color:#3b82f6;font-size:10px;">▶</span>
          <span>生成结果预览</span>
        </div>
        <span id="previewElapsedBadge" style="display:none;background:rgba(59,130,246,0.18);color:#60a5fa;border:1px solid rgba(59,130,246,0.35);padding:4px 12px;border-radius:999px;font-size:12px;font-weight:700;">
          ⏱ 生成耗时 --
        </span>
      </div>

      <!-- 中部超大尺寸 16:9 播放器 (height: 520px) -->
      <div id="videoEmptyPlaceholder" style="height:520px;border-radius:12px;background:#000000;border:1px dashed #334155;display:flex;flex-direction:column;align-items:center;justify-content:center;color:var(--muted);font-size:13.5px;text-align:center;padding:24px;">
        <div style="font-size:40px;margin-bottom:10px;">🎬</div>
        <div style="color:#e2e8f0;font-weight:700;">点击右侧「本地历史」即可直接播放已生成的视频，或在上方点击「开始生成音视频」</div>
        <div style="font-size:12px;margin-top:6px;color:#94a3b8;">本机视频保存位置：<code style="color:#93c5fd;">~/Movies/EasyColab/</code>（同时自动备份至 Google Drive）</div>
      </div>

      <div id="videoResultWrap" style="display:none;">
        <video id="mmVideoPlayer" controls preload="metadata" style="width:100%;height:520px;border-radius:12px;background:#000000;object-fit:contain;display:block;"></video>
        <img id="mmImagePreview" alt="Generated Image" style="display:none;width:100%;height:520px;border-radius:12px;background:#000000;object-fit:contain;">
      </div>

      <!-- 底部文件信息栏：单行紧凑对齐 -->
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:nowrap;gap:10px;margin-top:12px;padding-top:4px;">
        <div style="display:flex;align-items:center;gap:6px;min-width:0;flex:1;overflow:hidden;">
          <span style="color:#94a3b8;font-size:14px;flex-shrink:0;">🎞️</span>
          <span id="mmVideoMeta" title="" style="font-size:12.5px;color:#cbd5e1;font-family:'SF Mono',Menlo,monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">尚未选择视频</span>
          <span id="previewBottomShortBadge" style="display:none;background:#1e293b;color:#94a3b8;border:1px solid var(--border);padding:1px 7px;border-radius:999px;font-size:11px;font-weight:700;flex-shrink:0;">--</span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;flex-shrink:0;white-space:nowrap;">
          <div id="imageActionBtns" style="display:none;align-items:center;gap:6px;">
            <button type="button" onclick="sendCurrentPreviewToDramaRole()" style="padding:5px 10px;font-size:12px;font-weight:800;border-radius:6px;border:1px solid rgba(192,132,252,0.65);background:rgba(168,85,247,0.22);color:#e9d5ff;cursor:pointer;" title="将当前预览图直接锁定为短剧角色A/B/C/D/E的 16:9 基准定妆照">📌 锁入短剧角色</button>
            <button type="button" onclick="sendCurrentImageToAvatar()" style="padding:5px 10px;font-size:12px;font-weight:800;border-radius:6px;border:1px solid rgba(16,185,129,0.55);background:rgba(16,185,129,0.2);color:#6ee7b7;cursor:pointer;" title="将当前图片直接设为 LongCat-Avatar-1.6 数字人参考图，驱动她开口说话">🎭 送入数字人说话</button>
            <button type="button" onclick="sendCurrentImageToH3Edit()" style="padding:5px 10px;font-size:12px;font-weight:800;border-radius:6px;border:1px solid rgba(236,72,153,0.55);background:rgba(236,72,153,0.2);color:#fbcfe8;cursor:pointer;" title="将当前图片设为 &lt;Picture 1&gt;，免涂 Mask 像素级锁脸换装">👗 设为&lt;Picture 1&gt;换装</button>
          </div>
          <div id="videoUpscaleActionWrap" style="display:inline-flex;align-items:center;gap:4px;background:rgba(139,92,246,0.14);border:1px solid rgba(139,92,246,0.45);border-radius:8px;padding:2px 5px;">
            <select id="upscalePresetSelect" class="cu-select" style="border:none;background:transparent;color:#ddd6fe;font-size:11.5px;font-weight:700;padding:2px 3px;cursor:pointer;" onchange="onUpscalePresetChange(this.value)">
              <option value="2k" selected>2K (1440P)</option>
              <option value="1080p">1080P</option>
              <option value="4k">4K 极清</option>
            </select>
            <button type="button" id="btnUpscaleVideo" onclick="upscaleCurrentVideo()" style="padding:4px 9px;font-size:12px;font-weight:800;border-radius:6px;border:none;background:linear-gradient(135deg,#8b5cf6,#6366f1);color:#ffffff;cursor:pointer;">
              ✨ 超分至 2K
            </button>
          </div>
          <input type="file" id="inpExternalVideo" accept="video/*,.mp4,.mov,.webm,.mkv" style="display:none;" onchange="handleExternalVideoImport(this)">
          <button type="button" id="btnContinueLastFrame" class="btn-outline" title="截取当前视频最后一帧作为「首帧参考图」，无缝续写下一段视频" style="padding:5px 10px;font-size:12px;font-weight:700;background:rgba(59,130,246,0.14);border-color:rgba(59,130,246,0.45);color:#93c5fd;" onclick="continueFromCurrentVideoLastFrame()">🔗 尾帧续写</button>
          <button type="button" id="btnReplayVideo" class="btn-outline" style="padding:5px 10px;font-size:12px;font-weight:700;background:#1e293b;border-color:#334155;color:#e2e8f0;" onclick="replayCurrentVideo()">重播</button>
          <button type="button" class="btn-outline" style="padding:5px 10px;font-size:12px;font-weight:700;background:#1e293b;border-color:#334155;color:#e2e8f0;" onclick="revealInFinder(false)">Finder</button>
          <a id="mmVideoDownload" href="#" download="minimax_h3_output.mp4" class="btn-outline" style="display:none;padding:5px 10px;font-size:12px;text-decoration:none;border-color:#10b981;color:#10b981;">⬇️ 另存为</a>
        </div>
      </div>
    </div>

    <!-- 右卡片：🕒 本地历史 (~/Movies/EasyColab) -->
    <div style="background:#0f172a;border:1px solid var(--border);border-radius:14px;padding:16px;display:flex;flex-direction:column;min-width:0;overflow:hidden;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:6px;">
          <span style="color:#3b82f6;font-size:15px;">🕒</span>
          <span style="font-size:14.5px;font-weight:800;color:#f8fafc;">本地历史</span>
          <span id="localHistoryCount" style="font-size:11.5px;color:#94a3b8;">(0)</span>
        </div>
        <div style="display:flex;align-items:center;gap:4px;">
          <button type="button" title="导入本机其他视频到列表中 (支持超分与尾帧续写)" onclick="triggerImportExternalVideo(false)" style="background:rgba(59,130,246,0.14);border:1px solid rgba(59,130,246,0.4);color:#93c5fd;font-size:11.5px;font-weight:700;cursor:pointer;padding:3px 9px;border-radius:6px;">＋ 导入视频</button>
          <button type="button" title="刷新本地历史列表" onclick="loadLocalVideoHistory(false)" style="background:transparent;border:none;color:#94a3b8;font-size:15px;cursor:pointer;padding:2px 6px;border-radius:6px;">↻</button>
        </div>
      </div>

      <div id="mmHistoryList" style="display:flex;flex-direction:column;gap:8px;max-height:515px;overflow-y:auto;padding-right:2px;flex:1;">
        <div style="font-size:12px;color:var(--muted);padding:12px 0;text-align:center;">正在扫描...</div>
      </div>

      <div style="margin-top:10px;padding-top:10px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#64748b;">
        <span title="~/Movies/EasyColab">📂 ~/Movies/EasyColab</span>
        <a href="javascript:void(0)" onclick="revealInFinder(true)" style="color:#60a5fa;text-decoration:none;font-weight:600;">打开文件夹 ↗</a>
      </div>
    </div>
  </div>
</div>

<div class="card">
  <h2>
    <span>📜 实时任务进度与部署日志</span>
    <div style="display:inline-flex;align-items:center;gap:8px;">
      <span id="deployStateText" style="font-size:12px;color:var(--muted)">空闲</span>
      <button type="button" onclick="clearDeployLogs()" style="padding:2px 8px;font-size:11px;border-radius:6px;border:1px solid var(--border);background:transparent;color:#94a3b8;cursor:pointer;" title="清空下方历史日志">🧹 清空</button>
      <button type="button" id="btnStopTaskBottom" onclick="stopActiveVideoTask()" style="display:none;padding:3px 10px;font-size:11.5px;font-weight:800;border-radius:6px;border:1px solid rgba(239,68,68,0.65);background:rgba(239,68,68,0.18);color:#fca5a5;cursor:pointer;">🛑 停止当前任务</button>
    </div>
  </h2>
  <div id="taskProgressArea" style="margin-bottom:14px;padding:12px 14px;background:#0f172a;border:1px solid var(--border);border-radius:10px;">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:14px;font-size:13px;margin-bottom:6px;height:20px;overflow:hidden;">
      <span id="taskProgressLabel" style="font-weight:600;color:#93c5fd;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;min-width:0;">当前进度：已就绪 (100%)</span>
      <span id="taskProgressSpeed" style="font-family:'SF Mono',monospace;color:var(--green);white-space:nowrap;flex-shrink:0;">50.90 / 50.90 GB · 服务运行中</span>
    </div>
    <div class="progress-bar" style="height:10px;">
      <div class="progress-fill green" id="taskProgressBar" style="width:100%;transition:width 0.35s ease;"></div>
    </div>
  </div>
  <pre class="log-box" id="logBox">等待操作... 你可以点击上方「🔄 刷新显存与硬盘」、「▶️ 一键开机 / 启动服务」或「🛑 关闭服务器」。</pre>
</div>

<!-- 添加自定义 HuggingFace 模型仓库 模态弹窗 (对齐 EasyColab) -->
<div id="customHFModal" style="display:none;position:fixed;inset:0;background:rgba(2,6,23,0.78);backdrop-filter:blur(4px);z-index:9999;align-items:center;justify-content:center;padding:20px;">
  <div style="width:100%;max-width:760px;max-height:90vh;overflow-y:auto;background:#1e293b;border:1px solid #334155;border-radius:16px;padding:20px;box-shadow:0 25px 50px -12px rgba(0,0,0,0.6);">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
      <div>
        <div style="font-size:17px;font-weight:800;color:#f8fafc;">➕ 添加自定义 HuggingFace 模型仓库</div>
        <div style="font-size:12px;color:var(--muted);margin-top:2px;">支持任意 HuggingFace GGUF 或 Safetensors 仓库，自动探测分片体积与 RTX PRO 6000 显存匹配度</div>
      </div>
      <button class="btn-outline" style="padding:4px 10px;font-size:14px;" onclick="closeCustomHFModal()">✕</button>
    </div>

    <!-- 实时趋势榜与在线搜索区 -->
    <div style="background:#0f172a;border:1px solid var(--border);border-radius:12px;padding:12px;margin-bottom:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px;">
        <div style="display:flex;gap:6px;flex-wrap:wrap;" id="hfCatTabs">
          <button class="btn-outline" style="padding:4px 10px;font-size:12px;border-color:#3b82f6;color:#60a5fa;" onclick="fetchTrendingHF('all', false)" data-cat="all">🔥 综合热榜</button>
          <button class="btn-outline" style="padding:4px 10px;font-size:12px;" onclick="fetchTrendingHF('gguf', false)" data-cat="gguf">📦 GGUF 量化</button>
          <button class="btn-outline" style="padding:4px 10px;font-size:12px;" onclick="fetchTrendingHF('llm_vlm', false)" data-cat="llm_vlm">🧠 对话/视觉</button>
          <button class="btn-outline" style="padding:4px 10px;font-size:12px;" onclick="fetchTrendingHF('image_video', false)" data-cat="image_video">🎨 生图/视频</button>
        </div>
        <div style="display:flex;gap:6px;flex:1;min-width:220px;">
          <input id="hfSearchInput" type="text" placeholder="在线搜索 HF 模型 (如 Qwen3, DeepSeek, Flux)..." style="flex:1;padding:5px 10px;font-size:12px;border-radius:8px;border:1px solid var(--border);background:#1e293b;color:#f8fafc;" onkeydown="if(event.key==='Enter')fetchTrendingHF(currentHFCat, true)">
          <button class="btn-primary" style="padding:5px 12px;font-size:12px;" onclick="fetchTrendingHF(currentHFCat, true)">🔍 搜索</button>
        </div>
      </div>

      <div id="hfTrendingGrid" style="display:grid;grid-template-columns:repeat(2, 1fr);gap:8px;max-height:195px;overflow-y:auto;padding-right:4px;">
        <div style="color:var(--muted);font-size:12px;padding:8px;">正在加载 HuggingFace 实时热榜...</div>
      </div>
    </div>

    <!-- 仓库分片详情探测器 (点击后展开) -->
    <div id="hfDetailBox" style="display:none;background:#0f172a;border:1px solid #3b82f6;border-radius:10px;padding:10px 12px;margin-bottom:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <span id="hfDetailTitle" style="font-size:12.5px;font-weight:700;color:#93c5fd;">📦 仓库分片明细</span>
        <span style="font-size:11px;color:var(--muted);">点击任意文件可直接填入下方过滤条件</span>
      </div>
      <div id="hfDetailFiles" style="max-height:140px;overflow-y:auto;display:flex;flex-wrap:wrap;gap:6px;"></div>
    </div>

    <!-- 表单配置区 -->
    <div class="form-group">
      <label>1. HuggingFace 仓库地址 (Repo ID 或完整 URL)</label>
      <div style="display:flex;gap:8px;">
        <input id="custRepo" type="text" placeholder="例如: unsloth/Qwen3.8-27B-GGUF 或 https://huggingface.co/..." style="flex:1;" oninput="autoFillCustomFolder()">
        <button class="btn-outline" style="padding:6px 12px;font-size:12px;border-color:#3b82f6;color:#93c5fd;white-space:nowrap;" onclick="inspectHFRepo()">🔎 查看模型详情与分片体积</button>
      </div>
    </div>

    <div class="form-group">
      <label>2. 文件名 / 通配符过滤 (Include Pattern)</label>
      <input id="custPattern" type="text" value="*Q4_K_M*.gguf" placeholder="例如: *Q4_K_M*.gguf 或 *.safetensors">
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;">
        <span style="font-size:11px;color:var(--muted);line-height:22px;">快捷过滤:</span>
        <button type="button" class="btn-outline" style="padding:2px 8px;font-size:11px;" onclick="document.getElementById('custPattern').value='*Q4_K_M*.gguf'">*Q4_K_M*.gguf</button>
        <button type="button" class="btn-outline" style="padding:2px 8px;font-size:11px;" onclick="document.getElementById('custPattern').value='*Q8_0*.gguf'">*Q8_0*.gguf</button>
        <button type="button" class="btn-outline" style="padding:2px 8px;font-size:11px;" onclick="document.getElementById('custPattern').value='*bf16*.gguf'">*bf16*.gguf</button>
        <button type="button" class="btn-outline" style="padding:2px 8px;font-size:11px;" onclick="document.getElementById('custPattern').value='*.safetensors'">*.safetensors (全量分片)</button>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
      <div class="form-group">
        <label>3. 自定义显示名称 (可选)</label>
        <input id="custName" type="text" placeholder="留空则默认使用仓库名">
      </div>
      <div class="form-group">
        <label>4. 云盘 `MyDrive/models/` 子文件夹名</label>
        <input id="custFolder" type="text" placeholder="自动与仓库名同步">
      </div>
    </div>

    <div class="form-group" style="margin-bottom:16px;">
      <label>5. HuggingFace Token (`hf_...` 受限/私有模型选填，填写后自动保存)</label>
      <input id="custToken" type="password" placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxx (公开模型无需填写)">
    </div>

    <div style="display:flex;justify-content:flex-end;gap:10px;border-top:1px solid var(--border);padding-top:14px;">
      <button class="btn-outline" onclick="closeCustomHFModal()">取消</button>
      <button class="btn-primary" onclick="saveCustomHFModel()">➕ 添加到开源模型中心（并自动选中）</button>
    </div>
  </div>
</div>

<!-- Google Colab Pro & 云盘多账号管理与秒切模态弹窗 -->
<div id="accountSwitchModal" style="display:none;position:fixed;inset:0;background:rgba(2,6,23,0.78);backdrop-filter:blur(4px);z-index:9999;align-items:center;justify-content:center;padding:20px;">
  <div style="width:100%;max-width:560px;max-height:90vh;overflow-y:auto;background:#1e293b;border:1px solid #334155;border-radius:14px;padding:18px;box-shadow:0 25px 50px -12px rgba(0,0,0,0.6);">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
      <div style="font-size:16px;font-weight:800;color:#f8fafc;">👤 Google Colab 账号管理与切换</div>
      <div style="display:flex;align-items:center;gap:6px;">
        <button type="button" class="btn-outline" id="btnRefreshAllAccCu" style="padding:3px 9px;font-size:11.5px;border-color:#3b82f6;color:#93c5fd;" onclick="refreshAllAccountsQuota(false)" title="向 Google Colab 服务端强制刷新所有已绑定账号的实时 CU 算力余额与 G4 权限">🔄 刷新全部账号额度</button>
        <button class="btn-outline" style="padding:3px 9px;font-size:13px;" onclick="closeAccountModal()">✕</button>
      </div>
    </div>

    <label style="display:flex;align-items:center;gap:8px;cursor:pointer;padding:8px 12px;margin-bottom:12px;border-radius:8px;background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);font-size:12px;color:#d1fae5;">
      <input type="checkbox" id="chkShareExistingDrive" checked style="width:14px;height:14px;accent-color:#10b981;flex-shrink:0;">
      <span><b>🔗 共享当前 Google Drive 云盘模型库</b>（切换账号免重下 134GB 模型）</span>
    </label>

    <div style="margin-bottom:12px;">
      <div id="savedAccountsList" style="display:flex;flex-direction:column;gap:6px;">
        <div style="font-size:12px;color:var(--muted);padding:10px;text-align:center;">正在读取已绑定的账号...</div>
      </div>
    </div>

    <!-- 简洁添加新账号区 -->
    <div style="padding:12px;margin-bottom:10px;border-radius:10px;background:#0f172a;border:1px solid #334155;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
        <span style="font-size:12.5px;font-weight:700;color:#f8fafc;">➕ 添加新账号</span>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <a id="linkOAuthEasyColab" href="#" target="_blank" rel="noopener" style="padding:5px 11px;border-radius:6px;background:#2563eb;color:#ffffff;font-size:12px;font-weight:700;text-decoration:none;">
            🌐 1. 打开授权页选号取码 ↗
          </a>
          <a id="linkOAuthSilent" href="#" target="_blank" rel="noopener" style="padding:5px 10px;border-radius:6px;background:#1e293b;border:1px solid #475569;color:#cbd5e1;font-size:11.5px;text-decoration:none;" title="不弹账号选择器，直接沿用当前浏览器默认已登录账号">
            ⚡ 当前已登录号直通 ↗
          </a>
        </div>
      </div>
      <div style="display:flex;gap:8px;">
        <input id="inpRemoteAuthCode" type="text" placeholder="2. 粘贴网页弹出的授权码 (4/0A...)" style="flex:1;min-width:180px;padding:7px 10px;font-size:12.5px;border-radius:7px;border:1px solid #334155;background:#1e293b;color:#f8fafc;" onkeydown="if(event.key==='Enter')submitRemoteAuthCode()">
        <button type="button" class="btn-success" style="padding:7px 16px;font-size:12.5px;font-weight:800;white-space:nowrap;" onclick="submitRemoteAuthCode()" id="btnSubmitRemoteCode">✅ 绑定</button>
      </div>
    </div>

    <!-- 折叠收纳：仅在个别账号卡验证时展开 -->
    <details style="margin-bottom:12px;border-radius:8px;background:#0f172a;border:1px solid #334155;padding:8px 12px;">
      <summary style="cursor:pointer;font-size:12px;color:#94a3b8;font-weight:600;user-select:none;">
        💡 个别账号卡安全码验证？点此展开「网页版 Colab 隧道直连」
      </summary>
      <div style="margin-top:8px;padding-top:8px;border-top:1px solid #1e293b;">
        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;">
          <button type="button" class="btn-primary" style="padding:5px 10px;font-size:11.5px;" onclick="copyColabWebBootstrapCode()">📋 1. 复制启动代码</button>
          <a href="https://colab.research.google.com/github/wwintj/colab-g4-studio/blob/main/Colab_G4_Server.ipynb" target="_blank" rel="noopener" style="padding:5px 10px;border-radius:6px;background:#1e293b;border:1px solid #3b82f6;color:#93c5fd;font-size:11.5px;font-weight:600;text-decoration:none;">
            📓 2. 打开 Colab 预置笔记本 (选 G4 运行) ↗
          </a>
        </div>
        <div style="display:flex;gap:6px;">
          <input id="inpManualTunnelUrl" type="text" placeholder="3. 粘贴跑完输出的 trycloudflare.com 地址..." style="flex:1;padding:6px 10px;font-size:12px;border-radius:6px;border:1px solid #334155;background:#1e293b;color:#f8fafc;" onkeydown="if(event.key==='Enter')connectManualTunnelUrl()">
          <button type="button" class="btn-success" style="padding:6px 12px;font-size:12px;font-weight:700;white-space:nowrap;" onclick="connectManualTunnelUrl()" id="btnConnectTunnel">⚡ 接管</button>
        </div>
      </div>
    </details>

    <div id="accountActionStatus" style="display:none;padding:8px 10px;margin-bottom:10px;border-radius:8px;background:#0f172a;border:1px solid var(--border);font-size:12px;color:#93c5fd;"></div>

    <div style="display:flex;justify-content:space-between;align-items:center;border-top:1px solid var(--border);padding-top:12px;">
      <button class="btn-outline" style="padding:5px 10px;font-size:11.5px;border-color:rgba(245,158,11,0.4);color:#fcd34d;" onclick="rebindGoogleDriveAccount()" id="btnRebindDrive">☁️ 单独更换云盘账号</button>
      <button class="btn-outline" style="padding:5px 14px;font-size:12px;" onclick="closeAccountModal()">关闭</button>
    </div>
  </div>
</div>

<script>
let currentHFCat = 'all';
let customModelsCache = [];
let cachedAccountInfo = {active_email: '', accounts: []};

function renderAccountBadgeAndModal(accInfo) {
  if (!accInfo) return;
  cachedAccountInfo = accInfo;
  const lbl = document.getElementById('headerAccountLabel');
  if (lbl) {
    lbl.innerText = accInfo.active_email || '未登录 (点击绑定)';
  }
  const listEl = document.getElementById('savedAccountsList');
  if (!listEl) return;
  const arr = Array.isArray(accInfo.accounts) ? accInfo.accounts : [];
  if (!arr.length) {
    listEl.innerHTML = `<div style="font-size:12px;color:var(--muted);padding:12px;background:#0f172a;border-radius:8px;text-align:center;">暂未检测到已保存账号，请点击下方「➕ 登录并添加新 Gmail Pro 账号」</div>`;
    return;
  }
  listEl.innerHTML = arr.map(it => {
    const isAct = Boolean(it.is_active);
    const borderCol = isAct ? '#10b981' : 'var(--border)';
    const bgCol = isAct ? 'rgba(16,185,129,0.12)' : '#0f172a';
    const cuNum = (it.cu_balance !== null && typeof it.cu_balance !== 'undefined') ? Number(it.cu_balance) : null;
    const hasG4 = (cuNum !== null && cuNum > 0) || Boolean(it.g4_eligible);
    const g4Label = (cuNum !== null && cuNum > 0) ? `⚡ ${cuNum.toFixed(2)} CU · RTX 6000` : `⚡ Ultra · RTX 6000`;
    const cuBadge = cuNum === null && !hasG4
      ? `<span style="font-size:11px;padding:1px 7px;border-radius:999px;background:rgba(148,163,184,0.18);color:#cbd5e1;font-weight:700;">-- CU</span>`
      : (hasG4
          ? `<span style="font-size:11px;padding:1px 7px;border-radius:999px;background:rgba(16,185,129,0.2);color:#34d399;font-weight:700;">${g4Label}</span>`
          : `<span style="font-size:11px;padding:1px 7px;border-radius:999px;background:rgba(59,130,246,0.18);color:#93c5fd;font-weight:700;" title="Google Colab 服务端当前返回该账号 0.00 CU · eligibleGpus=['T4']">⚡ T4 (16GB · ≈3~5h/天)</span>`);
    const actBtn = isAct
      ? `<span style="padding:3px 9px;border-radius:6px;background:rgba(16,185,129,0.22);color:#34d399;font-size:11.5px;font-weight:700;">✅ 当前使用</span>`
      : `<button type="button" class="btn-primary" style="padding:4px 10px;font-size:11.5px;" onclick="switchColabAccount('${escapeHtml(it.email)}')">⚡ 切换</button>`;
    return `
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;padding:8px 12px;border-radius:8px;border:1px solid ${borderCol};background:${bgCol};">
        <div style="display:flex;align-items:center;gap:8px;min-width:0;flex-wrap:wrap;">
          <span style="font-size:13px;font-weight:700;color:#f8fafc;">${escapeHtml(it.email)}</span>
          ${cuBadge}
        </div>
        <div style="display:flex;align-items:center;gap:6px;flex-shrink:0;">
          ${actBtn}
          <button type="button" title="移除账号" onclick="deleteSavedAccount('${escapeHtml(it.email)}')" style="background:transparent;border:none;color:#64748b;font-size:13px;cursor:pointer;padding:2px 5px;" onmouseover="this.style.color='#ef4444'" onmouseout="this.style.color='#64748b'">✕</button>
        </div>
      </div>`;
  }).join('');
}

async function refreshModalOAuthLinks() {
  try {
    const r = await fetch('/api/accounts/remote_url', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({open_browser: false}),
    });
    const d = await r.json();
    if (d.ok) {
      const aSilent = document.getElementById('linkOAuthSilent');
      const aCli = document.getElementById('linkOAuthColabCli');
      const aEasy = document.getElementById('linkOAuthEasyColab');
      if (aSilent && d.url_silent_light) aSilent.href = d.url_silent_light;
      if (aCli && d.url_colab_cli) aCli.href = d.url_colab_cli;
      if (aEasy && d.url_easycolab) aEasy.href = d.url_easycolab;
    }
  } catch (e) {}
}

function openAccountModal() {
  const m = document.getElementById('accountSwitchModal');
  if (m) m.style.display = 'flex';
  const st = document.getElementById('accountActionStatus');
  if (st) st.style.display = 'none';
  renderAccountModalList();
  refreshModalOAuthLinks();
  refreshAllAccountsQuota(true);
}

async function refreshAllAccountsQuota(silent = false) {
  const btn = document.getElementById('btnRefreshAllAccCu');
  const st = document.getElementById('accountActionStatus');
  if (btn) {
    btn.disabled = true;
    btn.innerText = '⏳ 正在向 Google 刷新...';
  }
  if (!silent && st) {
    st.style.display = 'block';
    st.style.color = '#60a5fa';
    st.innerText = '⏳ 正在实时刷新全部已绑定账号的 Google Colab 算力余额 (CU) 与 G4 硬件权限...';
  }
  try {
    const r = await fetch('/api/accounts/refresh_all', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: '{}',
    });
    const d = await r.json();
    if (d.ok && d.account_info) {
      renderAccountBadgeAndModal(d.account_info);
      const actAcc = (d.account_info.accounts || []).find(a => a.is_active);
      if (actAcc && actAcc.g4_eligible) {
        switchPresetTierTab('g4');
      }
      if (!silent && st) {
        if (actAcc && !actAcc.g4_eligible) {
          st.style.color = '#f59e0b';
          st.innerText = `⚠️ 已向 Google Colab 实时刷新：当前账号 (${actAcc.email}) 在 Colab 后端接口 (/tun/m/ccu-info) 返回余额仍为 0.00 CU、可用 GPU 为 ["T4"]。若您刚升级 Google AI Ultra，请先在已登录该账号的浏览器打开一次 https://colab.research.google.com/ 激活同步 Colab 算力钱包（若网页端已能选 G4，也可直接展开下方「网页版 Colab 隧道直连」10秒接管）！`;
        } else {
          st.style.color = '#10b981';
          st.innerText = '✅ 已成功同步全部账号的最新算力配额与 G4 权限！';
        }
      }
      await refreshStatus(true);
    }
  } catch (e) {
    if (!silent && st) {
      st.style.color = '#ef4444';
      st.innerText = '❌ 刷新失败: ' + e;
    }
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = '🔄 刷新全部账号额度';
    }
  }
}

async function copyColabWebBootstrapCode() {
  const st = document.getElementById('accountActionStatus');
  try {
    const r = await fetch('/api/accounts/notebook_bootstrap', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
    const d = await r.json();
    if (!d.ok || !d.code) {
      if (st) { st.style.display = 'block'; st.style.color = '#ef4444'; st.innerText = '❌ 生成启动代码失败: ' + (d.error || '未知错误'); }
      return;
    }
    await navigator.clipboard.writeText(d.code);
    if (st) {
      st.style.display = 'block';
      st.style.color = '#10b981';
      st.innerText = '✅ 已复制 Colab 网页端一键启动代码（已内置您现有的 134GB 云盘模型权限）！请打开 colab.research.google.com 新建笔记本（选 G4 GPU），粘贴到单元格运行，然后把输出的 https://xxxx.trycloudflare.com 粘到上方第 2 步框内！';
    }
  } catch (e) {
    if (st) { st.style.display = 'block'; st.style.color = '#ef4444'; st.innerText = '❌ 复制失败: ' + e; }
  }
}

async function connectManualTunnelUrl() {
  const inp = document.getElementById('inpManualTunnelUrl');
  const raw = (inp ? inp.value : '').trim();
  const st = document.getElementById('accountActionStatus');
  const btn = document.getElementById('btnConnectTunnel');
  if (!raw || !raw.includes('trycloudflare.com')) {
    alert('请粘贴有效的 https://xxxx.trycloudflare.com 隧道地址');
    return;
  }
  if (btn) { btn.disabled = true; btn.innerText = '⏳ 正在接管连接...'; }
  if (st) {
    st.style.display = 'block';
    st.style.color = '#60a5fa';
    st.innerText = '⏳ 正在连接并接管该 Colab G4 实例隧道...';
  }
  try {
    const r = await fetch('/api/accounts/connect_tunnel', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({tunnel_input: raw}),
    });
    const d = await r.json();
    if (!d.ok) {
      if (st) { st.style.color = '#ef4444'; st.innerText = '❌ 连接失败: ' + (d.error || '未知错误'); }
      return;
    }
    if (st) {
      st.style.color = '#10b981';
      st.innerText = `✅ 已成功接管新账号的 G4 实例 (${d.tunnel_url})！您现在可以直接在下方工作台生成视频、超分或对话了！`;
    }
    await refreshStatus(true);
  } catch (e) {
    if (st) { st.style.color = '#ef4444'; st.innerText = '❌ 请求异常: ' + e; }
  } finally {
    if (btn) { btn.disabled = false; btn.innerText = '⚡ 接管'; }
  }
}

async function generateRemoteAuthUrl() {
  const st = document.getElementById('accountActionStatus');
  const urlInp = document.getElementById('inpRemoteAuthUrl');
  try {
    const r = await fetch('/api/accounts/remote_url', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
    const d = await r.json();
    if (!d.ok || !d.auth_url) {
      if (st) { st.style.display = 'block'; st.style.color = '#ef4444'; st.innerText = '❌ 生成授权链接失败: ' + (d.error || '未知错误'); }
      return;
    }
    if (urlInp) {
      urlInp.value = d.auth_url;
      urlInp.select();
    }
    try { await navigator.clipboard.writeText(d.auth_url); } catch (e) {}
    if (st) {
      st.style.display = 'block';
      st.style.color = '#10b981';
      st.innerText = '✅ 已生成并复制授权链接！请将它粘贴到已登录该 Gmail 账号的浏览器（或发给家人手机/电脑打开），点「允许」后把授权码粘贴到上方第 2 步框内！';
    }
  } catch (e) {
    if (st) { st.style.display = 'block'; st.style.color = '#ef4444'; st.innerText = '❌ 请求异常: ' + e; }
  }
}

async function submitRemoteAuthCode() {
  const codeInp = document.getElementById('inpRemoteAuthCode');
  const code = (codeInp ? codeInp.value : '').trim();
  const st = document.getElementById('accountActionStatus');
  const btn = document.getElementById('btnSubmitRemoteCode');
  if (!code) {
    alert('请先粘贴网页上显示的 Google 授权码');
    return;
  }
  if (btn) { btn.disabled = true; btn.innerText = '⏳ 正在验证绑定...'; }
  if (st) {
    st.style.display = 'block';
    st.style.color = '#60a5fa';
    st.innerText = '⏳ 正在向 Google 验证授权码并获取新账号算力信息...';
  }
  try {
    const r = await fetch('/api/accounts/remote_exchange', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({code}),
    });
    const d = await r.json();
    if (!d.ok) {
      if (st) { st.style.color = '#ef4444'; st.innerText = '❌ 授权码验证失败: ' + (d.error || '请重新点击第 1 步生成新链接后再试'); }
      return;
    }
    if (codeInp) codeInp.value = '';
    if (d.account_info) renderAccountBadgeAndModal(d.account_info);
    refreshModalOAuthLinks();
    if (st) {
      st.style.color = '#10b981';
      st.innerText = `✅ 新账号 ${d.email} 绑定成功并已设为当前激活账号！（已为您永久保存在列表中，随时 1 秒互切）`;
    }
    await refreshStatus(true);
  } catch (e) {
    if (st) { st.style.color = '#ef4444'; st.innerText = '❌ 请求异常: ' + e; }
  } finally {
    if (btn) { btn.disabled = false; btn.innerText = '✅ 绑定'; }
  }
}

function closeAccountModal() {
  const m = document.getElementById('accountSwitchModal');
  if (m) m.style.display = 'none';
}

function renderAccountModalList() {
  renderAccountBadgeAndModal(cachedAccountInfo);
}

async function switchColabAccount(targetEmail) {
  const shareDrive = document.getElementById('chkShareExistingDrive') ? document.getElementById('chkShareExistingDrive').checked : true;
  const st = document.getElementById('accountActionStatus');
  if (st) {
    st.style.display = 'block';
    st.style.color = '#60a5fa';
    st.innerText = `⏳ 正在切换 Colab Pro 算力账号至 ${targetEmail}...`;
  }
  try {
    const r = await fetch('/api/accounts/switch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({email: targetEmail, switch_drive: !shareDrive}),
    });
    const d = await r.json();
    if (!d.ok) {
      if (st) { st.style.color = '#ef4444'; st.innerText = '❌ 切换失败: ' + (d.error || '未知错误'); }
      return;
    }
    if (d.account_info) renderAccountBadgeAndModal(d.account_info);
    if (d.cu_info && typeof d.cu_info.paid_balance !== 'undefined') {
      if (Number(d.cu_info.paid_balance) <= 0) {
        switchPresetTierTab('t4');
        selectPresetModel('preset_t4_qwen38_q3', 'huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF', 'Qwen3.8-27B-T4-Q3', 'Huihui-Qwen3.8-27B-abliterated-UD-Q3_K_XL.gguf');
      } else {
        switchPresetTierTab('g4');
      }
    }
    if (st) {
      st.style.color = '#10b981';
      st.innerText = `✅ 已成功切换至 ${targetEmail}！正在刷新该账号的算力余额 (CU)...`;
    }
    await refreshStatus(true);
  } catch (e) {
    if (st) { st.style.color = '#ef4444'; st.innerText = '❌ 请求异常: ' + e; }
  }
}

async function tryZeroChallengeLocalLogin() {
  const shareDrive = document.getElementById('chkShareExistingDrive') ? document.getElementById('chkShareExistingDrive').checked : true;
  const btn = document.getElementById('btnZeroChallenge');
  const st = document.getElementById('accountActionStatus');
  if (btn) { btn.disabled = true; btn.innerText = '⏳ 请在浏览器中点击账号...'; }
  if (st) {
    st.style.display = 'block';
    st.style.color = '#60a5fa';
    st.innerText = '🌐 已打开极简授权页（无 cloud-platform 高危权限、无 remote 标记），请在已登录的浏览器中点击账号...';
  }
  try {
    const r = await fetch('/api/accounts/add_colab', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({bind_new_drive: !shareDrive}),
    });
    const d = await r.json();
    if (!d.ok) {
      if (st) { st.style.color = '#ef4444'; st.innerText = '❌ 登录未完成: ' + (d.error || '已取消或超时'); }
      return;
    }
    if (d.account_info) renderAccountBadgeAndModal(d.account_info);
    if (st) {
      st.style.color = '#10b981';
      st.innerText = `✅ 账号 ${d.email} 已永久绑定成功！以后随时在上方列表中 1 秒切换，无需再开网页！`;
    }
    await refreshStatus(true);
  } catch (e) {
    if (st) { st.style.color = '#ef4444'; st.innerText = '❌ 请求异常: ' + e; }
  } finally {
    if (btn) { btn.disabled = false; btn.innerText = '⚡ 免安全码极简直登 (推荐先试)'; }
  }
}

async function addNewColabAccount() {
  const btn = document.getElementById('btnAddColabAcc');
  const st = document.getElementById('accountActionStatus');
  const codeInp = document.getElementById('inpRemoteAuthCode');
  if (btn) { btn.disabled = true; btn.innerText = '⏳ 正在打开浏览器...'; }
  try {
    const r = await fetch('/api/accounts/remote_url', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({open_browser: true}),
    });
    const d = await r.json();
    if (!d.ok || !d.auth_url) {
      if (st) { st.style.display = 'block'; st.style.color = '#ef4444'; st.innerText = '❌ 获取 Google 授权链接失败: ' + (d.error || '未知错误'); }
      return;
    }
    if (codeInp) {
      codeInp.value = '';
      codeInp.focus();
    }
    if (st) {
      st.style.display = 'block';
      st.style.color = '#f59e0b';
      st.innerText = '🌐 已在浏览器打开 Google 账号选择页，请选择新账号并点击「允许」，然后将网页弹出的授权码粘贴到上方输入框点「确认」！';
    }
  } catch (e) {
    if (st) { st.style.display = 'block'; st.style.color = '#ef4444'; st.innerText = '❌ 请求异常: ' + e; }
  } finally {
    if (btn) { btn.disabled = false; btn.innerText = '🌐 1. 打开浏览器获取授权码'; }
  }
}

async function rebindGoogleDriveAccount() {
  const btn = document.getElementById('btnRebindDrive');
  const st = document.getElementById('accountActionStatus');
  if (btn) { btn.disabled = true; btn.innerText = '⏳ 请在浏览器授权 Google Drive...'; }
  if (st) {
    st.style.display = 'block';
    st.style.color = '#f59e0b';
    st.innerText = '🌐 正在打开浏览器授权新的 Google Drive 云盘账号，请在网页中点击允许...';
  }
  try {
    const r = await fetch('/api/accounts/bind_drive', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
    const d = await r.json();
    if (!d.ok) {
      if (st) { st.style.color = '#ef4444'; st.innerText = '❌ 云盘绑定失败: ' + (d.error || '未知错误'); }
      return;
    }
    if (st) { st.style.color = '#10b981'; st.innerText = '✅ Google Drive 云盘账号已更新！正在重新扫描云盘目录...'; }
    await refreshStatus(true);
  } catch (e) {
    if (st) { st.style.color = '#ef4444'; st.innerText = '❌ 请求异常: ' + e; }
  } finally {
    if (btn) { btn.disabled = false; btn.innerText = '☁️ 单独更换云盘存储账号'; }
  }
}

async function deleteSavedAccount(email) {
  if (!confirm(`确定要从已保存列表中移除账号 ${email} 吗？`)) return;
  try {
    const r = await fetch('/api/accounts/delete', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({email}),
    });
    const d = await r.json();
    if (d.account_info) renderAccountBadgeAndModal(d.account_info);
  } catch (e) {}
}

function renderCustomPresets(list) {
  customModelsCache = Array.isArray(list) ? list : [];
  const grid = document.getElementById('customPresetGrid');
  if (!grid) return;
  if (customModelsCache.length === 0) {
    grid.innerHTML = '';
    return;
  }
  const curRepo = (document.getElementById('inpRepo') || {}).value || '';
  const curPat = (document.getElementById('inpPattern') || {}).value || '';
  grid.innerHTML = customModelsCache.map(item => {
    const cid = 'cust_' + item.id;
    const isAct = (curRepo === item.repo && curPat === item.pattern);
    const bColor = isAct ? '#3b82f6' : 'var(--border)';
    const bg = isAct ? 'rgba(59,130,246,0.14)' : '#0f172a';
    const chkDisp = isAct ? 'inline' : 'none';
    const badgeTxt = item.subtitle || `${item.pattern}`;
    return `<div class="folder-tag preset-card" id="${cid}" onclick="selectPresetModel('${cid}', '${item.repo.replace(/'/g, "\\'")}', '${item.folder.replace(/'/g, "\\'")}', '${item.pattern.replace(/'/g, "\\'")}')" style="margin:0;padding:8px 10px;justify-content:space-between;border-color:${bColor};background:${bg};">
      <div style="overflow:hidden;padding-right:6px;">
        <div style="font-weight:700;color:#f8fafc;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">🧩 ${item.name}</div>
        <div style="color:var(--muted);font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${badgeTxt}</div>
      </div>
      <div style="display:flex;align-items:center;gap:6px;flex-shrink:0;">
        <span class="preset-check" style="color:#3b82f6;font-weight:700;display:${chkDisp};">✔</span>
        <span title="移除该自定义模型" onclick="deleteCustomHFModel('${item.id}', event)" style="color:#64748b;font-size:12px;padding:2px 5px;border-radius:4px;" onmouseover="this.style.color='#ef4444'" onmouseout="this.style.color='#64748b'">✕</span>
      </div>
    </div>`;
  }).join('');
}

function openCustomHFModal() {
  const modal = document.getElementById('customHFModal');
  modal.style.display = 'flex';
  fetchTrendingHF(currentHFCat, false);
}

function closeCustomHFModal() {
  document.getElementById('customHFModal').style.display = 'none';
}

function autoFillCustomFolder() {
  const raw = document.getElementById('custRepo').value.trim();
  const clean = raw.replace(/^https?:\/\/(www\.)?(huggingface\.co|hf\.co|hf-mirror\.com)\//i, '').replace(/\/+$/, '');
  const parts = clean.split('/').filter(Boolean);
  if (parts.length >= 2) {
    const repoName = parts[1];
    document.getElementById('custFolder').value = repoName;
    if (!document.getElementById('custName').value.trim()) {
      document.getElementById('custName').placeholder = repoName;
    }
    if (repoName.toLowerCase().includes('gguf')) {
      if (document.getElementById('custPattern').value === '*.safetensors') {
        document.getElementById('custPattern').value = '*Q4_K_M*.gguf';
      }
    } else if (document.getElementById('custPattern').value === '*Q4_K_M*.gguf') {
      document.getElementById('custPattern').value = '*.safetensors';
    }
  }
}

async function fetchTrendingHF(cat = 'all', force = false) {
  currentHFCat = cat;
  document.querySelectorAll('#hfCatTabs button').forEach(btn => {
    const active = btn.getAttribute('data-cat') === cat;
    btn.style.borderColor = active ? '#3b82f6' : 'var(--border)';
    btn.style.color = active ? '#60a5fa' : '#f8fafc';
  });
  const q = document.getElementById('hfSearchInput').value.trim();
  const grid = document.getElementById('hfTrendingGrid');
  grid.innerHTML = '<div style="color:var(--muted);font-size:12px;padding:8px;">⏳ 正在从 HuggingFace 获取实时模型列表...</div>';
  try {
    const url = `/api/hf_trending?category=${encodeURIComponent(cat)}&q=${encodeURIComponent(q)}${force ? '&force=1' : ''}`;
    const r = await fetch(url);
    const d = await r.json();
    const items = d.items || [];
    if (items.length === 0) {
      grid.innerHTML = '<div style="color:var(--muted);font-size:12px;padding:8px;">未找到匹配的模型，请尝试其它关键词或直接粘贴仓库地址。</div>';
      return;
    }
    grid.innerHTML = items.map(it => {
      const pBadge = it.param_label ? `<span style="background:rgba(59,130,246,0.2);color:#93c5fd;padding:1px 6px;border-radius:4px;font-size:10.5px;font-weight:700;">${it.param_label}</span>` : '';
      return `<div style="background:#1e293b;border:1px solid var(--border);border-radius:8px;padding:8px 10px;display:flex;flex-direction:column;justify-content:space-between;gap:4px;cursor:pointer;" onclick="pickTrendingHF('${it.repo_id.replace(/'/g, "\\'")}', '${it.name.replace(/'/g, "\\'")}', '${it.folder.replace(/'/g, "\\'")}', '${it.suggested_pattern.replace(/'/g, "\\'")}')" onmouseover="this.style.borderColor='#3b82f6'" onmouseout="this.style.borderColor='var(--border)'">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">
          <div style="font-weight:700;font-size:12.5px;color:#f8fafc;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="${it.repo_id}">${it.repo_id}</div>
          ${pBadge}
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;font-size:11px;">
          <span style="color:${it.fit_color || '#94a3b8'};font-weight:600;">${it.fit_label || ''}</span>
          <div style="display:flex;align-items:center;gap:6px;">
            <span style="color:var(--muted);">❤️ ${it.likes}</span>
            <button type="button" class="btn-outline" style="padding:1px 6px;font-size:10.5px;border-color:#3b82f6;color:#93c5fd;" onclick="event.stopPropagation();pickTrendingHF('${it.repo_id.replace(/'/g, "\\'")}', '${it.name.replace(/'/g, "\\'")}', '${it.folder.replace(/'/g, "\\'")}', '${it.suggested_pattern.replace(/'/g, "\\'")}');inspectHFRepo('${it.repo_id.replace(/'/g, "\\'")}')">详情/选分片</button>
          </div>
        </div>
      </div>`;
    }).join('');
  } catch (e) {
    grid.innerHTML = `<div style="color:#ef4444;font-size:12px;padding:8px;">加载失败: ${e}</div>`;
  }
}

function pickTrendingHF(repo, name, folder, pat) {
  document.getElementById('custRepo').value = repo;
  document.getElementById('custName').value = name;
  document.getElementById('custFolder').value = folder;
  document.getElementById('custPattern').value = pat;
}

async function inspectHFRepo(repoOverride = '') {
  const raw = (repoOverride || document.getElementById('custRepo').value).trim();
  if (!raw) {
    alert('请先输入或选择 HuggingFace 仓库地址');
    return;
  }
  const box = document.getElementById('hfDetailBox');
  const title = document.getElementById('hfDetailTitle');
  const filesDiv = document.getElementById('hfDetailFiles');
  box.style.display = 'block';
  title.innerText = `⏳ 正在探测 ${raw} 的分片文件与体积...`;
  filesDiv.innerHTML = '';
  try {
    const r = await fetch(`/api/hf_detail?repo=${encodeURIComponent(raw)}`);
    const d = await r.json();
    if (!d.ok) {
      title.innerText = `❌ ${d.error || '探测失败'}`;
      return;
    }
    document.getElementById('custRepo').value = d.repo_id;
    if (!document.getElementById('custFolder').value.trim()) {
      document.getElementById('custFolder').value = d.name;
    }
    title.innerText = `📦 ${d.repo_id} · 共 ${d.files.length} 个权重文件 (总计 ${d.total_gb} GB) · ${d.fit_label}`;
    if (d.files.length === 0) {
      filesDiv.innerHTML = '<span style="font-size:12px;color:var(--muted);">该仓库未找到 .gguf 或 .safetensors 权重文件</span>';
      return;
    }
    filesDiv.innerHTML = d.files.map(f => {
      return `<div class="folder-tag" style="margin:0;padding:4px 8px;font-size:11.5px;" onclick="document.getElementById('custPattern').value='${f.filename.replace(/'/g, "\\'")}'">
        📄 <b>${f.filename}</b> <span style="color:var(--green)">(${f.size_gb} GB)</span>
      </div>`;
    }).join('');
  } catch (e) {
    title.innerText = `❌ 探测异常: ${e}`;
  }
}

async function saveCustomHFModel() {
  const rawRepo = document.getElementById('custRepo').value.trim();
  if (!rawRepo) {
    alert('请输入 HuggingFace 仓库地址 (Repo ID 或 URL)');
    return;
  }
  const clean = rawRepo.replace(/^https?:\/\/(www\.)?(huggingface\.co|hf\.co|hf-mirror\.com)\//i, '').replace(/\/+$/, '');
  const parts = clean.split('/').filter(Boolean);
  const repo = parts.length >= 2 ? `${parts[0]}/${parts[1]}` : clean;
  const defaultName = repo.split('/').pop() || repo;
  const name = document.getElementById('custName').value.trim() || defaultName;
  const folder = document.getElementById('custFolder').value.trim() || defaultName;
  const pattern = document.getElementById('custPattern').value.trim() || '*';
  const hf_token = document.getElementById('custToken').value.trim();

  const r = await fetch('/api/custom_models', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({repo, name, folder, pattern, hf_token})
  });
  const d = await r.json();
  if (d.ok) {
    renderCustomPresets(d.custom_models || []);
    if (d.added) {
      selectPresetModel('cust_' + d.added.id, d.added.repo, d.added.folder, d.added.pattern);
    }
    closeCustomHFModal();
  } else {
    alert(d.error || '添加失败');
  }
}

async function deleteCustomHFModel(id, ev) {
  if (ev) ev.stopPropagation();
  const r = await fetch('/api/custom_models/delete', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id})
  });
  const d = await r.json();
  if (d.ok) {
    renderCustomPresets(d.custom_models || []);
  }
}

let liveUptimeSec = null;
let liveBurnRate = 0.0;
let basePaidBalance = null;
let baseBalanceSyncTs = 0;
let isFreeT4Available = true;
let liveIdleSec = 0;
let liveIdleTimeoutMin = 30;
let isInstanceOnline = false;

function formatHMS(sec) {
  const s = Math.max(0, Math.floor(sec || 0));
  const hh = String(Math.floor(s / 3600)).padStart(2, '0');
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  return `${hh}:${mm}:${ss}`;
}

function updateLiveSessionCuDisplay() {
  const usedEl = document.getElementById('cuSessionUsed');
  const subEl = document.getElementById('cuSessionUsedSub');
  const balEl = document.getElementById('cuBalance');
  const hrsEl = document.getElementById('cuRemainHours');
  if (!usedEl) return;
  if (isInstanceOnline && liveUptimeSec !== null && liveUptimeSec >= 0) {
    if (basePaidBalance !== null && basePaidBalance <= 0) {
      usedEl.innerText = '0.00 CU';
      usedEl.style.color = '#10b981';
      if (subEl) {
        subEl.innerText = 'T4 零算力消耗';
        subEl.style.color = '#34d399';
      }
      if (hrsEl) {
        const usedMin = Math.floor(liveUptimeSec / 60);
        const remLow = Math.max(0.1, 3.0 - (liveUptimeSec / 3600.0)).toFixed(1);
        const remHigh = Math.max(0.5, 5.0 - (liveUptimeSec / 3600.0)).toFixed(1);
        hrsEl.innerText = `已跑 ${usedMin}m · 剩 ≈${remLow}~${remHigh}h`;
      }
      return;
    }
    const effRate = liveBurnRate > 0 ? liveBurnRate : 10.05;
    const usedCu = (liveUptimeSec / 3600.0) * effRate;
    usedEl.innerText = `${usedCu.toFixed(2)} CU`;
    usedEl.style.color = '#f59e0b';
    if (subEl) {
      subEl.innerText = `实时累计: ${usedCu.toFixed(3)} CU`;
      subEl.style.color = '#94a3b8';
    }
    if (balEl && basePaidBalance !== null && baseBalanceSyncTs > 0) {
      const dtSec = Math.max(0, (Date.now() - baseBalanceSyncTs) / 1000.0);
      const liveBal = Math.max(0, basePaidBalance - (dtSec / 3600.0) * effRate);
      balEl.innerText = `${liveBal.toFixed(2)} CU`;
    }
  } else {
    usedEl.innerText = '0.00 CU';
    usedEl.style.color = 'var(--muted)';
    if (subEl) {
      subEl.innerText = '当前已关机零消耗';
      subEl.style.color = 'var(--muted)';
    }
  }
}

setInterval(() => {
  const upEl = document.getElementById('cuUptime');
  const cdEl = document.getElementById('idleCountdownText');
  const hintEl = document.getElementById('idleElapsedHint');
  if (isInstanceOnline && liveUptimeSec !== null) {
    liveUptimeSec += 1;
    liveIdleSec += 1;
    upEl.innerText = formatHMS(liveUptimeSec);
    upEl.style.color = '#f8fafc';
    updateLiveSessionCuDisplay();
    if (liveIdleTimeoutMin > 0) {
      const remainSec = Math.max(0, liveIdleTimeoutMin * 60 - liveIdleSec);
      const rMin = Math.floor(remainSec / 60);
      const rSec = remainSec % 60;
      cdEl.innerText = `自动休眠倒计时：${rMin} 分 ${String(rSec).padStart(2, '0')} 秒`;
      hintEl.innerText = `(静置 ${Math.floor(liveIdleSec / 60)}m)`;
    } else {
      cdEl.innerText = '已设为「从不」自动休眠（3分钟心跳持续保活中）';
      hintEl.innerText = '';
    }
  } else {
    upEl.innerText = '00:00:00';
    upEl.style.color = 'var(--muted)';
    cdEl.innerText = '当前实例已释放（不消耗算力单元）';
    hintEl.innerText = '';
    updateLiveSessionCuDisplay();
  }
}, 1000);

async function updateIdleConfig() {
  const mins = parseInt(document.getElementById('idleSelect').value, 10);
  liveIdleTimeoutMin = mins;
  liveIdleSec = 0;
  try {
    const r = await fetch('/api/idle_config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({idle_timeout_minutes: mins})
    });
    const d = await r.json();
    if (d.message) {
      document.getElementById('idleCountdownText').innerText = '✅ ' + d.message;
    }
  } catch (e) {}
}

function renderResetDayInfo(day) {
  const el = document.getElementById('cuResetDayLabel');
  if (!el) return;
  const dNum = parseInt(day || 0, 10);
  if (!dNum || dNum < 1 || dNum > 31) {
    el.innerText = '⚙️ 设重置日';
    return;
  }
  const now = new Date();
  let target = new Date(now.getFullYear(), now.getMonth(), dNum, 0, 0, 0);
  if (target <= now) {
    target = new Date(now.getFullYear(), now.getMonth() + 1, dNum, 0, 0, 0);
  }
  const diffDays = Math.ceil((target - now) / (1000 * 60 * 60 * 24));
  const mm = target.getMonth() + 1;
  const dd = target.getDate();
  el.innerText = `📅 下次重置: ${mm}月${dd}日 (剩${diffDays}天)`;
}

async function setBillingResetDay() {
  const val = prompt('请输入您在 Apple Store 订阅中看到的每月续订日（1 ~ 31 的数字，例如 18；输入 0 清除）：', '');
  if (val === null) return;
  const day = parseInt(val.trim(), 10);
  if (isNaN(day) || day < 0 || day > 31) {
    alert('请输入 1 到 31 之间的日期数字');
    return;
  }
  try {
    await fetch('/api/idle_config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({billing_reset_day: day})
    });
    renderResetDayInfo(day);
  } catch (e) {}
}

function copyText(id) {
  const txt = document.getElementById(id).innerText.trim();
  if (!txt || txt === '--') return;
  navigator.clipboard.writeText(txt);
  alert('已复制: ' + txt);
}

const savedDriveFolders = new Set(['Huihui-Qwen3.8-27B-abliterated-GGUF', 'MiniMax-H3', 'ReCaVSR']);
let lastDriveFoldersRaw = [];
let currentPresetTier = 'g4';

function switchPresetTierTab(tier, autoSelect = true) {
  currentPresetTier = tier;
  const gG4 = document.getElementById('presetModelGrid');
  const gT4 = document.getElementById('presetModelGridT4');
  const bG4 = document.getElementById('tabPresetG4');
  const bT4 = document.getElementById('tabPresetT4');
  if (tier === 't4') {
    if (gG4) gG4.style.display = 'none';
    if (gT4) gT4.style.display = 'grid';
    if (bT4) {
      bT4.style.borderColor = '#10b981';
      bT4.style.background = 'rgba(16,185,129,0.22)';
      bT4.style.color = '#34d399';
    }
    if (bG4) {
      bG4.style.borderColor = 'var(--border)';
      bG4.style.background = '#0f172a';
      bG4.style.color = '#94a3b8';
    }
    if (autoSelect) {
      const curF = (document.getElementById('inpFolder') || {}).value || '';
      if (!curF.includes('-T4-') && !curF.endsWith('-INT8') && curF !== 'ReCaVSR') {
        selectPresetModel('preset_t4_qwen38_q3', 'huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF', 'Qwen3.8-27B-T4-Q3', 'Huihui-Qwen3.8-27B-abliterated-UD-Q3_K_XL.gguf');
      }
    }
  } else {
    if (gG4) gG4.style.display = 'grid';
    if (gT4) gT4.style.display = 'none';
    if (bG4) {
      bG4.style.borderColor = '#3b82f6';
      bG4.style.background = 'rgba(59,130,246,0.22)';
      bG4.style.color = '#93c5fd';
    }
    if (bT4) {
      bT4.style.borderColor = 'var(--border)';
      bT4.style.background = '#0f172a';
      bT4.style.color = '#94a3b8';
    }
    if (autoSelect) {
      const curF = (document.getElementById('inpFolder') || {}).value || '';
      if (curF.includes('-T4-') || curF.endsWith('-INT8') || curF === 'ReCaVSR') {
        selectPresetModel('preset_qwen_bf16', 'huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF', 'Huihui-Qwen3.8-27B-abliterated-GGUF', 'Huihui-Qwen3.8-27B-abliterated-bf16.gguf');
      }
    }
  }
}

function renderGroupedDriveFolders(folders, updateButtons = true) {
  if (Array.isArray(folders)) lastDriveFoldersRaw = folders;
  const fBox = document.getElementById('driveFoldersList');
  if (!fBox) return;
  const curSelFolder = (document.getElementById('inpFolder') || {}).value || '';
  savedDriveFolders.clear();
  const g4List = [];
  const t4List = [];
  (lastDriveFoldersRaw || []).forEach(item => {
    savedDriveFolders.add(item.folder);
    const gb = typeof item.total_gb === 'number' ? item.total_gb : 0;
    const isT4Named = item.folder.includes('-T4-') || item.folder.endsWith('-INT8') || item.folder === 'ReCaVSR';
    if (!isT4Named && gb > 15.5) g4List.push(item);
    else t4List.push(item);
  });
  const renderTag = item => {
    const hasSt = item.files.some(f => f.name.toLowerCase().includes('.safetensors'));
    const mainFile = hasSt ? '*.safetensors' : ((item.files.find(f => !f.name.toLowerCase().includes('mmproj')) || item.files[0] || {}).name || '');
    const totGb = typeof item.total_gb === 'number' ? ` · ${item.total_gb} GB` : '';
    const filesDesc = item.files.slice(0, 12).map(f => `${f.name} (${f.size_gb}G)`).join(', ') || '空文件夹';
    const isAct = item.folder === curSelFolder;
    const bCol = isAct ? '#3b82f6' : 'var(--border)';
    const bgCol = isAct ? 'rgba(59,130,246,0.16)' : '#0f172a';
    const chkDisp = isAct ? 'inline' : 'none';
    return `<div class="folder-tag" data-folder="${item.folder}" onclick="selectDriveFolder('${item.folder}', '${mainFile}')" title="${filesDesc}" style="border-color:${bCol};background:${bgCol};">
      📁 <b>${item.folder}</b> <span style="color:var(--green)">(${item.files.length}个${totGb})</span>
      <span class="drive-folder-check" style="color:#3b82f6;font-weight:700;display:${chkDisp};margin-left:4px;">✔</span>
    </div>`;
  };
  let pendingG4Html = '';
  let pendingT4Html = '';
  if (curSelFolder && !savedDriveFolders.has(curSelFolder)) {
    const pendingTag = `<div class="folder-tag" data-folder="${curSelFolder}" title="该模型尚未存入您的 Google Drive 云盘；点击下方按钮即可从 HuggingFace 下载并永久保存到云盘" style="border:1.5px dashed #3b82f6;background:rgba(59,130,246,0.14);">
      ☁️ <b>${curSelFolder}</b> <span style="color:#93c5fd;">(尚未存云盘 · 点下方按钮下载入盘)</span>
      <span class="drive-folder-check" style="color:#3b82f6;font-weight:700;display:inline;margin-left:4px;">✔</span>
    </div>`;
    if (curSelFolder.includes('-T4-') || currentPresetTier === 't4') {
      pendingT4Html = pendingTag;
    } else {
      pendingG4Html = pendingTag;
    }
  }
  let html = '';
  if (g4List.length > 0 || pendingG4Html) {
    html += `<div style="margin-bottom:6px;"><div style="font-size:11px;font-weight:700;color:#93c5fd;margin-bottom:3px;">🔥 RTX 6000 (96GB) 云盘模型：</div><div>${g4List.map(renderTag).join('')}${pendingG4Html}</div></div>`;
  }
  if (t4List.length > 0 || pendingT4Html) {
    html += `<div><div style="font-size:11px;font-weight:700;color:#34d399;margin-bottom:3px;">⚡ T4 (16GB) 云盘模型：</div><div>${t4List.map(renderTag).join('')}${pendingT4Html}</div></div>`;
  }
  fBox.innerHTML = html || '当前 MyDrive/models/ 下暂无模型文件夹';
  if (updateButtons) syncDriveFolderHighlight(curSelFolder, false);
}

function syncDriveFolderHighlight(folderName, rerenderList = true) {
  if (rerenderList && lastDriveFoldersRaw.length > 0) {
    renderGroupedDriveFolders(lastDriveFoldersRaw, false);
  }
  document.querySelectorAll('#driveFoldersList .folder-tag').forEach(el => {
    const active = el.getAttribute('data-folder') === folderName;
    if (!el.style.borderStyle || el.style.borderStyle !== 'dashed') {
      el.style.borderColor = active ? '#3b82f6' : 'var(--border)';
      el.style.background = active ? 'rgba(59,130,246,0.16)' : '#0f172a';
    }
    const chk = el.querySelector('.drive-folder-check');
    if (chk) chk.style.display = active ? 'inline' : 'none';
  });
  const shortMap = {
    'Huihui-Qwen3.8-27B-abliterated-GGUF': 'Qwen3.8-27B',
    'MiniMax-H3': 'MiniMax-H3',
    'MiniMax-H3-Image': 'MiniMax-H3-Image(4K)',
    'LongCat-Video-Avatar-1.6': 'LongCat-Avatar-1.6',
    'LongCat-Video-Avatar-1.6-INT8': 'LongCat-Avatar(INT8)',
    'Qwen3.8-27B-T4-Q3': 'Qwen3.8-27B(T4)',
    'Qwen3.8-27B-T4-MTP': 'Qwen3.8-27B(MTP)',
    'Qwen3-14B-T4-Q6': 'Qwen3-14B(Q6)',
    'Qwen3-VL-8B-T4-Q8': 'Qwen3-VL-8B(Q8)',
    'MiniMax-H3-T4-GGUF': 'MiniMax-H3(T4)',
    'ReCaVSR': 'ReCaVSR',
  };
  const shortLabel = shortMap[folderName] || folderName || '当前模型';
  const inDrive = savedDriveFolders.has(folderName);
  const btnDl = document.getElementById('btnDownloadOnly');
  const btnDep = document.getElementById('btnDeployForm');
  if (btnDl && !btnDl.disabled) {
    if (inDrive) {
      btnDl.innerText = '✅ 云盘已存妥 (校验文件)';
      btnDl.style.borderColor = 'rgba(16,185,129,0.5)';
      btnDl.style.color = '#6ee7b7';
      btnDl.title = '该模型已保存在您的 Google Drive 云盘中；点击仅校验或补全云盘缺失文件，不加载进显存';
    } else {
      btnDl.innerText = '☁️ 仅从 HF 转存至云盘';
      btnDl.style.borderColor = '#3b82f6';
      btnDl.style.color = '#93c5fd';
      btnDl.title = '从 HuggingFace 下载模型并永久保存到您的 Google Drive 云盘，下载完成后不启动推理服务';
    }
  }
  if (btnDep && !btnDep.disabled) {
    if (inDrive) {
      btnDep.innerText = `🚀 从云盘加载并启动 (${shortLabel})`;
      btnDep.title = '直接从您的 Google Drive 云盘极速预热并装载至 GPU 显存启动运行（无需从 HF 重新下载）';
    } else {
      btnDep.innerText = `🚀 下载入盘并启动 (${shortLabel})`;
      btnDep.title = '从 HuggingFace 下载模型、自动备份至 Google Drive 云盘，并立即加载到 GPU 显存启动运行';
    }
  }
}

function selectPresetModel(cardId, repo, folder, pattern, keepCurrentTab = false) {
  document.getElementById('inpRepo').value = repo;
  document.getElementById('inpFolder').value = folder;
  document.getElementById('inpPattern').value = pattern;
  document.querySelectorAll('.preset-card').forEach(el => {
    const active = el.id === cardId;
    el.style.borderColor = active ? '#3b82f6' : 'var(--border)';
    el.style.background = active ? 'rgba(59,130,246,0.14)' : '#0f172a';
    const chk = el.querySelector('.preset-check');
    if (chk) chk.style.display = active ? 'inline' : 'none';
  });
  if (!keepCurrentTab && currentMiniMaxMode !== 'drama' && currentMiniMaxMode !== 'fusion') {
    if (folder.startsWith('LongCat-Video-Avatar')) {
      switchMiniMaxMode('avatar');
    } else if (folder === 'MiniMax-H3-Image') {
      switchMiniMaxMode('image');
    } else if (folder.startsWith('MiniMax-H3') && (currentMiniMaxMode === 'avatar' || currentMiniMaxMode === 'image')) {
      switchMiniMaxMode('t2va');
    }
  }
  syncDriveFolderHighlight(folder);
}

function selectDriveFolder(folderName, firstFile) {
  document.getElementById('inpFolder').value = folderName;
  if (firstFile) {
    document.getElementById('inpPattern').value = firstFile;
  }
  const repoMap = {
    'Huihui-Qwen3.8-27B-abliterated-GGUF': 'huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF',
    'MiniMax-H3': 'MiniMaxAI/MiniMax-H3',
    'MiniMaxAI': 'MiniMaxAI/MiniMax-H3',
    'MiniMax-H3-Image': 'Comfy-Org/MiniMax-H3-Image',
    'LongCat-Video-Avatar-1.6': 'meituan-longcat/LongCat-Video-Avatar-1.6',
    'LongCat-Video-Avatar-1.6-INT8': 'meituan-longcat/LongCat-Video-Avatar-1.6',
    'Qwen3.8-27B-T4-Q3': 'huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF',
    'Qwen3.8-27B-T4-MTP': 'huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF',
    'Qwen3-14B-T4-Q6': 'bartowski/huihui-ai_Qwen3-14B-abliterated-GGUF',
    'Qwen3-VL-8B-T4-Q8': 'mradermacher/Qwen3-VL-8B-Instruct-abliterated-v2.0-GGUF',
    'MiniMax-H3-T4-GGUF': 'unsloth/MiniMax-H3-GGUF',
    'ReCaVSR': 'Decorator/ReCaVSR',
  };
  const cardMap = {
    'Huihui-Qwen3.8-27B-abliterated-GGUF': 'preset_qwen_bf16',
    'MiniMax-H3': 'preset_minimax_h3',
    'MiniMaxAI': 'preset_minimax_h3',
    'MiniMax-H3-Image': 'preset_minimax_image',
    'LongCat-Video-Avatar-1.6': 'preset_longcat_avatar',
    'LongCat-Video-Avatar-1.6-INT8': 'preset_t4_longcat_int8',
    'Qwen3.8-27B-T4-Q3': 'preset_t4_qwen38_q3',
    'Qwen3.8-27B-T4-MTP': 'preset_t4_qwen38_mtp',
    'Qwen3-14B-T4-Q6': 'preset_t4_qwen14b',
    'Qwen3-VL-8B-T4-Q8': 'preset_t4_qwenvl8b',
    'MiniMax-H3-T4-GGUF': 'preset_t4_minimax_h3',
    'ReCaVSR': 'preset_t4_recavsr',
  };
  const patternMap = {
    'Huihui-Qwen3.8-27B-abliterated-GGUF': 'Huihui-Qwen3.8-27B-abliterated-bf16.gguf',
    'MiniMax-H3': 'FL2VA/*.safetensors',
    'MiniMax-H3-Image': '*.safetensors',
    'LongCat-Video-Avatar-1.6': 'base_model/*.safetensors',
    'LongCat-Video-Avatar-1.6-INT8': 'base_model_int8/*.safetensors',
    'Qwen3.8-27B-T4-Q3': 'Huihui-Qwen3.8-27B-abliterated-UD-Q3_K_XL.gguf',
    'Qwen3.8-27B-T4-MTP': 'Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_S-mtp.gguf',
    'Qwen3-14B-T4-Q6': 'huihui-ai_Qwen3-14B-abliterated-Q6_K.gguf',
    'Qwen3-VL-8B-T4-Q8': '*Q8_0.gguf',
    'MiniMax-H3-T4-GGUF': 'minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf',
    'ReCaVSR': '*.pth',
  };
  if (repoMap[folderName]) {
    document.getElementById('inpRepo').value = repoMap[folderName];
  }
  if (patternMap[folderName]) {
    document.getElementById('inpPattern').value = patternMap[folderName];
  }
  if (cardMap[folderName]) {
    if (cardMap[folderName].startsWith('preset_t4_')) switchPresetTierTab('t4');
    else switchPresetTierTab('g4');
    document.querySelectorAll('.preset-card').forEach(el => {
      const active = el.id === cardMap[folderName];
      el.style.borderColor = active ? '#3b82f6' : 'var(--border)';
      el.style.background = active ? 'rgba(59,130,246,0.14)' : '#0f172a';
      const chk = el.querySelector('.preset-check');
      if (chk) chk.style.display = active ? 'inline' : 'none';
    });
  }
  if (currentMiniMaxMode !== 'drama' && currentMiniMaxMode !== 'fusion') {
    if (folderName.startsWith('LongCat-Video-Avatar')) {
      switchMiniMaxMode('avatar');
    } else if (folderName === 'MiniMax-H3-Image') {
      switchMiniMaxMode('image');
    } else if (folderName.startsWith('MiniMax-H3') && (currentMiniMaxMode === 'avatar' || currentMiniMaxMode === 'image')) {
      switchMiniMaxMode('t2va');
    }
  }
  syncDriveFolderHighlight(folderName);
}

async function refreshStatus(force = false) {
  const btn = document.getElementById('btnRefresh');
  btn.disabled = true;
  btn.innerText = '🔄 刷新中...';
  try {
    const r = await fetch('/api/status' + (force ? '?force=1' : ''));
    const d = await r.json();

    document.getElementById('apiKey').innerText = d.endpoint_cfg.api_key || 'sk-colab-ed28e54eb298d0c1';
    document.getElementById('modelId').innerText = d.endpoint_cfg.model_id || 'huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF';
    document.getElementById('tunnelUrl').innerText = d.endpoint_cfg.tunnel_url ? (d.endpoint_cfg.tunnel_url + '/v1') : '--';
    if (!window._autoModeInit && d.endpoint_cfg && d.endpoint_cfg.model_folder) {
      window._autoModeInit = true;
      const mf = String(d.endpoint_cfg.model_folder);
      if (currentMiniMaxMode !== 'drama' && currentMiniMaxMode !== 'fusion') {
        if (mf.startsWith('LongCat-Video-Avatar')) {
          switchMiniMaxMode('avatar');
          selectDriveFolder(mf);
        } else if (mf === 'MiniMax-H3-Image') {
          switchMiniMaxMode('image');
          selectDriveFolder(mf);
        }
      }
    }
    renderCustomPresets(d.custom_models || []);
    if (d.account_info) {
      renderAccountBadgeAndModal(d.account_info);
    }

    if (typeof d.endpoint_cfg.idle_timeout_minutes !== 'undefined') {
      liveIdleTimeoutMin = parseInt(d.endpoint_cfg.idle_timeout_minutes, 10);
      const sel = document.getElementById('idleSelect');
      if (sel && String(sel.value) !== String(liveIdleTimeoutMin)) {
        sel.value = String(liveIdleTimeoutMin);
      }
    }
    renderResetDayInfo(d.endpoint_cfg.billing_reset_day || 0);
    if (typeof d.idle_seconds === 'number') {
      liveIdleSec = d.idle_seconds;
    }

    if (d.cu_info) {
      const balEl = document.getElementById('cuBalance');
      const rateEl = document.getElementById('cuBurnRate');
      const hrsEl = document.getElementById('cuRemainHours');
      const br = Number(d.cu_info.burn_rate_hourly || 0);
      liveBurnRate = br;
      if (d.cu_info.paid_balance !== null && typeof d.cu_info.paid_balance !== 'undefined') {
        const bal = Number(d.cu_info.paid_balance);
        const elGpus = Array.isArray(d.cu_info.eligible_gpus) ? d.cu_info.eligible_gpus : [];
        const hasG4Entitlement = bal > 0 || elGpus.includes('G4');
        basePaidBalance = bal;
        baseBalanceSyncTs = Date.now();
        isFreeT4Available = d.cu_info.t4_available !== false;
        if (hasG4Entitlement) {
          balEl.innerText = bal > 0 ? `${bal.toFixed(2)} CU` : 'Ultra (G4)';
          balEl.style.color = '#f8fafc';
          if (hrsEl) {
            if (bal > 0) {
              const refRate = br > 0 ? br : 10.05;
              const hrs = bal / refRate;
              hrsEl.innerText = br > 0
                ? `当前速率可跑 ≈ ${hrs.toFixed(1)} 小时`
                : `按 RTX PRO 6000 (10.05/h) ≈ ${hrs.toFixed(1)} 小时`;
            } else {
              hrsEl.innerText = '🔥 已解锁 RTX PRO 6000 (G4 · 96GB)';
            }
          }
        } else if (isFreeT4Available) {
          balEl.innerText = 'T4 (16GB)';
          balEl.style.color = '#34d399';
          if (hrsEl) {
            hrsEl.innerText = '🟢 今日配额可用 · 单次约 3~5 小时';
          }
          const curF = (document.getElementById('inpFolder') || {}).value || '';
          const g4OnlySet = new Set(['Huihui-Qwen3.8-27B-abliterated-GGUF', 'MiniMax-H3', 'MiniMax-H3-Image']);
          if (g4OnlySet.has(curF)) {
            switchPresetTierTab('t4', true);
          }
        } else {
          balEl.innerText = 'T4 冷却中';
          balEl.style.color = '#f59e0b';
          if (hrsEl) {
            hrsEl.innerText = '⏳ 今日 T4 配额已用完 (请切其他号)';
          }
        }
      }
      const effBr = (basePaidBalance !== null && basePaidBalance <= 0) ? 0.0 : br;
      rateEl.innerText = `${effBr.toFixed(2)} CU/h`;
      rateEl.style.color = effBr > 0 ? '#f59e0b' : 'var(--muted)';
    }

    const badge = document.getElementById('sessionBadge');
    const relBtn = document.getElementById('btnReleaseAll');
    const rawFolder = (d.endpoint_cfg && (d.endpoint_cfg.model_folder || d.endpoint_cfg.model_id)) || '';
    const friendlyModelMap = {
      'Huihui-Qwen3.8-27B-abliterated-GGUF': 'Qwen3.8-27B(无审查)',
      'huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF': 'Qwen3.8-27B(无审查)',
      'Qwen3.8-27B-T4-Q3': 'Qwen3.8-27B(T4)',
      'Qwen3.8-27B-T4-MTP': 'Qwen3.8-27B(MTP)',
      'Qwen3-14B-T4-Q6': 'Qwen3-14B(Q6)',
      'Qwen3-VL-8B-T4-Q8': 'Qwen3-VL-8B(Q8)',
      'MiniMax-H3-T4-GGUF': 'MiniMax-H3(T4)',
      'unsloth/MiniMax-H3-GGUF': 'MiniMax-H3(T4)',
      'ReCaVSR': 'ReCaVSR',
      'MiniMax-H3': 'MiniMax-H3(33B)',
      'MiniMaxAI/MiniMax-H3': 'MiniMax-H3(33B)',
      'MiniMax-H3-Image': 'MiniMax-H3-Image(4K)',
      'Comfy-Org/MiniMax-H3-Image': 'MiniMax-H3-Image(4K)',
      'LongCat-Video-Avatar-1.6': 'LongCat-Avatar-1.6',
      'LongCat-Video-Avatar-1.6-INT8': 'LongCat-Avatar(INT8)',
      'meituan-longcat/LongCat-Video-Avatar-1.6': 'LongCat-Avatar-1.6',
    };
    const folderShort = rawFolder.split('/').pop();
    const activeModelDisplay = friendlyModelMap[rawFolder] || friendlyModelMap[folderShort] || folderShort || 'Qwen3.8-27B(无审查)';
    if (d.sessions && d.sessions.length > 0) {
      isInstanceOnline = true;
      if (liveUptimeSec === null) liveUptimeSec = 1;
      badge.className = 'badge online';
      badge.innerText = `🟢 运行中: ${activeModelDisplay}`;
      if (relBtn) {
        relBtn.innerText = `⏻ 释放 (${activeModelDisplay})`;
        relBtn.style.borderColor = 'rgba(239,68,68,0.5)';
        relBtn.style.color = '#fca5a5';
      }
    } else {
      isInstanceOnline = false;
      liveUptimeSec = null;
      liveBurnRate = 0.0;
      badge.className = 'badge offline';
      badge.innerText = '⚪ 已关机 · 0.00 CU/h';
      document.getElementById('cuBurnRate').innerText = '0.00 CU/h';
      document.getElementById('cuBurnRate').style.color = 'var(--muted)';
      document.getElementById('cuUptime').innerText = '00:00:00';
      if (relBtn) {
        relBtn.innerText = '✅ 已释放 (0.00 CU/h)';
        relBtn.style.borderColor = 'var(--border)';
        relBtn.style.color = '#10b981';
      }
    }

    const m = d.metrics;
    if (m) {
      if (typeof m.uptime_sec === 'number' && m.uptime_sec > 0) {
        liveUptimeSec = m.uptime_sec;
        document.getElementById('cuUptime').innerText = formatHMS(liveUptimeSec);
      }
    }
    updateLiveSessionCuDisplay();
    if (m) {
      if (m.gpu) {
        document.getElementById('gpuVal').innerText = `${m.gpu.used_gb} / ${m.gpu.total_gb} GB (${m.gpu.pct}%)`;
        document.getElementById('gpuSub').innerText = `${m.gpu.name}\n负载 ${m.gpu.util}% · ${m.gpu.power}`;
        document.getElementById('gpuTemp').innerText = `${m.gpu.temp}°C`;
        document.getElementById('gpuBar').style.width = `${Math.min(100, m.gpu.pct)}%`;
      }
      if (m.ram) {
        document.getElementById('ramInfo').innerText = `系统内存: ${m.ram.used_gb}/${m.ram.total_gb} GB`;
      }
      if (m.shm) {
        document.getElementById('shmVal').innerText = `${m.shm.used_gb} / ${m.shm.total_gb} GB (${m.shm.pct}%)`;
        document.getElementById('shmSub').innerText = `剩余可用: ${m.shm.free_gb} GB`;
        document.getElementById('shmBar').style.width = `${Math.min(100, m.shm.pct)}%`;
      }
      if (m.disk_content) {
        document.getElementById('diskVal').innerText = `${m.disk_content.used_gb} / ${m.disk_content.total_gb} GB`;
        document.getElementById('diskSub').innerText = `剩余可用: ${m.disk_content.free_gb} GB (${m.disk_content.pct}%)`;
        document.getElementById('diskBar').style.width = `${Math.min(100, m.disk_content.pct)}%`;
      }
      if (m.disk_drive && m.drive_mounted) {
        document.getElementById('driveVal').innerText = `${m.disk_drive.used_gb} / ${m.disk_drive.total_gb} GB`;
        document.getElementById('driveSub').innerText = `剩余可用: ${m.disk_drive.free_gb} GB (${m.disk_drive.pct}%)`;
        document.getElementById('driveBar').style.width = `${Math.min(100, m.disk_drive.pct)}%`;
        document.getElementById('syncBadge').innerText = m.sync_running ? '⏳ 正在往云盘同步...' : '✅ 已挂载 (同步完成)';
      } else if (!(d.gdrive_info && d.gdrive_info.bound)) {
        document.getElementById('driveVal').innerText = '未挂载';
        document.getElementById('syncBadge').innerText = '未挂载';
      }

      const lBadge = document.getElementById('llamaBadge');
      if (m.llama_running) {
        lBadge.className = 'badge online';
        lBadge.innerText = '🟢 推理引擎运行中';
      } else {
        lBadge.className = 'badge offline';
        lBadge.innerText = '⚪ 推理引擎未运行';
      }

      if (m.drive_folders && m.drive_folders.length > 0) {
        renderGroupedDriveFolders(m.drive_folders);
      } else {
        const fBox = document.getElementById('driveFoldersList');
        if (fBox) fBox.innerText = m.drive_mounted ? '当前 MyDrive/models/ 下暂无子文件夹' : '挂载 Google Drive 后显示云盘模型文件夹';
      }
    } else if (!isInstanceOnline) {
      document.getElementById('gpuVal').innerText = '已关机';
      document.getElementById('gpuSub').innerText = '实例已释放 · 0.00W';
      document.getElementById('gpuTemp').innerText = '--°C';
      document.getElementById('gpuBar').style.width = '0%';
      document.getElementById('shmVal').innerText = '已关机';
      document.getElementById('shmSub').innerText = '实例未启动';
      document.getElementById('shmBar').style.width = '0%';
      document.getElementById('diskVal').innerText = '已关机';
      document.getElementById('diskSub').innerText = '实例未启动';
      document.getElementById('diskBar').style.width = '0%';
      const lBadge = document.getElementById('llamaBadge');
      lBadge.className = 'badge offline';
      lBadge.innerText = '⚪ 推理引擎已关机';
    }

    const gd = d.gdrive_info;
    if (gd && gd.bound) {
      if (gd.quota) {
        const totLabel = gd.quota.total_gb >= 1024 ? `${gd.quota.total_gb} GB (${Math.round(gd.quota.total_gb / 1024)}TB)` : `${gd.quota.total_gb} GB`;
        document.getElementById('driveVal').innerText = `${gd.quota.used_gb} / ${totLabel}`;
        document.getElementById('driveSub').innerText = `剩余可用: ${gd.quota.free_gb} GB (${gd.quota.pct}%)`;
        document.getElementById('driveBar').style.width = `${Math.min(100, gd.quota.pct)}%`;
      }
      document.getElementById('syncBadge').innerText = isInstanceOnline ? '✅ 已挂载 (永久免弹窗)' : '✅ 永久免弹窗已绑定 (实例关机)';
      if (gd.drive_folders && gd.drive_folders.length > 0) {
        renderGroupedDriveFolders(gd.drive_folders);
      }
    } else if (!isInstanceOnline) {
      document.getElementById('driveVal').innerText = '已关机';
      document.getElementById('driveSub').innerText = '开机后自动挂载 Google Drive';
      document.getElementById('driveBar').style.width = '0%';
      document.getElementById('syncBadge').innerText = '⚪ 实例已关机';
    }
  } catch (e) {
    console.error(e);
  } finally {
    btn.disabled = false;
    btn.innerText = '🔄 刷新';
  }
}

async function clearDeployLogs() {
  try {
    await fetch('/api/clear_logs', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
    const pLabel = document.getElementById('taskProgressLabel');
    const pSpeed = document.getElementById('taskProgressSpeed');
    const pBar = document.getElementById('taskProgressBar');
    const st = document.getElementById('deployStateText');
    if (pLabel) pLabel.innerText = '当前进度：已就绪 (100%)';
    if (pSpeed) pSpeed.innerText = '已就绪';
    if (pBar) { pBar.className = 'progress-fill green'; pBar.style.width = '100%'; }
    if (st) { st.innerText = '空闲'; st.style.color = 'var(--muted)'; }
    await pollLogs();
  } catch (e) {}
}

async function pollLogs() {
  try {
    const r = await fetch('/api/logs');
    const d = await r.json();
    const box = document.getElementById('logBox');
    if (d.logs && d.logs.length > 0) {
      const fullTxt = d.logs.join('\n');
      box.innerText = fullTxt;
      box.scrollTop = box.scrollHeight;
    }
    if (dramaPollTimer) {
      return;
    }
    if (d.logs && d.logs.length > 0) {

      // 解析最新一条任务状态：部署运行期间始终独立提取最新下载/预热/装载进度，防止被自动切换日志截断或互相覆盖闪烁
      let latestDeployMatch = null;
      let latestDeployPostStage = null;
      for (let i = d.logs.length - 1; i >= 0; i--) {
        const line = d.logs[i];
        if (!latestDeployPostStage && /正在(拉取最新 ComfyUI|检查 ComfyUI 运行依赖|启动 Cloudflare Tunnel|安装 cloudflared)/.test(line)) {
          latestDeployPostStage = line.replace(/^\[\d{2}:\d{2}:\d{2}\]\s*/, '').trim();
        }
        const m = line.match(/\[(HF 下载|云盘极速预热|云盘同步|显存装载)\]\s+([^:]+):\s+(\d+)%\s+\(([^,]+),\s*([^)]+)\)/);
        if (m) {
          latestDeployMatch = m;
          break;
        }
      }

      let latestVideoActive = null;
      let latestUpscaleActive = null;
      let latestVideoDone = null;
      let latestUpscaleDone = null;
      let latestUpscaleErr = null;
      for (let i = d.logs.length - 1; i >= 0; i--) {
        const line = d.logs[i];
        const mvSrErr = line.match(/\[视频超分失败\]\s+(.+)$/);
        if (mvSrErr) {
          latestUpscaleErr = mvSrErr[1];
          break;
        }
        const mvSrDone = line.match(/\[视频超分完成\]\s+(.+)$/);
        if (mvSrDone) {
          latestUpscaleDone = mvSrDone[1];
          break;
        }
        const mvSrAct = line.match(/\[(视频超分|提交 ReCaVSR 超分任务)\]\s+(.+)$/);
        if (mvSrAct) {
          latestUpscaleActive = { stage: mvSrAct[1], msg: mvSrAct[2] };
          break;
        }
        const mvAct = line.match(/\[(视频生成|图片生成|视频回传|图片回传|自动切换引擎)\]\s+(.+)$/);
        if (mvAct) {
          if (mvAct[1] === '视频回传' && mvAct[2].includes('_ReCaVSR_')) {
            latestUpscaleActive = { stage: mvAct[1], msg: mvAct[2] };
          } else {
            latestVideoActive = { stage: mvAct[1], msg: mvAct[2] };
          }
          break;
        }
        const mvDone = line.match(/\[(视频完成|图片完成|视频已落盘|图片已落盘)\]\s+(.+)$/);
        if (mvDone) {
          if (mvDone[2].includes('_ReCaVSR_')) {
            latestUpscaleDone = mvDone[2];
          } else {
            latestVideoDone = mvDone[2];
          }
          break;
        }
      }
      const pLabel = document.getElementById('taskProgressLabel');
      const pSpeed = document.getElementById('taskProgressSpeed');
      const pBar = document.getElementById('taskProgressBar');
      const st = document.getElementById('deployStateText');

      if (!d.running && (dramaPollTimer || videoPollTimer || upscalePollTimer || fusionBatchRunning)) {
        return;
      }
      if (!d.running && latestUpscaleErr) {
        pLabel.innerText = `❌ ${latestUpscaleErr}`;
        pSpeed.innerText = '超分中断，请重试';
        st.innerText = '❌ ReCaVSR 超分失败';
        st.style.color = '#ef4444';
        return;
      }
      if (!d.running && latestUpscaleActive) {
        const uMsg = latestUpscaleActive.msg;
        const pctM = uMsg.match(/(\d+)%/);
        const uPct = pctM ? Math.max(8, Math.min(99, parseInt(pctM[1], 10))) : 10;
        pLabel.innerText = `⏳ ${uMsg}`;
        pBar.className = 'progress-fill';
        pBar.style.width = `${uPct}%`;
        st.innerText = '✨ ReCaVSR 正在超分重绘中...';
        st.style.color = '#a78bfa';
        if (!upscalePollTimer) {
          upscalePollTimer = setInterval(pollUpscaleStatus, 1500);
          pollUpscaleStatus();
        }
        return;
      }
      if (!d.running && latestUpscaleDone && !upscalePollTimer) {
        pLabel.innerText = `✅ ${latestUpscaleDone}`;
        pSpeed.innerText = '已超分完成并保存至 ~/Movies/EasyColab';
        pBar.className = 'progress-fill green';
        pBar.style.width = '100%';
        st.innerText = '✅ ReCaVSR 超分已完成';
        st.style.color = '#10b981';
        return;
      }
      if (!d.running && latestVideoActive) {
        const vMsg = latestVideoActive.msg;
        const pctM = vMsg.match(/(\d+)%/);
        const segM = vMsg.match(/第\s*(\d+)\/(\d+)\s*段/);
        let vPct = 8;
        if (pctM) {
          vPct = Math.max(5, Math.min(99, parseInt(pctM[1], 10)));
        } else if (segM) {
          const sCur = parseInt(segM[1], 10) || 1;
          const sTot = Math.max(1, parseInt(segM[2], 10) || 1);
          vPct = Math.max(5, Math.min(96, 5 + Math.round(((sCur - 1) / sTot) * 90)));
        }
        pLabel.innerText = `⏳ ${vMsg}`;
        if (liveVideoElapsedSec === null && pSpeed) {
          pSpeed.innerText = `总进度 ${vPct}% · 正在生成中...`;
        }
        pBar.className = 'progress-fill';
        pBar.style.width = `${vPct}%`;
        st.innerText = '⏳ 视频正在生成中...';
        st.style.color = '#f59e0b';
        if (!videoPollTimer) {
          videoPollTimer = setInterval(pollVideoStatus, 2000);
          pollVideoStatus();
        }
        return;
      }
      if (!d.running && latestVideoDone && !videoPollTimer) {
        pLabel.innerText = `✅ ${latestVideoDone}`;
        pSpeed.innerText = '已完成并保存至 ~/Movies/EasyColab';
        pBar.className = 'progress-fill green';
        pBar.style.width = '100%';
        st.innerText = '✅ 视频生成已完成';
        st.style.color = '#10b981';
        return;
      }

      if (d.exit_code === 0 && !d.running) {
        pLabel.innerText = '✅ 模型已装载至显存并全部就绪 (100%)';
        pSpeed.innerText = '已就绪';
        pBar.style.background = '';
        pBar.className = 'progress-fill green';
        pBar.style.width = '100%';
      } else if (!d.running && d.exit_code !== null && d.exit_code !== 0) {
        pLabel.innerText = `❌ 部署连接中断 (退出码 ${d.exit_code}) · 点击「🚀 从云盘加载并启动」或「▶️ 一键开机」即可自动重连继续`;
        pSpeed.innerText = '连接中断';
        pBar.className = 'progress-fill';
        pBar.style.background = '#ef4444';
        pBar.style.width = '100%';
      } else if (latestDeployMatch) {
        const stage = latestDeployMatch[1];
        const rawFname = latestDeployMatch[2];
        const fname = rawFname
          .replace(/\((\d+\/\d+)\s*·\s*[^)]+\)/g, '($1 个核心组件)')
          .replace(/([a-zA-Z0-9_.-]{14})[a-zA-Z0-9_.-]{12,}([a-zA-Z0-9_.-]{10})/g, '$1...$2');
        const pct = parseInt(latestDeployMatch[3], 10);
        const gbStr = latestDeployMatch[4];
        const spdStr = latestDeployMatch[5];
        const icon = stage === '显存装载' ? '🎮' : '⏳';
        if (d.running && pct >= 100 && stage !== '显存装载' && latestDeployPostStage) {
          pLabel.innerText = `🎮 [环境与显存装载] ${latestDeployPostStage}`;
          pSpeed.innerText = `${gbStr} · 正在启动推理引擎`;
        } else {
          pLabel.innerText = `${icon} [${stage}] ${fname}: ${pct}%`;
          pSpeed.innerText = `${gbStr} · ${spdStr.includes('复用') || spdStr.includes('已就绪') ? spdStr : ('速度 ' + spdStr)}`;
        }
        pBar.style.background = '';
        pBar.className = stage === '显存装载' ? 'progress-fill green' : 'progress-fill';
        pBar.style.width = `${Math.min(100, pct)}%`;
        if (stage === '显存装载') {
          const vm = gbStr.match(/([\d.]+)\/([\d.]+)/);
          if (vm) {
            const usedV = parseFloat(vm[1]);
            const vPct = Math.min(100, Math.round((usedV / 96.0) * 100));
            document.getElementById('gpuVal').innerText = `${usedV.toFixed(1)} / 96.0 GB (${vPct}%)`;
            document.getElementById('gpuSub').innerText = `正在将模型权重与 KV 缓存装入 GPU 显存 (${pct}%)...`;
            document.getElementById('gpuBar').style.width = `${vPct}%`;
          }
        }
      } else if (d.running) {
        pLabel.innerText = '⏳ 正在初始化环境 / 解析仓库分片...';
        pSpeed.innerText = '请稍候';
        pBar.style.background = '';
      }
    }
    const st = document.getElementById('deployStateText');
    if (d.running) {
      st.innerText = '⏳ 任务正在执行中...';
      st.style.color = '#f59e0b';
    } else if (d.exit_code === 0) {
      st.innerText = '✅ 上次任务执行成功';
      st.style.color = '#10b981';
    } else if (d.exit_code !== null) {
      st.innerText = `❌ 任务退出码: ${d.exit_code}`;
      st.style.color = '#ef4444';
    }
  } catch (e) {}
}

async function startDeploy(downloadOnly = false) {
  const payload = {
    repo: document.getElementById('inpRepo').value.trim(),
    folder: document.getElementById('inpFolder').value.trim(),
    pattern: document.getElementById('inpPattern').value.trim(),
    ctx: parseInt(document.getElementById('inpCtx').value || '262144', 10),
    gpu: 'G4',
    download_only: Boolean(downloadOnly)
  };
  const r = await fetch('/api/deploy', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const d = await r.json();
  alert(d.message);
  pollLogs();
}

async function stopColab() {
  if (!confirm('确定要关闭当前运行的 Colab RTX PRO 6000 实例吗？\n关闭后将立即停止消耗算力单元 (CU)，已同步到 Google Drive 的模型不会丢失。')) return;
  const btn = document.getElementById('btnStop');
  const relBtn = document.getElementById('btnReleaseAll');
  if (btn) { btn.disabled = true; btn.innerText = '🛑 正在关机...'; }
  if (relBtn) { relBtn.disabled = true; relBtn.innerText = '🛑 正在释放远端实例...'; }
  try {
    const r = await fetch('/api/stop', {method: 'POST'});
    const d = await r.json();
    alert(d.message);
    await refreshStatus(true);
  } finally {
    if (btn) { btn.disabled = false; btn.innerText = '🛑 关机'; }
    if (relBtn) { relBtn.disabled = false; }
  }
}

let lastCleanChatReply = '';
let lastStructured4Prompts = null;
let selectedBatchCount = 1;

async function sendTestChat(structured4 = true) {
  const prompt = document.getElementById('chatPrompt').value.trim();
  if (!prompt) return;
  const out = document.getElementById('chatOutput');
  const btn = document.getElementById('btnChat');
  const btn4 = document.getElementById('btnExpand4');
  if (btn) btn.disabled = true;
  if (btn4) btn4.disabled = true;
  out.innerText = structured4
    ? '⏳ 正在由 Qwen3.8-27B (无审查) 自动生成【主体画面 / 环境音效 / 背景音乐 / 反向排除项】四项提示词...'
    : '⏳ 正在请求 RTX PRO 6000 模型生成回复...';
  try {
    const r = await fetch('/api/chat_test', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt, structured_video: Boolean(structured4)})
    });
    const d = await r.json();
    if (d.error) {
      out.innerText = '❌ 请求失败: ' + d.error;
    } else if (d.structured) {
      lastStructured4Prompts = d.structured;
      lastCleanChatReply = d.structured.visual || d.reply || '';
      applyStructured4ToBoxes(d.structured);
      out.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;color:#10b981;font-weight:700;">
          <span>✅ 已生成四项多模态提示词并自动填入下方四个独立框！</span>
          <span style="font-family:'SF Mono',monospace;font-size:11px;">⚡ ${d.speed_tps} tok/s (${d.tokens} toks)</span>
        </div>
        <div style="display:grid;gap:6px;font-size:11.5px;">
          <div style="padding:6px 8px;background:#1e293b;border-radius:6px;border:1px solid var(--border);"><b style="color:#93c5fd;">[1. 主体、构图、光影与风格描述]</b> ${escapeHtml(d.structured.visual)}</div>
          <div style="padding:6px 8px;background:#1e293b;border-radius:6px;border:1px solid var(--border);"><b style="color:#93c5fd;">[2. 环境音效与台词对白]</b> ${escapeHtml(d.structured.soundscape)}</div>
          <div style="padding:6px 8px;background:#1e293b;border-radius:6px;border:1px solid var(--border);"><b style="color:#93c5fd;">[3. 背景音乐风格]</b> ${escapeHtml(d.structured.music)}</div>
          <div style="padding:6px 8px;background:#1e293b;border-radius:6px;border:1px solid var(--border);"><b style="color:#fca5a5;">[4. 反向排除项 Negative]</b> ${escapeHtml(d.structured.negative)}</div>
        </div>`;
    } else {
      lastCleanChatReply = (d.reply || '').trim();
      out.innerText = `${d.reply}\n\n⚡ 实测生成速度: ${d.speed_tps} tokens/s (总计 ${d.tokens} tokens)`;
    }
  } catch (e) {
    out.innerText = '❌ 请求异常: ' + e;
  } finally {
    if (btn) btn.disabled = false;
    if (btn4) btn4.disabled = false;
  }
}

function escapeHtml(s) {
  return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function applyStructured4ToBoxes(s) {
  if (!s) return;
  if (s.visual) document.getElementById('videoPrompt').value = s.visual;
  if (typeof s.soundscape === 'string') document.getElementById('videoSoundPrompt').value = s.soundscape;
  if (typeof s.music === 'string') document.getElementById('videoMusicPrompt').value = s.music;
  if (typeof s.negative === 'string') document.getElementById('videoNegPrompt').value = s.negative;
  const st = document.getElementById('videoGenStatus');
  if (st) {
    st.innerText = '✅ 已将 Qwen3.8 生成的四项提示词全部填入左侧四个框！切换到 MiniMax-H3 后点击「🎬 开始生成音视频」即可。';
    st.style.color = '#10b981';
  }
}

function copyChatToVideoPrompt() {
  if (lastStructured4Prompts) {
    applyStructured4ToBoxes(lastStructured4Prompts);
    document.getElementById('minimaxVideoBox').scrollIntoView({behavior: 'smooth', block: 'center'});
    return;
  }
  const txt = lastCleanChatReply || document.getElementById('chatOutput').innerText.split('⚡ 实测生成速度')[0].trim();
  if (!txt || txt.startsWith('输入视频想法') || txt.startsWith('❌') || txt.startsWith('⏳')) {
    alert('请先在上方点击「✨ 生成四项提示词并填入下方」让 Qwen3.8 生成提示词！');
    return;
  }
  document.getElementById('videoPrompt').value = txt;
  const st = document.getElementById('videoGenStatus');
  st.innerText = '✅ 已从上方 Qwen3.8 回复填入画面提示词！切换到 MiniMax-H3 后点击「🎬 开始生成音视频」即可。';
  st.style.color = '#10b981';
  document.getElementById('minimaxVideoBox').scrollIntoView({behavior: 'smooth', block: 'center'});
}

function clearSingleVideoPrompt(boxId) {
  const el = document.getElementById(boxId);
  if (el) {
    el.value = '';
    el.focus();
  }
}

function clearAllVideoPrompts() {
  ['videoPrompt', 'videoSoundPrompt', 'videoMusicPrompt', 'videoNegPrompt'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = '';
  });
  const st = document.getElementById('videoGenStatus');
  if (st) {
    st.innerText = '🗑️ 已一键清空全部四个提示词框';
    st.style.color = '#94a3b8';
  }
  const firstEl = document.getElementById('videoPrompt');
  if (firstEl) firstEl.focus();
}

function toggleNegPromptBox() {
  const wrap = document.getElementById('negPromptWrap');
  const icon = document.getElementById('negArrowIcon');
  if (!wrap) return;
  const hidden = wrap.style.display === 'none';
  wrap.style.display = hidden ? 'block' : 'none';
  if (icon) icon.innerText = hidden ? '⌄' : '›';
}

function onFramesSliderChange(val) {
  const f = parseInt(val, 10) || 124;
  const sec = (f / 24.0).toFixed(1);
  const lbl = document.getElementById('framesValLabel');
  if (lbl) lbl.innerText = `${f} 帧 · 约 ${sec} 秒`;
  document.querySelectorAll('#durationPills .dur-pill').forEach(btn => {
    const active = parseInt(btn.getAttribute('data-frames'), 10) === f;
    btn.style.borderColor = active ? '#3b82f6' : 'var(--border)';
    btn.style.background = active ? 'rgba(59,130,246,0.22)' : '#0f172a';
    btn.style.color = active ? '#60a5fa' : '#cbd5e1';
    btn.style.fontWeight = active ? '700' : '600';
  });
}

function selectDurationFrames(frames) {
  const slider = document.getElementById('videoFrames');
  if (slider) slider.value = String(frames);
  onFramesSliderChange(frames);
}

function selectBatchCount(count) {
  selectedBatchCount = parseInt(count, 10) || 1;
  document.querySelectorAll('#batchCountGroup .batch-pill').forEach(btn => {
    const active = parseInt(btn.getAttribute('data-batch'), 10) === selectedBatchCount;
    btn.style.background = active ? '#3b82f6' : 'transparent';
    btn.style.color = active ? '#ffffff' : '#cbd5e1';
    btn.style.fontWeight = active ? '700' : '600';
  });
}

let currentMiniMaxMode = 't2va';
let firstFrameB64 = '';
let lastFrameB64 = '';
let avatarAudioB64 = '';
let avatarAudioExt = '.mp3';
let avatarAudioIsTts = false;
let avatarUseBuiltinSample = true;
let h3ImgRefList = ['', '', ''];

const VIDEO_RES_OPTIONS_HTML = `
  <option value="832x480" selected>832×480 (16:9 横屏·推荐)</option>
  <option value="480x832">480×832 (9:16 竖屏)</option>
  <option value="1280x720">1280×720 (720P 横屏)</option>
  <option value="720x1280">720×1280 (720P 竖屏)</option>
  <option value="640x640">640×640 (1:1 方形)</option>
`;

const H3_IMAGE_RES_OPTIONS_HTML = `
  <option value="3552x4736" selected>3552×4736 (3:4 竖版4K人像 · 1682万像素·推荐)</option>
  <option value="3360x5024">3360×5024 (2:3 写真4K竖幅 · 1688万像素)</option>
  <option value="3072x5472">3072×5472 (9:16 全屏4K竖屏 · 1681万像素)</option>
  <option value="4096x4096">4096×4096 (1:1 方形4K海报 · 1677万像素)</option>
  <option value="5472x3072">5472×3072 (16:9 宽幅4K夜景 · 1681万像素)</option>
  <option value="4736x3552">4736×3552 (4:3 横版4K展板 · 1682万像素)</option>
  <option value="2048x1152">2048×1152 (16:9 极速2K电影横屏 · ~2.3s · 短剧分镜/数字人底图推荐)</option>
  <option value="1152x2048">1152×2048 (9:16 极速2K短剧竖屏 · ~2.3s · 竖屏短剧/数字人底图推荐)</option>
  <option value="1536x2048">1536×2048 (3:4 极速2K人像 · ~2.5s)</option>
`;

function switchMiniMaxMode(mode) {
  const prevMode = currentMiniMaxMode;
  currentMiniMaxMode = mode;
  const isT2 = mode === 't2va';
  const isFl = mode === 'fl2va';
  const isAv = mode === 'avatar';
  const isImg = mode === 'image';
  const isDrama = mode === 'drama';
  const isFusion = mode === 'fusion';
  const flRow = document.getElementById('fl2vaImageRow');
  const avRow = document.getElementById('avatarControlRow');
  const imgRow = document.getElementById('h3ImageControlRow');
  const dramaRow = document.getElementById('autoDramaDirectorRow');
  const fusionRow = document.getElementById('fusionStudioRow');
  const promptCard = document.getElementById('promptSectionCard');
  const specCard = document.getElementById('specSectionCard');
  if (flRow) flRow.style.display = isFl ? 'grid' : 'none';
  if (avRow) avRow.style.display = isAv ? 'grid' : 'none';
  if (imgRow) imgRow.style.display = isImg ? 'grid' : 'none';
  if (dramaRow) dramaRow.style.display = isDrama ? 'block' : 'none';
  if (fusionRow) fusionRow.style.display = isFusion ? 'block' : 'none';
  if (promptCard) promptCard.style.display = (isDrama || isFusion) ? 'none' : 'block';
  if (specCard) specCard.style.display = (isDrama || isFusion) ? 'none' : 'block';
  const tabConfigs = [
    {id: 'tabBtnImage', active: isImg, border: '#ec4899', bg: 'linear-gradient(135deg,rgba(236,72,153,0.34),rgba(15,23,42,0.95))', shadow: 'rgba(236,72,153,0.45)'},
    {id: 'tabBtnFusion', active: isFusion, border: '#c084fc', bg: 'linear-gradient(135deg,rgba(168,85,247,0.38),rgba(15,23,42,0.95))', shadow: 'rgba(168,85,247,0.48)'},
    {id: 'tabBtnT2va', active: (!isFl && !isAv && !isImg && !isDrama && !isFusion), border: '#3b82f6', bg: 'linear-gradient(135deg,rgba(37,99,235,0.36),rgba(15,23,42,0.95))', shadow: 'rgba(59,130,246,0.45)'},
    {id: 'tabBtnFl2va', active: isFl, border: '#38bdf8', bg: 'linear-gradient(135deg,rgba(14,165,233,0.34),rgba(15,23,42,0.95))', shadow: 'rgba(56,189,248,0.45)'},
    {id: 'tabBtnAvatar', active: isAv, border: '#10b981', bg: 'linear-gradient(135deg,rgba(16,185,129,0.34),rgba(15,23,42,0.95))', shadow: 'rgba(16,185,129,0.45)'},
    {id: 'tabBtnAutoDrama', active: isDrama, border: '#f59e0b', bg: 'linear-gradient(135deg,rgba(245,158,11,0.36),rgba(15,23,42,0.95))', shadow: 'rgba(245,158,11,0.48)'},
  ];
  tabConfigs.forEach(tc => {
    const el = document.getElementById(tc.id);
    if (!el) return;
    el.style.borderColor = tc.active ? tc.border : '#334155';
    el.style.background = tc.active ? tc.bg : 'rgba(15,23,42,0.75)';
    el.style.boxShadow = tc.active ? `0 8px 20px -4px ${tc.shadow}` : 'none';
    el.style.transform = tc.active ? 'translateY(-1px)' : 'none';
  });
  const titleEl = document.getElementById('videoStudioTitle');
  const badgeEl = document.getElementById('videoStudioBadge');
  if (isDrama) {
    if (titleEl) titleEl.innerText = '🎬 通用分镜短剧全自动工作流导演台 (H3-Image 锁脸 + LongCat-Avatar 对白 + H3-FL2VA 动作)';
    if (badgeEl) badgeEl.innerText = '任意剧本智能拆解 + 可视化分镜节点编辑 + 单步独立重抽 + 0.1秒卡点混音合片';
    const drawer = document.getElementById('dramaWorkflowEditorDrawer');
    if (drawer) drawer.style.display = 'block';
    loadDramaWorkflowEditor();
    pollAutoDramaStatus();
    return;
  }
  if (isFusion) {
    if (titleEl) titleEl.innerText = '🪄 AI 智能换脸 · 试衣换装 · 实景光影融合工坊 (MiniMax-H3-Image 33B 多图空间主语解耦)';
    if (badgeEl) badgeEl.innerText = '免涂 Mask 像素级骨相锁脸 + 服装纹理剥离 + 实景环境光重打光';
    syncFusionRoleDropdown();
    renderFusionTargetCards();
    return;
  }
  const pEl = document.getElementById('videoPrompt');
  const pSecTitle = document.getElementById('promptSectionTitle');
  const fourBtns = document.getElementById('fourBoxActionBtns');
  const box1Wrap = document.getElementById('videoPromptBox1Wrap');
  const box1Lbl = document.getElementById('videoPromptBox1Label');
  const box2Wrap = document.getElementById('videoSoundBoxWrap');
  const box3Wrap = document.getElementById('videoMusicBoxWrap');
  const box4Wrap = document.getElementById('videoNegBoxWrap');
  const specRow1 = document.getElementById('specRow1Grid');
  const stepsWrap = document.getElementById('stepsControlWrap');
  const batchWrap = document.getElementById('batchControlWrap');
  const framesInner = document.getElementById('framesSliderInner');
  const avSpecHint = document.getElementById('avatarSpecHint');
  const sr2kLbl = document.getElementById('autoUpscale2kLabel');
  const pureImgTag = document.getElementById('h3ImagePureTag');
  const btnGen = document.getElementById('btnVideoGen');
  const btnImgGen = document.getElementById('btnImageOnlyGen');
  const resSel = document.getElementById('videoRes');
  if (resSel) {
    if (isImg && prevMode !== 'image') {
      resSel.innerHTML = H3_IMAGE_RES_OPTIONS_HTML;
    } else if (!isImg && prevMode === 'image') {
      resSel.innerHTML = VIDEO_RES_OPTIONS_HTML;
      if (isAv) resSel.value = '480x832';
    }
  }
  if (isImg) {
    if (titleEl) titleEl.innerText = '📸 MiniMax-H3-Image (33B) 原生 4K 无审查写真 & 免Mask锁脸换装工作台';
    if (badgeEl) badgeEl.innerText = '11.6秒原生1680万像素4K直出 + Heretic 32B无审查编码器 + Picture+Subject免Mask锁脸换装';
    if (pSecTitle) pSecTitle.innerText = '3. 4K 写真画面提示词 / Picture + Subject 免Mask锁脸换装指令';
    if (fourBtns) fourBtns.style.display = 'none';
    if (box1Wrap) box1Wrap.style.marginBottom = '0';
    if (box1Lbl) box1Lbl.innerText = '画面描述或改图指令（已自动开启去磨砂真皮过滤；若上方上传了 <Picture 1> 并填写了换装栏，将自动合成三段式锁脸语法）';
    if (box2Wrap) box2Wrap.style.display = 'none';
    if (box3Wrap) box3Wrap.style.display = 'none';
    if (box4Wrap) box4Wrap.style.display = 'none';
    if (stepsWrap) stepsWrap.style.display = 'none';
    if (batchWrap) batchWrap.style.display = 'none';
    if (specRow1) specRow1.style.gridTemplateColumns = '1.4fr 1fr';
    if (framesInner) framesInner.style.display = 'none';
    if (sr2kLbl) sr2kLbl.style.display = 'none';
    if (pureImgTag) pureImgTag.style.display = 'inline-flex';
    if (avSpecHint) {
      avSpecHint.style.display = 'block';
      avSpecHint.innerText = '⚡ MiniMax-H3-Image：固定 6 步蒸馏 (CFG=1.0 · 免负面词)';
    }
    if (btnImgGen && btnImgGen.getAttribute('data-running') !== '1') {
      btnImgGen.innerText = '📸 仅生成 4K 写真图片';
      btnImgGen.style.flex = '1.35';
    }
    if (btnGen && btnGen.getAttribute('data-running') !== '1') {
      btnGen.innerText = '🎬 生成有声视频';
      btnGen.style.flex = '0.95';
    }
    if (pEl) {
      pEl.rows = 4;
      pEl.style.minHeight = '95px';
      if (pEl.value.includes('Makoto Shinkai anime style') || pEl.value.includes('固定镜头，近景居中构图')) {
        pEl.value = 'A high-resolution, real-life portrait photograph of an elegant 20-year-old Chinese young woman with a classic East Asian oval face, smooth fair skin, expressive dark brown almond eyes, and a serene, natural expression, standing beneath the carved wooden eaves of a traditional Suzhou courtyard at dusk. Her long, sleek black hair is styled in a graceful half-up bun secured with a carved white jade hairpin. She wears a pale moonlight-white silk Chinese stand-collar jacket with subtle woven floral patterns and holds a warm glowing bamboo paper lantern near her chest. Behind her, soft out-of-focus red lanterns and grey tiled roofs glow warmly in the evening twilight, captured on a Canon EOS R5 with an 85mm f/1.4 lens, clean natural lighting, and shallow depth of field.';
      }
    }
  } else if (isAv) {
    if (titleEl) titleEl.innerText = '🎭 LongCat-Video-Avatar-1.6 数字人音频驱动视频创作工作台';
    if (badgeEl) badgeEl.innerText = 'Whisper-v3 唇形同步 + DMD/OPSD 8步双LoRA蒸馏 + Kim_Vocal_2 人声分离';
    if (pSecTitle) pSecTitle.innerText = '3. 纯画面与人物神态描述 Prompt（仅写镜头/表情/眼神/光影，不要把台词写在这里）';
    if (fourBtns) fourBtns.style.display = 'none';
    if (box1Wrap) box1Wrap.style.marginBottom = '0';
    if (box1Lbl) box1Lbl.innerText = '画面与人物神态描述（要说的话请填在上方第 1 步黄色「💬 台词配音框」内；若一起粘贴会自动为您分离）';
    if (box2Wrap) box2Wrap.style.display = 'none';
    if (box3Wrap) box3Wrap.style.display = 'none';
    if (box4Wrap) box4Wrap.style.display = 'none';
    if (stepsWrap) stepsWrap.style.display = 'none';
    if (batchWrap) batchWrap.style.display = 'none';
    if (specRow1) specRow1.style.gridTemplateColumns = '1.3fr 1fr';
    if (framesInner) framesInner.style.display = 'none';
    if (sr2kLbl) sr2kLbl.style.display = 'inline-flex';
    if (pureImgTag) pureImgTag.style.display = 'none';
    if (avSpecHint) {
      avSpecHint.style.display = 'block';
      avSpecHint.innerText = '⚡ LongCat-Avatar-1.6：固定 8 步双蒸馏 (DMD + OPSD) · 声音与口型由上方驱动音频决定 · 视频时长由上方「生成段数」控制';
    }
    if (btnImgGen && btnImgGen.getAttribute('data-running') !== '1') {
      btnImgGen.innerText = '📸 仅生成 4K 图片';
      btnImgGen.style.flex = '1';
    }
    if (btnGen && btnGen.getAttribute('data-running') !== '1') {
      btnGen.innerText = '🎬 生成数字人视频';
      btnGen.style.flex = '1.2';
    }
    if (pEl) {
      pEl.rows = 3;
      pEl.style.minHeight = '76px';
      if (pEl.value.includes('Makoto Shinkai anime style') || pEl.value.includes('Suzhou courtyard at dusk')) {
        pEl.value = '固定镜头，近景居中构图。画面中的人物直视镜头自然说话，眼神温柔灵动，伴随说话节奏有轻微自然的眨眼、轻轻点头和呼吸起伏，嘴型与语音高度同步，面部柔和暖光，五官清晰稳定，过渡丝滑真实。';
      }
      autoSplitAvatarPromptAndTts(pEl);
    }
  } else {
    if (titleEl) titleEl.innerText = '🎬 MiniMax-H3 多模态有声视频创作工作台';
    if (badgeEl) badgeEl.innerText = '32kHz 原生立体声音轨 + 32B 视觉编码器';
    if (pSecTitle) pSecTitle.innerText = '镜头与多模态提示词';
    if (fourBtns) fourBtns.style.display = 'flex';
    if (box1Wrap) box1Wrap.style.marginBottom = '14px';
    if (box1Lbl) box1Lbl.innerText = '1. 主体、构图、光影与风格描述';
    if (box2Wrap) box2Wrap.style.display = 'block';
    if (box3Wrap) box3Wrap.style.display = 'block';
    if (box4Wrap) box4Wrap.style.display = 'block';
    if (stepsWrap) stepsWrap.style.display = 'flex';
    if (batchWrap) batchWrap.style.display = 'flex';
    if (specRow1) specRow1.style.gridTemplateColumns = '1.3fr 1.15fr 1fr 1.15fr';
    if (framesInner) framesInner.style.display = 'flex';
    if (sr2kLbl) sr2kLbl.style.display = 'inline-flex';
    if (pureImgTag) pureImgTag.style.display = 'none';
    if (avSpecHint) avSpecHint.style.display = 'none';
    if (btnImgGen && btnImgGen.getAttribute('data-running') !== '1') {
      btnImgGen.innerText = '📸 仅生成 4K 图片';
      btnImgGen.style.flex = '1';
    }
    if (btnGen && btnGen.getAttribute('data-running') !== '1') {
      btnGen.innerText = '🎬 生成有声视频';
      btnGen.style.flex = '1.2';
    }
    if (pEl) {
      pEl.rows = 4;
      pEl.style.minHeight = '110px';
      if (pEl.value.includes('A western man stands on stage under dramatic lighting')) {
        pEl.value = 'Masterpiece Makoto Shinkai anime style, 2D cel-shaded animation. A teenage girl in a Japanese sailor school uniform standing at a sunlit railway crossing in late spring. A green commuter train speeds past in front of her, creating a strong gust of wind that blows her long black hair and a flurry of pink cherry blossom petals across the frame. Dramatic lens flare, glowing cumulous clouds in a deep blue sky, vibrant cinematic lighting, smooth 24fps keyframe animation.';
      }
    }
  }
}

let currentFusionSubMode = 'tryon';
let fusionSlotsB64 = ['', '', ''];
let fusionTargetList = [''];
let fusionBatchRunning = false;
let fusionBatchCancelled = false;
let fusionBatchCompletedFiles = [];

function syncFusionRoleDropdown() {
  const sel = document.getElementById('selFusionTargetRole');
  if (!sel) return;
  const roles = (Array.isArray(dramaCastState) && dramaCastState.length > 0)
    ? dramaCastState
    : [{id: 'A', name: '角色A'}, {id: 'B', name: '角色B'}, {id: 'C', name: '角色C'}, {id: 'D', name: '角色D'}, {id: 'E', name: '角色E'}];
  const curVal = sel.value || 'A';
  sel.innerHTML = roles.map(r => `<option value="${escapeHtmlAttr(r.id)}" ${r.id === curVal ? 'selected' : ''}>🎭 短剧角色${escapeHtmlAttr(r.id)}${r.name ? ' (' + escapeHtmlAttr(r.name) + ')' : ''}</option>`).join('');
}

function toggleFusionAdvancedDrawer() {
  const dr = document.getElementById('fusionAdvancedDrawer');
  const btn = document.getElementById('btnToggleFusionAdv');
  if (!dr) return;
  const open = dr.style.display === 'none';
  dr.style.display = open ? 'block' : 'none';
  if (btn) btn.innerText = open ? '⚙️ 收起高级微调 ▴' : '⚙️ 高级微调 (默认自动最优) ▾';
}

function updateFusionRunBtnLabel() {
  const btn = document.getElementById('btnRunFusionStudio');
  if (!btn || fusionBatchRunning) return;
  const filledCnt = fusionTargetList.filter(Boolean).length;
  const multiMode = ((document.getElementById('fusionMultiActionMode') || {}).value || 'batch');
  if (filledCnt >= 2 && multiMode === 'batch') {
    btn.innerText = `🪄 一键批量合成全部 (共 ${filledCnt} 张)`;
  } else if (filledCnt >= 2 && multiMode === 'combine') {
    btn.innerText = `🪄 一键多图融合合成 (${filledCnt} 图合 1)`;
  } else {
    btn.innerText = '🪄 一键开始合成';
  }
}

function renderFusionTargetCards() {
  const grid = document.getElementById('fusionTargetCardsGrid');
  const badge = document.getElementById('fusionTargetCountBadge');
  const multiSel = document.getElementById('fusionMultiActionMode');
  if (!grid) return;
  if (!Array.isArray(fusionTargetList) || fusionTargetList.length === 0) {
    fusionTargetList = [''];
  }
  const totalSlots = fusionTargetList.length;
  const filledCnt = fusionTargetList.filter(Boolean).length;
  if (badge) {
    badge.innerText = totalSlots > 1 ? `已传 ${filledCnt} / ${totalSlots} 张` : (filledCnt ? '已传 1 张' : '1 个槽位');
  }
  if (multiSel) {
    multiSel.style.display = totalSlots >= 2 ? 'inline-block' : 'none';
  }
  if (totalSlots === 1) {
    grid.style.gridTemplateColumns = '1fr';
  } else if (totalSlots === 2) {
    grid.style.gridTemplateColumns = 'repeat(2, 1fr)';
  } else {
    grid.style.gridTemplateColumns = 'repeat(auto-fill, minmax(155px, 1fr))';
  }
  const isSwap = currentFusionSubMode === 'faceswap';
  const isScene = currentFusionSubMode === 'scenefuse';
  const cardIcon = isSwap ? '🖼️' : '👗';
  const cardTitlePrefix = isSwap ? '被换脸图' : '服装/穿搭图';
  const emptyHint = isSwap
    ? '点击上传【被替换的目标照片】'
    : (isScene ? '点击上传【服装照片】(可选)' : '点击上传【服装/模特穿搭照】');
  const subHint = totalSlots === 1
    ? '可点右上角「➕ 新增照片槽位」或「📂 批量多选导入」一次合成多张；若上传了左右拼图可点「✂️拆2张」'
    : '支持独立替换或点右上角 ✕ 删除此槽位';

  grid.innerHTML = fusionTargetList.map((b64, idx) => {
    const hasImg = Boolean(b64);
    const cardHeight = totalSlots === 1 ? '165px' : '136px';
    return `
      <div style="padding:6px 8px;border-radius:8px;background:#0f172a;border:1.5px ${hasImg ? 'solid #ec4899' : 'dashed #be185d'};display:flex;flex-direction:column;gap:4px;">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:4px;">
          <span style="font-size:11px;font-weight:800;color:#fbcfe8;">${cardIcon} ${cardTitlePrefix} #${idx + 1}</span>
          <div style="display:flex;align-items:center;gap:3px;">
            ${hasImg ? `<button type="button" onclick="splitFusionTargetCardLeftRight(${idx})" style="padding:1px 6px;font-size:10px;font-weight:700;border-radius:4px;border:1px solid #38bdf8;background:rgba(56,189,248,0.16);color:#7dd3fc;cursor:pointer;" title="如果这张图是左右并排的两张照片截图（如推特/小红书双图），点击即可自动左、右一分为二拆成 2 个槽位！">✂️左右拆2张</button>` : ''}
            <button type="button" onclick="removeFusionTargetSlot(${idx})" style="padding:1px 6px;font-size:10px;font-weight:700;border-radius:4px;border:1px solid rgba(239,68,68,0.5);background:rgba(239,68,68,0.14);color:#fca5a5;cursor:pointer;" title="${totalSlots > 1 ? '删除此照片槽位' : '清空此槽位照片'}">✕</button>
          </div>
        </div>
        <label style="position:relative;height:${cardHeight};border-radius:6px;background:#020617;display:flex;flex-direction:column;align-items:center;justify-content:center;cursor:pointer;overflow:hidden;">
          ${hasImg
            ? `<img src="${b64}" style="width:100%;height:100%;object-fit:contain;display:block;">`
            : `<div style="text-align:center;color:#94a3b8;font-size:11.5px;padding:8px;">
                <div style="font-size:24px;margin-bottom:2px;">${cardIcon}</div>
                <div style="color:#fbcfe8;font-weight:700;">${emptyHint} #${idx + 1}</div>
                <div style="font-size:10px;color:#64748b;margin-top:2px;">${subHint}</div>
              </div>`
          }
          <input type="file" accept="image/*" multiple style="display:none;" onchange="onFusionTargetCardFileChange(${idx}, this)">
        </label>
      </div>
    `;
  }).join('');
  fusionSlotsB64[1] = fusionTargetList.find(Boolean) || '';
  updateFusionRunBtnLabel();
}

function addFusionTargetSlot() {
  fusionTargetList.push('');
  renderFusionTargetCards();
  showGlobalCopyToast(`➕ 已新增「槽位 #${fusionTargetList.length}」！您现在可以上传多张照片一次性批量生成`);
}

function removeFusionTargetSlot(idx) {
  if (fusionTargetList.length <= 1) {
    fusionTargetList = [''];
  } else {
    fusionTargetList.splice(idx, 1);
  }
  renderFusionTargetCards();
}

function clearAllFusionTargets() {
  fusionTargetList = [''];
  renderFusionTargetCards();
}

function onFusionTargetCardFileChange(slotIdx, inputEl) {
  const files = inputEl && inputEl.files ? Array.from(inputEl.files) : [];
  if (!files.length) return;
  if (files.length === 1) {
    const reader = new FileReader();
    reader.onload = ev => {
      fusionTargetList[slotIdx] = ev.target.result || '';
      renderFusionTargetCards();
    };
    reader.readAsDataURL(files[0]);
    return;
  }
  // 如果用户在单个卡片里一次框选了多张照片，自动把第 1 张放入当前卡，其余自动追加新卡片！
  Promise.all(files.map(f => new Promise(res => {
    const r = new FileReader();
    r.onload = e => res(e.target.result || '');
    r.readAsDataURL(f);
  }))).then(dataUrls => {
    const valid = dataUrls.filter(Boolean);
    if (!valid.length) return;
    fusionTargetList.splice(slotIdx, 1, ...valid);
    renderFusionTargetCards();
    showGlobalCopyToast(`📂 已一次性导入 ${valid.length} 张照片！点击「🪄 一键批量合成全部」即可一次全部生成`);
  });
}

function onFusionMultiFilesSelect(inputEl) {
  const files = inputEl && inputEl.files ? Array.from(inputEl.files) : [];
  if (!files.length) return;
  Promise.all(files.map(f => new Promise(res => {
    const r = new FileReader();
    r.onload = e => res(e.target.result || '');
    r.readAsDataURL(f);
  }))).then(dataUrls => {
    const valid = dataUrls.filter(Boolean);
    if (!valid.length) return;
    const existingFilled = fusionTargetList.filter(Boolean);
    fusionTargetList = existingFilled.concat(valid);
    renderFusionTargetCards();
    showGlobalCopyToast(`📂 已批量导入 ${valid.length} 张照片（当前共 ${fusionTargetList.length} 张）！`);
  });
  inputEl.value = '';
}

function splitFusionTargetCardLeftRight(idx) {
  const srcB64 = fusionTargetList[idx];
  if (!srcB64) return;
  const img = new Image();
  img.onload = () => {
    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;
    if (w < 40 || h < 40) return;
    // 自动切除社交平台双联截图四周的白边/圆角与中间白缝
    const topY = Math.floor(h * 0.045);
    const cropH = Math.floor(h * 0.91);
    const leftX = Math.floor(w * 0.035);
    const leftW = Math.floor(w * 0.452);
    const rightX = Math.ceil(w * 0.513);
    const rightW = Math.floor(w * 0.452);

    const c1 = document.createElement('canvas');
    c1.width = leftW;
    c1.height = cropH;
    c1.getContext('2d').drawImage(img, leftX, topY, leftW, cropH, 0, 0, leftW, cropH);
    const leftB64 = c1.toDataURL('image/jpeg', 0.95);

    const c2 = document.createElement('canvas');
    c2.width = rightW;
    c2.height = cropH;
    c2.getContext('2d').drawImage(img, rightX, topY, rightW, cropH, 0, 0, rightW, cropH);
    const rightB64 = c2.toDataURL('image/jpeg', 0.95);

    fusionTargetList.splice(idx, 1, leftB64, rightB64);
    renderFusionTargetCards();
    showGlobalCopyToast('✂️ 已将该左右双联截图拆分为 #1、#2 两张无白边独立竖版照片！点击「🪄 一键批量合成全部」即可一次生成两张');
  };
  img.src = srcB64;
}

function detectImageAspectRes(dataUrl) {
  return new Promise(resolve => {
    if (!dataUrl) {
      resolve('1152x2048');
      return;
    }
    const img = new Image();
    img.onload = () => {
      const w = img.naturalWidth || img.width || 1000;
      const h = img.naturalHeight || img.height || 1000;
      if (h > w * 1.12) {
        resolve('1152x2048');
      } else if (w > h * 1.12) {
        resolve('2048x1152');
      } else {
        resolve('1152x2048');
      }
    };
    img.onerror = () => resolve('1152x2048');
    img.src = dataUrl;
  });
}

function switchFusionSubMode(subMode) {
  currentFusionSubMode = subMode;
  const bTry = document.getElementById('btnFusionModeTryon');
  const bSwap = document.getElementById('btnFusionModeFaceswap');
  const bScene = document.getElementById('btnFusionModeScenefuse');
  const activeStyle = 'padding:6px 14px;font-size:12.5px;font-weight:800;border-radius:8px;border:1.5px solid #c084fc;background:linear-gradient(135deg,#9333ea,#c026d3);color:#ffffff;cursor:pointer;';
  const idleStyle = 'padding:6px 14px;font-size:12.5px;font-weight:700;border-radius:8px;border:1px solid #475569;background:#0f172a;color:#cbd5e1;cursor:pointer;';
  if (bTry) bTry.style.cssText = subMode === 'tryon' ? activeStyle : idleStyle;
  if (bSwap) bSwap.style.cssText = subMode === 'faceswap' ? activeStyle : idleStyle;
  if (bScene) bScene.style.cssText = subMode === 'scenefuse' ? activeStyle : idleStyle;

  const grid = document.getElementById('fusionSlotsGrid');
  const s3Card = document.getElementById('fusionSlot3Card');
  const s1Title = document.getElementById('fusionSlot1Title');
  const s2Title = document.getElementById('fusionSlot2Title');
  const s3Title = document.getElementById('fusionSlot3Title');
  const s3Main = document.getElementById('fusionSlot3MainHint');
  const framingSel = document.getElementById('fusionFramingSelect');
  const lightSel = document.getElementById('fusionLightingSelect');
  const hairSel = document.getElementById('fusionHairModeSelect');

  if (subMode === 'faceswap') {
    if (grid) grid.style.gridTemplateColumns = '1fr 1.65fr';
    if (s3Card) s3Card.style.display = 'none';
    if (s1Title) s1Title.innerText = '🧑 第一步：上传我的人脸照片';
    if (s2Title) s2Title.innerText = '🖼️ 第二步：上传被换脸的目标照片 (支持多张)';
    if (framingSel) framingSel.value = 'keep_target_frame';
    if (lightSel) lightSel.value = 'keep_target_light';
    if (hairSel) hairSel.value = 'keep_target_hair';
  } else if (subMode === 'scenefuse') {
    if (grid) grid.style.gridTemplateColumns = '1fr 1.35fr 1fr';
    if (s3Card) s3Card.style.display = 'flex';
    if (s1Title) s1Title.innerText = '🧑 第一步：上传人物/人脸照片';
    if (s2Title) s2Title.innerText = '👗 第二步：换套服装 (可选，支持多张)';
    if (s3Title) s3Title.innerText = '🏞️ 第三步：上传场景照片';
    if (s3Main) s3Main.innerText = '点击上传【场景环境照片】';
    if (framingSel) framingSel.value = 'keep_target_frame';
    if (lightSel) lightSel.value = 'clear_35mm_scene';
    if (hairSel) hairSel.value = 'keep_face_hair';
  } else {
    if (grid) grid.style.gridTemplateColumns = '1fr 1.65fr';
    if (s3Card) s3Card.style.display = 'none';
    if (s1Title) s1Title.innerText = '🧑 第一步：上传人脸照片';
    if (s2Title) s2Title.innerText = '👗 第二步：上传服装/穿搭参考图 (支持多张)';
    if (framingSel) framingSel.value = 'keep_target_frame';
    if (lightSel) lightSel.value = 'keep_target_light';
    if (hairSel) hairSel.value = 'keep_face_hair';
  }
  renderFusionTargetCards();
}

function onFusionSlotFileChange(slotIdx, inputEl) {
  const file = inputEl && inputEl.files && inputEl.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    setFusionSlotB64(slotIdx, ev.target.result);
  };
  reader.readAsDataURL(file);
}

function setFusionSlotB64(slotIdx, dataUrl) {
  if (slotIdx === 1) {
    fusionTargetList[0] = dataUrl || '';
    renderFusionTargetCards();
    return;
  }
  fusionSlotsB64[slotIdx] = dataUrl || '';
  const imgEl = document.getElementById(`fusionSlot${slotIdx + 1}Img`);
  const empEl = document.getElementById(`fusionSlot${slotIdx + 1}Empty`);
  if (dataUrl) {
    if (imgEl) { imgEl.src = dataUrl; imgEl.style.display = 'block'; }
    if (empEl) empEl.style.display = 'none';
  } else {
    if (imgEl) { imgEl.src = ''; imgEl.style.display = 'none'; }
    if (empEl) empEl.style.display = 'block';
  }
}

function clearFusionSlot(slotIdx) {
  if (slotIdx === 1) {
    clearAllFusionTargets();
    return;
  }
  setFusionSlotB64(slotIdx, '');
  const inp = document.getElementById(`inpFusionSlot${slotIdx + 1}`);
  if (inp) inp.value = '';
}

async function loadCurrentPreviewIntoFusionSlot(slotIdx) {
  const prevImg = document.getElementById('mmImagePreview');
  if (prevImg && prevImg.style.display !== 'none' && prevImg.src) {
    try {
      const r = await fetch(prevImg.src);
      const blob = await r.blob();
      const reader = new FileReader();
      reader.onload = ev => {
        setFusionSlotB64(slotIdx, ev.target.result);
        showGlobalCopyToast(`✅ 已将当前预览大图载入「槽位 ${slotIdx + 1}」！`);
      };
      reader.readAsDataURL(blob);
      return;
    } catch (e) {}
  }
  alert('请先在下方历史记录中点击选择一张已生成的图片，再点「📋取预览图」载入此槽位');
}

// 核心底层机制：MiniMaxH3ImageConditioning 中 image_1 (<Picture 1>) 永远是空间构图与姿势底图 ("image_1 is the picture to edit")！
// 因此当上传了服装/姿势参考图时，必须把【服装/姿势底图】作为 <Picture 1>，把【目标人脸照】作为 <Picture 2> 注入，才能 100% 锁住底图坐姿/动作，同时 100% 还原目标人脸的精致小V脸与神态！
function buildFusionRefsAndPrompt(targetListForJob) {
  const faceB64 = fusionSlotsB64[0] || '';
  const targets = Array.isArray(targetListForJob) ? targetListForJob.filter(Boolean) : [];
  const sceneB64 = fusionSlotsB64[2] || '';
  const hasFace = Boolean(faceB64);
  const hasSlot2 = targets.length > 0;
  const hasScene = Boolean(sceneB64);

  const exprVal = ((document.getElementById('fusionExprSelect') || {}).value || 'keep_orig');
  const gazeVal = ((document.getElementById('fusionGazeSelect') || {}).value || 'keep_target');
  const bodyVal = ((document.getElementById('fusionBodySelect') || {}).value || 'standard');
  const frameVal = ((document.getElementById('fusionFramingSelect') || {}).value || 'keep_target_frame');
  const lightVal = ((document.getElementById('fusionLightingSelect') || {}).value || 'keep_target_light');
  const hairVal = ((document.getElementById('fusionHairModeSelect') || {}).value || 'keep_face_hair');
  const accVal = ((document.getElementById('fusionAccessorySelect') || {}).value || 'auto_face');
  const extraTxt = (((document.getElementById('fusionPropsActionInput') || {}).value) || '').trim();

  const exprMap = {
    keep_orig: 'preserving the exact natural facial expression, smile, and lip posture from the face reference photo',
    natural_closed: 'serene, calm natural expression with relaxed gently closed lips (no teeth showing, ideal neutral mouth posture)',
    gentle_smile: 'warm, subtle gentle smile with expressive eyes',
    boss_serious: 'commanding, sharp, authoritative serious expression with steady piercing gaze',
    sly_smirk: 'subtle, clever, knowing half-smile with agile observant eyes',
    sad_teary: 'emotional, melancholic expression with glistening slightly reddened eyes',
  };
  const gazeMap = {
    keep_target: 'natural gaze direction matching the pose and camera composition',
    front_camera: 'looking directly at the camera at eye level',
    side_15_right: 'head turned slightly 15 degrees looking toward an off-camera person on the right',
    side_15_left: 'head turned slightly 15 degrees looking toward an off-camera person on the left',
  };
  const bodyMap = {
    standard: 'preserving the natural body proportions and posture of the pose reference',
    boss_stocky: 'solid, stocky, mature heavyweight boss build with broad shoulders',
    athletic_tall: 'tall, broad-shouldered, athletic muscular build',
    slim_youth: 'slender, lean, refined youthful build',
    curvy_tall: 'tall, graceful, slender and well-proportioned feminine silhouette',
  };
  const frameMap = {
    keep_target_frame: '100% preserve the exact camera framing, shot scale, seated/standing body pose, arm and hand placement, and leg posture from <Picture 1>',
    waist_medium: 'Three-quarter to waist-up fashion portrait clearly showcasing the complete upper garment, waistline, lower pants/skirt pairing, and body posture',
    bust_half: 'Medium close-up bust portrait (from chest up to top of head with natural headroom)',
    closeup_face: 'Cinematic facial close-up portrait (focusing on facial features, eyes, and collar)',
    full_body: 'Full-body to three-quarter cinematic shot naturally grounded in the environment',
  };
  const lightMap = {
    keep_target_light: '100% preserve the clean background environment, studio lighting, shadows, and color grading from <Picture 1>.',
    cinema_85mm_bokeh: 'Captured on an 85mm f/1.4 cinema portrait lens, shallow depth of field with soft creamy background bokeh, natural ambient rim light, realistic skin texture.',
    clear_35mm_scene: 'Captured on a 35mm cinema lens with deep focus, both the person and the surrounding architectural/scene details clearly visible, realistic spatial perspective.',
    dramatic_noir: 'Cinematic high-contrast chiaroscuro lighting, warm amber key light and cool rim light, rich atmospheric shadows.',
    soft_natural: 'Soft, even, diffused natural studio lighting across the face and body, clean and transparent skin tone.',
  };
  const accMap = {
    auto_face: '',
    no_glasses: 'No eyeglasses on the face, completely unobstructed eyes and eyebrows.',
    thin_glasses: 'Wearing refined thin metal-frame eyeglasses.',
    sunglasses: 'Wearing stylish dark sunglasses.',
    stubble_beard: 'With short, rugged dark facial stubble on the chin and jawline.',
    clean_shaven: 'Clean-shaven smooth jawline with no beard or mustache.',
  };

  const styleTail = `${frameMap[frameVal] || frameMap.keep_target_frame}. Facial expression: ${exprMap[exprVal] || exprMap.keep_orig}, ${gazeMap[gazeVal] || gazeMap.keep_target}. Body build: ${bodyMap[bodyVal] || bodyMap.standard}. ${accMap[accVal] || ''} ${extraTxt ? 'Additional requirements: ' + extraTxt + '.' : ''} ${lightMap[lightVal] || lightMap.keep_target_light} High-resolution real-life photograph, delicate skin texture.`;

  // 情形 A：人脸 + 1张服装/姿势目标图 (+ 可选场景图)
  // 将【服装/姿势目标图】放在 image_1 (<Picture 1> = 待编辑的空间与姿势底图)，将【目标人脸照】放在 image_2 (<Picture 2> = 脸型与五官身份源)！
  if (hasFace && targets.length === 1) {
    const poseOrOutfitB64 = targets[0];
    if (hasScene) {
      const refs = [poseOrOutfitB64, faceB64, sceneB64];
      const hairInstruction = hairVal === 'keep_target_hair'
        ? "Keep the hairstyle and headwear from <Picture 1>, replacing only the face shape, jawline contour, and facial features with <Subject 2> from <Picture 2>."
        : "Keep <Subject 2>'s exact slim face shape, narrow jawline contour, facial features, warm expression, and hairstyle from <Picture 2> 100% unchanged.";
      const prompt = (
        "Edit <Picture 1>. The person, body posture, hand gesture, and clothing outfit in <Picture 1> is <Subject 1>. " +
        "The person (exact slim V-shaped/oval face contour, narrow jawline, chin proportion, eyes, nose, lips, expression, and hairstyle) in <Picture 2> is <Subject 2>. " +
        "The background environment and lighting in <Picture 3> is <Subject 3>. " +
        `Replace the head and face of <Subject 1> in <Picture 1> with <Subject 2> from <Picture 2>, and place the figure naturally inside <Subject 3> (<Picture 3>). ${hairInstruction} ` +
        "Completely discard the original face shape or wider cheek contour of <Picture 1>, while 100% preserving the exact body posture, arm/hand gesture, and complete clothing outfit from <Picture 1>, relit naturally by <Picture 3>. " +
        styleTail
      );
      return {refs, prompt};
    }

    const refs = [poseOrOutfitB64, faceB64];
    if (hairVal === 'keep_target_hair') {
      const prompt = (
        "Edit <Picture 1>. The person sitting or standing in <Picture 1> is <Subject 1>. " +
        "The face (exact slim V-shaped/oval face contour, narrow jawline, chin proportion, cheekbones, eyes, nose, mouth, and natural expression) of the person in <Picture 2> is <Subject 2>. " +
        "Replace the face and facial contour of <Subject 1> in <Picture 1> with <Subject 2> from <Picture 2>. " +
        "Keep <Subject 2>'s exact slim face shape, narrow jawline contour, facial features, and natural expression from <Picture 2> 100% unchanged (do not inherit the wider face shape or jaw width from <Picture 1>), " +
        "while 100% preserving the exact body posture, arm and hand gesture, leg position, hairstyle, headwear, complete clothing outfit (both upper top and lower pants/jeans/skirt), camera framing, and clean studio/background environment from <Picture 1>. " +
        styleTail
      );
      return {refs, prompt};
    }

    const prompt = (
      "Edit <Picture 1>. The person sitting or standing in <Picture 1> is <Subject 1>. " +
      "The person (exact slim V-shaped/oval face contour, narrow jawline, chin proportion, facial features, natural expression, and hairstyle) in <Picture 2> is <Subject 2>. " +
      "Replace the head, face shape, facial features, and hairstyle of <Subject 1> in <Picture 1> with <Subject 2> from <Picture 2>. " +
      "Keep <Subject 2>'s exact slim V-shaped/oval face contour, narrow jawline, facial features, natural expression, and hairstyle from <Picture 2> 100% unchanged (completely discard the original face shape and hair of <Picture 1>), " +
      "while 100% preserving the exact seated/standing body posture, raised arm and hand gesture, leg position, complete clothing outfit (both upper garment and lower pants/jeans/skirt), camera framing, and background environment from <Picture 1>. " +
      styleTail
    );
    return {refs, prompt};
  }

  // 情形 B：多图合并为 1 张 (人脸 + 2张以上服装/配饰参考图)
  if (hasFace && targets.length >= 2) {
    const refs = [targets[0], faceB64, ...targets.slice(1), ...(hasScene ? [sceneB64] : [])];
    const extraItems = targets.slice(1).map((_, i) => `The additional clothing/styling item in <Picture ${i + 3}> is <Subject ${i + 3}>.`).join(' ');
    const extraNames = targets.slice(1).map((_, i) => `<Subject ${i + 3}>`).join(' and ');
    const sceneIdx = targets.length + 2;
    const sceneClause = hasScene ? ` Place the figure inside the background environment of <Picture ${sceneIdx}>.` : '';
    const prompt = (
      `Edit <Picture 1>. The base body posture and primary outfit in <Picture 1> is <Subject 1>. ` +
      `The person (exact slim face shape, jawline contour, facial features, expression, and hairstyle) in <Picture 2> is <Subject 2>. ${extraItems} ` +
      `Replace the head and face of <Subject 1> in <Picture 1> with <Subject 2> from <Picture 2>, wearing the combined outfit of <Subject 1> and ${extraNames}.${sceneClause} ` +
      `Keep <Subject 2>'s exact face shape, narrow jawline, facial features, and identity from <Picture 2> 100% unchanged, while preserving the body posture from <Picture 1>. ` +
      styleTail
    );
    return {refs, prompt};
  }

  // 情形 C：仅有人脸/人物图 + 场景图 (无新服装图)
  if (hasFace && !hasSlot2 && hasScene) {
    const refs = [faceB64, sceneB64];
    const prompt = (
      "Edit <Picture 1>. The person (exact slim face shape, jawline contour, facial features, hairstyle, and clothing) in <Picture 1> is <Subject 1>. " +
      "The real-world scene environment, architecture, and ambient lighting in <Picture 2> is <Subject 2>. " +
      "Place <Subject 1> from <Picture 1> naturally inside <Subject 2> (<Picture 2>) with realistic spatial perspective, grounded shadows, and ambient rim lighting matching <Picture 2>, while keeping <Subject 1>'s exact face shape, facial features, hairstyle, and identity from <Picture 1> 100% unchanged. " +
      styleTail
    );
    return {refs, prompt};
  }

  // 情形 D：仅上传了人脸/人物主照
  return {
    refs: [faceB64],
    prompt: (
      "Edit <Picture 1>. The person in <Picture 1> is <Subject 1>. " +
      "Keep <Subject 1>'s exact slim face shape, jawline contour, facial features, hairstyle, and identity from <Picture 1> 100% unchanged. " +
      styleTail
    ),
  };
}

function renderFusionBatchGallery() {
  const bar = document.getElementById('fusionBatchResultsBar');
  if (!bar) return;
  if (!fusionBatchCompletedFiles.length) {
    bar.style.display = 'none';
    return;
  }
  bar.style.display = 'block';
  bar.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
      <span style="font-size:12px;font-weight:800;color:#e9d5ff;">🖼️ 本次合成完成画廊 (共 ${fusionBatchCompletedFiles.length} 张 · 点击任意缩略图即可在下方大屏查看、设为角色定妆或送入数字人说话)：</span>
      <button type="button" onclick="fusionBatchCompletedFiles=[];renderFusionBatchGallery();" style="padding:1px 7px;font-size:10.5px;border-radius:4px;border:1px solid #475569;background:transparent;color:#94a3b8;cursor:pointer;">✕ 收起画廊</button>
    </div>
    <div style="display:flex;gap:10px;overflow-x:auto;padding-bottom:4px;">
      ${fusionBatchCompletedFiles.map((fname, i) => {
        const url = `/api/local_video_file?name=${encodeURIComponent(fname)}`;
        return `
          <div onclick="playHistoryVideo('${escapeHtmlAttr(fname)}')" style="cursor:pointer;padding:5px;border-radius:8px;background:#0f172a;border:1.5px solid #c084fc;display:flex;flex-direction:column;align-items:center;gap:4px;min-width:118px;max-width:140px;flex-shrink:0;" title="点击在下方大屏预览 ${escapeHtmlAttr(fname)}">
            <img src="${url}" style="width:108px;height:82px;object-fit:contain;border-radius:5px;background:#020617;">
            <span style="font-size:10.5px;font-weight:700;color:#f3e8ff;">✅ 成片 #${i + 1}</span>
          </div>
        `;
      }).join('')}
    </div>
  `;
}

async function stopFusionBatchGenerate() {
  fusionBatchCancelled = true;
  await stopActiveVideoTask('video');
  fusionBatchRunning = false;
  const btn = document.getElementById('btnRunFusionStudio');
  const stopBtn = document.getElementById('btnStopFusionStudio');
  const stEl = document.getElementById('fusionStudioStatusText');
  if (btn) btn.disabled = false;
  if (stopBtn) stopBtn.style.display = 'none';
  updateFusionRunBtnLabel();
  if (stEl) {
    stEl.innerText = '🛑 已停止当前合成队列（已生成的图片已保存在下方画廊与 ~/Movies/EasyColab/）';
    stEl.style.color = '#f59e0b';
  }
}

function waitForSingleFusionJobCompletion(jobIdx, totalJobs, prevLatestFile) {
  return new Promise(resolve => {
    const stEl = document.getElementById('fusionStudioStatusText');
    const fBar = document.getElementById('fusionProgressBar');
    const tLabel = document.getElementById('taskProgressLabel');
    const tSpeed = document.getElementById('taskProgressSpeed');
    const tBar = document.getElementById('taskProgressBar');
    let sawRunning = false;
    let pollTicks = 0;

    const timer = setInterval(async () => {
      pollTicks += 1;
      if (fusionBatchCancelled) {
        clearInterval(timer);
        resolve({ok: false, cancelled: true});
        return;
      }
      try {
        const r = await fetch('/api/video_status');
        const d = await r.json();
        await pollLogs();

        if (d.auto_switching) {
          sawRunning = true;
          const swPct = typeof d.pct === 'number' ? Math.max(8, Math.min(96, d.pct)) : 15;
          if (fBar) fBar.style.width = `${swPct}%`;
          if (stEl) {
            stEl.innerText = `🔄 [第 ${jobIdx}/${totalJobs} 张] ${d.message || '正在自动切换装载 MiniMax-H3-Image 图像引擎...'}`;
            stEl.style.color = '#f59e0b';
          }
          return;
        }

        if (d.running) {
          sawRunning = true;
          const jobPct = typeof d.pct === 'number' ? Math.max(10, Math.min(99, d.pct)) : 35;
          const overallPct = Math.round(((jobIdx - 1) * 100 + jobPct) / totalJobs);
          if (fBar) fBar.style.width = `${overallPct}%`;
          if (tBar) { tBar.className = 'progress-fill'; tBar.style.width = `${overallPct}%`; }
          if (tLabel) tLabel.innerText = `⏳ 角色合成总进度 (${jobIdx}/${totalJobs}张)：${overallPct}%`;
          if (tSpeed) tSpeed.innerText = `总进度 ${overallPct}%`;
          if (stEl) {
            stEl.innerText = `⏳ [正在合成第 ${jobIdx} / ${totalJobs} 张] ${d.message || '正在执行 33B 多图主语解耦采样...'} (总进度 ${overallPct}%)`;
            stEl.style.color = '#fbbf24';
          }
          return;
        }

        // d.running === false
        if (d.error) {
          clearInterval(timer);
          resolve({ok: false, error: d.error});
          return;
        }

        const hasNewFile = d.latest_video && d.latest_video !== prevLatestFile;
        if (sawRunning || hasNewFile || pollTicks >= 3) {
          clearInterval(timer);
          const doneFiles = Array.isArray(d.completed_videos) && d.completed_videos.length
            ? d.completed_videos
            : (d.latest_video ? [d.latest_video] : []);
          resolve({ok: true, files: doneFiles, latest: d.latest_video || ''});
        }
      } catch (e) {}
    }, 1500);
  });
}

async function startFusionStudioGenerate() {
  if (fusionBatchRunning) return;
  const btn = document.getElementById('btnRunFusionStudio');
  const stopBtn = document.getElementById('btnStopFusionStudio');
  const stEl = document.getElementById('fusionStudioStatusText');
  const fWrap = document.getElementById('fusionProgressWrap');
  const fBar = document.getElementById('fusionProgressBar');

  if (!fusionSlotsB64[0]) {
    alert('请先在左侧「第一步」上传目标人脸/人物照片！');
    return;
  }
  const activeTargets = fusionTargetList.filter(Boolean);
  if (currentFusionSubMode === 'faceswap' && activeTargets.length === 0) {
    alert('在「人脸替换他人照片 (换脸)」模式下，请在右侧「第二步」上传至少 1 张被替换的目标照片！');
    return;
  }
  if (currentFusionSubMode === 'scenefuse' && !fusionSlotsB64[2] && activeTargets.length === 0) {
    alert('在「融入场景照片」模式下，请上传场景照片或服装照片！');
    return;
  }

  const multiMode = ((document.getElementById('fusionMultiActionMode') || {}).value || 'batch');
  const resChoice = ((document.getElementById('fusionResSelect') || {}).value || 'auto');
  const batchCnt = parseInt(((document.getElementById('fusionBatchSelect') || {}).value || '1'), 10) || 1;

  // 构建任务队列：
  // - 若选了“逐张批量生成 (batch)”且上传了 N 张目标图 -> 拆成 N 个独立合成任务依次执行
  // - 若选了“多图合并为1张 (combine)”或只传了 0~1 张目标图 -> 作为 1 个任务执行
  let jobTargetLists = [];
  if (activeTargets.length >= 2 && multiMode === 'batch') {
    jobTargetLists = activeTargets.map(t => [t]);
  } else if (activeTargets.length >= 1) {
    jobTargetLists = [activeTargets];
  } else {
    jobTargetLists = [[]];
  }

  if (videoPollTimer) {
    clearInterval(videoPollTimer);
    videoPollTimer = null;
  }
  fusionBatchRunning = true;
  fusionBatchCancelled = false;
  fusionBatchCompletedFiles = [];
  renderFusionBatchGallery();

  if (btn) {
    btn.disabled = true;
    btn.innerText = `⏳ 正在队列合成 (共 ${jobTargetLists.length} 组)...`;
  }
  if (stopBtn) stopBtn.style.display = 'inline-block';
  if (fWrap) fWrap.style.display = 'block';
  if (fBar) fBar.style.width = '8%';

  try {
    let prevLatest = '';
    try {
      const r0 = await fetch('/api/video_status');
      const d0 = await r0.json();
      prevLatest = d0.latest_video || '';
    } catch (e) {}

    for (let i = 0; i < jobTargetLists.length; i++) {
      if (fusionBatchCancelled) break;
      const targetsForJob = jobTargetLists[i];
      const refForAspect = (currentFusionSubMode === 'scenefuse' && fusionSlotsB64[2])
        ? fusionSlotsB64[2]
        : (targetsForJob[0] || fusionSlotsB64[0]);
      const actualRes = (resChoice === 'auto')
        ? await detectImageAspectRes(refForAspect)
        : resChoice;
      const [wStr, hStr] = actualRes.split('x');

      const built = buildFusionRefsAndPrompt(targetsForJob);
      const refsPayload = built.refs;
      const promptForJob = built.prompt;
      const payload = {
        mode: 'image',
        prompt: promptForJob,
        resolution: actualRes,
        width: parseInt(wStr, 10) || 1152,
        height: parseInt(hStr, 10) || 2048,
        steps: 6,
        seed: -1,
        randomize_seed: true,
        batch_count: batchCnt,
        ref_images_b64: refsPayload,
        edit_outfit: '',
        edit_scene: '',
        clean_matte: true,
      };

      if (stEl) {
        stEl.innerText = `⏳ [第 ${i + 1} / ${jobTargetLists.length} 张] 已提交至 MiniMax-H3-Image (${actualRes})，正在合成...`;
        stEl.style.color = '#fbbf24';
      }

      const r = await fetch('/api/video_generate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      });
      const d = await r.json();
      if (!d.ok) {
        if (stEl) {
          stEl.innerText = `❌ 第 ${i + 1} 张启动失败: ` + (d.error || '未知错误');
          stEl.style.color = '#ef4444';
        }
        break;
      }
      if (d.auto_switching) {
        selectPresetModel('preset_minimax_image', 'Comfy-Org/MiniMax-H3-Image', 'MiniMax-H3-Image', '*.safetensors', true);
        if (stEl && d.message) {
          stEl.innerText = `🔄 [第 ${i + 1} / ${jobTargetLists.length} 张] ` + d.message;
        }
      }

      const waitRes = await waitForSingleFusionJobCompletion(i + 1, jobTargetLists.length, prevLatest);
      if (waitRes.cancelled) break;
      if (!waitRes.ok) {
        if (stEl) {
          stEl.innerText = `❌ 第 ${i + 1} 张合成出错: ` + (waitRes.error || '未知错误');
          stEl.style.color = '#ef4444';
        }
        break;
      }
      if (waitRes.latest) prevLatest = waitRes.latest;
      if (Array.isArray(waitRes.files)) {
        waitRes.files.forEach(fn => {
          if (fn && !fusionBatchCompletedFiles.includes(fn)) {
            fusionBatchCompletedFiles.push(fn);
          }
        });
      }
      renderFusionBatchGallery();
      await loadLocalVideoHistory(true);
    }

    if (!fusionBatchCancelled && fusionBatchCompletedFiles.length > 0) {
      if (fBar) fBar.style.width = '100%';
      const tBar = document.getElementById('taskProgressBar');
      const tLabel = document.getElementById('taskProgressLabel');
      const tSpeed = document.getElementById('taskProgressSpeed');
      if (tBar) { tBar.className = 'progress-fill green'; tBar.style.width = '100%'; }
      if (tLabel) tLabel.innerText = `✅ 角色合成完成 (共生成 ${fusionBatchCompletedFiles.length} 张)：100%`;
      if (tSpeed) tSpeed.innerText = '总进度 100%';
      if (stEl) {
        stEl.innerText = `✅ 全部合成完成（共生成 ${fusionBatchCompletedFiles.length} 张）！可点击下方画廊任意缩略图预览、设为短剧角色定妆或送入数字人开口说话！`;
        stEl.style.color = '#34d399';
      }
    }
  } catch (e) {
    if (stEl) {
      stEl.innerText = '❌ 合成请求异常: ' + e;
      stEl.style.color = '#ef4444';
    }
  } finally {
    fusionBatchRunning = false;
    if (btn) btn.disabled = false;
    if (stopBtn) stopBtn.style.display = 'none';
    updateFusionRunBtnLabel();
  }
}

async function sendCurrentPreviewToDramaRole() {
  const roleSel = document.getElementById('selFusionTargetRole');
  const roleId = (roleSel && roleSel.value) ? roleSel.value : 'A';
  const prevImg = document.getElementById('mmImagePreview');
  if (!currentSelectedVideoName && (!prevImg || prevImg.style.display === 'none' || !prevImg.src)) {
    alert('请先在上方生成或从右侧「本地历史」点选一张图片！');
    return;
  }
  try {
    const imgSrc = (prevImg && prevImg.src) ? prevImg.src : `/api/local_video_file?name=${encodeURIComponent(currentSelectedVideoName)}`;
    const resp = await fetch(imgSrc);
    const blob = await resp.blob();
    const reader = new FileReader();
    reader.onload = async ev => {
      const b64 = ev.target.result;
      const r2 = await fetch('/api/drama_upload_char_photo', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({role_id: roleId, image_b64: b64}),
      });
      const d2 = await r2.json();
      if (!d2.ok) {
        alert('锁入角色定妆卡槽失败: ' + (d2.error || '未知错误'));
        return;
      }
      dramaCastPhotoCacheBust = Date.now();
      let item = dramaCastState.find(c => c.id === roleId);
      if (!item) {
        item = {id: roleId, name: `角色${roleId}`, voice: roleId === 'A' ? 'ming:灵小柔' : 'ming:灵信远_中性', desc: '', user_uploaded: true};
        dramaCastState.push(item);
      } else {
        item.user_uploaded = true;
      }
      syncCastToWorkflow();
      renderDramaCastCards();
      showGlobalCopyToast(`✅ 已将当前预览大图锁定为「短剧角色${roleId} (${item.name || ''})」的 16:9 定妆照！`);
    };
    reader.readAsDataURL(blob);
  } catch (e) {
    alert('锁入短剧角色异常: ' + e);
  }
}

function handleH3RefImageSelect(slotIdx, inputEl) {
  const file = inputEl && inputEl.files && inputEl.files[0];
  const prevEl = document.getElementById(`h3ImgRef${slotIdx + 1}Preview`);
  const stEl = document.getElementById('h3ImgRefStatus');
  if (!file) {
    h3ImgRefList[slotIdx] = '';
    if (prevEl) prevEl.style.display = 'none';
    return;
  }
  const reader = new FileReader();
  reader.onload = function(ev) {
    h3ImgRefList[slotIdx] = ev.target.result;
    if (prevEl) {
      prevEl.src = ev.target.result;
      prevEl.style.display = 'block';
    }
    const cnt = h3ImgRefList.filter(Boolean).length;
    if (stEl) {
      stEl.innerText = `👗 已载入 ${cnt} 张参考图 (<Picture 1>..${cnt}) · 已激活「免涂 Mask 像素级锁脸换装/多图融合」模式！`;
    }
    syncH3EditPreviewPrompt();
  };
  reader.readAsDataURL(file);
}

function clearAllH3RefImages() {
  h3ImgRefList = ['', '', ''];
  for (let i = 1; i <= 3; i++) {
    const inp = document.getElementById(`inpH3ImgRef${i}`);
    const img = document.getElementById(`h3ImgRef${i}Preview`);
    if (inp) inp.value = '';
    if (img) { img.src = ''; img.style.display = 'none'; }
  }
  const stEl = document.getElementById('h3ImgRefStatus');
  if (stEl) stEl.innerText = '✨ 当前为「纯文生 4K 超清写真」模式（上传 <Picture 1> 即自动切换为免涂 Mask 锁脸换装）';
}

function syncH3EditPreviewPrompt() {
  const outfit = ((document.getElementById('inpH3EditOutfit') || {}).value || '').trim();
  const scene = ((document.getElementById('inpH3EditScene') || {}).value || '').trim();
  if (!outfit && !scene) return;
  const pEl = document.getElementById('videoPrompt');
  if (!pEl) return;
  const tgtOutfit = outfit || 'a tailored black silk modern qipao with gold phoenix embroidery';
  if (scene) {
    pEl.value = `The young woman in <Picture 1> is <Subject 1>. The original clothing worn by <Subject 1> in <Picture 1> is <Subject 2>. The target image shows <Subject 1> wearing ${tgtOutfit} in ${scene}, captured as a high-resolution real-life portrait photograph with clean natural lighting. <Subject 1> is partially preserved; her exact East Asian facial features, oval face shape, smooth fair skin, serene gaze, and hairstyle remain completely unchanged, while <Subject 2> is replaced by ${tgtOutfit}, and the surrounding environment is ${scene}.`;
  } else {
    pEl.value = `The young woman in <Picture 1> is <Subject 1>. The original clothing worn by <Subject 1> in <Picture 1> is <Subject 2>. The background and lighting environment in <Picture 1> is <Subject 3>. The target image shows <Subject 1> wearing ${tgtOutfit} in <Subject 3>. <Subject 1> is partially preserved; her exact East Asian facial features, oval face shape, smooth fair skin, serene gaze, and hairstyle remain completely unchanged, while <Subject 2> is replaced by ${tgtOutfit}. <Subject 3> is fully preserved.`;
  }
}

function applyH3ImagePresetTemplate(tplKey) {
  if (!tplKey) return;
  const pEl = document.getElementById('videoPrompt');
  const outEl = document.getElementById('inpH3EditOutfit');
  const scEl = document.getElementById('inpH3EditScene');
  const resEl = document.getElementById('videoRes');
  const templates = {
    edit_qipao: {
      outfit: 'a luxurious black-and-gold phoenix embroidered royal silk qipao with a high mandarin collar and imperial emerald jadeite teardrop earrings',
      scene: '',
      prompt: 'The young Chinese woman in <Picture 1> is <Subject 1>. The original clothing worn by <Subject 1> in <Picture 1> is <Subject 2>. The background in <Picture 1> is <Subject 3>. The target image shows <Subject 1> wearing a luxurious black-and-gold phoenix embroidered royal silk qipao and imperial emerald jadeite earrings in <Subject 3>. <Subject 1> is partially preserved; her exact Chinese East Asian facial features, oval face, smooth fair skin, serene gaze, and hairstyle remain completely unchanged, while <Subject 2> is replaced by a tailored black silk modern qipao with a high mandarin collar and gold phoenix embroidery across the chest, plus a pair of glowing imperial green jadeite teardrop earrings. <Subject 3> is fully preserved.'
    },
    edit_silk: {
      outfit: 'a deep burgundy silk slip robe gracefully sliding off one shoulder with delicate black Chantilly lace trim',
      scene: '',
      prompt: 'The young woman in <Picture 1> is <Subject 1>. The original clothing worn by <Subject 1> in <Picture 1> is <Subject 2>. The background in <Picture 1> is <Subject 3>. The target image shows <Subject 1> wearing a deep burgundy pure silk slip robe with delicate black Chantilly lace trim along the neckline in <Subject 3>. <Subject 1> is partially preserved; her exact East Asian facial features, face shape, smooth fair skin, and hairstyle remain completely unchanged, while <Subject 2> is replaced by a burgundy silk slip robe with subtle black lace trim and natural silk drape folds. <Subject 3> is fully preserved.'
    },
    edit_lace: {
      outfit: 'an elegant pure white floral hollow-out lace camisole dress with a delicate pearl collarbone necklace',
      scene: '',
      prompt: 'The young woman in <Picture 1> is <Subject 1>. The original clothing worn by <Subject 1> in <Picture 1> is <Subject 2>. The background in <Picture 1> is <Subject 3>. The target image shows <Subject 1> wearing an elegant pure white floral lace camisole dress and a delicate pearl necklace in <Subject 3>. <Subject 1> is partially preserved; her exact East Asian facial features, face shape, smooth fair skin, and hairstyle remain completely unchanged, while <Subject 2> is replaced by a white floral hollow-out lace camisole dress with soft translucent lace straps. <Subject 3> is fully preserved.'
    },
    edit_ol: {
      outfit: 'a sleek fitted black velvet off-shoulder evening gown with a sparkling diamond choker necklace',
      scene: '',
      prompt: 'The young woman in <Picture 1> is <Subject 1>. The original clothing worn by <Subject 1> in <Picture 1> is <Subject 2>. The background in <Picture 1> is <Subject 3>. The target image shows <Subject 1> wearing a sleek fitted black velvet off-shoulder evening gown and a diamond choker in <Subject 3>. <Subject 1> is partially preserved; her exact East Asian facial features, face shape, smooth fair skin, and hairstyle remain completely unchanged, while <Subject 2> is replaced by a tailored black velvet off-shoulder evening dress and a diamond choker. <Subject 3> is fully preserved.'
    },
    t2i_courtyard: {
      res: '3552x4736',
      prompt: 'A high-resolution, real-life portrait photograph of an elegant 20-year-old Chinese young woman with a classic East Asian oval face, smooth fair skin, expressive dark brown almond eyes, and a serene, natural expression, standing beneath the carved wooden eaves of a traditional Suzhou courtyard at dusk. Her long, sleek black hair is styled in a graceful half-up bun secured with a carved white jade hairpin. She wears a pale moonlight-white silk Chinese stand-collar jacket with subtle woven floral patterns and holds a warm glowing bamboo paper lantern near her chest. Behind her, soft out-of-focus red lanterns and grey tiled roofs glow warmly in the evening twilight, captured on a Canon EOS R5 with an 85mm f/1.4 lens, clean natural lighting, and shallow depth of field.'
    },
    t2i_boudoir: {
      res: '3360x5024',
      prompt: 'A high-resolution, real-life portrait photograph of a graceful 21-year-old Chinese young woman with a delicate East Asian face, smooth fair skin, soft expressive eyes, and wavy dark hair resting over her shoulder, seated in a sunlit room beside wooden Venetian blinds in late afternoon. She wears a burgundy pure silk slip robe trimmed with fine black Chantilly lace that drapes naturally across her collarbone. Warm golden sunlight filters through the blinds, casting soft natural stripes of light and warm diffuse reflections across her smooth skin and silk fabric, captured on a Hasselblad X2D 100C with a 90mm f/2.5 lens, clean natural lighting, shallow depth of field.'
    },
    t2i_white_lace: {
      res: '3552x4736',
      prompt: 'A high-resolution, real-life close-up portrait photograph of a sweet 20-year-old East Asian young woman with a soft oval face, smooth fair porcelain skin, bright clear dark eyes, and a gentle smile, with shoulder-length layered black hair, wearing a pure white floral lace camisole top with delicate straps in front of a softly lit studio backdrop. Clean natural window lighting illuminates her face and collarbone with soft, natural optical depth, captured on a Canon EOS R5 with an 85mm f/1.4 lens.'
    },
    t2i_chongqing: {
      res: '5472x3072',
      prompt: 'A high-resolution, real-life documentary street photograph taken at Chongqing Hongyadong at night in light rain, captured on a Sony A7R V with a 35mm f/1.4 lens. In the foreground, a beautiful young Chinese woman with smooth fair skin and long dark hair holds a clear transparent umbrella, looking back over her shoulder with a gentle smile. Behind her, traditional wooden stilt buildings glow with warm lanterns and crisp Chinese neon signs reading "重庆老火锅", "洪崖洞码头", and a street stall sign reading "正宗重庆小面 24H" with steaming bowls and warm reflections on the wet stone pavement.'
    },
  };
  const t = templates[tplKey];
  if (!t) return;
  if (outEl) outEl.value = t.outfit || '';
  if (scEl) scEl.value = t.scene || '';
  if (t.res && resEl) resEl.value = t.res;
  if (pEl && t.prompt) pEl.value = t.prompt;
}

let mingCloneAudioB64 = '';

function getCombinedAvatarTtsSpec() {
  const vRaw = ((document.getElementById('avatarTtsVoiceSelect') || {}).value || 'ming:灵雨汐_中性').trim();
  if (vRaw.startsWith('ming:')) {
    return vRaw;
  }
  const sRaw = ((document.getElementById('avatarTtsStyleSelect') || {}).value || '').trim();
  if (sRaw && !vRaw.includes('|')) {
    return `${vRaw}|${sRaw}`;
  }
  return vRaw;
}

function insertMingParalinguisticTag(tag) {
  const el = document.getElementById('inpAvatarTtsText');
  if (!el) return;
  const start = typeof el.selectionStart === 'number' ? el.selectionStart : el.value.length;
  const end = typeof el.selectionEnd === 'number' ? el.selectionEnd : el.value.length;
  const before = el.value.slice(0, start);
  const after = el.value.slice(end);
  const padL = (before && !before.endsWith(' ')) ? ' ' : '';
  const padR = (after && !after.startsWith(' ')) ? ' ' : '';
  el.value = `${before}${padL}${tag}${padR}${after}`;
  const newPos = (before + padL + tag + padR).length;
  el.focus();
  el.setSelectionRange(newPos, newPos);
  // 若当前选中的是微软 Edge-TTS，自动切换到 Ming-3.0 音色以发挥副语言气口表现力
  const vSel = document.getElementById('avatarTtsVoiceSelect');
  if (vSel && !String(vSel.value || '').startsWith('ming:')) {
    vSel.value = 'ming:灵雨汐_中性';
  }
  onAvatarTtsInput(el);
}

function handleMingCloneAudioSelect(inputEl) {
  const file = inputEl && inputEl.files && inputEl.files[0];
  const stEl = document.getElementById('avatarAudioStatus');
  if (!file) {
    mingCloneAudioB64 = '';
    return;
  }
  const reader = new FileReader();
  reader.onload = function(ev) {
    mingCloneAudioB64 = ev.target.result || '';
    avatarAudioB64 = '';
    avatarUseBuiltinSample = false;
    const vSel = document.getElementById('avatarTtsVoiceSelect');
    if (vSel) vSel.value = 'ming:__clone__';
    if (stEl) {
      stEl.innerText = `🧬 已载入声音克隆样本「${file.name}」！输入台词后点「🔊 合成并试听」或「🎬 生成数字人视频」即可复刻该音色`;
    }
  };
  reader.readAsDataURL(file);
}

function clearMingCloneAudio() {
  mingCloneAudioB64 = '';
  const inp = document.getElementById('inpMingCloneAudio');
  const pTxt = document.getElementById('inpMingClonePromptText');
  const vSel = document.getElementById('avatarTtsVoiceSelect');
  const stEl = document.getElementById('avatarAudioStatus');
  if (inp) inp.value = '';
  if (pTxt) pTxt.value = '';
  if (vSel && vSel.value === 'ming:__clone__') vSel.value = 'ming:灵雨汐_中性';
  if (stEl) stEl.innerText = '✅ 已清除克隆样本，已切回 Ming-3.0 内置戏剧人设音色';
}

function onAvatarVoiceOrStyleChange() {
  avatarAudioB64 = '';
  const vSel = document.getElementById('avatarTtsVoiceSelect');
  const sSel = document.getElementById('avatarTtsStyleSelect');
  const isMing = vSel && String(vSel.value || '').startsWith('ming:');
  if (sSel) {
    sSel.style.display = isMing ? 'none' : 'inline-block';
  }
  const stEl = document.getElementById('avatarAudioStatus');
  if (stEl) {
    stEl.innerText = isMing
      ? '🎭 已切换至 Ming-3.0 百灵戏剧/克隆音色 (支持呼吸/笑声/哼声标签)！可点「🔊 合成并试听」或直接点下方生成视频'
      : '🎙️ 已切换至微软 Edge-TTS 极速音色！可点「🔊 合成并试听」试听新音色，或直接点下方开始生成';
  }
}

function onAvatarTtsInput(el) {
  const txt = (el ? el.value : '').trim();
  const stEl = document.getElementById('avatarAudioStatus');
  if (txt) {
    avatarAudioB64 = '';
    avatarUseBuiltinSample = false;
    const audInp = document.getElementById('inpAvatarAudio');
    if (audInp) audInp.value = '';
    if (stEl) {
      stEl.innerText = `💬 已输入配音台词 (${txt.length} 字) · 可点「🔊 合成并试听」先听声音，或直接点下方开始生成`;
    }
  }
}

function autoSplitAvatarPromptAndTts(promptEl) {
  if (currentMiniMaxMode !== 'avatar') return;
  const pEl = promptEl || document.getElementById('videoPrompt');
  const ttsEl = document.getElementById('inpAvatarTtsText');
  const stEl = document.getElementById('avatarAudioStatus');
  if (!pEl || !ttsEl) return;
  const raw = pEl.value || '';
  const m = raw.match(/^(?:([\s\S]*?)\n+)?\s*(?:台词如下|说话内容如下|对白如下|台词内容|开口说话|说话内容|台词|对白)\s*[:：]\s*([\s\S]+)$/);
  if (m && m[2] && m[2].trim()) {
    const visPart = (m[1] || '').trim() || '固定镜头，近景居中构图。画面中的人物直视镜头自然说话，眼神温柔灵动，伴随说话节奏有轻微自然的眨眼、轻轻点头和呼吸起伏，嘴型与语音高度同步，面部柔和暖光，五官清晰稳定，过渡丝滑真实。';
    const spokenPart = m[2].trim();
    pEl.value = visPart;
    ttsEl.value = spokenPart;
    avatarAudioB64 = '';
    avatarUseBuiltinSample = false;
    if (stEl) {
      stEl.innerText = `✨ 已自动将「台词 (${spokenPart.length}字)」分离至上方黄色配音框！画面描述已保留在第 3 框`;
    }
  }
}

async function previewAvatarTts() {
  const ttsEl = document.getElementById('inpAvatarTtsText');
  const voiceEl = document.getElementById('avatarTtsVoiceSelect');
  const btn = document.getElementById('btnPreviewTts');
  const audEl = document.getElementById('avatarAudioPreview');
  const stEl = document.getElementById('avatarAudioStatus');
  autoSplitAvatarPromptAndTts(document.getElementById('videoPrompt'));
  const text = (ttsEl ? ttsEl.value : '').trim();
  const voice = getCombinedAvatarTtsSpec();
  if (!text) {
    alert('请先在黄色「💬 台词配音框」内粘贴或输入要说的台词文字');
    if (ttsEl) ttsEl.focus();
    return;
  }
  if (btn) {
    btn.disabled = true;
    btn.innerText = '⏳ 合成中...';
  }
  if (stEl) stEl.innerText = `⏳ 正在使用 ${voice} 合成高拟真语音...`;
  const clonePromptText = ((document.getElementById('inpMingClonePromptText') || {}).value || '').trim();
  try {
    const r = await fetch('/api/tts_preview', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        text,
        voice,
        clone_ref_b64: mingCloneAudioB64,
        clone_prompt_text: clonePromptText,
      }),
    });
    const d = await r.json();
    if (!d.ok || !d.audio_b64) {
      if (stEl) stEl.innerText = '❌ 语音合成失败: ' + (d.error || '未知错误');
      return;
    }
    avatarAudioB64 = d.used_edge_fallback ? '' : d.audio_b64;
    avatarAudioExt = d.audio_ext || '.mp3';
    avatarAudioIsTts = !d.used_edge_fallback;
    avatarUseBuiltinSample = false;
    if (audEl) {
      audEl.src = d.audio_b64;
      audEl.style.display = 'block';
      audEl.play().catch(() => {});
    }
    const durInfo = d.duration_sec ? ` · 时长 ${d.duration_sec} 秒 (自动匹配 ${d.auto_segments || 1} 段)` : '';
    if (stEl) {
      stEl.innerText = d.note
        ? `🎵 ${d.note}${durInfo}`
        : `🎵 台词语音已就绪${durInfo}！可直接点下方「🎬 生成数字人视频」`;
    }
  } catch (e) {
    if (stEl) stEl.innerText = '❌ 语音合成请求异常: ' + e;
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = '🔊 合成并试听';
    }
  }
}

function handleAvatarAudioSelect(inputEl) {
  const file = inputEl.files && inputEl.files[0];
  const audEl = document.getElementById('avatarAudioPreview');
  const stEl = document.getElementById('avatarAudioStatus');
  if (!file) {
    avatarAudioB64 = '';
    if (audEl) audEl.style.display = 'none';
    if (stEl) stEl.innerText = '✅ 默认已启用内置示例音频 (man.mp3)';
    return;
  }
  const mExt = file.name.match(/\.[a-zA-Z0-9]+$/);
  avatarAudioExt = mExt ? mExt[0].toLowerCase() : '.mp3';
  avatarAudioIsTts = false;
  avatarUseBuiltinSample = false;
  const reader = new FileReader();
  reader.onload = function(ev) {
    avatarAudioB64 = ev.target.result;
    if (audEl) {
      audEl.src = avatarAudioB64;
      audEl.style.display = 'block';
    }
    if (stEl) stEl.innerText = `🎵 已载入驱动音频: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
  };
  reader.readAsDataURL(file);
}

function useBuiltinAvatarSample() {
  avatarAudioB64 = '';
  avatarUseBuiltinSample = true;
  firstFrameB64 = '';
  const audInp = document.getElementById('inpAvatarAudio');
  const audEl = document.getElementById('avatarAudioPreview');
  const imgInp = document.getElementById('inpAvatarImage');
  const imgEl = document.getElementById('avatarImagePreview');
  const stEl = document.getElementById('avatarAudioStatus');
  const hintEl = document.getElementById('avatarImageHint');
  if (audInp) audInp.value = '';
  if (audEl) audEl.style.display = 'none';
  if (imgInp) imgInp.value = '';
  if (imgEl) imgEl.style.display = 'none';
  if (stEl) stEl.innerText = '✅ 已载入官方内置示例人声 (man.mp3) + 示例舞台歌手人像 (man.png)！直接点击下方「🎬 开始生成音视频」即可测试';
  if (hintEl) hintEl.innerText = '✅ 当前将使用官方内置示例舞台歌手照片 (man.png · AI2V 模式)；您也可以随时上传自己的照片替换。';
}

function handleFrameFileSelect(inputEl, previewId) {
  const file = inputEl.files && inputEl.files[0];
  const imgEl = document.getElementById(previewId);
  if (!file) {
    if (previewId === 'firstFramePreview' || previewId === 'avatarImagePreview') firstFrameB64 = '';
    if (previewId === 'lastFramePreview') lastFrameB64 = '';
    if (imgEl) imgEl.style.display = 'none';
    return;
  }
  const reader = new FileReader();
  reader.onload = function(ev) {
    const dataUrl = ev.target.result;
    if (previewId === 'firstFramePreview' || previewId === 'avatarImagePreview') {
      firstFrameB64 = dataUrl;
      avatarUseBuiltinSample = false;
    }
    if (previewId === 'lastFramePreview') lastFrameB64 = dataUrl;
    if (imgEl) {
      imgEl.src = dataUrl;
      imgEl.style.display = 'block';
    }
  };
  reader.readAsDataURL(file);
}

function clearRefFrame(which) {
  if (which === 'first') {
    firstFrameB64 = '';
    avatarUseBuiltinSample = false;
    const inp = document.getElementById('inpFirstFrame');
    const img = document.getElementById('firstFramePreview');
    const avInp = document.getElementById('inpAvatarImage');
    const avImg = document.getElementById('avatarImagePreview');
    const hintEl = document.getElementById('avatarImageHint');
    if (inp) inp.value = '';
    if (img) { img.src = ''; img.style.display = 'none'; }
    if (avInp) avInp.value = '';
    if (avImg) { avImg.src = ''; avImg.style.display = 'none'; }
    if (hintEl) hintEl.innerText = '✨ 已切换为纯「文+音生数字人 (AT2V)」模式（不传照片，直接根据提示词和音频生成说话人物）。';
  } else {
    lastFrameB64 = '';
    const inp = document.getElementById('inpLastFrame');
    const img = document.getElementById('lastFramePreview');
    if (inp) inp.value = '';
    if (img) { img.src = ''; img.style.display = 'none'; }
  }
}

async function continueFromCurrentVideoLastFrame() {
  if (!currentSelectedVideoName) {
    alert('请先在右侧「本地历史」中选择一条已生成的视频');
    return;
  }
  const st = document.getElementById('videoGenStatus');
  if (st) {
    st.innerText = `⏳ 正在无损提取 ${currentSelectedVideoName} 的最后一帧画面...`;
    st.style.color = '#60a5fa';
  }
  try {
    const r = await fetch('/api/extract_last_frame', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: currentSelectedVideoName}),
    });
    const d = await r.json();
    if (!d.ok || !d.data_url) {
      alert('提取尾帧失败: ' + (d.error || '未知错误'));
      return;
    }
    firstFrameB64 = d.data_url;
    const imgEl = document.getElementById('firstFramePreview');
    if (imgEl) {
      imgEl.src = d.data_url;
      imgEl.style.display = 'block';
    }
    clearRefFrame('last');
    switchMiniMaxMode('fl2va');
    const flRow = document.getElementById('fl2vaImageRow');
    if (flRow) flRow.scrollIntoView({behavior: 'smooth', block: 'center'});
    if (st) {
      st.innerText = `🔗 已将「${currentSelectedVideoName}」最后一帧设为首帧参考图！可修改下方提示词后点击「🎬 开始生成音视频」续写下一段 15~20s！`;
      st.style.color = '#10b981';
    }
  } catch (e) {
    alert('提取尾帧请求失败: ' + e);
  }
}

let localHistoryItems = [];
let currentSelectedVideoName = '';
let autoUpscaleAfterImport = false;

function triggerImportExternalVideo(autoUpscale = false) {
  autoUpscaleAfterImport = Boolean(autoUpscale);
  const inp = document.getElementById('inpExternalVideo');
  if (inp) {
    inp.value = '';
    inp.click();
  }
}

async function handleExternalVideoImport(inputEl) {
  const file = inputEl && inputEl.files && inputEl.files[0];
  if (!file) return;
  if (file.size > 120 * 1024 * 1024) {
    alert('导入的视频文件过大（建议小于 120MB），请直接将该文件拖入 ~/Movies/EasyColab/ 文件夹中再点刷新');
    return;
  }
  const st = document.getElementById('videoGenStatus');
  const tLabel = document.getElementById('taskProgressLabel');
  if (st) {
    st.innerText = `⏳ 正在导入外部视频「${file.name}」...`;
    st.style.color = '#60a5fa';
  }
  if (tLabel) tLabel.innerText = `⏳ 正在导入外部视频「${file.name}」至 ~/Movies/EasyColab/...`;
  const reader = new FileReader();
  reader.onload = async function(ev) {
    try {
      const r = await fetch('/api/import_external_video', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({filename: file.name, video_b64: ev.target.result}),
      });
      const d = await r.json();
      await pollLogs();
      if (!d.ok || !d.filename) {
        alert('导入外部视频失败: ' + (d.error || '未知错误'));
        return;
      }
      await loadLocalVideoHistory(false);
      const idx = localHistoryItems.findIndex(x => x.name === d.filename);
      if (idx >= 0) selectLocalHistoryItem(idx);
      else playHistoryVideo(d.filename);
      if (autoUpscaleAfterImport) {
        const presetVal = (document.getElementById('upscalePresetSelect') || {}).value || '2k';
        upscaleVideoByName(d.filename, presetVal);
      } else if (st) {
        st.innerText = `✅ 已成功导入外部视频「${d.filename}」！点击下方「✨ ReCaVSR 超分」即可放大至 2K/4K，或点「🔗 提取尾帧续写」`;
        st.style.color = '#10b981';
      }
    } catch (e) {
      alert('导入外部视频请求异常: ' + e);
    }
  };
  reader.readAsDataURL(file);
}

async function loadLocalVideoHistory(autoSelectFirst = false) {
  const hBox = document.getElementById('mmHistoryList');
  const cntEl = document.getElementById('localHistoryCount');
  try {
    const r = await fetch('/api/local_videos');
    const d = await r.json();
    localHistoryItems = (d && Array.isArray(d.items)) ? d.items : [];
    if (cntEl) cntEl.innerText = `(${localHistoryItems.length})`;
    renderLocalHistoryList();
    if (autoSelectFirst && localHistoryItems.length > 0) {
      selectLocalHistoryItem(0);
    } else if (currentSelectedVideoName && !localHistoryItems.some(x => x.name === currentSelectedVideoName)) {
      if (localHistoryItems.length > 0) {
        selectLocalHistoryItem(0);
      } else {
        currentSelectedVideoName = '';
        const player = document.getElementById('mmVideoPlayer');
        const dl = document.getElementById('mmVideoDownload');
        const emptyEl = document.getElementById('videoEmptyPlaceholder');
        const rWrap = document.getElementById('videoResultWrap');
        const topBadge = document.getElementById('previewElapsedBadge');
        const botBadge = document.getElementById('previewBottomShortBadge');
        const metaEl = document.getElementById('mmVideoMeta');
        if (player) { player.pause(); player.removeAttribute('src'); player.removeAttribute('data-fname'); }
        if (rWrap) rWrap.style.display = 'none';
        if (emptyEl) emptyEl.style.display = 'flex';
        if (dl) dl.style.display = 'none';
        if (topBadge) topBadge.style.display = 'none';
        if (botBadge) botBadge.style.display = 'none';
        if (metaEl) { metaEl.innerText = '尚未选择视频'; metaEl.title = ''; }
      }
    }
  } catch (e) {
    if (hBox) hBox.innerHTML = `<div style="font-size:12px;color:var(--muted);">读取本地历史失败: ${e}</div>`;
  }
}

function renderLocalHistoryList() {
  const hBox = document.getElementById('mmHistoryList');
  if (!hBox) return;
  if (!localHistoryItems.length) {
    hBox.innerHTML = `<div style="font-size:12px;color:var(--muted);padding:24px 0;text-align:center;">暂无本地视频记录</div>`;
    return;
  }
  hBox.innerHTML = localHistoryItems.map((it, idx) => {
    const active = it.name === currentSelectedVideoName;
    const isImgFile = /\.(png|jpe?g|webp)$/i.test(it.name);
    const isSr = it.name.includes('_ReCaVSR_');
    const rowBg = active ? 'rgba(59,130,246,0.20)' : 'transparent';
    const rowBorder = active ? '1px solid rgba(59,130,246,0.5)' : '1px solid transparent';
    const iconColor = active ? '#60a5fa' : (isImgFile ? '#f472b6' : (isSr ? '#a78bfa' : '#94a3b8'));
    const badgeBg = active ? 'rgba(59,130,246,0.28)' : '#1e293b';
    const badgeColor = active ? '#60a5fa' : '#94a3b8';
    const shortBadge = it.elapsed_short
      ? `<span style="background:${badgeBg};color:${badgeColor};padding:2px 7px;border-radius:999px;font-size:11px;font-weight:700;flex-shrink:0;">${escapeHtml(it.elapsed_short)}</span>`
      : '';
    const srAction = isImgFile
      ? `<button type="button" title="一键将该照片送入 LongCat 数字人开口说话" onclick="event.stopPropagation();selectLocalHistoryItem(${idx});sendCurrentImageToAvatar()" style="background:rgba(16,185,129,0.18);color:#6ee7b7;border:1px solid rgba(16,185,129,0.45);padding:2px 7px;border-radius:6px;font-size:10.5px;font-weight:700;cursor:pointer;flex-shrink:0;">🎭数字人</button>`
      : (isSr
        ? `<span style="background:rgba(139,92,246,0.22);color:#c4b5fd;border:1px solid rgba(139,92,246,0.45);padding:1px 6px;border-radius:6px;font-size:10.5px;font-weight:800;flex-shrink:0;">✨超清</span>`
        : `<button type="button" title="使用 ReCaVSR 将该视频超分至 2K 超清" onclick="event.stopPropagation();upscaleVideoByName('${escapeHtml(it.name)}')" style="background:rgba(139,92,246,0.16);color:#c4b5fd;border:1px solid rgba(139,92,246,0.42);padding:2px 7px;border-radius:6px;font-size:10.5px;font-weight:700;cursor:pointer;flex-shrink:0;">✨2K</button>`);
    return `
      <div onclick="selectLocalHistoryItem(${idx})" style="display:flex;justify-content:space-between;align-items:center;gap:6px;padding:9px 10px;border-radius:10px;background:${rowBg};border:${rowBorder};cursor:pointer;transition:all 0.12s;">
        <div style="display:flex;align-items:center;gap:8px;min-width:0;flex:1;">
          <span style="color:${iconColor};font-size:14px;flex-shrink:0;">${isImgFile ? '🖼️' : '🎞️'}</span>
          <div style="min-width:0;">
            <div style="font-size:12.5px;font-weight:700;color:#f8fafc;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="${escapeHtml(it.name)}">${escapeHtml(it.short_name || it.name)}</div>
            <div style="font-size:11px;color:#94a3b8;margin-top:2px;">${escapeHtml(it.date_str || '')}</div>
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:5px;flex-shrink:0;">
          ${srAction}
          ${shortBadge}
        </div>
      </div>`;
  }).join('');
}

function _updatePreviewMediaMode(fname, url) {
  const isImg = /\.(png|jpe?g|webp)$/i.test(fname || '');
  const player = document.getElementById('mmVideoPlayer');
  const imgPrev = document.getElementById('mmImagePreview');
  const imgBtns = document.getElementById('imageActionBtns');
  const srWrap = document.getElementById('videoUpscaleActionWrap');
  const btnCont = document.getElementById('btnContinueLastFrame');
  const btnRep = document.getElementById('btnReplayVideo');
  if (isImg) {
    if (player) { player.pause(); player.style.display = 'none'; }
    if (imgPrev) { imgPrev.src = url; imgPrev.style.display = 'block'; }
    if (imgBtns) imgBtns.style.display = 'inline-flex';
    if (srWrap) srWrap.style.display = 'none';
    if (btnCont) btnCont.style.display = 'none';
    if (btnRep) btnRep.style.display = 'none';
  } else {
    if (imgPrev) { imgPrev.style.display = 'none'; }
    if (player) {
      player.style.display = 'block';
      if (player.getAttribute('data-fname') !== fname) {
        player.src = url;
        player.setAttribute('data-fname', fname);
      }
    }
    if (imgBtns) imgBtns.style.display = 'none';
    if (srWrap) srWrap.style.display = 'inline-flex';
    if (btnCont) btnCont.style.display = 'inline-flex';
    if (btnRep) btnRep.style.display = 'inline-flex';
  }
}

async function _fetchCurrentSelectedImageAsDataUrl() {
  if (!currentSelectedVideoName) return '';
  const url = `/api/local_video_file?name=${encodeURIComponent(currentSelectedVideoName)}`;
  const r = await fetch(url);
  const blob = await r.blob();
  return await new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result || '');
    reader.readAsDataURL(blob);
  });
}

async function sendCurrentImageToAvatar() {
  try {
    const dataUrl = await _fetchCurrentSelectedImageAsDataUrl();
    if (!dataUrl) {
      alert('请先在本地历史中选择一张已生成的图片');
      return;
    }
    firstFrameB64 = dataUrl;
    avatarUseBuiltinSample = false;
    const avImg = document.getElementById('avatarImagePreview');
    const hintEl = document.getElementById('avatarImageHint');
    if (avImg) {
      avImg.src = dataUrl;
      avImg.style.display = 'block';
    }
    if (hintEl) {
      hintEl.innerText = `✅ 已载入刚生成的 4K 写真「${currentSelectedVideoName}」作为数字人参考照片！在左侧黄色框输入台词即可驱动她开口说话！`;
    }
    switchMiniMaxMode('avatar');
    const pEl = document.getElementById('videoPrompt');
    if (pEl) {
      pEl.value = '固定镜头，近景居中构图。画面中的人物直视镜头自然说话，眼神温柔灵动，伴随说话节奏有轻微自然的眨眼、轻轻点头和呼吸起伏，嘴型与语音高度同步，面部柔和暖光，五官清晰稳定，过渡丝滑真实。';
    }
    const avRow = document.getElementById('avatarControlRow');
    if (avRow) avRow.scrollIntoView({behavior: 'smooth', block: 'center'});
  } catch (e) {
    alert('载入数字人参考照片失败: ' + e);
  }
}

async function sendCurrentImageToH3Edit() {
  try {
    const dataUrl = await _fetchCurrentSelectedImageAsDataUrl();
    if (!dataUrl) {
      alert('请先在本地历史中选择一张已生成的图片');
      return;
    }
    h3ImgRefList[0] = dataUrl;
    const prevEl = document.getElementById('h3ImgRef1Preview');
    const stEl = document.getElementById('h3ImgRefStatus');
    if (prevEl) {
      prevEl.src = dataUrl;
      prevEl.style.display = 'block';
    }
    if (stEl) {
      stEl.innerText = `👗 已将「${currentSelectedVideoName}」设为 <Picture 1> 主人物底图！在右侧填写新衣服或选择锁脸换装模板即可一键换装！`;
    }
    switchMiniMaxMode('image');
    const imgRow = document.getElementById('h3ImageControlRow');
    if (imgRow) imgRow.scrollIntoView({behavior: 'smooth', block: 'center'});
  } catch (e) {
    alert('设为 <Picture 1> 失败: ' + e);
  }
}

function selectLocalHistoryItem(idx) {
  const it = localHistoryItems[idx];
  if (!it) return;
  currentSelectedVideoName = it.name;
  renderLocalHistoryList();
  const url = `/api/local_video_file?name=${encodeURIComponent(it.name)}`;
  const dl = document.getElementById('mmVideoDownload');
  const emptyEl = document.getElementById('videoEmptyPlaceholder');
  const rWrap = document.getElementById('videoResultWrap');
  const topBadge = document.getElementById('previewElapsedBadge');
  const botBadge = document.getElementById('previewBottomShortBadge');
  if (emptyEl) emptyEl.style.display = 'none';
  if (rWrap) rWrap.style.display = 'block';
  _updatePreviewMediaMode(it.name, url);
  dl.href = url;
  dl.download = it.name;
  dl.style.display = 'inline-flex';
  const metaEl = document.getElementById('mmVideoMeta');
  if (metaEl) {
    metaEl.innerText = it.name;
    metaEl.title = it.name;
  }
  if (topBadge) {
    if (it.elapsed_long) {
      topBadge.innerText = `⏱ ${it.elapsed_long}`;
      topBadge.style.display = 'inline-flex';
    } else {
      topBadge.style.display = 'none';
    }
  }
  if (botBadge) {
    if (it.elapsed_short) {
      botBadge.innerText = it.elapsed_short;
      botBadge.style.display = 'inline-block';
    } else {
      botBadge.style.display = 'none';
    }
  }
}

function replayCurrentVideo() {
  const player = document.getElementById('mmVideoPlayer');
  if (player && player.src) {
    player.currentTime = 0;
    player.play().catch(() => {});
  }
}

async function revealInFinder(openFolderOnly = false) {
  const fname = openFolderOnly ? '' : (currentSelectedVideoName || '');
  try {
    await fetch('/api/reveal_in_finder', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: fname})
    });
  } catch (e) {}
}

function playHistoryVideo(fname) {
  if (!fname) return;
  const idx = localHistoryItems.findIndex(x => x.name === fname);
  if (idx >= 0) {
    selectLocalHistoryItem(idx);
    return;
  }
  currentSelectedVideoName = fname;
  const url = `http://127.0.0.1:30000/v1/video/file/${encodeURIComponent(fname)}`;
  const dl = document.getElementById('mmVideoDownload');
  const emptyEl = document.getElementById('videoEmptyPlaceholder');
  const rWrap = document.getElementById('videoResultWrap');
  if (emptyEl) emptyEl.style.display = 'none';
  if (rWrap) rWrap.style.display = 'block';
  _updatePreviewMediaMode(fname, url);
  dl.href = url;
  dl.download = fname;
  dl.style.display = 'inline-flex';
  document.getElementById('mmVideoMeta').innerText = fname;
}

let videoPollTimer = null;
let liveVideoElapsedSec = null;
let liveVideoEtaSec = null;
let liveVideoPct = null;
let liveVideoBaseMsg = '';

function formatCnDur(sec) {
  const s = Math.max(0, Math.floor(sec || 0));
  if (s >= 3600) {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const r = s % 60;
    return `${h}小时${m}分${String(r).padStart(2, '0')}秒`;
  }
  if (s >= 60) {
    const m = Math.floor(s / 60);
    const r = s % 60;
    return `${m}分${String(r).padStart(2, '0')}秒`;
  }
  return `${s}秒`;
}

function renderLiveVideoTimingText() {
  if (dramaPollTimer || liveVideoElapsedSec === null) return;
  const tLabel = document.getElementById('taskProgressLabel');
  const tSpeed = document.getElementById('taskProgressSpeed');
  const st = document.getElementById('videoGenStatus');
  const elStr = `已耗时 ${formatCnDur(liveVideoElapsedSec)}`;
  const etaStr = (liveVideoEtaSec && liveVideoEtaSec > 0) ? ` · 预计剩余 ≈ ${formatCnDur(liveVideoEtaSec)}` : '';
  const pctVal = liveVideoPct !== null ? liveVideoPct : 0;
  if (tLabel) {
    tLabel.innerText = `⏳ 当前任务总进度：${pctVal}%`;
  }
  if (tSpeed) {
    tSpeed.innerText = `总进度 ${pctVal}%`;
  }
  if (st && liveVideoBaseMsg) {
    st.innerText = `⏳ ${liveVideoBaseMsg} · ${elStr}${etaStr}`;
    st.style.color = '#f59e0b';
  }
}

setInterval(() => {
  if (liveVideoElapsedSec !== null) {
    liveVideoElapsedSec += 1;
    if (liveVideoEtaSec !== null && liveVideoEtaSec > 1) {
      liveVideoEtaSec -= 1;
    }
    renderLiveVideoTimingText();
  }
}, 1000);

function setStopButtonsVisible(visible) {
  const bTop = document.getElementById('btnStopTaskTop');
  const bBot = document.getElementById('btnStopTaskBottom');
  if (bTop) bTop.style.display = visible ? 'inline-flex' : 'none';
  if (bBot) bBot.style.display = visible ? 'inline-flex' : 'none';
}

function setVideoBtnRunningState(running, customLabel) {
  const btnVid = document.getElementById('btnVideoGen');
  const btnImg = document.getElementById('btnImageOnlyGen');
  const btnFus = document.getElementById('btnRunFusionStudio');
  const fusSt = document.getElementById('fusionStudioStatusText');
  const isImgTask = (currentMiniMaxMode === 'image') || (currentMiniMaxMode === 'fusion') || /写真|图片|4K/i.test(customLabel || '');
  if (running) {
    if (btnFus && currentMiniMaxMode === 'fusion') {
      btnFus.disabled = true;
      btnFus.innerText = '⏳ 正在执行 33B 多图空间解耦合成...';
    }
    if (isImgTask) {
      if (btnImg) {
        btnImg.setAttribute('data-running', '1');
        btnImg.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
        btnImg.style.boxShadow = '0 6px 16px -4px rgba(239,68,68,0.55)';
        btnImg.innerText = customLabel || '🛑 正在生成 4K 图片... (点击停止)';
      }
      if (btnVid) {
        btnVid.setAttribute('data-running', '0');
        btnVid.style.opacity = '0.45';
      }
    } else {
      if (btnVid) {
        btnVid.setAttribute('data-running', '1');
        btnVid.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
        btnVid.style.boxShadow = '0 6px 16px -4px rgba(239,68,68,0.55)';
        btnVid.innerText = customLabel || '🛑 正在合成视频... (点击停止)';
      }
      if (btnImg) {
        btnImg.setAttribute('data-running', '0');
        btnImg.style.opacity = '0.45';
      }
    }
    setStopButtonsVisible(true);
  } else {
    if (btnImg) {
      btnImg.setAttribute('data-running', '0');
      btnImg.style.opacity = '1';
      btnImg.style.background = 'linear-gradient(135deg, #ec4899, #9333ea)';
      btnImg.style.boxShadow = '0 6px 16px -4px rgba(236,72,153,0.45)';
      btnImg.innerText = currentMiniMaxMode === 'image' ? '📸 仅生成 4K 写真图片' : '📸 仅生成 4K 图片';
    }
    if (btnFus) {
      btnFus.disabled = false;
      btnFus.innerText = '🪄 立即执行 AI 换脸 / 换装 / 场景融合';
    }
    if (fusSt && currentMiniMaxMode === 'fusion') {
      fusSt.innerText = '✅ 多图融合完成！已在下方大屏预览并保存至 ~/Movies/EasyColab/（可点右侧「📌 锁入该角色定妆卡槽」或「🗣️ 让TA开口说话」）';
      fusSt.style.color = '#34d399';
    }
    if (btnVid) {
      btnVid.setAttribute('data-running', '0');
      btnVid.style.opacity = '1';
      btnVid.style.background = '';
      btnVid.style.boxShadow = '0 6px 16px -4px rgba(59,130,246,0.5)';
      btnVid.innerText = currentMiniMaxMode === 'avatar' ? '🎬 生成数字人视频' : '🎬 生成有声视频';
    }
    if (!upscalePollTimer) setStopButtonsVisible(false);
  }
}

function setUpscaleBtnRunningState(running) {
  const btn = document.getElementById('btnUpscaleVideo');
  if (!btn) return;
  btn.disabled = false;
  if (running) {
    btn.setAttribute('data-running', '1');
    btn.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
    btn.innerText = '🛑 正在超分中... (点击停止)';
    setStopButtonsVisible(true);
  } else {
    btn.setAttribute('data-running', '0');
    btn.style.background = 'linear-gradient(135deg,#8b5cf6,#6366f1)';
    onUpscalePresetChange((document.getElementById('upscalePresetSelect') || {}).value || '2k');
    if (!videoPollTimer) setStopButtonsVisible(false);
  }
}

async function stopActiveVideoTask(targetMode = 'auto') {
  const isUpscaling = (targetMode === 'upscale') || (targetMode === 'auto' && upscalePollTimer && !videoPollTimer);
  const st = document.getElementById('videoGenStatus');
  const tLabel = document.getElementById('taskProgressLabel');
  const tSpeed = document.getElementById('taskProgressSpeed');
  const dState = document.getElementById('deployStateText');
  const endpoint = isUpscaling ? '/api/video_upscale_cancel' : '/api/video_cancel';
  if (st) {
    st.innerText = isUpscaling ? '🛑 正在停止 ReCaVSR 超分任务...' : '🛑 正在停止当前生成任务...';
    st.style.color = '#f59e0b';
  }
  if (tLabel) tLabel.innerText = isUpscaling ? '🛑 正在安全中断 ReCaVSR 超分任务...' : '🛑 正在安全中断当前生成任务 (保留模型显存常驻)...';
  try {
    const r = await fetch(endpoint, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
    const d = await r.json();
    await pollLogs();
    if (isUpscaling) {
      if (upscalePollTimer) { clearInterval(upscalePollTimer); upscalePollTimer = null; }
      setUpscaleBtnRunningState(false);
    } else {
      if (videoPollTimer) { clearInterval(videoPollTimer); videoPollTimer = null; }
      setVideoBtnRunningState(false);
    }
    liveVideoElapsedSec = null;
    liveVideoEtaSec = null;
    liveVideoPct = null;
    liveVideoBaseMsg = '';
    const msg = d.message || '🛑 已手动停止当前任务';
    if (st) { st.innerText = msg; st.style.color = '#f59e0b'; }
    if (tLabel) tLabel.innerText = msg;
    if (tSpeed) tSpeed.innerText = '已手动停止 · 显存已就绪';
    if (dState) { dState.innerText = '🛑 已手动停止'; dState.style.color = '#f59e0b'; }
  } catch (e) {
    if (st) { st.innerText = '❌ 停止请求异常: ' + e; st.style.color = '#ef4444'; }
  }
}

async function startVideoGenerate(forceTarget = '') {
  const btnVid = document.getElementById('btnVideoGen');
  const btnImg = document.getElementById('btnImageOnlyGen');
  if ((btnVid && btnVid.getAttribute('data-running') === '1') || (btnImg && btnImg.getAttribute('data-running') === '1')) {
    await stopActiveVideoTask('video');
    return;
  }
  const savedUserPrompt = (document.getElementById('videoPrompt') || {}).value || '';
  if (forceTarget === 'image' && currentMiniMaxMode !== 'image') {
    switchMiniMaxMode('image');
    if (savedUserPrompt.trim() && !savedUserPrompt.includes('Makoto Shinkai anime style')) {
      document.getElementById('videoPrompt').value = savedUserPrompt;
    }
  } else if (forceTarget === 'video' && currentMiniMaxMode === 'image') {
    switchMiniMaxMode('t2va');
    if (savedUserPrompt.trim() && !savedUserPrompt.includes('Suzhou courtyard at dusk')) {
      document.getElementById('videoPrompt').value = savedUserPrompt;
    }
  }
  if (currentMiniMaxMode === 'avatar') {
    autoSplitAvatarPromptAndTts(document.getElementById('videoPrompt'));
  }
  const prompt = document.getElementById('videoPrompt').value.trim();
  const isImgMode = (forceTarget === 'image') || (currentMiniMaxMode === 'image');
  if (!prompt && !isImgMode) {
    alert('请输入画面提示词');
    return;
  }
  const resVal = document.getElementById('videoRes').value || (isImgMode ? '3552x4736' : '832x480');
  const [wStr, hStr] = resVal.split('x');
  const randSeed = document.getElementById('videoRandSeed').checked;
  const seedVal = randSeed ? -1 : (parseInt(document.getElementById('videoSeed').value || '42', 10) || 42);
  const payload = {
    prompt,
    soundscape: document.getElementById('videoSoundPrompt').value.trim(),
    music: document.getElementById('videoMusicPrompt').value.trim(),
    negative_prompt: document.getElementById('videoNegPrompt').value.trim(),
    resolution: resVal,
    width: parseInt(wStr, 10) || 832,
    height: parseInt(hStr, 10) || 480,
    num_frames: parseInt(document.getElementById('videoFrames').value || '124', 10),
    steps: isImgMode ? 6 : parseInt(document.getElementById('videoSteps').value || '30', 10),
    fps: parseInt(document.getElementById('videoFps').value || '24', 10),
    seed: seedVal,
    randomize_seed: randSeed,
    batch_count: isImgMode ? 1 : selectedBatchCount,
  };
  if (currentMiniMaxMode === 'fl2va' && firstFrameB64) {
    payload.first_frame_b64 = firstFrameB64;
    if (lastFrameB64) payload.last_frame_b64 = lastFrameB64;
  } else if (currentMiniMaxMode === 'image') {
    payload.mode = 'image';
    payload.ref_images_b64 = h3ImgRefList.filter(Boolean);
    payload.edit_outfit = ((document.getElementById('inpH3EditOutfit') || {}).value || '').trim();
    payload.edit_scene = ((document.getElementById('inpH3EditScene') || {}).value || '').trim();
    const chkClean = document.getElementById('chkCleanMatteWords');
    payload.clean_matte = chkClean ? Boolean(chkClean.checked) : true;
  } else if (currentMiniMaxMode === 'avatar') {
    payload.mode = 'avatar';
    payload.audio_b64 = avatarAudioB64;
    payload.audio_ext = avatarAudioExt;
    payload.is_tts_audio = Boolean(avatarAudioIsTts);
    payload.tts_text = ((document.getElementById('inpAvatarTtsText') || {}).value || '').trim();
    payload.tts_voice = getCombinedAvatarTtsSpec();
    payload.clone_ref_b64 = mingCloneAudioB64;
    payload.clone_prompt_text = ((document.getElementById('inpMingClonePromptText') || {}).value || '').trim();
    payload.use_builtin_sample = avatarUseBuiltinSample;
    payload.num_segments = parseInt((document.getElementById('avatarSegmentsSelect') || {}).value ?? '0', 10);
    if (isNaN(payload.num_segments)) payload.num_segments = 0;
    if (firstFrameB64) payload.first_frame_b64 = firstFrameB64;
  }
  const st = document.getElementById('videoGenStatus');
  const pWrap = document.getElementById('videoProgressWrap');
  const pBar = document.getElementById('videoProgressBar');
  const isAvMode = currentMiniMaxMode === 'avatar';
  const runningBtnText = isImgMode
    ? '🛑 正在生成 4K 写真大图... (点击停止)'
    : (isAvMode ? '🛑 正在生成数字人视频... (点击停止)' : (selectedBatchCount > 1 ? `🛑 正在连抽 ${selectedBatchCount} 条... (点击停止)` : '🛑 正在合成有声视频... (点击停止)'));
  setVideoBtnRunningState(true, runningBtnText);
  pWrap.style.display = 'block';
  pBar.style.width = '8%';
  st.innerText = isImgMode
    ? '⏳ 已提交至 MiniMax-H3-Image (33B · Heretic无审查) 引擎，正在进行 6 步极速扩散采样...'
    : (isAvMode
      ? '⏳ 已提交至 LongCat-Video-Avatar-1.6 引擎，正在分离人声并提取 Whisper-v3 唇形特征...'
      : `⏳ 已提交至 MiniMax-H3 引擎 (连抽 ${selectedBatchCount} 条)，正在调用多模态编码器...`);
  st.style.color = '#f59e0b';
  try {
    const r = await fetch('/api/video_generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const d = await r.json();
    if (!d.ok) {
      if (String(d.error || '').includes('正在生成中')) {
        st.innerText = '⏳ ' + d.error + '（已自动为您恢复实时进度监控）';
        st.style.color = '#f59e0b';
        if (videoPollTimer) clearInterval(videoPollTimer);
        pollVideoStatus();
        videoPollTimer = setInterval(pollVideoStatus, 2000);
        return;
      }
      st.innerText = '❌ ' + (d.error || '启动失败（请确认模型服务已就绪）');
      st.style.color = '#ef4444';
      setVideoBtnRunningState(false);
      return;
    }
    if (d.auto_switching) {
      if (isImgMode) {
        selectPresetModel('preset_minimax_image', 'Comfy-Org/MiniMax-H3-Image', 'MiniMax-H3-Image', '*.safetensors');
      } else if (isAvMode) {
        if (currentPresetTier === 't4') {
          selectPresetModel('preset_t4_longcat_int8', 'meituan-longcat/LongCat-Video-Avatar-1.6', 'LongCat-Video-Avatar-1.6-INT8', 'base_model_int8/*.safetensors');
        } else {
          selectPresetModel('preset_longcat_avatar', 'meituan-longcat/LongCat-Video-Avatar-1.6', 'LongCat-Video-Avatar-1.6', 'base_model/*.safetensors');
        }
      } else {
        if (currentPresetTier === 't4') {
          selectPresetModel('preset_t4_minimax_h3', 'unsloth/MiniMax-H3-GGUF', 'MiniMax-H3-T4-GGUF', 'minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf');
        } else {
          selectPresetModel('preset_minimax_h3', 'MiniMaxAI/MiniMax-H3', 'MiniMax-H3', 'FL2VA/*.safetensors');
        }
      }
      if (d.message) {
        st.innerText = d.message;
        st.style.color = '#f59e0b';
      }
    }
    if (videoPollTimer) clearInterval(videoPollTimer);
    videoPollTimer = setInterval(pollVideoStatus, 2000);
  } catch (e) {
    st.innerText = '❌ 请求异常: ' + e;
    st.style.color = '#ef4444';
    setVideoBtnRunningState(false);
  }
}

async function pollVideoStatus() {
  if (dramaPollTimer) {
    if (videoPollTimer) {
      clearInterval(videoPollTimer);
      videoPollTimer = null;
    }
    liveVideoElapsedSec = null;
    liveVideoEtaSec = null;
    liveVideoPct = null;
    liveVideoBaseMsg = '';
    return;
  }
  const btn = document.getElementById('btnVideoGen');
  const st = document.getElementById('videoGenStatus');
  const pWrap = document.getElementById('videoProgressWrap');
  const pBar = document.getElementById('videoProgressBar');
  const tLabel = document.getElementById('taskProgressLabel');
  const tSpeed = document.getElementById('taskProgressSpeed');
  const tBar = document.getElementById('taskProgressBar');
  const dState = document.getElementById('deployStateText');
  try {
    const r = await fetch('/api/video_status');
    const d = await r.json();
    await pollLogs();
    if (d.error && !d.running) {
      if (videoPollTimer) {
        clearInterval(videoPollTimer);
        videoPollTimer = null;
      }
      liveVideoElapsedSec = null;
      liveVideoEtaSec = null;
      liveVideoPct = null;
      liveVideoBaseMsg = '';
      const isCancel = String(d.error).includes('已手动停止');
      st.innerText = isCancel ? d.error : ('❌ ' + d.error);
      st.style.color = isCancel ? '#f59e0b' : '#ef4444';
      if (tLabel) tLabel.innerText = isCancel ? d.error : ('❌ ' + d.error);
      if (tSpeed) tSpeed.innerText = isCancel ? '已手动停止 · 显存已就绪' : '生成中断';
      if (dState) {
        dState.innerText = isCancel ? '🛑 已手动停止' : '❌ 视频生成中断';
        dState.style.color = isCancel ? '#f59e0b' : '#ef4444';
      }
      setVideoBtnRunningState(false);
      await loadLocalVideoHistory(false);
      return;
    }
    if (d.running) {
      const isImgMode = currentMiniMaxMode === 'image';
      const isAvMode = currentMiniMaxMode === 'avatar';
      const runLbl = isImgMode ? '🛑 正在生成 4K 写真大图... (点击停止)' : (isAvMode ? '🛑 正在生成数字人视频... (点击停止)' : '🛑 正在合成有声视频... (点击停止)');
      setVideoBtnRunningState(true, runLbl);
      if (pWrap) pWrap.style.display = 'block';
      if (!videoPollTimer) {
        videoPollTimer = setInterval(pollVideoStatus, 2000);
      }
    }
    if (d.auto_switching) {
      // 模型自动切换/预热/装载期间，底部主进度条由 pollLogs() 独占驱动，严禁在此覆盖导致左右百分比与进度条来回跳变闪烁
      liveVideoElapsedSec = null;
      liveVideoEtaSec = null;
      liveVideoPct = null;
      liveVideoBaseMsg = '';
      const swPct = typeof d.pct === 'number' ? Math.max(8, Math.min(99, d.pct)) : 15;
      if (pBar) pBar.style.width = `${swPct}%`;
      if (st && d.message) {
        st.innerText = d.message;
        st.style.color = '#f59e0b';
      }
      return;
    }
    const prog = typeof d.pct === 'number' ? d.pct : (typeof d.progress === 'number' ? d.progress : null);
    if (prog !== null) {
      const clamped = Math.max(5, Math.min(100, prog));
      liveVideoPct = clamped;
      pBar.style.width = `${clamped}%`;
      if (tBar) {
        tBar.className = d.running ? 'progress-fill' : 'progress-fill green';
        tBar.style.width = `${clamped}%`;
      }
    }
    if (d.seed && !d.running) {
      document.getElementById('videoSeed').value = d.seed;
    }
    if (d.running) {
      if (typeof d.elapsed_sec === 'number' && d.elapsed_sec > 0) {
        liveVideoElapsedSec = d.elapsed_sec;
      } else if (liveVideoElapsedSec === null) {
        liveVideoElapsedSec = 1;
      }
      if (typeof d.eta_sec === 'number' && d.eta_sec > 0) {
        liveVideoEtaSec = d.eta_sec;
      }
    } else {
      liveVideoElapsedSec = null;
      liveVideoEtaSec = null;
      liveVideoPct = null;
      liveVideoBaseMsg = '';
    }
    if (d.message) {
      if (d.running) {
        liveVideoBaseMsg = d.message;
        renderLiveVideoTimingText();
      } else {
        const doneDur = (typeof d.elapsed_s === 'number' && d.elapsed_s > 0) ? ` (总耗时 ${formatCnDur(d.elapsed_s)})` : '';
        st.innerText = `${d.message}`;
        st.style.color = '#10b981';
        if (tLabel) {
          tLabel.innerText = `✅ 当前任务总进度：100%`;
        }
        if (tSpeed) {
          tSpeed.innerText = `总进度 100%`;
        }
      }
      if (dState) {
        const isImgDoneOrRun = (currentMiniMaxMode === 'image') || /MiniMax-H3-Image|写真|大图|\.(png|jpe?g|webp)/i.test(d.message || '');
        dState.innerText = d.running
          ? (isImgDoneOrRun ? '⏳ 4K 写真图片正在生成中...' : '⏳ 视频正在生成中...')
          : (isImgDoneOrRun ? '✅ 4K 写真图片生成已完成' : '✅ 视频生成已完成');
        dState.style.color = d.running ? '#f59e0b' : '#10b981';
      }
    }
    if (!d.running) {
      if (videoPollTimer) {
        clearInterval(videoPollTimer);
        videoPollTimer = null;
      }
      setVideoBtnRunningState(false);
      pBar.style.width = '100%';
      if (tBar) {
        tBar.className = 'progress-fill green';
        tBar.style.width = '100%';
      }
      await loadLocalVideoHistory(true);
      const autoSrChk = document.getElementById('autoUpscale2kChk');
      const isMp4Video = d.latest_video && /\.mp4$/i.test(d.latest_video);
      if (autoSrChk && autoSrChk.checked && isMp4Video && !d.latest_video.includes('_ReCaVSR_') && lastAutoUpscaledVideo !== d.latest_video) {
        lastAutoUpscaledVideo = d.latest_video;
        const presetVal = (document.getElementById('upscalePresetSelect') || {}).value || '2k';
        upscaleVideoByName(d.latest_video, presetVal);
      }
    }
  } catch (e) {}
}

let upscalePollTimer = null;
let lastAutoUpscaledVideo = '';

function onUpscalePresetChange(val) {
  const btn = document.getElementById('btnUpscaleVideo');
  if (!btn || btn.getAttribute('data-running') === '1') return;
  const labelMap = {'2k': '✨ 超分至 2K', '1080p': '✨ 超分至 1080P', '4k': '✨ 超分至 4K'};
  btn.innerText = labelMap[val] || '✨ 超分至 2K';
}

async function upscaleCurrentVideo() {
  const btn = document.getElementById('btnUpscaleVideo');
  if (btn && btn.getAttribute('data-running') === '1') {
    await stopActiveVideoTask('upscale');
    return;
  }
  if (!currentSelectedVideoName) {
    alert('请先在右侧「本地历史」中选择一条要超分的视频');
    return;
  }
  const presetVal = (document.getElementById('upscalePresetSelect') || {}).value || '2k';
  upscaleVideoByName(currentSelectedVideoName, presetVal);
}

async function upscaleVideoByName(fname, presetOverride) {
  if (!fname) return;
  const presetVal = presetOverride || (document.getElementById('upscalePresetSelect') || {}).value || '2k';
  const st = document.getElementById('videoGenStatus');
  const pWrap = document.getElementById('videoProgressWrap');
  const pBar = document.getElementById('videoProgressBar');
  const tLabel = document.getElementById('taskProgressLabel');
  const tSpeed = document.getElementById('taskProgressSpeed');
  const tBar = document.getElementById('taskProgressBar');
  const dState = document.getElementById('deployStateText');
  setUpscaleBtnRunningState(true);
  if (pWrap) pWrap.style.display = 'block';
  if (pBar) pBar.style.width = '10%';
  if (tBar) {
    tBar.className = 'progress-fill';
    tBar.style.width = '10%';
  }
  const startMsg = `正在启动 ReCaVSR (${presetVal.toUpperCase()}) 单步流式超分: ${fname}`;
  if (tLabel) tLabel.innerText = `⏳ ${startMsg}`;
  if (dState) {
    dState.innerText = '✨ ReCaVSR 正在超分重绘中...';
    dState.style.color = '#a78bfa';
  }
  const curFrames = parseInt((document.getElementById('videoFrames') || {}).value || '362', 10) || 362;
  const estPerFrame = presetVal === '4k' ? 0.22 : (presetVal === '1080p' ? 0.085 : 0.115);
  liveVideoElapsedSec = 1;
  liveVideoEtaSec = Math.max(8, Math.round(curFrames * estPerFrame + 6));
  liveVideoPct = 10;
  liveVideoBaseMsg = startMsg;
  renderLiveVideoTimingText();
  try {
    const r = await fetch('/api/video_upscale', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filename: fname, target_preset: presetVal}),
    });
    const d = await r.json();
    await pollLogs();
    if (!d.ok) {
      const errTxt = d.error || '超分启动失败';
      if (st) {
        st.innerText = '❌ ' + errTxt;
        st.style.color = '#ef4444';
      }
      if (tLabel) tLabel.innerText = '❌ ' + errTxt;
      if (tSpeed) tSpeed.innerText = '超分未启动';
      if (dState) {
        dState.innerText = '❌ ReCaVSR 超分启动失败';
        dState.style.color = '#ef4444';
      }
      liveVideoElapsedSec = null;
      liveVideoEtaSec = null;
      setUpscaleBtnRunningState(false);
      return;
    }
    if (upscalePollTimer) clearInterval(upscalePollTimer);
    pollUpscaleStatus();
    upscalePollTimer = setInterval(pollUpscaleStatus, 1500);
  } catch (e) {
    if (st) {
      st.innerText = '❌ 超分请求异常: ' + e;
      st.style.color = '#ef4444';
    }
    if (tLabel) tLabel.innerText = '❌ 超分请求异常: ' + e;
    if (dState) {
      dState.innerText = '❌ ReCaVSR 超分请求异常';
      dState.style.color = '#ef4444';
    }
    liveVideoElapsedSec = null;
    setUpscaleBtnRunningState(false);
  }
}

async function pollUpscaleStatus() {
  const st = document.getElementById('videoGenStatus');
  const pWrap = document.getElementById('videoProgressWrap');
  const pBar = document.getElementById('videoProgressBar');
  const tLabel = document.getElementById('taskProgressLabel');
  const tSpeed = document.getElementById('taskProgressSpeed');
  const tBar = document.getElementById('taskProgressBar');
  const dState = document.getElementById('deployStateText');
  try {
    const r = await fetch('/api/video_upscale_status');
    const d = await r.json();
    await pollLogs();
    if (d.error && !d.running) {
      if (upscalePollTimer) {
        clearInterval(upscalePollTimer);
        upscalePollTimer = null;
      }
      liveVideoElapsedSec = null;
      liveVideoEtaSec = null;
      liveVideoPct = null;
      liveVideoBaseMsg = '';
      const isCancel = String(d.error).includes('已手动停止');
      if (st) {
        st.innerText = isCancel ? d.error : ('❌ ' + d.error);
        st.style.color = isCancel ? '#f59e0b' : '#ef4444';
      }
      if (tLabel) tLabel.innerText = isCancel ? d.error : ('❌ ' + (d.message || d.error));
      if (tSpeed) tSpeed.innerText = isCancel ? '已手动停止 · 显存已就绪' : '超分中断，请查看下方日志或重试';
      if (dState) {
        dState.innerText = isCancel ? '🛑 已手动停止' : '❌ ReCaVSR 超分失败';
        dState.style.color = isCancel ? '#f59e0b' : '#ef4444';
      }
      setUpscaleBtnRunningState(false);
      return;
    }
    const prog = typeof d.pct === 'number' ? d.pct : 10;
    const clamped = Math.max(5, Math.min(100, prog));
    liveVideoPct = clamped;
    if (pBar) pBar.style.width = `${clamped}%`;
    if (tBar) {
      tBar.className = d.running ? 'progress-fill' : 'progress-fill green';
      tBar.style.width = `${clamped}%`;
    }
    if (d.running) {
      if (pWrap) pWrap.style.display = 'block';
      setUpscaleBtnRunningState(true);
      if (typeof d.elapsed_sec === 'number' && d.elapsed_sec > 0) {
        liveVideoElapsedSec = d.elapsed_sec;
      } else if (liveVideoElapsedSec === null) {
        liveVideoElapsedSec = 1;
      }
      if (typeof d.eta_sec === 'number' && d.eta_sec > 0) {
        liveVideoEtaSec = d.eta_sec;
      }
      if (d.message) {
        liveVideoBaseMsg = d.message;
        if (tLabel) tLabel.innerText = `⏳ ${d.message}`;
        renderLiveVideoTimingText();
      }
      if (dState) {
        dState.innerText = '✨ ReCaVSR 正在超分重绘中...';
        dState.style.color = '#a78bfa';
      }
    } else {
      if (upscalePollTimer) {
        clearInterval(upscalePollTimer);
        upscalePollTimer = null;
      }
      liveVideoElapsedSec = null;
      liveVideoEtaSec = null;
      liveVideoPct = null;
      liveVideoBaseMsg = '';
      setUpscaleBtnRunningState(false);
      if (pBar) pBar.style.width = '100%';
      if (tBar) {
        tBar.className = 'progress-fill green';
        tBar.style.width = '100%';
      }
      if (d.message) {
        const doneDur = (typeof d.elapsed_s === 'number' && d.elapsed_s > 0) ? ` (超分耗时 ${formatCnDur(d.elapsed_s)})` : '';
        if (st) {
          st.innerText = d.message;
          st.style.color = '#10b981';
        }
        if (tLabel) tLabel.innerText = `✅ ${d.message}`;
        if (tSpeed) tSpeed.innerText = `已超分完成并保存至 ~/Movies/EasyColab${doneDur}`;
      }
      if (dState) {
        dState.innerText = '✅ ReCaVSR 超分已完成';
        dState.style.color = '#10b981';
      }
      await loadLocalVideoHistory(true);
      try {
        const rv = await fetch('/api/video_status');
        const dv = await rv.json();
        if (dv && dv.running && !videoPollTimer) {
          pollVideoStatus();
          videoPollTimer = setInterval(pollVideoStatus, 2000);
        }
      } catch (e) {}
    }
  } catch (e) {}
}

let dramaPollTimer = null;
let currentDramaWorkflow = null;
let currentDramaWfTab = 'img';
let dramaCastPhotoCacheBust = Date.now();
let dramaCastState = [
  {id: 'A', name: '林晚', voice: 'ming:灵小柔', desc: '27岁江南修伞铺女老板，鹅蛋脸，左眉尾小痣，低髻插素银玉兰簪，月白立领盘扣短衫配浅灰蓝围裙', has_photo: true, user_uploaded: false},
  {id: 'B', name: '沈砚', voice: 'ming:灵孤影_悲伤', desc: '30岁青年，剑眉丹凤眼，右侧颧骨下浅淡旧疤，炭黑色羊毛长大衣内搭米白高领，左腕系褪色红绳', has_photo: true, user_uploaded: false},
];

const CAST_VOICE_OPTIONS = [
  {val: 'ming:灵小柔', lbl: '🌸 灵小柔 (清亮甜美女主)'},
  {val: 'ming:灵雨汐_中性', lbl: '☁️ 灵雨汐·温柔 (贴耳软妹)'},
  {val: 'ming:灵黛嗔_生气', lbl: '👑 灵黛嗔·傲娇 (大小姐娇嗔)'},
  {val: 'ming:灵采薇_悲伤', lbl: '🌙 灵采薇·凄美 (深情破碎女声)'},
  {val: 'ming:灵梦', lbl: '💕 灵梦 (撒娇粘人女友)'},
  {val: 'ming:妩媚妲己', lbl: '🔥 妩媚妲己 (酥麻魅惑女声)'},
  {val: 'ming:灵惜梦_中性', lbl: '💎 灵惜梦·御姐 (干练都市姐姐)'},
  {val: 'ming:灵心澜_厌恶', lbl: '👠 灵心澜·女王 (高贵冷艳女声)'},
  {val: 'ming:灵信远_中性', lbl: '☀️ 灵信远·暖男 (温润公子/男友)'},
  {val: 'ming:灵朗', lbl: '🥂 灵朗 (深情沉稳霸总男主)'},
  {val: 'ming:灵宇', lbl: '🌹 灵宇 (自信含笑撩人学长)'},
  {val: 'ming:灵楷', lbl: '📻 灵楷 (温润磁性百灵官方男主)'},
  {val: 'ming:灵空', lbl: '🎧 灵空 (低音炮哄睡男神)'},
  {val: 'ming:灵孤影_悲伤', lbl: '🌧️ 灵孤影·心碎 (深情虐恋男主)'},
  {val: 'ming:灵孤影_厌恶', lbl: '❄️ 灵孤影·高冷 (禁欲系冷漠霸总)'},
  {val: 'ming:灵孤影_生气', lbl: '🔥 灵孤影·震怒 (低沉压迫感霸总)'},
  {val: 'ming:甄嬛传_四郎', lbl: '🎬 甄嬛传·四郎 (沉稳帝王原声)'},
  {val: 'ming:康熙王朝_康熙', lbl: '🎬 康熙大帝 (霸气帝王原声)'},
  {val: 'ming:水浒传_武松', lbl: '🎬 水浒传·武松 (豪迈硬汉原声)'},
  {val: 'ming:西游记_唐僧', lbl: '🎬 西游记·唐僧 (温润慈悲原声)'},
];

function renderDramaCastCards() {
  const grid = document.getElementById('dramaCastCardsGrid');
  if (!grid) return;
  if (!Array.isArray(dramaCastState) || dramaCastState.length === 0) {
    grid.innerHTML = `<div style="grid-column:1 / -1;padding:12px 14px;border-radius:8px;background:rgba(2,6,23,0.75);border:1px dashed #475569;color:#94a3b8;font-size:11.5px;text-align:center;line-height:1.6;">
      ✨ 当前已清空所有预设角色（0 个角色）。<br>
      <span style="color:#cbd5e1;">此时点击「🤖 复制外部 AI 编剧指令」将完全按您的原创故事自由生成角色，粘贴剧本后点「🔍 解析并生成工作流」会自动提取新角色填入此处；您也可随时点右上角「+ 新增角色」手动添加。</span>
    </div>`;
    return;
  }
  grid.innerHTML = dramaCastState.map((c, idx) => {
    const imgUrl = `/api/drama_work_file?name=char_${encodeURIComponent(c.id)}_half.jpg&t=${dramaCastPhotoCacheBust}`;
    const voiceOpts = CAST_VOICE_OPTIONS.map(o => `<option value="${o.val}" ${c.voice === o.val ? 'selected' : ''}>${o.lbl}</option>`).join('');
    const badgeHtml = c.user_uploaded
      ? `<span style="font-size:9.5px;padding:1px 5px;border-radius:4px;background:rgba(16,185,129,0.22);color:#6ee7b7;border:1px solid #10b981;font-weight:700;">📌已锁定上传照</span>`
      : `<span style="font-size:9.5px;padding:1px 5px;border-radius:4px;background:rgba(56,189,248,0.16);color:#7dd3fc;font-weight:600;">🎨AI定妆/可传照</span>`;
    return `
      <div style="padding:7px 9px;border-radius:8px;background:#020617;border:1px solid ${c.user_uploaded ? '#10b981' : '#334155'};display:flex;gap:8px;align-items:stretch;">
        <!-- 左：16:9 定妆缩略图 + 上传/复制生图词按钮 -->
        <div style="width:92px;flex-shrink:0;display:flex;flex-direction:column;justify-content:space-between;gap:4px;">
          <div style="position:relative;width:92px;height:52px;border-radius:5px;overflow:hidden;background:#0f172a;border:1px solid #475569;display:flex;align-items:center;justify-content:center;">
            <img src="${imgUrl}" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';" onload="this.style.display='block';this.nextElementSibling.style.display='none';" style="width:100%;height:100%;object-fit:cover;display:block;">
            <div style="display:none;flex-direction:column;align-items:center;justify-content:center;font-size:10px;color:#64748b;text-align:center;line-height:1.2;">📷<br>角色${escapeHtmlAttr(c.id)}</div>
          </div>
          <div style="display:flex;gap:3px;">
            <label style="flex:1;text-align:center;padding:2px 0;font-size:10px;font-weight:700;border-radius:4px;border:1px solid #10b981;background:rgba(16,185,129,0.18);color:#6ee7b7;cursor:pointer;" title="上传您在 ChatGPT/Gemini 生成好的角色${escapeHtmlAttr(c.id)}定妆照（直接锁脸，跳过云端捏脸）">
              📤传照
              <input type="file" accept="image/*" style="display:none;" onchange="uploadDramaCharPhoto('${escapeHtmlAttr(c.id)}', this)">
            </label>
            <button type="button" onclick="copyCharacterGoldenPrompt('${escapeHtmlAttr(c.id)}', this)" style="flex:1;padding:2px 0;font-size:10px;font-weight:700;border-radius:4px;border:1px solid #f472b6;background:rgba(244,114,182,0.16);color:#fbcfe8;cursor:pointer;" title="一键复制角色${escapeHtmlAttr(c.id)}的【5条黄金标准·定妆照生图提示词】，去 ChatGPT/Gemini 生成标准半身照">📋生图词</button>
          </div>
        </div>
        <!-- 右：角色代号、姓名、专属音色、外貌与服装描述 -->
        <div style="flex:1;min-width:0;display:flex;flex-direction:column;justify-content:space-between;gap:4px;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:4px;flex-wrap:wrap;">
            <div style="display:flex;align-items:center;gap:4px;">
              <span style="font-size:11px;font-weight:900;padding:1px 6px;border-radius:4px;background:#f472b6;color:#020617;">角色${escapeHtmlAttr(c.id)}</span>
              <input type="text" value="${escapeHtmlAttr(c.name || '')}" onchange="dramaCastState[${idx}].name=this.value.trim();syncCastToWorkflow();" placeholder="姓名(如:博哥)" style="width:68px;padding:2px 5px;font-size:11px;font-weight:700;border-radius:4px;border:1px solid #475569;background:#0f172a;color:#f8fafc;">
              ${badgeHtml}
            </div>
            <div style="display:flex;align-items:center;gap:3px;">
              ${c.user_uploaded ? `<button type="button" onclick="deleteDramaCharPhoto('${escapeHtmlAttr(c.id)}')" style="padding:1px 5px;font-size:9.5px;border-radius:3px;border:1px solid #f59e0b;background:transparent;color:#fcd34d;cursor:pointer;" title="取消上传的照片，改由云端 AI 生成定妆照">↺改AI生</button>` : ''}
              <button type="button" onclick="removeDramaCastRole('${escapeHtmlAttr(c.id)}')" style="padding:1px 6px;font-size:10px;border-radius:3px;border:1px solid #ef4444;background:rgba(239,68,68,0.14);color:#fca5a5;cursor:pointer;" title="删除角色${escapeHtmlAttr(c.id)}">✕</button>
            </div>
          </div>
          <select onchange="dramaCastState[${idx}].voice=this.value;syncCastToWorkflow();" style="width:100%;padding:2px 5px;font-size:10.5px;border-radius:4px;border:1px solid #475569;background:#0f172a;color:#fde68a;font-weight:600;">
            ${voiceOpts}
          </select>
          <input type="text" value="${escapeHtmlAttr(c.desc || '')}" onchange="dramaCastState[${idx}].desc=this.value.trim();syncCastToWorkflow();" placeholder="填写角色${escapeHtmlAttr(c.id)}年龄、五官特征、发型发饰、上衣穿搭..." style="width:100%;padding:3px 6px;font-size:10.5px;border-radius:4px;border:1px solid #334155;background:#0f172a;color:#cbd5e1;">
        </div>
      </div>
    `;
  }).join('');
}

function syncCastToWorkflow() {
  if (currentDramaWorkflow) {
    currentDramaWorkflow.cast_list = dramaCastState;
    fetch('/api/drama_workflow_save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({workflow: currentDramaWorkflow}),
    }).catch(() => {});
  }
}

function addDramaCastRole() {
  const usedIds = new Set(dramaCastState.map(c => c.id));
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  let nextId = 'A';
  for (let i = 0; i < letters.length; i++) {
    if (!usedIds.has(letters[i])) {
      nextId = letters[i];
      break;
    }
  }
  const isMaleDefault = (dramaCastState.length % 2 === 1);
  dramaCastState.push({
    id: nextId,
    name: `角色${nextId}`,
    voice: isMaleDefault ? 'ming:灵朗' : 'ming:灵小柔',
    desc: isMaleDefault ? '28岁青年男性，五官立体轮廓分明，沉稳干练穿搭' : '24岁青年女性，五官精致灵动，气质温婉穿搭',
    has_photo: false,
    user_uploaded: false,
  });
  syncCastToWorkflow();
  renderDramaCastCards();
  showGlobalCopyToast(`🎭 已新增「角色${nextId}」定妆卡槽！可为其上传照片、复制生图词或绑定专属音色`);
}

function removeDramaCastRole(roleId) {
  dramaCastState = dramaCastState.filter(c => c.id !== roleId);
  syncCastToWorkflow();
  renderDramaCastCards();
  showGlobalCopyToast(`🗑️ 已删除「角色${roleId}」（当前剩余 ${dramaCastState.length} 个角色）`);
}

function clearAllDramaCastRoles() {
  dramaCastState = [];
  syncCastToWorkflow();
  renderDramaCastCards();
  showGlobalCopyToast('🗑️ 已清空全部预设角色！现在复制编剧指令时将 100% 由您的故事自由决定角色');
}

async function uploadDramaCharPhoto(roleId, inputEl) {
  const file = inputEl && inputEl.files && inputEl.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async () => {
    const b64 = reader.result;
    try {
      const r = await fetch('/api/drama_upload_char_photo', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({role_id: roleId, image_b64: b64}),
      });
      const d = await r.json();
      if (!d.ok) {
        alert('上传定妆照失败: ' + (d.error || '未知错误'));
        return;
      }
      dramaCastPhotoCacheBust = Date.now();
      const item = dramaCastState.find(c => c.id === roleId);
      if (item) {
        item.has_photo = true;
        item.user_uploaded = true;
      }
      syncCastToWorkflow();
      renderDramaCastCards();
      await pollAutoDramaStatus();
      showGlobalCopyToast(`✅ 角色${roleId} 定妆照已上传并锁定为 16:9 基准图！开拍时将跳过AI捏脸，100%使用此照片锁脸`);
    } catch (e) {
      alert('上传定妆照异常: ' + e);
    }
  };
  reader.readAsDataURL(file);
}

async function deleteDramaCharPhoto(roleId) {
  try {
    await fetch('/api/drama_delete_char_photo', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({role_id: roleId}),
    });
    dramaCastPhotoCacheBust = Date.now();
    const item = dramaCastState.find(c => c.id === roleId);
    if (item) {
      item.user_uploaded = false;
    }
    syncCastToWorkflow();
    renderDramaCastCards();
    await pollAutoDramaStatus();
    showGlobalCopyToast(`↺ 已解除角色${roleId} 的上传照片锁定，开拍时将由云端 MiniMax-H3-Image 自动生成定妆照`);
  } catch (e) {}
}

async function copyCharacterGoldenPrompt(roleId, btnEl) {
  const item = roleId ? dramaCastState.find(c => c.id === roleId) : null;
  const roleTitle = item ? `角色${item.id}${item.name ? '（' + item.name + '）' : ''}` : '剧集主角';
  const roleDesc = (item && item.desc && item.desc.trim())
    ? item.desc.trim()
    : '（在此填写角色造型与穿搭，例如：25岁中国古装侠女，乌黑高马尾系绯红发带，身穿正红色织金劲装长衫 / 或：28岁现代豪门男总裁，黑色短发，身穿炭黑色双排扣羊毛西装内搭白衬衫）';
  const visStyle = ((document.getElementById('selAiVisualStyle') || {}).value || '写实真人电影级质感，85mm浅景深，真实细腻皮肤纹理，自然柔和光影').trim();

  const goldenPrompt = `请为我生成一张用于影视分镜锁脸与数字人驱动的【16:9 横屏电影级标准半身定妆照】（如果我同时上传了参考人脸照片，请严格保持参考图中人物的面部五官、脸型轮廓、骨相与身份特征 100% 一致，不要改变长相，仅按以下设定为其换上对应造型与场景）：

【角色名称】：${roleTitle}
【整体画风】：${visStyle}
【角色外貌与服装造型设定】：
${roleDesc}

【必须严格遵守的 5 条摄影与构图黄金标准】：
1. 画幅与景别：16:9 宽屏横幅（Landscape 16:9），正面居中半身胸像构图（Medium Close-up / Waist-up Shot），画面上方完整保留头顶与发饰（严禁切头顶），下方拍至胸口与腰部上方，完整展示上衣领口、肩膀与服装材质细节。
2. 朝向与机位：85mm f/1.4 定焦电影人像镜头，平视机位（Eye-level），人物正对镜头或极轻微侧身（10度以内），双眼清澈有神地直视镜头。
3. 面部布光与无遮挡：采用柔和均匀的电影级正面主光（Soft even key lighting），面部两侧受光均衡，严禁出现大面积阴影、阴阳脸或强逆光黑脸；额头、眉毛、眼睛、鼻梁、脸颊与下巴完整清晰，无碎发挡眼、无手部或道具遮挡面部。
4. 表情与闭合唇形基线：神情自然沉静（或带极轻微的温和浅笑），双唇必须自然闭合（Mouth naturally closed，严禁张嘴或露齿大笑），下颌线清晰自然，真实细腻的皮肤纹理。
5. 纯净单人浅景深背景：画面中严格只有该角色 1 人，背景为柔和虚化的电影级浅景深环境光斑（绝不允许背景中出现任何其他路人、多余人脸、相框边线、字幕或水印）。`;

  let copiedOk = false;
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(goldenPrompt);
      copiedOk = true;
    }
  } catch (e) {}
  if (!copiedOk) {
    try {
      const ta = document.createElement('textarea');
      ta.value = goldenPrompt;
      ta.style.position = 'fixed';
      ta.style.left = '-9999px';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    } catch (e2) {}
  }
  showGlobalCopyToast(`✅ 已复制【${roleTitle} · 5条黄金标准定妆照生图词】到剪贴板！可直接连同人脸照发给 ChatGPT / Gemini 生图`);
  if (btnEl) {
    const oldTxt = btnEl.innerText;
    btnEl.innerText = '✅已复制';
    setTimeout(() => { btnEl.innerText = oldTxt; }, 2500);
  }
}

function toggleAutoDramaDirector() {
  switchMiniMaxMode('drama');
}

async function toggleDramaWorkflowEditor() {
  const drawer = document.getElementById('dramaWorkflowEditorDrawer');
  if (!drawer) return;
  if (drawer.style.display === 'none') {
    drawer.style.display = 'block';
    await loadDramaWorkflowEditor();
  } else {
    drawer.style.display = 'none';
  }
}

function switchDramaWfTab(tab) {
  currentDramaWfTab = tab === 'vid' ? 'vid' : 'img';
  const imgC = document.getElementById('wfNodesImgContainer');
  const vidC = document.getElementById('wfNodesVidContainer');
  const bImg = document.getElementById('btnWfTabImg');
  const bVid = document.getElementById('btnWfTabVid');
  if (imgC) imgC.style.display = currentDramaWfTab === 'img' ? 'flex' : 'none';
  if (vidC) vidC.style.display = currentDramaWfTab === 'vid' ? 'flex' : 'none';
  if (bImg) {
    bImg.style.borderColor = currentDramaWfTab === 'img' ? '#f472b6' : '#475569';
    bImg.style.background = currentDramaWfTab === 'img' ? 'rgba(244,114,182,0.18)' : '#0f172a';
    bImg.style.color = currentDramaWfTab === 'img' ? '#fbcfe8' : '#94a3b8';
  }
  if (bVid) {
    bVid.style.borderColor = currentDramaWfTab === 'vid' ? '#38bdf8' : '#475569';
    bVid.style.background = currentDramaWfTab === 'vid' ? 'rgba(56,189,248,0.18)' : '#0f172a';
    bVid.style.color = currentDramaWfTab === 'vid' ? '#7dd3fc' : '#94a3b8';
  }
}

function escapeHtmlAttr(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function showGlobalCopyToast(msg) {
  let toast = document.getElementById('globalCopyToastBanner');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'globalCopyToastBanner';
    toast.style.cssText = 'position:fixed;top:22px;left:50%;transform:translateX(-50%);z-index:99999;padding:11px 22px;border-radius:999px;background:linear-gradient(135deg,#059669,#10b981);color:#ffffff;font-size:13.5px;font-weight:800;box-shadow:0 10px 30px rgba(16,185,129,0.45);border:1.5px solid #6ee7b7;pointer-events:none;transition:opacity 0.25s,transform 0.25s;opacity:0;';
    document.body.appendChild(toast);
  }
  toast.innerText = msg;
  toast.style.opacity = '1';
  toast.style.transform = 'translateX(-50%) translateY(0)';
  if (window._copyToastTimer) clearTimeout(window._copyToastTimer);
  window._copyToastTimer = setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(-50%) translateY(-8px)';
  }, 3600);
}

async function copyExternalAiScriptPrompt(btnEl) {
  const drawer = document.getElementById('dramaWorkflowEditorDrawer');
  if (drawer && drawer.style.display === 'none') {
    drawer.style.display = 'block';
    await loadDramaWorkflowEditor();
  }
  const ideaBoxVal = ((document.getElementById('inpAiIdeaTheme') || {}).value || '').trim();
  const rawBoxVal = ((document.getElementById('inpDramaRawScript') || {}).value || '').trim();
  const savedWfScript = (currentDramaWorkflow && currentDramaWorkflow.raw_script) ? String(currentDramaWorkflow.raw_script).trim() : '';

  // 智能提取用户填写的原著故事与台词：
  // 1. 若用户在上方紫色框和下方大文本框都写了内容（且下方大文本框不是上一次已保存的旧分镜模板），将两者合并；
  // 2. 若用户直接把完整的故事和台词写在了下方大文本框 (inpDramaRawScript) 中，直接提取大文本框里的全部内容！
  let userStoryPieces = [];
  if (ideaBoxVal) {
    userStoryPieces.push(ideaBoxVal);
  }
  if (rawBoxVal && !rawBoxVal.startsWith('你现在是一位顶级的影视分镜导演')) {
    const isUnmodifiedOldTemplate = (rawBoxVal === savedWfScript) && rawBoxVal.includes('【分镜】') && Boolean(ideaBoxVal);
    if (!isUnmodifiedOldTemplate) {
      userStoryPieces.push(rawBoxVal);
    }
  }
  const userOriginalStory = userStoryPieces.join('\n\n').trim();
  const hasUserStory = Boolean(userOriginalStory);

  const targetDur = ((document.getElementById('selAiTargetDur') || {}).value || '不限总时长（由剧情节奏自然决定镜头数与秒数）').trim();
  const visStyle = ((document.getElementById('selAiVisualStyle') || {}).value || '写实真人电影级质感，85mm浅景深，真实细腻皮肤纹理，自然柔和光影').trim();

  const storySection = hasUserStory
    ? `【用户提供的原著故事、人设与台词原稿（必须 100% 忠实保留，严禁另编其他故事！）】：
====================
${userOriginalStory}
====================

⚠️ 核心改编铁律（请严格遵守）：
1. 你必须 100% 忠实于上方【用户提供的原著故事、人设与台词原稿】进行分镜拆解，严禁脱离原稿另编一个无关的新故事！
2. 原稿中用户已经写好的每一句台词、角色姓名/外貌、关键动作与剧情走向，必须原封不动地保留并精准分配到对应分镜的【对白】与【画面】中！
3. 严禁擅自加戏或捏造原稿中没有写的镜头！不要自作主张添加原稿未提及的片头城市航拍、走廊过渡、无台词反应插镜或片尾拉远镜头！用户原稿写了几段动作和台词，就严格只拆解原稿包含的镜头，100% 忠实于原稿的起止范围！
4. 【画风】一行只允许写摄影机质感、镜头焦段与色彩影调（如：写实真人电影级质感，85mm浅景深，琥珀暖金暗调），严禁在【画风】里塞入具体的室内墙板、桌子、酒杯或冰桶等家具道具（具体场景家具请单独写在【场景】一行），否则会导致室外镜头出现左右分屏假图！`
    : `【创作主题/剧情梗概】：请自由构思一个富有戏剧张力、画面唯美且有情绪递进的高口碑短剧片段`;

  const castSectionTxt = (Array.isArray(dramaCastState) && dramaCastState.length > 0)
    ? `\n【剧组角色卡槽参考（若以下角色与上方【用户提供的原著故事】中的角色不一致且未标注“已上传照片锁定”，请直接以原著故事里的真实角色为准重新编号为 角色A、角色B、角色C...，切勿保留原著里不出场的旧角色！）】：\n${dramaCastState.map(c => `- 角色${c.id}${c.name ? '（' + c.name + (c.user_uploaded ? ' · 已上传照片锁定' : '') + '）' : ''}：${c.desc || '请根据剧情补充外貌与穿搭'} 【绑定音色】${c.voice || 'ming:灵小柔'}`).join('\n')}\n`
    : `\n【角色设定说明】：当前未预设任何角色卡槽，请直接根据上方【用户提供的原著故事】中出现的角色，按 角色A、角色B、角色C... 依次定义外貌穿搭并绑定专属 Ming-3.0 音色。\n`;

  const metaPrompt = `你现在是一位顶级的影视分镜导演与 AI 视频工作流编剧。请根据以下创作要求，为我编写一份可直接导入「EasyColab 多分镜短剧全自动工作流引擎」的完整分镜剧本。

${storySection}

【目标总时长与分镜规模】：${targetDur}（单镜时长可在 1.5秒 ~ 10.0秒 之间根据戏剧节奏自由设定，总时长不限死）
【全局视觉画风】：${visStyle}
${castSectionTxt}
请严格按照以下结构化格式输出（不要省略标签，以便工作流引擎 100% 自动拆解为多角色锁脸定妆卡、首尾关键帧、数字人配音与环境音视频节点）：

【剧名】《在此填写剧名》
【全局设定】
画风：${visStyle}
角色A：（填写第一位出场角色的名字、年龄、脸型五官、发型、服装颜色与材质）
角色B：（填写第二位出场角色的名字、年龄、脸型五官、发型、服装颜色与材质，支持继续列出 角色C / 角色D / 角色E...）

【分镜】
镜头01【时长】2.5秒
【出场角色】A（请填该镜头画面中实际出现的角色代号，如 A 或 A, B 或 B, C, D；纯空镜填“无”；注意单镜头同框人物尽量控制在 1~3 人以内以保证最高锁脸精度）
【运镜】固定镜头 / 缓慢推镜 / 倒退跟拍 / 越肩正反打（写清机位与景别，如全景/中近景/面部特写）
【画面】（详细描述画面中人物的位置、动作起止、表情神态变化、光影与环境细节；若该镜需要明显的前后焦点转移或大跨度连续动作，可用“起幅...；随后...”描述，系统会自动生成首尾双关键帧）
【环境音】（描述该镜头的纯自然环境音效，如雨声、溪水声、市集人群声、脚步声等）
【对白】无（若有角色开口说话，请严格保留用户原稿台词，写如 A："台词内容" 或 B："台词内容" 或 C："台词内容"；支持在台词开头或中间加入 [Dissatisfaction-hnn] 傲娇哼声、[Sigh] 叹气、[Laughter] 轻笑、[Breathing] 呼吸、[Surprise-ah] 惊呼 等真人气口标签）
【音色】（若有对白，请优先使用该角色上方绑定的音色，或从以下官方音色中选填：ming:灵小柔 / ming:灵雨汐_中性 / ming:灵黛嗔_生气 / ming:灵采薇_悲伤 / ming:灵梦 / ming:妩媚妲己 / ming:灵信远_中性 / ming:灵朗 / ming:灵宇 / ming:灵楷 / ming:灵孤影_厌恶 / ming:灵孤影_悲伤 / ming:灵孤影_生气 / ming:甄嬛传_四郎）

镜头02【时长】3.5秒
...（请按原著故事与台词自然展开所有分镜，镜头数量与总秒数不设硬性上限）`;

  let copiedOk = false;
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(metaPrompt);
      copiedOk = true;
    }
  } catch (e) {}
  if (!copiedOk) {
    try {
      const ta = document.createElement('textarea');
      ta.value = metaPrompt;
      ta.style.position = 'fixed';
      ta.style.left = '-9999px';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      copiedOk = document.execCommand('copy');
      document.body.removeChild(ta);
    } catch (e2) {}
  }

  const cleanPreview = userOriginalStory.replace(/\s+/g, ' ').slice(0, 22);
  const toastMsg = hasUserStory
    ? `✅ 已将您的故事与台词（「${cleanPreview}...」共 ${userOriginalStory.length} 字）打包复制到剪贴板！`
    : '📋 已复制「自由构思模式」指令！（如需按您的故事与台词写，请先在左框输入您的故事再点复制）';
  showGlobalCopyToast(toastMsg);

  const bTop = document.getElementById('btnCopyExtAiTop');
  const bDrw = document.getElementById('btnCopyExtAiDrawer');
  if (bTop) {
    bTop.innerText = hasUserStory ? `✅ 已打包复制您的故事 (${userOriginalStory.length}字)！` : '✅ 已复制到剪贴板！';
    bTop.style.background = 'linear-gradient(135deg,#059669,#10b981)';
    bTop.style.borderColor = '#34d399';
    bTop.style.color = '#ffffff';
  }
  if (bDrw) {
    bDrw.innerText = hasUserStory ? `✅ 已将您的故事与台词 (${userOriginalStory.length}字) 复制到剪贴板！` : '✅ 已成功复制到剪贴板！';
    bDrw.style.background = 'linear-gradient(135deg,#059669,#10b981)';
  }
  if (window._extAiBtnResetTimer) clearTimeout(window._extAiBtnResetTimer);
  window._extAiBtnResetTimer = setTimeout(() => {
    if (bTop) {
      bTop.innerText = '🤖 复制外部 AI 编剧指令';
      bTop.style.background = 'rgba(168,85,247,0.18)';
      bTop.style.borderColor = '#a855f7';
      bTop.style.color = '#e9d5ff';
    }
    if (bDrw) {
      bDrw.innerText = '📋 一键复制【发给外部 AI 的编剧指令】';
      bDrw.style.background = 'linear-gradient(135deg,#9333ea,#6366f1)';
    }
  }, 4000);

  const stEl = document.getElementById('dramaStatusText');
  if (stEl) {
    stEl.innerText = toastMsg + ' 请粘贴给外部 AI，再将它生成的分镜剧本粘回左框并点「🔍 智能解析」即可！';
    stEl.style.color = '#34d399';
  }
}

function renderDramaWfEditorTables() {
  if (!currentDramaWorkflow) return;
  const titleInp = document.getElementById('inpDramaProjectTitle');
  const hdrEl = document.getElementById('dramaTitleHeader');
  const sumEl = document.getElementById('dramaDynamicSummaryBox');
  const btnStart = document.getElementById('btnStartAutoDrama');
  const btnRemux = document.getElementById('btnRemuxAutoDrama');
  const imgTasks = currentDramaWorkflow.image_tasks || [];
  const vidTasks = currentDramaWorkflow.video_tasks || [];
  const totalDur = vidTasks.reduce((acc, v) => acc + (parseFloat(v.duration) || 3.0), 0).toFixed(1);
  const totalSteps = imgTasks.length + vidTasks.length + 1;
  const projTitle = currentDramaWorkflow.title || '《未命名短剧》';
  if (titleInp) titleInp.value = projTitle;
  if (hdrEl) {
    hdrEl.innerText = `🎬 通用分镜短剧全自动工作流引擎 · ${projTitle} (${vidTasks.length} 镜 · ${totalDur} 秒)`;
  }
  if (btnStart && !btnStart.disabled) {
    btnStart.innerText = `🚀 一键全自动拍摄 ${projTitle} (${vidTasks.length}镜 · ${totalDur}秒 · 共 ${totalSteps} 步)`;
  }
  if (btnRemux) {
    btnRemux.innerText = `🎞️ 仅用已有分镜重新合成成片 (${totalDur}s)`;
  }
  if (sumEl) {
    const avList = vidTasks.filter(v => v.engine === 'avatar');
    const flList = vidTasks.filter(v => v.engine !== 'avatar');
    const durBreakdown = vidTasks.map(v => `${(parseFloat(v.duration) || 3.0).toFixed(1)}s`).join(' + ');
    sumEl.innerHTML = `
      <div><b>🤖 当前工作流四阶段全自动执行蓝图（${projTitle} · 共 ${vidTasks.length} 镜 · 总时长 ${totalDur} 秒 · 总计 ${totalSteps} 步）：</b></div>
      <div>① <b>阶段一 & 二 (MiniMax-H3-Image)</b>：自动生成角色锁脸定妆卡与各镜首尾帧，共 <b>${imgTasks.length} 张原生 16:9 / 9:16 锁脸关键帧</b>；</div>
      <div>② <b>阶段三-A (LongCat-Avatar-1.6 + Ming-3.0)</b>：自动切换数字人引擎，生成 <b>${avList.length} 个口型对白镜头</b>${avList.length ? '（' + avList.map(v => v.label.split(':')[0]).join('、') + '）' : '（当前剧本无特写对白镜头）'}；</div>
      <div>③ <b>阶段三-B (MiniMax-H3 FL2VA)</b>：自动切换首尾帧有声视频引擎，按各镜时长动态采样生成 <b>${flList.length} 个动作与原生环境音镜头</b>；</div>
      <div>④ <b>阶段四 (FFmpeg 自动精剪混音合片)</b>：自动按各镜时长（<code>${durBreakdown} = ${totalDur}s</code>）0.1秒级精准卡点并将环境底噪铺入对白背景，输出完整成片至 <code>~/Movies/EasyColab/</code>！</div>
    `;
  }
  const cImgEl = document.getElementById('wfImgCount');
  const cVidEl = document.getElementById('wfVidCount');
  if (cImgEl) cImgEl.innerText = String(imgTasks.length);
  if (cVidEl) cVidEl.innerText = String(vidTasks.length);

  const imgC = document.getElementById('wfNodesImgContainer');
  if (imgC) {
    imgC.innerHTML = imgTasks.map((it, idx) => `
      <div style="padding:7px 9px;border-radius:7px;background:#0f172a;border:1px solid #334155;display:flex;flex-direction:column;gap:5px;">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap;">
          <div style="display:flex;align-items:center;gap:6px;flex:1;min-width:180px;">
            <span style="font-size:11px;font-weight:800;color:#f472b6;">#${idx + 1}</span>
            <input type="text" value="${escapeHtmlAttr(it.label)}" onchange="currentDramaWorkflow.image_tasks[${idx}].label=this.value" style="padding:2px 6px;font-size:11.5px;font-weight:700;border-radius:4px;border:1px solid #475569;background:#1e293b;color:#f8fafc;width:160px;" title="节点名称">
            <input type="text" value="${escapeHtmlAttr(it.file)}" onchange="currentDramaWorkflow.image_tasks[${idx}].file=this.value" style="padding:2px 6px;font-size:11px;border-radius:4px;border:1px solid #334155;background:#020617;color:#94a3b8;width:125px;" title="保存文件名">
          </div>
          <div style="display:flex;align-items:center;gap:5px;">
            <span style="font-size:10.5px;color:#94a3b8;">参考图:</span>
            <input type="text" value="${escapeHtmlAttr((it.refs || []).join(','))}" onchange="currentDramaWorkflow.image_tasks[${idx}].refs=this.value.split(',').map(s=>s.trim()).filter(Boolean)" placeholder="如 char_A_half.jpg" style="padding:2px 6px;font-size:10.5px;border-radius:4px;border:1px solid #475569;background:#1e293b;color:#cbd5e1;width:165px;" title="锁脸/锁道具参考图文件名 (多个用英文逗号分隔)">
            <button type="button" onclick="rerunSingleDramaStep('${escapeHtmlAttr(it.file)}')" style="padding:2px 7px;font-size:10.5px;border-radius:4px;border:1px solid #f59e0b;background:rgba(245,158,11,0.16);color:#fcd34d;cursor:pointer;" title="单独重新生成此张图片">🔄 重抽此图</button>
            <button type="button" onclick="deleteDramaWorkflowNode('img', ${idx})" style="padding:2px 6px;font-size:10.5px;border-radius:4px;border:1px solid #ef4444;background:transparent;color:#fca5a5;cursor:pointer;" title="删除此节点">✕</button>
          </div>
        </div>
        <textarea rows="2" onchange="currentDramaWorkflow.image_tasks[${idx}].prompt=this.value" style="width:100%;padding:4px 7px;border-radius:5px;border:1px solid #334155;background:#020617;color:#e2e8f0;font-size:11px;line-height:1.4;resize:vertical;">${escapeHtmlAttr(it.prompt)}</textarea>
      </div>
    `).join('');
  }

  const vidC = document.getElementById('wfNodesVidContainer');
  if (vidC) {
    vidC.innerHTML = vidTasks.map((vt, idx) => `
      <div style="padding:7px 9px;border-radius:7px;background:#0f172a;border:1px solid #334155;display:flex;flex-direction:column;gap:5px;">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap;">
          <div style="display:flex;align-items:center;gap:5px;flex-wrap:wrap;">
            <span style="font-size:11px;font-weight:800;color:#38bdf8;">镜#${idx + 1}</span>
            <input type="text" value="${escapeHtmlAttr(vt.label)}" onchange="currentDramaWorkflow.video_tasks[${idx}].label=this.value" style="padding:2px 6px;font-size:11.5px;font-weight:700;border-radius:4px;border:1px solid #475569;background:#1e293b;color:#f8fafc;width:150px;">
            <span style="font-size:10.5px;color:#94a3b8;">时长(秒):</span>
            <input type="number" step="0.1" min="0.5" max="15" value="${vt.duration || 3.0}" onchange="currentDramaWorkflow.video_tasks[${idx}].duration=parseFloat(this.value)||3.0" style="padding:2px 5px;font-size:11px;border-radius:4px;border:1px solid #475569;background:#1e293b;color:#fde68a;width:52px;font-weight:700;">
            <select onchange="currentDramaWorkflow.video_tasks[${idx}].engine=this.value;renderDramaWfEditorTables()" style="padding:2px 6px;font-size:11px;border-radius:4px;border:1px solid #475569;background:#1e293b;color:#f8fafc;">
              <option value="fl2va" ${vt.engine === 'fl2va' ? 'selected' : ''}>🎬 MiniMax-H3 FL2VA (首尾帧动作+环境音)</option>
              <option value="avatar" ${vt.engine === 'avatar' ? 'selected' : ''}>🎭 LongCat-Avatar + Ming-3.0 (特写口型对白)</option>
            </select>
          </div>
          <div style="display:flex;align-items:center;gap:5px;">
            <span style="font-size:10.5px;color:#94a3b8;">首帧:</span>
            <input type="text" value="${escapeHtmlAttr(vt.first_img || '')}" onchange="currentDramaWorkflow.video_tasks[${idx}].first_img=this.value.trim()" style="padding:2px 5px;font-size:10.5px;border-radius:4px;border:1px solid #475569;background:#1e293b;color:#cbd5e1;width:105px;" title="起幅首帧图片文件名">
            <span style="font-size:10.5px;color:#94a3b8;">尾帧:</span>
            <input type="text" value="${escapeHtmlAttr(vt.last_img || '')}" onchange="currentDramaWorkflow.video_tasks[${idx}].last_img=this.value.trim()" placeholder="可选" style="padding:2px 5px;font-size:10.5px;border-radius:4px;border:1px solid #475569;background:#1e293b;color:#cbd5e1;width:95px;" title="落幅尾帧图片文件名 (留空则单首帧)">
            <button type="button" onclick="rerunSingleDramaStep('${escapeHtmlAttr(vt.file)}')" style="padding:2px 7px;font-size:10.5px;border-radius:4px;border:1px solid #38bdf8;background:rgba(56,189,248,0.16);color:#7dd3fc;cursor:pointer;" title="单独重新生成此镜视频">🎬 重跑此镜</button>
            <button type="button" onclick="deleteDramaWorkflowNode('vid', ${idx})" style="padding:2px 6px;font-size:10.5px;border-radius:4px;border:1px solid #ef4444;background:transparent;color:#fca5a5;cursor:pointer;">✕</button>
          </div>
        </div>
        ${vt.engine === 'avatar' ? `
        <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
          <span style="font-size:10.5px;color:#fcd34d;font-weight:700;">💬 台词与音色:</span>
          <input type="text" value="${escapeHtmlAttr(vt.tts_voice || 'ming:灵小柔')}" onchange="currentDramaWorkflow.video_tasks[${idx}].tts_voice=this.value.trim()" style="padding:2px 6px;font-size:10.5px;border-radius:4px;border:1px solid #f59e0b;background:#1e293b;color:#fde68a;width:115px;" title="Ming-3.0 音色 (如 ming:灵小柔 / ming:灵逍遥)">
          <input type="text" value="${escapeHtmlAttr(vt.tts_text || '')}" onchange="currentDramaWorkflow.video_tasks[${idx}].tts_text=this.value" placeholder="输入该镜角色要说的台词 (支持 [Dissatisfaction-hnn] [Breathing] [Sigh] 气口)" style="padding:2px 8px;font-size:11px;border-radius:4px;border:1px solid #f59e0b;background:#020617;color:#fef08a;flex:1;min-width:200px;">
        </div>
        ` : ''}
        <textarea rows="2" onchange="currentDramaWorkflow.video_tasks[${idx}].prompt=this.value" style="width:100%;padding:4px 7px;border-radius:5px;border:1px solid #334155;background:#020617;color:#e2e8f0;font-size:11px;line-height:1.4;resize:vertical;">${escapeHtmlAttr(vt.prompt)}</textarea>
      </div>
    `).join('');
  }
}

async function loadDramaWorkflowEditor() {
  try {
    const r = await fetch('/api/drama_workflow_get');
    const d = await r.json();
    if (d && d.ok && d.workflow) {
      currentDramaWorkflow = d.workflow;
      if (Array.isArray(d.workflow.cast_list)) {
        dramaCastState = d.workflow.cast_list;
      }
      dramaCastPhotoCacheBust = Date.now();
      renderDramaCastCards();
      renderDramaWfEditorTables();
    }
  } catch (e) {}
}

function addDramaWorkflowNode() {
  if (!currentDramaWorkflow) return;
  if (currentDramaWfTab === 'img') {
    const n = (currentDramaWorkflow.image_tasks || []).length + 1;
    currentDramaWorkflow.image_tasks.push({
      file: `custom_img_${n}.jpg`,
      label: `📷 自定义关键帧 #${n}`,
      refs: ['char_A_half.jpg'],
      prompt: 'Keep the exact face of <Picture 1>. 写实古装剧电影级画面，柔和阳光浅景深。',
    });
  } else {
    const n = (currentDramaWorkflow.video_tasks || []).length + 1;
    currentDramaWorkflow.video_tasks.push({
      file: `shot${String(n).padStart(2, '0')}_raw.mp4`,
      label: `🎞️ 镜${String(n).padStart(2, '0')}(3.0s):新分镜`,
      engine: 'fl2va',
      duration: 3.0,
      first_img: `shot01_first.jpg`,
      last_img: '',
      tts_text: '',
      tts_voice: 'ming:灵小柔',
      mix_ambience: false,
      prompt: '固定镜头，画面自然生动，电影级光影与真实环境音效，无背景音乐。',
    });
  }
  renderDramaWfEditorTables();
}

function deleteDramaWorkflowNode(kind, idx) {
  if (!currentDramaWorkflow) return;
  if (kind === 'img') {
    currentDramaWorkflow.image_tasks.splice(idx, 1);
  } else {
    currentDramaWorkflow.video_tasks.splice(idx, 1);
  }
  renderDramaWfEditorTables();
}

async function saveDramaWorkflowEdits() {
  if (!currentDramaWorkflow) return;
  const titleInp = document.getElementById('inpDramaProjectTitle');
  if (titleInp && titleInp.value.trim()) {
    currentDramaWorkflow.title = titleInp.value.trim();
  }
  const stEl = document.getElementById('dramaStatusText');
  try {
    const r = await fetch('/api/drama_workflow_save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({workflow: currentDramaWorkflow}),
    });
    const d = await r.json();
    if (d.ok) {
      renderDramaWfEditorTables();
      await pollAutoDramaStatus();
      if (stEl) {
        stEl.innerText = '💾 工作流节点修改已保存！点击「🚀 一键全自动拍摄」或任意节点的「🔄 重抽」即可按最新配置执行';
        stEl.style.color = '#34d399';
      }
    }
  } catch (e) {}
}

async function parseRawScriptToWorkflow() {
  const rawEl = document.getElementById('inpDramaRawScript');
  const titleInp = document.getElementById('inpDramaProjectTitle');
  const clearOld = Boolean(((document.getElementById('chkDramaClearOldAssets') || {}).checked) ?? true);
  const stEl = document.getElementById('dramaStatusText');
  const rawScript = (rawEl ? rawEl.value : '').trim();
  if (!rawScript) {
    alert('请先在左侧文本框粘贴您的分镜剧本内容！');
    return;
  }
  if (stEl) {
    stEl.innerText = '🔍 正在智能解析分镜剧本并拆解为角色定妆卡、首尾帧节点与视频分流节点...';
    stEl.style.color = '#38bdf8';
  }
  try {
    const r = await fetch('/api/drama_workflow_parse', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        raw_script: rawScript,
        title: (titleInp ? titleInp.value : '').trim(),
        clear_old_assets: clearOld,
        cast_list: dramaCastState,
      }),
    });
    const d = await r.json();
    if (!d.ok) {
      if (stEl) {
        stEl.innerText = '❌ 剧本解析失败: ' + (d.error || '未知错误');
        stEl.style.color = '#ef4444';
      }
      return;
    }
    currentDramaWorkflow = d.workflow;
    if (Array.isArray(d.workflow.cast_list) && d.workflow.cast_list.length > 0) {
      dramaCastState = d.workflow.cast_list;
    }
    renderDramaCastCards();
    renderDramaWfEditorTables();
    await pollAutoDramaStatus();
    if (stEl) {
      stEl.innerText = d.message || '✅ 剧本已成功解析为工作流节点！可在上方查看或直接点击「🚀 一键全自动拍摄」';
      stEl.style.color = '#34d399';
    }
  } catch (e) {
    if (stEl) {
      stEl.innerText = '❌ 解析异常: ' + e;
      stEl.style.color = '#ef4444';
    }
  }
}

async function resetDefaultDramaWorkflow() {
  try {
    const r = await fetch('/api/drama_workflow_parse', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({reset_default: true}),
    });
    const d = await r.json();
    if (d.ok && d.workflow) {
      currentDramaWorkflow = d.workflow;
      renderDramaWfEditorTables();
      await pollAutoDramaStatus();
    }
  } catch (e) {}
}

async function rerunSingleDramaStep(targetFile) {
  if (!targetFile) return;
  if (currentDramaWorkflow) {
    await saveDramaWorkflowEdits();
  }
  const aspect = ((document.getElementById('dramaAspectSelect') || {}).value || '16:9').trim();
  const stEl = document.getElementById('dramaStatusText');
  if (stEl) {
    stEl.innerText = `🔄 正在单独重跑节点: ${targetFile}...`;
    stEl.style.color = '#fbbf24';
  }
  try {
    const r = await fetch('/api/drama_workflow_rerun_one', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({target_file: targetFile, aspect}),
    });
    const d = await r.json();
    if (!d.ok) {
      if (stEl) {
        stEl.innerText = '❌ ' + (d.error || '单步重跑启动失败');
        stEl.style.color = '#ef4444';
      }
      return;
    }
    await pollAutoDramaStatus();
    if (!dramaPollTimer) {
      dramaPollTimer = setInterval(pollAutoDramaStatus, 2000);
    }
  } catch (e) {}
}

function renderDramaStepsGrid(stepsList) {
  const grid = document.getElementById('dramaStepsGrid');
  if (!grid || !Array.isArray(stepsList)) return;
  grid.innerHTML = stepsList.map(item => {
    const st = item.status || 'pending';
    let bg = 'rgba(30,41,59,0.65)';
    let bd = '#334155';
    let col = '#94a3b8';
    let icon = '⏳';
    if (st === 'done') {
      bg = 'rgba(16,185,129,0.14)';
      bd = 'rgba(16,185,129,0.45)';
      col = '#6ee7b7';
      icon = '✅';
    } else if (st === 'running') {
      bg = 'rgba(245,158,11,0.22)';
      bd = '#f59e0b';
      col = '#fde68a';
      icon = '🎬';
    } else if (st === 'error') {
      bg = 'rgba(239,68,68,0.18)';
      bd = '#ef4444';
      col = '#fca5a5';
      icon = '❌';
    }
    const canRerun = item.file && item.file !== 'ancient_drama_30s_final.mp4';
    return `<div style="padding:5px 7px;border-radius:6px;background:${bg};border:1px solid ${bd};color:${col};display:flex;align-items:center;justify-content:space-between;gap:4px;white-space:nowrap;overflow:hidden;" title="${escapeHtmlAttr(item.label || '')} (${escapeHtmlAttr(item.file || '')})">
      <div style="display:flex;align-items:center;gap:4px;overflow:hidden;text-overflow:ellipsis;">
        <span>${icon}</span>
        <span style="overflow:hidden;text-overflow:ellipsis;font-weight:${st === 'running' ? '700' : '500'};">${item.label || ''}</span>
      </div>
      ${canRerun ? `<button type="button" onclick="rerunSingleDramaStep('${escapeHtmlAttr(item.file)}')" style="padding:1px 4px;font-size:10px;border-radius:3px;border:1px solid rgba(148,163,184,0.3);background:rgba(15,23,42,0.6);color:#cbd5e1;cursor:pointer;flex-shrink:0;" title="单独重抽此步">🔄</button>` : ''}
    </div>`;
  }).join('');
}

async function startAutoDramaPipeline() {
  const aspect = ((document.getElementById('dramaAspectSelect') || {}).value || '16:9').trim();
  const resume = Boolean(((document.getElementById('chkDramaResume') || {}).checked) ?? true);
  const auto2k = Boolean(((document.getElementById('chkDramaAuto2K') || {}).checked) ?? false);
  const stEl = document.getElementById('dramaStatusText');
  const btnStart = document.getElementById('btnStartAutoDrama');
  const btnStop = document.getElementById('btnStopAutoDrama');
  if (btnStart) btnStart.disabled = true;
  if (stEl) {
    stEl.innerText = '🚀 正在启动 11 镜 30 秒古装短剧全自动导演流水线...';
    stEl.style.color = '#fbbf24';
  }
  try {
    const r = await fetch('/api/drama_auto_start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({aspect, resume, auto_2k: auto2k}),
    });
    const d = await r.json();
    if (!d.ok) {
      if (stEl) {
        stEl.innerText = '❌ ' + (d.error || '启动失败');
        stEl.style.color = '#ef4444';
      }
      if (btnStart) btnStart.disabled = false;
      return;
    }
    if (btnStop) btnStop.style.display = 'inline-block';
    await pollAutoDramaStatus();
    if (!dramaPollTimer) {
      dramaPollTimer = setInterval(pollAutoDramaStatus, 2000);
    }
  } catch (e) {
    if (stEl) {
      stEl.innerText = '❌ 请求异常: ' + e;
      stEl.style.color = '#ef4444';
    }
    if (btnStart) btnStart.disabled = false;
  }
}

async function stopAutoDramaPipeline() {
  const stEl = document.getElementById('dramaStatusText');
  if (stEl) {
    stEl.innerText = '🛑 正在发送暂停信号 (已生成的定妆照与分镜将完整保留，下次点击可直接断点续拍)...';
    stEl.style.color = '#f59e0b';
  }
  try {
    await fetch('/api/drama_auto_stop', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
    await pollAutoDramaStatus();
  } catch (e) {}
}

async function remuxAutoDramaOnly() {
  const aspect = ((document.getElementById('dramaAspectSelect') || {}).value || '16:9').trim();
  const auto2k = Boolean(((document.getElementById('chkDramaAuto2K') || {}).checked) ?? false);
  const stEl = document.getElementById('dramaStatusText');
  if (stEl) {
    stEl.innerText = '🎞️ 正在使用已生成的 11 个分镜执行 0.1 秒级卡点裁切 + 市集环境底噪混音合成 30.0 秒成片...';
    stEl.style.color = '#38bdf8';
  }
  try {
    const r = await fetch('/api/drama_auto_remux', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({aspect, auto_2k: auto2k}),
    });
    const d = await r.json();
    await pollAutoDramaStatus();
    await loadLocalVideoHistory(true);
    if (!d.ok && stEl) {
      stEl.innerText = '❌ ' + (d.error || '合片失败');
      stEl.style.color = '#ef4444';
    }
  } catch (e) {
    if (stEl) {
      stEl.innerText = '❌ 合片请求异常: ' + e;
      stEl.style.color = '#ef4444';
    }
  }
}

async function pollAutoDramaStatus() {
  const stEl = document.getElementById('dramaStatusText');
  const pBar = document.getElementById('dramaProgressBar');
  const btnStart = document.getElementById('btnStartAutoDrama');
  const btnStop = document.getElementById('btnStopAutoDrama');
  const tLabel = document.getElementById('taskProgressLabel');
  const tSpeed = document.getElementById('taskProgressSpeed');
  const tBar = document.getElementById('taskProgressBar');
  const dState = document.getElementById('deployStateText');
  try {
    const r = await fetch('/api/drama_auto_status');
    const d = await r.json();
    await pollLogs();
    if (Array.isArray(d.steps)) {
      renderDramaStepsGrid(d.steps);
    }
    const pct = Math.max(0, Math.min(100, parseInt(d.pct || 0, 10)));
    if (pBar) pBar.style.width = `${pct}%`;
    const hdrEl = document.getElementById('dramaTitleHeader');
    if (hdrEl && d.title && d.shot_count && d.total_duration) {
      hdrEl.innerText = `🎬 通用分镜短剧全自动工作流引擎 · ${d.title} (${d.shot_count} 镜 · ${d.total_duration} 秒)`;
    }
    const btnRemux = document.getElementById('btnRemuxAutoDrama');
    if (btnRemux && d.total_duration) {
      btnRemux.innerText = `🎞️ 仅用已有分镜重新合成成片 (${d.total_duration}s)`;
    }
    if (d.running) {
      if (videoPollTimer) {
        clearInterval(videoPollTimer);
        videoPollTimer = null;
      }
      liveVideoElapsedSec = null;
      liveVideoEtaSec = null;
      liveVideoPct = null;
      liveVideoBaseMsg = '';
      if (!window._dramaRunningSwitched && currentMiniMaxMode !== 'drama') {
        window._dramaRunningSwitched = true;
        switchMiniMaxMode('drama');
      }
      if (btnStart) {
        btnStart.disabled = true;
        btnStart.innerText = `🎬 正在全自动拍摄中 (${d.done_count || 0}/${d.total_count || 26} 步 · 总进度 ${pct}%)...`;
      }
      if (btnStop) btnStop.style.display = 'inline-block';
      if (stEl && d.message) {
        stEl.innerText = d.message;
        stEl.style.color = '#fde68a';
      }
      if (tLabel) tLabel.innerText = `🎬 全自动短剧拍摄总进度：${pct}%（已完成 ${d.done_count || 0} / ${d.total_count || 26} 步）`;
      if (tSpeed) tSpeed.innerText = `总进度 ${pct}%`;
      if (dState) {
        dState.innerText = `🎬 总进度 ${pct}%`;
        dState.style.color = '#f59e0b';
      }
      if (tBar) {
        tBar.className = 'progress-fill';
        tBar.style.width = `${Math.max(2, pct)}%`;
      }
    } else {
      if (dramaPollTimer) {
        clearInterval(dramaPollTimer);
        dramaPollTimer = null;
      }
      if (d.done_count && d.total_count && d.done_count >= d.total_count) {
        if (tLabel) tLabel.innerText = `✅ 全自动短剧拍摄总进度：100%（全部 ${d.total_count} 步已完成）`;
        if (tSpeed) tSpeed.innerText = `总进度 100%`;
        if (dState) {
          dState.innerText = '✅ 总进度 100%';
          dState.style.color = '#10b981';
        }
        if (tBar) {
          tBar.className = 'progress-fill green';
          tBar.style.width = '100%';
        }
      }
      if (btnStart) {
        btnStart.disabled = false;
        const shotCnt = d.shot_count || (currentDramaWorkflow && currentDramaWorkflow.video_tasks ? currentDramaWorkflow.video_tasks.length : 11);
        const totDur = d.total_duration || (currentDramaWorkflow && currentDramaWorkflow.video_tasks ? currentDramaWorkflow.video_tasks.reduce((a, v) => a + (parseFloat(v.duration) || 3.0), 0).toFixed(1) : '30.0');
        const totSt = d.total_count || 27;
        const pTitle = d.title || (currentDramaWorkflow && currentDramaWorkflow.title) || '当前短剧';
        btnStart.innerText = `🚀 一键全自动拍摄 ${pTitle} (${shotCnt}镜 · ${totDur}秒 · 共 ${totSt} 步)`;
      }
      const btnRemux = document.getElementById('btnRemuxAutoDrama');
      if (btnRemux && d.total_duration) {
        btnRemux.innerText = `🎞️ 仅用已有分镜重新合成成片 (${d.total_duration}s)`;
      }
      const hdrEl = document.getElementById('dramaTitleHeader');
      if (hdrEl && d.title && d.shot_count && d.total_duration) {
        hdrEl.innerText = `🎬 通用分镜短剧全自动工作流引擎 · ${d.title} (${d.shot_count} 镜 · ${d.total_duration} 秒)`;
      }
      if (btnStop) btnStop.style.display = 'none';
      if (d.error) {
        if (stEl) {
          stEl.innerText = '❌ ' + d.error;
          stEl.style.color = '#ef4444';
        }
      } else if (d.message) {
        if (stEl) {
          stEl.innerText = d.message;
          stEl.style.color = '#34d399';
        }
      }
      if (d.final_video) {
        await loadLocalVideoHistory(true);
      }
    }
  } catch (e) {}
}

renderFusionTargetCards();
refreshStatus(false);
loadLocalVideoHistory(true);
(async () => {
  let dramaIsRunning = false;
  try {
    await loadDramaWorkflowEditor();
    await pollAutoDramaStatus();
    const rd = await fetch('/api/drama_auto_status');
    const dd = await rd.json();
    if (dd && Array.isArray(dd.steps)) renderDramaStepsGrid(dd.steps);
    if (dd && dd.running) {
      dramaIsRunning = true;
      if (!dramaPollTimer) {
        dramaPollTimer = setInterval(pollAutoDramaStatus, 2000);
      }
    }
  } catch (e) {}
  if (!dramaIsRunning) {
    try {
      const rv = await fetch('/api/video_status');
      const dv = await rv.json();
      if (dv && dv.running && !videoPollTimer) {
        pollVideoStatus();
        videoPollTimer = setInterval(pollVideoStatus, 2000);
      }
    } catch (e) {}
  }
  try {
    const r = await fetch('/api/video_upscale_status');
    const d = await r.json();
    if (d && d.running && !upscalePollTimer) {
      pollUpscaleStatus();
      upscalePollTimer = setInterval(pollUpscaleStatus, 1500);
    }
  } catch (e) {}
})();
setInterval(pollLogs, 2000);
setInterval(() => refreshStatus(false), 20000);
</script>
</body>
</html>
"""


class PanelRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index.html"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/status"):
            force = "force=1" in self.path
            sessions = _list_active_sessions()
            cfg = _load_endpoint_cfg()
            cu_info = _query_cu_usage(force=force)
            metrics = None
            if sessions:
                now = time.time()
                if _DEPLOY_STATE["running"] or (not force and _LAST_METRICS_CACHE["data"] and (now - _LAST_METRICS_CACHE["timestamp"] < 12)):
                    metrics = _LAST_METRICS_CACHE["data"]
                else:
                    try:
                        metrics = _query_remote_metrics(sessions[0]["name"])
                        _LAST_METRICS_CACHE["data"] = metrics
                        _LAST_METRICS_CACHE["timestamp"] = now
                        cfg = _load_endpoint_cfg()
                    except Exception:
                        metrics = _LAST_METRICS_CACHE["data"]
            idle_sec = int(max(0.0, time.time() - float(cfg.get("last_activity_ts", time.time()) or time.time())))
            acc_info = _get_accounts_status(
                force=force,
                cu_balance=(cu_info or {}).get("paid_balance"),
                eligible_gpus=(cu_info or {}).get("eligible_gpus"),
            )
            self._send_json({
                "sessions": sessions,
                "endpoint_cfg": cfg,
                "cu_info": cu_info,
                "account_info": acc_info,
                "idle_seconds": idle_sec,
                "metrics": metrics,
                "gdrive_info": _query_local_gdrive_info(force=force),
                "custom_models": _load_custom_models(),
            })
            return

        if self.path.startswith("/api/hf_trending"):
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            cat = (qs.get("category", ["all"])[0] or "all").strip()
            q = (qs.get("q", [""])[0] or "").strip()
            force = (qs.get("force", ["0"])[0] == "1")
            items = _fetch_hf_trending(category=cat, query=q, force=force)
            self._send_json({"ok": True, "items": items})
            return

        if self.path.startswith("/api/hf_detail"):
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            repo = (qs.get("repo", [""])[0] or "").strip()
            self._send_json(_fetch_hf_repo_detail(repo))
            return

        if self.path.startswith("/api/logs"):
            with _DEPLOY_LOCK:
                self._send_json({
                    "running": _DEPLOY_STATE["running"],
                    "logs": _DEPLOY_STATE["logs"],
                    "exit_code": _DEPLOY_STATE["exit_code"],
                })
            return

        if self.path.startswith("/api/local_videos"):
            self._send_json({
                "ok": True,
                "dir": str(LOCAL_MOVIES_DIR),
                "items": _list_local_easycolab_videos(),
            })
            return

        if self.path.startswith("/api/local_video_file"):
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            fname = os.path.basename((qs.get("name", [""])[0] or "").strip())
            fpath = LOCAL_MOVIES_DIR / fname
            if not fname or not fpath.is_file():
                self.send_error(404)
                return
            fsize = fpath.stat().st_size
            ext_low = fpath.suffix.lower()
            mime_type = (
                "image/png" if ext_low == ".png"
                else ("image/jpeg" if ext_low in (".jpg", ".jpeg") else ("image/webp" if ext_low == ".webp" else "video/mp4"))
            )
            range_hdr = self.headers.get("Range")
            if range_hdr and range_hdr.startswith("bytes="):
                try:
                    parts = range_hdr.replace("bytes=", "").split("-")
                    start = int(parts[0]) if parts[0] else 0
                    end = int(parts[1]) if (len(parts) > 1 and parts[1]) else (fsize - 1)
                    end = min(end, fsize - 1)
                    length = max(0, end - start + 1)
                    with open(fpath, "rb") as f:
                        f.seek(start)
                        chunk = f.read(length)
                    self.send_response(206)
                    self.send_header("Content-Type", mime_type)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{fsize}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
                    return
                except Exception:
                    pass
            data = fpath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path.startswith("/api/drama_work_file"):
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            fname = os.path.basename((qs.get("name", [""])[0] or "").strip())
            fpath = DRAMA_WORK_DIR / fname
            if not fname or not fpath.is_file():
                self.send_error(404)
                return
            ext_low = fpath.suffix.lower()
            mime_type = (
                "image/png" if ext_low == ".png"
                else ("image/jpeg" if ext_low in (".jpg", ".jpeg") else ("image/webp" if ext_low == ".webp" else "video/mp4"))
            )
            data = fpath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path.startswith("/api/video_status"):
            if _PENDING_VIDEO_JOB.get("active"):
                pct_val = 4
                p_job = _PENDING_VIDEO_JOB.get("payload") or {}
                c_tmp = _load_endpoint_cfg()
                is_img_job = (p_job.get("mode") == "image") or str(c_tmp.get("model_folder") or "") == "MiniMax-H3-Image"
                is_av_job = (p_job.get("mode") == "avatar") or str(c_tmp.get("model_folder") or "").startswith("LongCat-Video-Avatar")
                target_eng = "MiniMax-H3-Image (4K无审查)" if is_img_job else ("LongCat-Video-Avatar-1.6" if is_av_job else "MiniMax-H3")
                stage_msg = f"🔄 正在自动从 Google Drive 云盘切换并装载 {target_eng}，加载完成后将立即自动开始生成..."
                with _DEPLOY_LOCK:
                    logs_snap = list(_DEPLOY_STATE.get("logs") or [])
                for line in reversed(logs_snap):
                    m = re.search(r"\[(HF 下载|云盘极速预热|云盘同步|显存装载)\]\s+([^:]+):\s+(\d+)%", line)
                    if m:
                        st_type, _, st_pct_s = m.group(1), m.group(2), m.group(3)
                        st_pct = int(st_pct_s)
                        pct_val = max(8, min(98, st_pct))
                        stage_msg = f"🔄 正在自动切换加载 {target_eng} ({st_type}: {st_pct}%)，就绪后将立即自动开始生成..."
                        break
                self._send_json({
                    "running": True,
                    "auto_switching": True,
                    "pct": pct_val,
                    "message": stage_msg,
                })
                return
            if _PENDING_VIDEO_JOB.get("error"):
                err_txt = str(_PENDING_VIDEO_JOB.get("error"))
                _PENDING_VIDEO_JOB["error"] = ""
                self._send_json({"running": False, "error": err_txt})
                return

            cfg = _load_endpoint_cfg()
            req = urllib.request.Request(
                "http://127.0.0.1:30000/v1/video/status",
                headers={"Authorization": f"Bearer {cfg.get('api_key', 'sk-colab-ed28e54eb298d0c1')}"},
                method="GET",
            )
            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    d = json.loads(resp.read().decode("utf-8"))
                    is_img_eng = str(cfg.get("model_folder") or "") == "MiniMax-H3-Image"
                    gen_tag = "图片生成" if is_img_eng else "视频生成"
                    done_log_tag = "图片完成" if is_img_eng else "视频完成"
                    if d.get("running"):
                        try:
                            cfg["last_activity_ts"] = time.time()
                            ENDPOINT_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                        except Exception:
                            pass
                        if d.get("message"):
                            _append_panel_log(f"[{gen_tag}] {d.get('message')}")
                        timing_info = _compute_video_timing_from_logs(int(d.get("pct") or 8))
                        d["elapsed_sec"] = timing_info["elapsed_sec"]
                        d["eta_sec"] = timing_info["eta_sec"]
                    else:
                        vids = d.get("completed_videos") or ([d["latest_video"]] if d.get("latest_video") else []) or (d.get("history") or [])[:1]
                        el_s = float(d.get("elapsed_s") or 0.0)
                        per_vid_el = (el_s / max(1, len(vids))) if el_s > 0 else 0.0
                        missing_vids = []
                        for fn in vids:
                            if fn:
                                safe_fn = os.path.basename(str(fn))
                                lp = LOCAL_MOVIES_DIR / safe_fn
                                if not (lp.exists() and lp.stat().st_size > 1024):
                                    missing_vids.append(safe_fn)
                        if missing_vids:
                            if not _VIDEO_DL_STATE.get("active"):
                                def _bg_dl_all(fn_list, el_val):
                                    for f_item in fn_list:
                                        _download_remote_video_to_local(f_item, el_val)
                                threading.Thread(target=_bg_dl_all, args=(missing_vids, per_vid_el), daemon=True).start()
                            dl_pct = int(_VIDEO_DL_STATE.get("pct") or 0)
                            done_m = float(_VIDEO_DL_STATE.get("done_mb") or 0.0)
                            tot_m = float(_VIDEO_DL_STATE.get("tot_mb") or 12.0)
                            dl_el = max(0.5, time.time() - float(_VIDEO_DL_STATE.get("start_ts") or time.time()))
                            spd_mbs = (done_m / dl_el) if done_m >= 0.25 else 1.5
                            d["running"] = True
                            d["downloading_local"] = True
                            d["pct"] = max(95, min(99, 95 + int(dl_pct * 0.04)))
                            media_noun = "4K 写真图片" if is_img_eng else "视频"
                            d["message"] = f"⬇️ 云端{media_noun}已生成！正在回传保存至本机 ~/Movies/EasyColab/ ({done_m:.1f}/{tot_m:.1f} MB, {dl_pct}%)"
                            timing_info = _compute_video_timing_from_logs(int(d["pct"]))
                            d["elapsed_sec"] = timing_info["elapsed_sec"]
                            d["eta_sec"] = max(1, int(round(max(0.0, tot_m - done_m) / max(0.15, spd_mbs))))
                        elif vids:
                            latest_fn = os.path.basename(str(vids[-1]))
                            d["latest_video"] = latest_fn
                            d["pct"] = 100
                            if d.get("message") and not str(d.get("message")).startswith("就绪") and latest_fn not in _LOGGED_COMPLETED_VIDEOS:
                                _LOGGED_COMPLETED_VIDEOS.add(latest_fn)
                                _append_panel_log(f"[{done_log_tag}] {d.get('message')}")
                    self._send_json(d)
            except Exception as e:
                err_s = str(e)
                if "502" in err_s or "503" in err_s or "404" in err_s or "Connection refused" in err_s or "timed out" in err_s:
                    self._send_json({"running": False, "engine_ready": False, "error": ""})
                else:
                    self._send_json({"running": False, "error": err_s})
            return

        if self.path.startswith("/api/drama_auto_status"):
            self._send_json(_get_drama_status_payload())
            return

        if self.path.startswith("/api/drama_workflow_get"):
            self._send_json({"ok": True, "workflow": _load_drama_workflow()})
            return

        if self.path.startswith("/api/video_upscale_status"):
            cfg = _load_endpoint_cfg()
            base_url = _get_recavsr_base_url()
            req = urllib.request.Request(
                f"{base_url}/v1/video/upscale_status",
                headers={
                    "Authorization": f"Bearer {cfg.get('api_key', 'sk-colab-ed28e54eb298d0c1')}",
                    "User-Agent": "ColabPanel/1.0",
                },
                method="GET",
            )
            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    d = json.loads(resp.read().decode("utf-8"))
                    if d.get("running"):
                        try:
                            cfg["last_activity_ts"] = time.time()
                            ENDPOINT_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                        except Exception:
                            pass
                        _LAST_LOGGED_UPSCALE_ERR[0] = ""
                        if d.get("message"):
                            _append_panel_log(f"[视频超分] {d.get('message')}")
                        pct_now = int(d.get("pct") or 10)
                        el_cloud = float(d.get("elapsed_s") or 0.0)
                        now_ts = time.time()
                        if _UPSCALE_TIMING["start_ts"] <= 0.0 or (el_cloud > 0 and abs((now_ts - _UPSCALE_TIMING["start_ts"]) - el_cloud) > 30):
                            _UPSCALE_TIMING["start_ts"] = now_ts - max(1.0, el_cloud)
                        if pct_now > 10 and _UPSCALE_TIMING["compute_start_ts"] <= 0.0:
                            _UPSCALE_TIMING["compute_start_ts"] = now_ts
                        el_now = max(1, int(round(max(el_cloud, now_ts - _UPSCALE_TIMING["start_ts"]))))
                        d["elapsed_sec"] = el_now
                        tot_frames = max(30, int(d.get("total_steps") or 362))
                        target_res = str(d.get("target_res") or "")
                        is_4k = ("3328" in target_res) or (_UPSCALE_TIMING.get("preset") == "4k")
                        is_1080p = ("1108" in target_res or "1080" in target_res) or (_UPSCALE_TIMING.get("preset") == "1080p")
                        sec_per_frame = 0.215 if is_4k else (0.082 if is_1080p else 0.112)
                        est_gpu_total = max(8.0, tot_frames * sec_per_frame)
                        if pct_now <= 12:
                            rem_gpu = est_gpu_total
                        elif pct_now >= 93:
                            rem_gpu = 2.0
                        else:
                            # pct_now 12%..92% 对应 80% 的线性块进度 (Stage 1 扩散 + Stage 2 FlashDecoder)
                            work_frac = max(0.02, min(0.99, (pct_now - 12.0) / 80.0))
                            comp_t0 = _UPSCALE_TIMING["compute_start_ts"] or _UPSCALE_TIMING["start_ts"]
                            comp_el = max(1.0, now_ts - comp_t0)
                            measured_total = comp_el / work_frac
                            w = min(1.0, max(0.0, (pct_now - 14.0) / 20.0))
                            blended_total = w * measured_total + (1.0 - w) * est_gpu_total
                            rem_gpu = max(1.0, blended_total * (1.0 - work_frac))
                        d["eta_sec"] = max(2, int(round(rem_gpu + 4.0)))
                    elif d.get("error"):
                        _UPSCALE_TIMING["start_ts"] = 0.0
                        _UPSCALE_TIMING["compute_start_ts"] = 0.0
                        err_msg = str(d.get("error") or "").strip()
                        if err_msg and _LAST_LOGGED_UPSCALE_ERR[0] != err_msg:
                            _LAST_LOGGED_UPSCALE_ERR[0] = err_msg
                            _append_panel_log(f"❌ [视频超分失败] {err_msg}")
                    else:
                        fn = os.path.basename(str(d.get("latest_video") or "").strip())
                        el_s = float(d.get("elapsed_s") or 0.0)
                        if fn:
                            lp = LOCAL_MOVIES_DIR / fn
                            if not (lp.exists() and lp.stat().st_size > 1024):
                                if not _VIDEO_DL_STATE.get("active"):
                                    threading.Thread(
                                        target=_download_remote_video_to_local,
                                        args=(fn, el_s),
                                        daemon=True,
                                    ).start()
                                dl_pct = int(_VIDEO_DL_STATE.get("pct") or 0)
                                done_m = float(_VIDEO_DL_STATE.get("done_mb") or 0.0)
                                tot_m = float(_VIDEO_DL_STATE.get("tot_mb") or 18.0)
                                dl_start = float(_VIDEO_DL_STATE.get("start_ts") or time.time())
                                dl_el = max(0.5, time.time() - dl_start)
                                spd_mbs = (done_m / dl_el) if done_m >= 0.25 else 1.5
                                d["running"] = True
                                d["downloading_local"] = True
                                d["pct"] = max(95, min(99, 95 + int(dl_pct * 0.04)))
                                d["message"] = f"⬇️ 云端 ReCaVSR 超分已完成 (云端重绘耗时 {el_s:.1f}s)！正在回传保存至本机 ~/Movies/EasyColab/ ({done_m:.1f}/{tot_m:.1f} MB, {dl_pct}%)"
                                if _UPSCALE_TIMING["start_ts"] > 0:
                                    d["elapsed_sec"] = max(int(round(el_s + dl_el)), int(round(time.time() - _UPSCALE_TIMING["start_ts"])))
                                else:
                                    d["elapsed_sec"] = max(1, int(round(el_s + dl_el)))
                                d["eta_sec"] = max(1, int(round(max(0.0, tot_m - done_m) / max(0.15, spd_mbs))))
                            else:
                                _UPSCALE_TIMING["start_ts"] = 0.0
                                _UPSCALE_TIMING["compute_start_ts"] = 0.0
                                d["latest_video"] = fn
                                d["pct"] = 100
                                if d.get("message") and not str(d.get("message")).startswith("就绪") and fn not in _LOGGED_COMPLETED_VIDEOS:
                                    _LOGGED_COMPLETED_VIDEOS.add(fn)
                                    _append_panel_log(f"[视频超分完成] {d.get('message')}")
                    self._send_json(d)
            except Exception as e:
                err_s = str(e)
                if "502" in err_s or "503" in err_s or "404" in err_s or "Connection refused" in err_s or "timed out" in err_s:
                    self._send_json({"running": False, "engine_ready": False, "error": ""})
                else:
                    self._send_json({"running": False, "error": err_s})
            return

        self.send_error(404)

    def do_POST(self):
        cl = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(cl).decode("utf-8") if cl > 0 else "{}"
        try:
            payload = json.loads(raw_body) if raw_body else {}
        except Exception:
            payload = {}

        if self.path == "/api/accounts/notebook_bootstrap":
            try:
                import base64
                rclone_p = BASE_DIR / ".gdrive_rclone.conf"
                rclone_b64 = base64.b64encode(rclone_p.read_bytes()).decode("ascii") if rclone_p.exists() else ""
                cfg = _load_endpoint_cfg()
                model_repo = cfg.get("model_id", "MiniMaxAI/MiniMax-H3")
                model_folder = cfg.get("model_folder", "MiniMax-H3")
                pattern = "FL2VA/*.safetensors" if "minimax" in model_repo.lower() else "*.gguf"
                api_key = cfg.get("api_key", "sk-colab-ed28e54eb298d0c1")
                cell_code = f'''# 🚀 Colab G4 Pro 网页端免验证一键启动脚本 (已自动内置您的共享 Google Drive 134GB 模型库)
import base64, os, re, subprocess, sys, time
if "{rclone_b64}":
    os.makedirs("/root/.config/rclone", exist_ok=True)
    open("/root/.config/rclone/rclone.conf", "wb").write(base64.b64decode("{rclone_b64}"))
    print("✅ 已成功注入共享 Google Drive 模型库凭证 (免重新下载 134GB 模型)")
subprocess.run("curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -qo /tmp/rc.zip -d /tmp && cp /tmp/rclone-*-linux-amd64/rclone /usr/local/bin/rclone && chmod +x /usr/local/bin/rclone", shell=True, check=False)
if not os.path.isdir("/content/colab-g4-studio"):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/wwintj/colab-g4-studio.git", "/content/colab-g4-studio"], check=True)
else:
    subprocess.run(["git", "-C", "/content/colab-g4-studio", "pull", "--ff-only"], check=False)
env = os.environ.copy()
env.update({{"HF_REPO": "{model_repo}", "MODEL_FOLDER": "{model_folder}", "FILE_PATTERN": "{pattern}", "COLAB_API_KEY": "{api_key}"}})
p = subprocess.Popen([sys.executable, "-u", "/content/colab-g4-studio/deploy_colab_model.py"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="", flush=True)
p.wait()
'''
                self._send_json({"ok": True, "code": cell_code})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})
            return

        if self.path == "/api/tts_preview":
            try:
                import base64
                import math
                import tempfile
                tts_text = str(payload.get("text") or "").strip()
                tts_voice_raw = str(payload.get("voice") or "ming:灵雨汐_中性").strip()
                clone_ref_b64 = str(payload.get("clone_ref_b64") or "").strip()
                clone_prompt_text = str(payload.get("clone_prompt_text") or "").strip()
                if not tts_text:
                    self._send_json({"ok": False, "error": "台词内容不能为空"})
                    return

                cfg = _load_endpoint_cfg()
                try:
                    cfg["last_activity_ts"] = time.time()
                    ENDPOINT_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                except Exception:
                    pass

                is_ming_req = tts_voice_raw.startswith("ming:") or bool(clone_ref_b64)
                if is_ming_req:
                    # 1. 若本地保存的 tunnel_url 失效但 Colab 实例在线，自动从实例 /content/cf_tunnel.log 修复最新隧道地址
                    try:
                        from colab_cli.common import state
                        sess_map = state.store.list()
                        if sess_map:
                            s_obj = state.get_session(list(sess_map.keys())[0])
                            if s_obj and getattr(s_obj, "url", None) and getattr(s_obj, "token", None):
                                cf_req = urllib.request.Request(
                                    f"{s_obj.url.rstrip('/')}/api/contents/content/cf_tunnel.log?colab-runtime-proxy-token={urllib.parse.quote(s_obj.token)}",
                                    headers={"X-Colab-Runtime-Proxy-Token": s_obj.token},
                                )
                                with urllib.request.urlopen(cf_req, timeout=5) as cf_r:
                                    cf_txt = json.loads(cf_r.read().decode("utf-8")).get("content") or ""
                                    m_urls = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", cf_txt)
                                    if m_urls and m_urls[-1] != cfg.get("tunnel_url"):
                                        cfg["tunnel_url"] = m_urls[-1]
                                        ENDPOINT_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    except Exception:
                        pass

                    # 2. 请求云端 G4 GPU 上运行的 Ming-3.0-Talker (/v1/audio/ming_tts，MiniMax-H3-Image 与 LongCat-Avatar 均已原生挂载)
                    try:
                        req_m = urllib.request.Request(
                            "http://127.0.0.1:30000/v1/audio/ming_tts",
                            data=json.dumps({
                                "text": tts_text,
                                "voice": tts_voice_raw,
                                "clone_ref_b64": clone_ref_b64,
                                "clone_prompt_text": clone_prompt_text,
                            }).encode("utf-8"),
                            headers={
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {cfg.get('api_key', 'sk-colab-ed28e54eb298d0c1')}",
                            },
                            method="POST",
                        )
                        with urllib.request.urlopen(req_m, timeout=120) as r_m:
                            d_m = json.loads(r_m.read().decode("utf-8"))
                            if d_m.get("ok") and d_m.get("audio_b64"):
                                d_m["note"] = f"🎭 [Ming-3.0 GPU 原生戏剧语音] 已在云端 G4 合成「{tts_voice_raw.replace('ming:', '')}」与真人气口"
                                self._send_json(d_m)
                                return
                    except Exception:
                        pass

                # 3. 本机 Edge-TTS 极速合成（或当云端 G4 实例处于关机休眠状态时的离线预听：将气口标签转为拟声词并为不同人设匹配差异化声线）
                paraling_spoken_map = {
                    "[Breathing]": "，呼…，",
                    "[Cough]": "，咳咳，",
                    "[Sigh]": "，唉…，",
                    "[Laughter]": "，呵呵~，",
                    "[Surprise-ah]": "，啊~！",
                    "[Surprise-oh]": "，噢~！",
                    "[Dissatisfaction-hnn]": "，哼~！",
                    "[Uhm]": "，唔…，",
                    "[Confirmation-en]": "，嗯~，",
                    "[Question-ei]": "，诶？",
                    "[Question-ah]": "，啊？",
                    "[Question-en]": "，嗯？",
                    "[Question-oh]": "，哦？",
                    "[Question-huh]": "，哈？",
                }
                clean_edge_text = tts_text
                for k_tag, v_spoken in paraling_spoken_map.items():
                    clean_edge_text = clean_edge_text.replace(k_tag, v_spoken)
                clean_edge_text = re.sub(r"，{2,}", "，", clean_edge_text).strip("， ") or tts_text

                v_parts = [p.strip() for p in tts_voice_raw.split("|") if p.strip()]
                extra_flags = []
                if is_ming_req:
                    # 离线状态下按人设精准分配不同底模音色与音高/语速，避免离线预听听起来千篇一律
                    if any(k in tts_voice_raw for k in ("妲己", "潘金莲", "绿茶")):
                        tts_voice, extra_flags = "zh-TW-HsiaoChenNeural", ["--pitch=+3Hz", "--rate=-8%"]
                    elif any(k in tts_voice_raw for k in ("灵梦", "灵恬", "开心", "吃货", "小柔")):
                        tts_voice, extra_flags = "zh-CN-XiaoyiNeural", ["--pitch=+7Hz", "--rate=-4%"]
                    elif any(k in tts_voice_raw for k in ("嗔怒", "傲娇", "吃醋")):
                        tts_voice, extra_flags = "zh-CN-XiaoyiNeural", ["--pitch=+5Hz", "--rate=+8%"]
                    elif any(k in tts_voice_raw for k in ("委屈", "破碎", "凄美")):
                        tts_voice, extra_flags = "zh-TW-HsiaoYuNeural", ["--pitch=-2Hz", "--rate=-14%"]
                    elif any(k in tts_voice_raw for k in ("御姐", "女王")):
                        tts_voice, extra_flags = "zh-CN-XiaoxiaoNeural", ["--pitch=-7Hz", "--rate=-6%"]
                    elif any(k in tts_voice_raw for k in ("灵空", "高冷", "心碎")):
                        tts_voice, extra_flags = "zh-CN-YunxiNeural", ["--pitch=-9Hz", "--rate=-8%"]
                    elif any(k in tts_voice_raw for k in ("四郎", "康熙", "雍正", "武松", "震怒", "霸总")):
                        tts_voice, extra_flags = "zh-CN-YunjianNeural", ["--pitch=-6Hz", "--rate=-5%"]
                    elif any(k in tts_voice_raw for k in ("灵宇", "灵朗", "暖男", "唐僧", "余则成", "灵楷")):
                        tts_voice, extra_flags = "zh-CN-YunyangNeural", ["--pitch=-3Hz", "--rate=-4%"]
                    elif "小新" in tts_voice_raw:
                        tts_voice, extra_flags = "zh-CN-YunxiaNeural", ["--pitch=-4Hz", "--rate=-10%"]
                    elif "佩奇" in tts_voice_raw:
                        tts_voice, extra_flags = "zh-CN-XiaoyiNeural", ["--pitch=+12Hz", "--rate=+5%"]
                    elif "Stella" in tts_voice_raw:
                        tts_voice, extra_flags = "en-US-AvaMultilingualNeural", []
                    elif "James" in tts_voice_raw:
                        tts_voice, extra_flags = "en-US-AndrewMultilingualNeural", []
                    else:
                        tts_voice, extra_flags = "zh-CN-XiaoxiaoNeural", ["--pitch=+2Hz", "--rate=-6%"]
                else:
                    tts_voice = v_parts[0] if v_parts else "zh-CN-XiaoyiNeural"
                    for vp in v_parts[1:]:
                        if vp.startswith("pitch="):
                            extra_flags.append(f"--{vp}")
                        elif vp.startswith("rate="):
                            extra_flags.append(f"--{vp}")
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
                    tmp_mp3 = tf.name
                subprocess.run(
                    [str(PYTHON_BIN), "-m", "edge_tts", "--voice", tts_voice, *extra_flags, "--text", clean_edge_text, "--write-media", tmp_mp3],
                    check=True,
                    timeout=30,
                )
                mp3_bytes = Path(tmp_mp3).read_bytes()
                dur_sec = round(len(mp3_bytes) / 6000.0, 1)
                try:
                    dur_out = subprocess.check_output(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", tmp_mp3],
                        text=True,
                        timeout=5,
                    ).strip()
                    dur_sec = round(float(dur_out), 1)
                except Exception:
                    pass
                try:
                    os.unlink(tmp_mp3)
                except Exception:
                    pass
                auto_segs = 1 if dur_sec <= 3.85 else max(1, min(200, 1 + int(math.ceil((dur_sec - 3.72) / 3.20))))
                b64_str = "data:audio/mp3;base64," + base64.b64encode(mp3_bytes).decode("ascii")
                note_str = (
                    f"已在本机极速预听台词节奏（点击下方「🎬 生成数字人视频」时，将在云端 G4 GPU 上自动调用 {tts_voice_raw} 合成满血戏剧/克隆语音）"
                    if is_ming_req
                    else ""
                )
                self._send_json({
                    "ok": True,
                    "audio_b64": b64_str,
                    "audio_ext": ".mp3",
                    "duration_sec": dur_sec,
                    "auto_segments": auto_segs,
                    "used_edge_fallback": is_ming_req,
                    "note": note_str,
                })
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})
            return

        if self.path == "/api/accounts/refresh_all":
            try:
                acc_info = _refresh_all_saved_accounts_cu()
                self._send_json({"ok": True, "account_info": acc_info})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})
            return

        if self.path == "/api/accounts/connect_tunnel":
            raw_in = str(payload.get("tunnel_input") or "").strip()
            urls = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", raw_in)
            if not urls:
                self._send_json({"ok": False, "error": "未识别到有效的 https://xxxx.trycloudflare.com 地址"})
                return
            main_url = urls[0]
            sr_url = urls[1] if len(urls) > 1 else ""
            cfg = _load_endpoint_cfg()
            cfg["tunnel_url"] = main_url
            if sr_url:
                cfg["recavsr_tunnel_url"] = sr_url
            cfg["last_activity_ts"] = time.time()
            ENDPOINT_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            from deploy_colab_model import ensure_local_proxy_running
            ensure_local_proxy_running(30000)
            _append_panel_log(f"⚡ [网页端隧道直连] 已成功接管 Colab G4 实例隧道: {main_url}")
            self._send_json({"ok": True, "tunnel_url": main_url, "recavsr_tunnel_url": sr_url})
            return

        if self.path == "/api/accounts/remote_url":
            try:
                import base64
                import hashlib
                from importlib import resources
                from google_auth_oauthlib.flow import InstalledAppFlow
                from colab_cli.auth import PUBLIC_SCOPES, REMOTE_REDIRECT_URI

                _get_accounts_status(force=True)
                cfg_res = resources.files("colab_cli").joinpath("oauth_config.json")
                client_config = json.loads(cfg_res.read_text())

                light_scopes = [
                    "openid",
                    "https://www.googleapis.com/auth/userinfo.profile",
                    "https://www.googleapis.com/auth/userinfo.email",
                    "https://www.googleapis.com/auth/colaboratory",
                ]
                f_light = InstalledAppFlow.from_client_config(client_config, light_scopes)
                f_light.redirect_uri = REMOTE_REDIRECT_URI
                url_silent_light, _ = f_light.authorization_url()
                verifier = f_light.code_verifier
                challenge = (
                    base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
                    .rstrip(b"=")
                    .decode("ascii")
                )

                f_cli = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
                f_cli.redirect_uri = REMOTE_REDIRECT_URI
                f_cli.code_verifier = verifier
                url_colab_cli, _ = f_cli.authorization_url(
                    code_challenge=challenge,
                    code_challenge_method="S256",
                    prompt="consent",
                    token_usage="remote",
                )

                f_easy = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
                f_easy.redirect_uri = REMOTE_REDIRECT_URI
                f_easy.code_verifier = verifier
                url_easycolab, _ = f_easy.authorization_url(
                    code_challenge=challenge,
                    code_challenge_method="S256",
                    prompt="select_account consent",
                    token_usage="remote",
                )

                pkce_path = os.path.expanduser("~/.config/colab-cli/pkce_verifier.txt")
                os.makedirs(os.path.dirname(pkce_path), exist_ok=True)
                with open(pkce_path, "w", encoding="utf-8") as vf:
                    vf.write(verifier)
                _PENDING_REMOTE_OAUTH["flow"] = f_cli
                if payload.get("open_browser", False):
                    webbrowser.open(url_silent_light)
                self._send_json({
                    "ok": True,
                    "auth_url": url_silent_light,
                    "url_silent_light": url_silent_light,
                    "url_colab_cli": url_colab_cli,
                    "url_easycolab": url_easycolab,
                })
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})
            return

        if self.path == "/api/accounts/remote_exchange":
            raw_code = str(payload.get("code") or "").strip()
            if not raw_code:
                self._send_json({"ok": False, "error": "授权码不能为空"})
                return
            if "code=" in raw_code:
                try:
                    qs = urllib.parse.parse_qs(urllib.parse.urlparse(raw_code).query or raw_code)
                    if qs.get("code"):
                        raw_code = qs["code"][0]
                except Exception:
                    pass
            try:
                from importlib import resources
                import keyring
                from google_auth_oauthlib.flow import InstalledAppFlow
                from colab_cli.auth import KEYRING_SERVICE_NAME, KEYRING_USERNAME, PUBLIC_SCOPES, REMOTE_REDIRECT_URI

                os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
                cfg_res = resources.files("colab_cli").joinpath("oauth_config.json")
                client_config = json.loads(cfg_res.read_text())
                flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
                flow.redirect_uri = REMOTE_REDIRECT_URI
                pkce_path = os.path.expanduser("~/.config/colab-cli/pkce_verifier.txt")
                if os.path.exists(pkce_path):
                    with open(pkce_path, "r", encoding="utf-8") as vf:
                        flow.code_verifier = vf.read().strip()
                elif _PENDING_REMOTE_OAUTH.get("flow") and getattr(_PENDING_REMOTE_OAUTH["flow"], "code_verifier", None):
                    flow.code_verifier = _PENDING_REMOTE_OAUTH["flow"].code_verifier
                flow.fetch_token(code=raw_code)
                creds = flow.credentials
                tok_dict = json.loads(creds.to_json())
                tok_dict.pop("scopes", None)
                token_json = json.dumps(tok_dict)
                keyring.set_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME, token_json)
                tok_file = os.path.expanduser("~/.config/colab-cli/token.json")
                os.makedirs(os.path.dirname(tok_file), exist_ok=True)
                with open(tok_file, "w", encoding="utf-8") as tf:
                    tf.write(token_json)
                _PENDING_REMOTE_OAUTH["flow"] = None
                _reset_colab_cli_state()
                cu_now = _query_cu_usage(force=True)
                acc_now = _get_accounts_status(force=True, cu_balance=(cu_now or {}).get("paid_balance"))
                new_email = acc_now.get("active_email") or ""
                _append_panel_log(f"👤 [授权码绑定] 已成功登录并切换至 Google Colab Pro 账号: {new_email} (余额: {(cu_now or {}).get('paid_balance', '--')} CU)")
                self._send_json({"ok": True, "email": new_email, "account_info": acc_now, "cu_info": cu_now})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})
            return

        if self.path == "/api/accounts/switch":
            target_email = str(payload.get("email") or "").strip()
            switch_drive = bool(payload.get("switch_drive", False))
            acc_map = _load_saved_accounts()
            if not target_email or target_email not in acc_map:
                self._send_json({"ok": False, "error": f"未找到已保存的账号: {target_email}"})
                return
            try:
                import keyring
                from colab_cli.auth import KEYRING_SERVICE_NAME, KEYRING_USERNAME
                info = acc_map[target_email]
                token_json = str(info.get("token_json") or "").strip()
                if not token_json:
                    self._send_json({"ok": False, "error": "该账号的授权凭证为空，请重新点击添加登录"})
                    return
                keyring.set_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME, token_json)
                if switch_drive and info.get("rclone_conf"):
                    (BASE_DIR / ".gdrive_rclone.conf").write_text(str(info["rclone_conf"]), encoding="utf-8")
                    _LAST_GDRIVE_CACHE["data"] = None
                    _LAST_GDRIVE_CACHE["timestamp"] = 0.0
                _reset_colab_cli_state()
                cu_now = _query_cu_usage(force=True)
                acc_now = _get_accounts_status(force=True, cu_balance=(cu_now or {}).get("paid_balance"))
                _append_panel_log(f"👤 [账号切换] 已将当前 Colab Pro 算力账号切换为: {target_email} (余额: {(cu_now or {}).get('paid_balance', '--')} CU)")
                self._send_json({"ok": True, "email": target_email, "account_info": acc_now, "cu_info": cu_now})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})
            return

        if self.path == "/api/accounts/add_colab":
            bind_new_drive = bool(payload.get("bind_new_drive", False))
            try:
                from importlib import resources
                import keyring
                from google_auth_oauthlib.flow import InstalledAppFlow
                from colab_cli.auth import KEYRING_SERVICE_NAME, KEYRING_USERNAME, PUBLIC_SCOPES

                # 先确保当前已登录账号被快照保存下来，避免被冲掉
                _get_accounts_status(force=True)
                os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
                light_scopes = [
                    "openid",
                    "https://www.googleapis.com/auth/userinfo.profile",
                    "https://www.googleapis.com/auth/userinfo.email",
                    "https://www.googleapis.com/auth/colaboratory",
                ]
                cfg_res = resources.files("colab_cli").joinpath("oauth_config.json")
                client_config = json.loads(cfg_res.read_text())
                flow = InstalledAppFlow.from_client_config(client_config, light_scopes)
                creds = flow.run_local_server(
                    host="localhost",
                    port=0,
                    prompt="select_account",
                    open_browser=True,
                    success_message="✅ Google Colab Pro 账号授权成功！您可以直接关闭此标签页并回到 Colab G4 控制台 (http://127.0.0.1:18080)。",
                )
                tok_dict = json.loads(creds.to_json())
                tok_dict.pop("scopes", None)
                token_json = json.dumps(tok_dict)
                keyring.set_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME, token_json)
                _reset_colab_cli_state()

                if bind_new_drive:
                    rclone_bin = BASE_DIR / ".venv" / "bin" / "rclone"
                    if not rclone_bin.exists():
                        rclone_bin = Path(shutil.which("rclone") or "rclone")
                    rclone_conf = BASE_DIR / ".gdrive_rclone.conf"
                    subprocess.run(
                        [str(rclone_bin), "config", "create", "gdrive", "drive", "scope=drive", "--config", str(rclone_conf)],
                        check=False,
                        timeout=120,
                    )
                    _LAST_GDRIVE_CACHE["data"] = None
                    _LAST_GDRIVE_CACHE["timestamp"] = 0.0

                cu_now = _query_cu_usage(force=True)
                acc_now = _get_accounts_status(force=True, cu_balance=(cu_now or {}).get("paid_balance"))
                new_email = acc_now.get("active_email") or ""
                if bind_new_drive and new_email and (BASE_DIR / ".gdrive_rclone.conf").exists():
                    acc_map = _load_saved_accounts()
                    if new_email in acc_map:
                        acc_map[new_email]["rclone_conf"] = (BASE_DIR / ".gdrive_rclone.conf").read_text(encoding="utf-8")
                        _save_saved_accounts(acc_map)
                _append_panel_log(f"👤 [新账号绑定] 已成功登录并切换至 Google Colab Pro 账号: {new_email} (余额: {(cu_now or {}).get('paid_balance', '--')} CU)")
                self._send_json({"ok": True, "email": new_email, "account_info": acc_now, "cu_info": cu_now})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})
            return

        if self.path == "/api/accounts/bind_drive":
            try:
                import shutil
                rclone_bin = BASE_DIR / ".venv" / "bin" / "rclone"
                if not rclone_bin.exists():
                    rclone_bin = Path(shutil.which("rclone") or "rclone")
                rclone_conf = BASE_DIR / ".gdrive_rclone.conf"
                subprocess.run(
                    [str(rclone_bin), "config", "create", "gdrive", "drive", "scope=drive", "--config", str(rclone_conf)],
                    check=True,
                    timeout=120,
                )
                _LAST_GDRIVE_CACHE["data"] = None
                _LAST_GDRIVE_CACHE["timestamp"] = 0.0
                acc_now = _get_accounts_status(force=True)
                act_em = acc_now.get("active_email") or ""
                if act_em and rclone_conf.exists():
                    acc_map = _load_saved_accounts()
                    if act_em in acc_map:
                        acc_map[act_em]["rclone_conf"] = rclone_conf.read_text(encoding="utf-8")
                        _save_saved_accounts(acc_map)
                _query_local_gdrive_info(force=True)
                _append_panel_log("☁️ [云盘绑定] 已重新授权绑定 Google Drive 云盘账号")
                self._send_json({"ok": True, "account_info": _get_accounts_status(force=True)})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})
            return

        if self.path == "/api/accounts/delete":
            em = str(payload.get("email") or "").strip()
            acc_map = _load_saved_accounts()
            if em in acc_map:
                del acc_map[em]
                _save_saved_accounts(acc_map)
            _ACCOUNT_INFO_CACHE["data"] = None
            self._send_json({"ok": True, "account_info": _get_accounts_status(force=True)})
            return

        if self.path == "/api/reveal_in_finder":
            LOCAL_MOVIES_DIR.mkdir(parents=True, exist_ok=True)
            fname = os.path.basename(str(payload.get("name") or "").strip())
            target_path = LOCAL_MOVIES_DIR / fname if fname else LOCAL_MOVIES_DIR
            if fname and target_path.exists():
                subprocess.Popen(["open", "-R", str(target_path)])
            else:
                subprocess.Popen(["open", str(LOCAL_MOVIES_DIR)])
            self._send_json({"ok": True, "path": str(target_path)})
            return

        if self.path == "/api/idle_config":
            cfg = _load_endpoint_cfg()
            if "billing_reset_day" in payload:
                cfg["billing_reset_day"] = max(0, min(31, int(payload.get("billing_reset_day") or 0)))
            if "idle_timeout_minutes" in payload:
                mins = max(0, int(payload.get("idle_timeout_minutes", 30)))
                cfg["idle_timeout_minutes"] = mins
            else:
                mins = int(cfg.get("idle_timeout_minutes", 30))
            cfg["last_activity_ts"] = time.time()
            ENDPOINT_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            msg = f"已设置闲置 {mins} 分钟后自动释放实例" if mins > 0 else "已设为「从不」自动释放（持续保活）"
            self._send_json({"ok": True, "idle_timeout_minutes": mins, "billing_reset_day": cfg.get("billing_reset_day", 0), "message": msg})
            return

        if self.path == "/api/clear_logs":
            with _DEPLOY_LOCK:
                _DEPLOY_STATE["logs"] = [f"[{time.strftime('%H:%M:%S')}] ✅ 日志已清空，等待下一步操作..."]
                _DEPLOY_STATE["exit_code"] = None
                _save_panel_logs_snapshot()
            _PENDING_VIDEO_JOB["active"] = False
            _PENDING_VIDEO_JOB["error"] = ""
            self._send_json({"ok": True})
            return

        if self.path == "/api/custom_models":
            repo = str(payload.get("repo") or "").strip()
            if not repo or "/" not in repo:
                self._send_json({"ok": False, "error": "请输入有效的 HuggingFace 仓库地址 (如 Owner/Repo)"})
                return
            default_name = repo.split("/")[-1]
            name = str(payload.get("name") or default_name).strip() or default_name
            folder = str(payload.get("folder") or default_name).strip() or default_name
            pattern = str(payload.get("pattern") or "*").strip() or "*"
            hf_token = str(payload.get("hf_token") or "").strip()
            if hf_token:
                cfg = _load_endpoint_cfg()
                cfg["hf_token"] = hf_token
                ENDPOINT_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            items = _load_custom_models()
            item_id = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{repo}_{pattern}")[:64]
            new_item = {
                "id": item_id,
                "name": name,
                "repo": repo,
                "folder": folder,
                "pattern": pattern,
                "subtitle": f"{repo.split('/')[0]} · {pattern}",
            }
            items = [x for x in items if x.get("id") != item_id]
            items.append(new_item)
            _save_custom_models(items)
            self._send_json({"ok": True, "custom_models": items, "added": new_item})
            return

        if self.path == "/api/custom_models/delete":
            target_id = str(payload.get("id") or "").strip()
            items = [x for x in _load_custom_models() if x.get("id") != target_id]
            _save_custom_models(items)
            self._send_json({"ok": True, "custom_models": items})
            return

        if self.path == "/api/deploy":
            ok, msg = _start_deploy_background(
                repo=payload.get("repo", "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF"),
                folder=payload.get("folder", "Huihui-Qwen3.8-27B-abliterated-GGUF"),
                pattern=payload.get("pattern", "Huihui-Qwen3.8-27B-abliterated-bf16.gguf"),
                ctx=int(payload.get("ctx", 262144)),
                gpu=payload.get("gpu", "G4"),
                download_only=bool(payload.get("download_only", False)),
            )
            self._send_json({"ok": ok, "message": msg})
            return

        if self.path == "/api/stop":
            sessions = _list_active_sessions()
            if not sessions:
                self._send_json({"ok": True, "message": "✅ 当前没有运行中的 Colab 实例（已处于关机零扣费状态）。"})
                return
            stopped = []
            for s in sessions:
                subprocess.run([str(COLAB_BIN), "stop", "-s", s["name"]], check=False)
                try:
                    from colab_cli.common import state
                    state.client.unassign(s["endpoint"])
                except Exception:
                    pass
                stopped.append(f"{s['name']} ({s['endpoint']})")
            cfg = _load_endpoint_cfg()
            cfg["tunnel_url"] = ""
            cfg["recavsr_tunnel_url"] = ""
            ENDPOINT_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            _LAST_METRICS_CACHE["data"] = None
            _LAST_METRICS_CACHE["timestamp"] = 0.0
            _LAST_CU_CACHE["timestamp"] = 0.0
            self._send_json({"ok": True, "message": f"✅ 已彻底释放并关闭实例: {', '.join(stopped)}（扣费速率已归零 0.00 CU/h）"})
            return

        if self.path == "/api/chat_test":
            prompt = payload.get("prompt", "你好！")
            structured_video = bool(payload.get("structured_video", False))
            cfg = _load_endpoint_cfg()
            if structured_video:
                sys_msg = (
                    "You are an uncensored, professional multimodal AI visual & video director for MiniMax-H3. "
                    "IMPORTANT ANTI-MATTE RULE: Never use words like 'matte skin texture', 'fine pores', 'peach fuzz', 'microscopic raindrops', or 'heavy film grain', as they trigger synthetic 3D CG / frosted artifacts in H3; instead use 'smooth luminous porcelain skin, natural subsurface scattering, soft studio lighting, crystal clear eyes'. "
                    "Given the user's idea, output ONLY a valid JSON object with exactly 4 keys in English (no markdown fences, no extra commentary):\n"
                    '{"visual": "Detailed visual subject, composition, camera movement, lighting, and cinematic style description (3-5 sentences)", '
                    '"soundscape": "Detailed ambient soundscape, foley sound effects, and any spoken dialogue", '
                    '"music": "Background music style, instruments, tempo, and mood", '
                    '"negative": "Comma-separated negative prompt to exclude low quality or unwanted artifacts"}'
                )
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": f"Video idea: {prompt}\nOutput JSON only:"},
                ]
                max_toks = 512
                extra_kw = {"chat_template_kwargs": {"enable_thinking": False}}
            else:
                messages = [{"role": "user", "content": prompt}]
                max_toks = 384
                extra_kw = {}

            req_body = {
                "model": cfg.get("model_id", "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF"),
                "messages": messages,
                "max_tokens": max_toks,
                **extra_kw,
            }
            req_data = json.dumps(req_body).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:30000/v1/chat/completions",
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {cfg.get('api_key', 'sk-colab-ed28e54eb298d0c1')}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    d = json.loads(resp.read().decode("utf-8"))
                    msg = d.get("choices", [{}])[0].get("message", {})
                    reply = (msg.get("content") or msg.get("reasoning_content") or "").strip()
                    timings = d.get("timings", {})
                    tps = round(float(timings.get("predicted_per_second", 0.0)), 1)
                    toks = d.get("usage", {}).get("completion_tokens", 0)
                    structured_obj = None
                    if structured_video and d.get("id") in ("chatcmpl-minimax", "chatcmpl-minimax-t4"):
                        clean_idea = str(prompt or "").strip()
                        structured_obj = {
                            "visual": f"{clean_idea}. Masterpiece cinematic composition, realistic lighting and shadows, rich texture details, smooth 24fps camera movement, 4k film quality.",
                            "soundscape": f"Realistic spatial ambient soundscape and natural foley sound effects matching: {clean_idea}.",
                            "music": "Cinematic atmospheric score matching the mood, emotion, and rhythm of the scene.",
                            "negative": "low quality, worst quality, blurry, watermark, text, signature, deformed, bad anatomy, extra limbs, distorted face, jittery motion",
                        }
                    elif structured_video and reply:
                        cleaned = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
                        m_json = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
                        if m_json:
                            try:
                                parsed = json.loads(m_json.group(0))
                                if isinstance(parsed, dict) and parsed.get("visual"):
                                    structured_obj = {
                                        "visual": str(parsed.get("visual") or "").strip(),
                                        "soundscape": str(parsed.get("soundscape") or "").strip(),
                                        "music": str(parsed.get("music") or "").strip(),
                                        "negative": str(parsed.get("negative") or "low quality, worst quality, blurry, watermark, text, deformed, bad anatomy, jittery motion").strip(),
                                    }
                            except Exception:
                                pass
                        if not structured_obj:
                            structured_obj = {
                                "visual": cleaned,
                                "soundscape": "Natural ambient sound effects matching the scene motion and environment.",
                                "music": "Cinematic atmospheric score matching the mood of the scene.",
                                "negative": "low quality, worst quality, blurry, watermark, text, deformed, bad anatomy, jittery motion",
                            }
                    self._send_json({
                        "reply": reply or "(空回复)",
                        "structured": structured_obj,
                        "speed_tps": tps,
                        "tokens": toks,
                    })
            except Exception as e:
                self._send_json({"error": str(e)})
            return

        if self.path == "/api/video_generate":
            is_t4_account = False
            try:
                cu_chk = _query_cu_usage(force=False)
                if cu_chk.get("paid_balance") is not None and float(cu_chk.get("paid_balance")) <= 0.0:
                    is_t4_account = True
            except Exception:
                pass
            cfg = _load_endpoint_cfg()
            req_mode = str(payload.get("mode") or "t2va").strip()
            active_folder = str(cfg.get("model_folder") or "").strip()
            if req_mode == "image":
                _append_panel_log(
                    f"📸 [提交 4K 写真/换装任务] 分辨率 {payload.get('resolution', '3552x4736')} · 6步蒸馏直出 · 参考图 {len(payload.get('ref_images_b64') or [])}张"
                )
            else:
                _append_panel_log(
                    f"🎬 [提交视频生成任务] 分辨率 {payload.get('resolution', '832x480')} · {payload.get('num_frames', 124)}帧 · {payload.get('steps', 30)}步 · 连抽 {payload.get('batch_count', 1)}条"
                )
            engine_mismatch = (
                (req_mode == "image" and active_folder != "MiniMax-H3-Image")
                or (req_mode == "avatar" and not active_folder.startswith("LongCat-Video-Avatar"))
                or (req_mode not in ("image", "avatar") and (active_folder == "MiniMax-H3-Image" or not active_folder.startswith("MiniMax-H3")))
            )
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:30000/v1/video/generations",
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {cfg.get('api_key', 'sk-colab-ed28e54eb298d0c1')}",
                },
                method="POST",
            )
            need_auto_switch = bool(engine_mismatch)
            if not need_auto_switch:
                try:
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        resp_obj = json.loads(resp.read().decode("utf-8"))
                        if resp_obj.get("ok"):
                            self._send_json(resp_obj)
                            return
                        need_auto_switch = True
                except urllib.error.HTTPError as he:
                    if he.code in (400, 409):
                        try:
                            err_obj = json.loads(he.read().decode("utf-8"))
                            self._send_json(err_obj)
                            return
                        except Exception:
                            pass
                    need_auto_switch = True
                except Exception:
                    need_auto_switch = True

            if need_auto_switch:
                _PENDING_VIDEO_JOB["active"] = True
                _PENDING_VIDEO_JOB["payload"] = payload
                _PENDING_VIDEO_JOB["error"] = ""
                _PENDING_VIDEO_JOB["started_at"] = time.time()
                want_image = (req_mode == "image")
                want_avatar = (req_mode == "avatar") or (not want_image and active_folder.startswith("LongCat-Video-Avatar") and req_mode not in ("t2va", "fl2va"))
                if want_image:
                    target_label = "MiniMax-H3-Image (4K无审查黄金铁三角)"
                elif want_avatar:
                    target_label = "LongCat-Avatar-1.6 (INT8版)" if is_t4_account else "LongCat-Video-Avatar-1.6"
                else:
                    target_label = "MiniMax-H3 (T4 量化版)" if is_t4_account else "MiniMax-H3"
                _append_panel_log(f"🔄 [自动切换引擎] 检测到当前未运行 {target_label}，正在自动加载并在就绪后立即执行任务...")
                with _DEPLOY_LOCK:
                    already_deploying = bool(_DEPLOY_STATE.get("running"))
                if not already_deploying:
                    if want_image:
                        _start_deploy_background(
                            repo="Comfy-Org/MiniMax-H3-Image",
                            folder="MiniMax-H3-Image",
                            pattern="*.safetensors",
                            ctx=262144,
                            gpu="G4",
                            download_only=False,
                        )
                    elif want_avatar:
                        _start_deploy_background(
                            repo="meituan-longcat/LongCat-Video-Avatar-1.6",
                            folder="LongCat-Video-Avatar-1.6-INT8" if is_t4_account else "LongCat-Video-Avatar-1.6",
                            pattern="base_model_int8/*.safetensors" if is_t4_account else "base_model/*.safetensors",
                            ctx=262144,
                            gpu="T4" if is_t4_account else "G4",
                            download_only=False,
                        )
                    elif is_t4_account:
                        _start_deploy_background(
                            repo="unsloth/MiniMax-H3-GGUF",
                            folder="MiniMax-H3-T4-GGUF",
                            pattern="minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf",
                            ctx=262144,
                            gpu="T4",
                            download_only=False,
                        )
                    else:
                        _start_deploy_background(
                            repo="MiniMaxAI/MiniMax-H3",
                            folder="MiniMax-H3",
                            pattern="FL2VA/*.safetensors",
                            ctx=262144,
                            gpu="G4",
                            download_only=False,
                        )

                def _auto_switch_and_submit(job_payload):
                    try:
                        for _ in range(900):
                            time.sleep(2)
                            with _DEPLOY_LOCK:
                                d_run = bool(_DEPLOY_STATE.get("running"))
                                d_code = _DEPLOY_STATE.get("exit_code")
                            if not d_run:
                                if d_code not in (0, None):
                                    _PENDING_VIDEO_JOB["error"] = f"❌ 自动切换加载 {target_label} 失败，请查看下方实时部署日志"
                                    _PENDING_VIDEO_JOB["active"] = False
                                    return
                                break
                        # 视频引擎部署完成，等待 /v1/video/generations 接收任务
                        c_now = _load_endpoint_cfg()
                        body_bytes = json.dumps(job_payload).encode("utf-8")
                        for _ in range(60):
                            try:
                                r2 = urllib.request.Request(
                                    "http://127.0.0.1:30000/v1/video/generations",
                                    data=body_bytes,
                                    headers={
                                        "Content-Type": "application/json",
                                        "Authorization": f"Bearer {c_now.get('api_key', 'sk-colab-ed28e54eb298d0c1')}",
                                    },
                                    method="POST",
                                )
                                with urllib.request.urlopen(r2, timeout=20) as resp2:
                                    res2 = json.loads(resp2.read().decode("utf-8"))
                                    if res2.get("ok"):
                                        done_act_txt = "4K 写真图片" if want_image else ("数字人视频" if want_avatar else "有声视频")
                                        _append_panel_log(f"✅ [自动切换完成] {target_label} 已就绪并自动开始采样生成{done_act_txt}！")
                                        _PENDING_VIDEO_JOB["active"] = False
                                        return
                            except Exception:
                                pass
                            time.sleep(2)
                        _PENDING_VIDEO_JOB["error"] = f"❌ {target_label} 加载后提交视频生成超时，请再点一次「开始生成」"
                        _PENDING_VIDEO_JOB["active"] = False
                    except Exception as ex:
                        _PENDING_VIDEO_JOB["error"] = f"❌ 自动切换异常: {ex}"
                        _PENDING_VIDEO_JOB["active"] = False

                threading.Thread(target=_auto_switch_and_submit, args=(payload,), daemon=True).start()
                self._send_json({
                    "ok": True,
                    "auto_switching": True,
                    "message": f"🔄 检测到当前尚未运行 {target_label}，正在自动加载 {target_label}，就绪后将自动开始生成视频...",
                })
                return

        if self.path == "/api/video_upscale":
            import base64
            cfg = _load_endpoint_cfg()
            fname = os.path.basename(str(payload.get("filename") or "").strip())
            preset = str(payload.get("target_preset") or "2k").lower().strip()
            if not fname:
                self._send_json({"ok": False, "error": "请先选择要超分的视频文件"})
                return
            _UPSCALE_TIMING["start_ts"] = time.time()
            _UPSCALE_TIMING["compute_start_ts"] = 0.0
            _UPSCALE_TIMING["preset"] = preset
            _append_panel_log(
                f"✨ [提交 ReCaVSR 超分任务] 原视频 {fname} → 目标规格 {preset.upper()} (Wan2.2-5B 单步流式重绘 + 原生立体声无损合成)"
            )
            req_payload = {
                "filename": fname,
                "target_preset": preset,
                "color_fix": payload.get("color_fix", "wavelet"),
            }
            base_url = _get_recavsr_base_url()
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.get('api_key', 'sk-colab-ed28e54eb298d0c1')}",
                "User-Agent": "ColabPanel/1.0",
            }
            try:
                local_p = LOCAL_MOVIES_DIR / fname
                chk_req = urllib.request.Request(
                    "http://127.0.0.1:30000/v1/video/status",
                    headers=headers,
                    method="GET",
                )
                remote_hist = []
                try:
                    with urllib.request.urlopen(chk_req, timeout=5) as cr:
                        cd = json.loads(cr.read().decode("utf-8"))
                        remote_hist = cd.get("history") or []
                except Exception:
                    pass
                if fname not in remote_hist and local_p.is_file() and local_p.stat().st_size < 80 * 1024 * 1024:
                    req_payload["video_b64"] = base64.b64encode(local_p.read_bytes()).decode("ascii")

                req = urllib.request.Request(
                    f"{base_url}/v1/video/upscale",
                    data=json.dumps(req_payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    self._send_json(json.loads(resp.read().decode("utf-8")))
                    return
            except urllib.error.HTTPError as he:
                try:
                    self._send_json(json.loads(he.read().decode("utf-8")))
                except Exception:
                    self._send_json({"ok": False, "error": f"HTTP {he.code}: {he}"})
                return
            except Exception as e:
                self._send_json({"ok": False, "error": f"连接云端 ReCaVSR 超分服务失败: {e}"})
                return

        if self.path == "/api/import_external_video":
            import base64
            raw_name = os.path.basename(str(payload.get("filename") or "imported_video.mp4").strip())
            video_b64 = str(payload.get("video_b64") or "").strip()
            if not video_b64:
                self._send_json({"ok": False, "error": "未收到视频文件数据"})
                return
            LOCAL_MOVIES_DIR.mkdir(parents=True, exist_ok=True)
            stem, ext = os.path.splitext(raw_name)
            safe_stem = re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fa5-]", "_", stem).strip("_") or "external_video"
            ext_low = ext.lower() if ext else ".mp4"
            raw_bytes = base64.b64decode(video_b64.split(",", 1)[-1])
            final_name = f"{safe_stem}.mp4"
            final_path = LOCAL_MOVIES_DIR / final_name
            if final_path.exists():
                final_name = f"{safe_stem}_{int(time.time())}.mp4"
                final_path = LOCAL_MOVIES_DIR / final_name
            try:
                if ext_low == ".mp4":
                    final_path.write_bytes(raw_bytes)
                else:
                    tmp_in = LOCAL_MOVIES_DIR / f".tmp_import_{int(time.time())}{ext_low}"
                    tmp_in.write_bytes(raw_bytes)
                    try:
                        subprocess.run(
                            [
                                "ffmpeg", "-y", "-v", "error",
                                "-i", str(tmp_in),
                                "-c:v", "libx264", "-crf", "17", "-preset", "fast",
                                "-c:a", "aac", "-b:a", "192k",
                                "-movflags", "+faststart",
                                str(final_path),
                            ],
                            check=True,
                            timeout=60,
                        )
                    finally:
                        if tmp_in.exists():
                            tmp_in.unlink()
                sz_mb = round(final_path.stat().st_size / (1024 * 1024), 1)
                _append_panel_log(f"📤 [外部视频导入] 已导入 {final_name} ({sz_mb} MB) 至 ~/Movies/EasyColab/")
                self._send_json({"ok": True, "filename": final_name})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})
            return

        if self.path == "/api/extract_last_frame":
            import base64
            fname = os.path.basename(str(payload.get("name") or "").strip())
            fpath = LOCAL_MOVIES_DIR / fname
            if not fname or not fpath.is_file():
                self._send_json({"ok": False, "error": f"本地未找到视频文件: {fname}"})
                return
            try:
                out_png = subprocess.check_output(
                    [
                        "ffmpeg", "-v", "error",
                        "-sseof", "-0.12",
                        "-i", str(fpath),
                        "-frames:v", "1",
                        "-f", "image2pipe",
                        "-vcodec", "png",
                        "-",
                    ],
                    timeout=10,
                )
                if not out_png:
                    out_png = subprocess.check_output(
                        [
                            "ffmpeg", "-v", "error",
                            "-i", str(fpath),
                            "-frames:v", "1",
                            "-f", "image2pipe",
                            "-vcodec", "png",
                            "-",
                        ],
                        timeout=10,
                    )
                b64_png = base64.b64encode(out_png).decode("ascii")
                _append_panel_log(f"🔗 [尾帧续写] 已无损提取 {fname} 的最后一帧作为首帧参考图")
                self._send_json({"ok": True, "data_url": f"data:image/png;base64,{b64_png}"})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})
            return

        if self.path == "/api/video_cancel":
            _PENDING_VIDEO_JOB["active"] = False
            _PENDING_VIDEO_JOB["error"] = "🛑 已手动停止视频生成任务"
            cfg = _load_endpoint_cfg()
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.get('api_key', 'sk-colab-ed28e54eb298d0c1')}",
                "User-Agent": "ColabPanel/1.0",
            }
            for cancel_url in (
                "http://127.0.0.1:30000/v1/video/cancel",
                f"{_get_recavsr_base_url()}/v1/video/cancel_minimax",
            ):
                try:
                    req = urllib.request.Request(cancel_url, data=b"{}", headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=8) as _:
                        pass
                except Exception:
                    pass
            _append_panel_log("🛑 [手动停止] 已发送中断信号停止当前生成任务 (模型继续常驻显存)")
            self._send_json({"ok": True, "message": "🛑 已手动停止当前生成任务 (模型继续常驻显存)"})
            return

        if self.path == "/api/video_upscale_cancel":
            cfg = _load_endpoint_cfg()
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.get('api_key', 'sk-colab-ed28e54eb298d0c1')}",
                "User-Agent": "ColabPanel/1.0",
            }
            try:
                req = urllib.request.Request(
                    f"{_get_recavsr_base_url()}/v1/video/upscale_cancel",
                    data=b"{}",
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=8) as _:
                    pass
            except Exception:
                pass
            _append_panel_log("🛑 [手动停止] 已发送中断信号停止当前 ReCaVSR 超分任务")
            self._send_json({"ok": True, "message": "🛑 已手动停止 ReCaVSR 视频超分任务"})
            return

        if self.path == "/api/drama_auto_start":
            if _DRAMA_STATE["running"]:
                self._send_json({"ok": True, "already_running": True, "message": "全自动导演流水线已在运行中"})
                return
            aspect = str(payload.get("aspect") or "16:9").strip()
            resume = bool(payload.get("resume", True))
            auto_2k = bool(payload.get("auto_2k", False))
            _DRAMA_STATE["running"] = True
            _DRAMA_STATE["cancel"] = False
            _DRAMA_STATE["error"] = ""
            _DRAMA_STATE["pct"] = 2
            _DRAMA_STATE["message"] = "🚀 已启动《溪边与市集白花簪》11 镜 30 秒全自动导演流水线..."
            threading.Thread(
                target=_run_auto_drama_pipeline_bg,
                args=(aspect, resume, auto_2k),
                daemon=True,
            ).start()
            self._send_json({"ok": True, "message": "已启动 11 镜 30 秒全自动导演流水线"})
            return

        if self.path == "/api/drama_auto_stop":
            _DRAMA_STATE["cancel"] = True
            _DRAMA_STATE["message"] = "🛑 正在安全暂停全自动导演流水线 (已生成的定妆照与分镜均已保存，下次可断点续拍)..."
            _append_panel_log("🛑 [自动导演] 已收到暂停信号，当前单步完成后将停止并保留全部已生成素材")
            self._send_json({"ok": True, "message": "已发送暂停信号"})
            return

        if self.path == "/api/drama_auto_remux":
            aspect = str(payload.get("aspect") or "16:9").strip()
            auto_2k = bool(payload.get("auto_2k", False))
            try:
                final_p = _drama_assemble_30s_movie(DRAMA_WORK_DIR, aspect=aspect, auto_2k=auto_2k)
                _DRAMA_STATE["final_video"] = final_p.name
                _DRAMA_STATE["pct"] = 100
                _DRAMA_STATE["error"] = ""
                _DRAMA_STATE["message"] = f"🎉 完整短剧成片已重新精剪混音合成：{final_p.name} (已保存至 ~/Movies/EasyColab/)"
                self._send_json({"ok": True, "final_video": final_p.name, "message": _DRAMA_STATE["message"]})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})
            return

        if self.path == "/api/drama_workflow_save":
            wf_in = payload.get("workflow")
            if not isinstance(wf_in, dict):
                self._send_json({"ok": False, "error": "无效的工作流 JSON 数据"})
                return
            _save_drama_workflow(wf_in)
            _append_panel_log(f"💾 [工作流已保存] 剧目「{wf_in.get('title') or '未命名短剧'}」({len(wf_in.get('image_tasks') or [])} 个图片节点, {len(wf_in.get('video_tasks') or [])} 个视频分镜节点)")
            self._send_json({"ok": True, "workflow": _load_drama_workflow()})
            return

        if self.path == "/api/drama_upload_char_photo":
            role_id = re.sub(r"[^A-Z0-9]", "", str(payload.get("role_id") or "A").strip().upper()) or "A"
            img_b64 = str(payload.get("image_b64") or "").strip()
            if not img_b64:
                self._send_json({"ok": False, "error": "未收到图片数据"})
                return
            try:
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                raw_bytes = base64.b64decode(img_b64)
                DRAMA_WORK_DIR.mkdir(parents=True, exist_ok=True)
                target_fn = f"char_{role_id}_half.jpg"
                target_path = DRAMA_WORK_DIR / target_fn
                try:
                    import io
                    from PIL import Image, ImageOps
                    im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    im_half = ImageOps.fit(im, (2048, 1152), method=Image.Resampling.LANCZOS, centering=(0.5, 0.22))
                    im_half.save(target_path, format="JPEG", quality=95)
                    if role_id == "A":
                        im_close = ImageOps.fit(im, (2048, 1152), method=Image.Resampling.LANCZOS, centering=(0.5, 0.16))
                        im_close.save(DRAMA_WORK_DIR / "char_A_closeup.jpg", format="JPEG", quality=95)
                except Exception:
                    target_path.write_bytes(raw_bytes)
                    if role_id == "A":
                        (DRAMA_WORK_DIR / "char_A_closeup.jpg").write_bytes(raw_bytes)

                wf = _load_drama_workflow()
                uploaded_set = set(wf.get("user_uploaded_chars") or [])
                uploaded_set.add(role_id)
                wf["user_uploaded_chars"] = sorted(list(uploaded_set))
                cast_list = wf.get("cast_list") or []
                found_role = False
                for c_item in cast_list:
                    if str(c_item.get("id") or "").upper() == role_id:
                        c_item["user_uploaded"] = True
                        found_role = True
                if not found_role:
                    cast_list.append({
                        "id": role_id,
                        "name": f"角色{role_id}",
                        "voice": "ming:灵小柔" if role_id == "A" else "ming:灵信远_中性",
                        "desc": "",
                        "file": target_fn,
                        "user_uploaded": True,
                    })
                wf["cast_list"] = cast_list
                if not any(it.get("file") == target_fn for it in (wf.get("image_tasks") or [])):
                    wf.setdefault("image_tasks", []).insert(0, {
                        "file": target_fn,
                        "label": f"🎭 角色{role_id}指定定妆照",
                        "refs": [],
                        "prompt": f"用户指定角色{role_id}定妆照 ({target_fn})",
                    })
                _save_drama_workflow(wf)
                _append_panel_log(f"📸 [角色定妆锁定] 已上传并归一化 16:9 (2048x1152) 角色{role_id}定妆照 -> {target_fn}")
                self._send_json({"ok": True, "file": target_fn, "role_id": role_id})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})
            return

        if self.path == "/api/drama_delete_char_photo":
            role_id = re.sub(r"[^A-Z0-9]", "", str(payload.get("role_id") or "A").strip().upper()) or "A"
            try:
                for fn in (f"char_{role_id}_half.jpg", f"char_{role_id}_closeup.jpg" if role_id == "A" else ""):
                    if fn and (DRAMA_WORK_DIR / fn).is_file():
                        (DRAMA_WORK_DIR / fn).unlink()
                wf = _load_drama_workflow()
                uploaded_set = set(wf.get("user_uploaded_chars") or [])
                uploaded_set.discard(role_id)
                wf["user_uploaded_chars"] = sorted(list(uploaded_set))
                for c_item in (wf.get("cast_list") or []):
                    if str(c_item.get("id") or "").upper() == role_id:
                        c_item["user_uploaded"] = False
                _save_drama_workflow(wf)
                _append_panel_log(f"🗑️ [角色定妆解除] 已移除角色{role_id}的上传照片，将恢复由 AI 自动生图定妆")
                self._send_json({"ok": True, "role_id": role_id})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})
            return

        if self.path == "/api/drama_workflow_parse":
            if payload.get("reset_default"):
                wf_def = _get_default_drama_workflow()
                _save_drama_workflow(wf_def)
                self._send_json({"ok": True, "workflow": wf_def, "message": "✅ 已恢复内置 11 镜 30 秒古装短剧工作流示例"})
                return
            raw_script = str(payload.get("raw_script") or "").strip()
            title_in = str(payload.get("title") or "").strip()
            clear_old = bool(payload.get("clear_old_assets", True))
            cast_list_in = payload.get("cast_list")
            if not raw_script:
                self._send_json({"ok": False, "error": "剧本内容不能为空"})
                return
            try:
                old_wf = _load_drama_workflow()
                locked_char_ids = set(old_wf.get("user_uploaded_chars") or [])
                if isinstance(cast_list_in, list):
                    for c_item in cast_list_in:
                        if isinstance(c_item, dict) and c_item.get("user_uploaded"):
                            locked_char_ids.add(str(c_item.get("id") or "").upper())
                protected_files = {"workflow.json"}
                for cid in locked_char_ids:
                    if (DRAMA_WORK_DIR / f"char_{cid}_half.jpg").is_file():
                        protected_files.add(f"char_{cid}_half.jpg")
                    if cid == "A" and (DRAMA_WORK_DIR / "char_A_closeup.jpg").is_file():
                        protected_files.add("char_A_closeup.jpg")

                wf_parsed = _parse_raw_script_to_workflow(
                    raw_script,
                    title=title_in,
                    cast_list=cast_list_in if isinstance(cast_list_in, list) else (old_wf.get("cast_list") or None),
                    user_uploaded_chars=sorted(list(locked_char_ids)),
                )
                if clear_old and DRAMA_WORK_DIR.is_dir():
                    for old_f in DRAMA_WORK_DIR.glob("*"):
                        if old_f.is_file() and old_f.name not in protected_files:
                            try:
                                old_f.unlink()
                            except Exception:
                                pass
                _save_drama_workflow(wf_parsed)
                n_img = len(wf_parsed.get("image_tasks") or [])
                n_vid = len(wf_parsed.get("video_tasks") or [])
                n_cast = len(wf_parsed.get("cast_list") or [])
                tot_dur = round(sum(float(x.get("duration") or 3.0) for x in (wf_parsed.get("video_tasks") or [])), 1)
                msg = f"✅ 已成功解析新剧本「{wf_parsed.get('title')}」：{n_cast} 位角色 + {n_img} 个图片节点 + {n_vid} 个分镜视频节点 (总长 {tot_dur}s)"
                _DRAMA_STATE["message"] = msg
                _DRAMA_STATE["error"] = ""
                _append_panel_log(f"🔍 [剧本工作流解析] {msg}")
                self._send_json({"ok": True, "workflow": wf_parsed, "message": msg})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})
            return

        if self.path == "/api/drama_workflow_rerun_one":
            if _DRAMA_STATE["running"]:
                self._send_json({"ok": False, "error": "当前已有工作流任务正在运行中，请先等待完成或暂停"})
                return
            target_file = str(payload.get("target_file") or "").strip()
            aspect = str(payload.get("aspect") or "16:9").strip()
            if not target_file:
                self._send_json({"ok": False, "error": "未指定要重跑的节点文件名"})
                return
            _DRAMA_STATE["running"] = True
            _DRAMA_STATE["cancel"] = False
            _DRAMA_STATE["error"] = ""
            _DRAMA_STATE["active_file"] = target_file
            _DRAMA_STATE["message"] = f"🔄 正在单独重抽节点: {target_file}..."
            threading.Thread(
                target=_run_single_drama_step_bg,
                args=(target_file, aspect),
                daemon=True,
            ).start()
            self._send_json({"ok": True, "message": f"已启动单节点重抽: {target_file}"})
            return

        self.send_error(404)


DRAMA_WORK_DIR = LOCAL_MOVIES_DIR / "drama_30s_ancient"
DRAMA_WORKFLOW_FILE = DRAMA_WORK_DIR / "workflow.json"

_DRAMA_STATE: Dict[str, Any] = {
    "running": False,
    "cancel": False,
    "pct": 0,
    "active_file": "",
    "message": "✅ 已加载分镜短剧工作流（支持粘贴任意新剧本自动拆解、单镜独立重抽与 0.1s 卡点合片）！",
    "error": "",
    "final_video": "",
}


def _get_default_drama_workflow() -> Dict[str, Any]:
    return {
        "title": "《溪边与市集白花簪》",
        "global_style": "写实真人，古装爱情短剧质感，电影级浅景深，阳光暖调，画面通透柔和。",
        "ambience_source_file": "shot05_raw.mp4",
        "image_tasks": [
            {
                "file": "char_A_half.jpg",
                "label": "① 图1:女主A半身(小粉花)",
                "refs": [],
                "prompt": "写实真人摄影，古装爱情剧质感，85mm人像镜头，电影级浅景深，阳光暖调，画面通透柔和。一位19岁中国古装少女（女主A）的正面中近景半身定妆照，性格活泼俏皮、灵动甜美，鹅蛋脸，清澈有神的大眼睛，白皙细腻的真实皮肤纹理，带着俏皮明朗的浅笑。她梳着精致少女古装发髻，发间右侧固定簪着一朵小巧精致的浅粉色五瓣小花（右侧小粉花清晰可见，左侧发间无任何发饰），身穿浅青与淡粉交织的轻盈宋制齐胸衫裙，刺绣纹理清晰，自然户外暖阳，背景为干净柔和的浅绿色虚化光斑。",
            },
            {
                "file": "char_A_closeup.jpg",
                "label": "② 图2:女主A锁脸特写",
                "refs": ["char_A_half.jpg"],
                "prompt": "Keep the exact facial identity, hairstyle, and the small pink flower on the right side of the hair of the woman in <Picture 1>. 写实真人电影级面部特写镜头（Close-up），古装少女（女主A）直视镜头，五官与 <Picture 1> 完全一致，眼神灵动俏皮，嘴角微扬，细腻真实的皮肤质感，发间右侧依然簪着那朵小巧的浅粉色小花，左侧发间无发饰，柔和暖阳轮廓光，背景柔和虚化。",
            },
            {
                "file": "char_B_half.jpg",
                "label": "③ 图3:男主B古装定妆",
                "refs": [],
                "prompt": "写实真人摄影，古装爱情剧质感，85mm人像镜头，电影级浅景深，阳光暖调，画面通透柔和。一位23岁中国古装青年公子（男主B）的正面中近景半身定妆照，面容俊朗温润，剑眉星目，气质沉静温和、内敛克制，嘴角带着一丝温柔克制的浅笑。他束着整洁的古装墨发玉冠，身穿深藏青色与墨色交织的质感暗纹古装直裰长袍，真实细腻的皮肤纹理与发丝质感，柔和户外暖阳自后方打来形成金色发丝轮廓光，背景为柔和的深绿色林间虚化光斑。",
            },
            {
                "file": "shot01_first.jpg",
                "label": "📷 镜01首帧:溪边A歪头",
                "refs": ["char_A_half.jpg", "char_A_closeup.jpg", "char_B_half.jpg"],
                "prompt": "Keep the exact face and right-side small pink hair flower of the woman in <Picture 1> and <Picture 2>, and the man in <Picture 3> in the blurred background. 写实古装剧电影画面，林间溪边，阳光暖调。女主A（来自 <Picture 1> <Picture 2>，右侧发间簪着小粉花）的面部特写占画面左侧，前景左缘有一片虚化的浅色树叶遮挡；A的头向她自己的右肩侧微微歪着，脸朝着镜头，双唇微张、眼神俏皮灵动地望向镜头方向；男主B（来自 <Picture 3>）在画面右后方远处虚化成一团深色古装身影。暖阳自A身后打来，发丝边缘有一圈明亮的金色轮廓光，背景是虚化的溪边绿色光斑，电影级浅景深。",
            },
            {
                "file": "shot02_first.jpg",
                "label": "📷 镜02首帧:小鱼跃水面",
                "refs": ["char_A_half.jpg", "char_B_half.jpg"],
                "prompt": "Keep the woman from <Picture 1> in the soft out-of-focus background. 写实古装剧画面，机位贴近清澈溪水水面的超低角度镜头。焦点清晰落在前景水面上：一条灵动的小溪鱼正自清浅的溪水中跃起，在半空划出一道水线，水面溅起晶莹的水花与一圈涟漪；后景处于浅景深柔和虚化状态，女主A（来自 <Picture 1>，穿浅青淡粉古装裙，右侧发间有小粉花）蹲在溪边卵石滩上，裙摆铺开遮住双腿，正笑着把手伸向水面；男主B（来自 <Picture 2>）在更后方青苔大石处虚化。阳光洒在水面与树叶间，暖调通透。",
            },
            {
                "file": "shot02_last.jpg",
                "label": "📷 镜02尾帧:转焦A弯眼笑",
                "refs": ["char_A_half.jpg", "char_A_closeup.jpg", "char_B_half.jpg"],
                "prompt": "Keep the exact face and outfit of the woman in <Picture 1> and <Picture 2>. 写实古装剧画面，机位贴近溪水水面的低角度镜头。前景溪水水面处于柔和虚化并带有余波涟漪；焦点清晰对准后景蹲在溪边卵石滩上的女主A（来自 <Picture 1> <Picture 2>，右侧发间簪着小粉花），她裙摆铺开遮住双腿，手停在水面上方，笑得眉眼弯弯、俏皮可爱；男主B（来自 <Picture 3>）在更后方远处虚化。阳光透过树叶洒下温暖光斑。",
            },
            {
                "file": "shot03_first.jpg",
                "label": "📷 镜03首帧:全景A撩水",
                "refs": ["char_A_half.jpg", "char_A_closeup.jpg", "char_B_half.jpg"],
                "prompt": "Keep the woman from <Picture 1> <Picture 2> and the man from <Picture 3>. 写实古装剧溪边全景镜头，画面左右两缘有虚化的翠绿树叶作前景框景。画面左中位置，女主A（来自 <Picture 1>，右侧发间有小粉花）站在清浅的溪水卵石滩边，正开心地笑着俯身用双手撩起溪水向镜头方向泼洒，晶莹的水花在阳光下发亮；画面右侧岸边的青苔大石上，男主B（来自 <Picture 3>，深色古装）安稳坐着，侧身温和地看着A，嘴角带着一丝克制温柔的笑意。溪水在两人前方缓缓流过，倒映着金色阳光。",
            },
            {
                "file": "shot04_first.jpg",
                "label": "📷 镜04首帧:溪边B逆光",
                "refs": ["char_B_half.jpg"],
                "prompt": "Keep the exact facial identity and outfit of the man in <Picture 1>. 写实古装剧电影级面部特写镜头，林间溪边。男主B（来自 <Picture 1>）的面部特写，他的目光温柔地望向画面左下方的方向，神情沉静温和，嘴角微露一丝克制的浅笑意，眼神柔和。温暖的阳光自他身后打来，墨发边缘有一圈金色轮廓光，背景是虚化的森林绿色光斑，电影级浅景深。",
            },
            {
                "file": "shot05_first.jpg",
                "label": "📷 镜05首帧:市集A前B随",
                "refs": ["char_A_half.jpg", "char_A_closeup.jpg", "char_B_half.jpg"],
                "prompt": "Keep the exact woman from <Picture 1> <Picture 2> in front and the man from <Picture 3> walking behind her. 写实古装剧白天的古代市集街道，中景正面镜头。女主A（来自 <Picture 1> <Picture 2>，右侧发间簪着小粉花，左侧无发饰）走在前面处于画面中景，笑容明朗俏皮，正开心看向两旁摊位；男主B（来自 <Picture 3>，深色古装）跟在她身后两三步的距离，神情平静温和地目光落在她身上。街道两旁摊位林立、商贩与路人全部处于浅景深虚化状态，不看镜头；远处可见石拱桥与古代酒楼招牌，阳光明亮，街面有斑驳暖调光影。",
            },
            {
                "file": "shot06_first.jpg",
                "label": "📷 镜06首帧:摊前捧白花簪",
                "refs": ["char_A_half.jpg", "char_A_closeup.jpg"],
                "prompt": "Keep the exact face and outfit of the woman in <Picture 1> <Picture 2>. 写实古装剧古代市集小摊前的中近景镜头，摊位上方挂着一排色彩雅致的古风香囊与小饰物。女主A（来自 <Picture 1> <Picture 2>，右侧发间簪着一朵小粉花，左侧发间无发饰）站在摊前，正微微低头，双手小心捧着一支刚从摊上拿起的「白花发簪」（细长的簪杆，顶端是一朵精致的白色五瓣小花），神情专注欣喜地端详着发簪。背景为虚化的阳光市集街道，电影级暖调浅景深。",
            },
            {
                "file": "shot06_last.jpg",
                "label": "📷 镜06尾帧:举簪招手唤B",
                "refs": ["char_A_half.jpg", "char_A_closeup.jpg", "char_B_half.jpg"],
                "prompt": "Keep the exact woman from <Picture 1> <Picture 2> and the man from <Picture 3> in the blurred background. 写实古装剧市集香囊小摊前的中近景镜头。女主A（来自 <Picture 1> <Picture 2>，右侧发间有小粉花）一只手把那支「白花发簪」（细簪杆顶端一朵白色五瓣小花）举在自己左侧头边比试，同时转头望向画面左后方，笑得明朗俏皮，另一只手抬起朝左后方招手示意；画面左后方背景中，男主B（来自 <Picture 3>）正应声走近停在她对面（处于浅景深柔和虚化中）。阳光暖调。",
            },
            {
                "file": "shot07_first.jpg",
                "label": "📷 镜07首帧:越肩A举簪",
                "refs": ["char_A_half.jpg", "char_A_closeup.jpg", "char_B_half.jpg"],
                "prompt": "Keep the exact face of the woman in <Picture 1> <Picture 2>. 写实古装剧越肩镜头（Over-the-shoulder shot）。画面左侧前景是男主B（来自 <Picture 3>）虚化的深色古装后脑与左肩背；镜头越过B的肩膀对准对面的女主A（来自 <Picture 1> <Picture 2>，面部与上身清晰实焦）。A发间右侧簪着小粉花，她右手正把那支「白花发簪」（细簪杆顶端一朵白色五瓣小花）举在自己左侧发髻旁比试，笑得眉眼弯弯，眼神亮晶晶地望着对面的B。背景是虚化的市集香囊小摊，阳光暖调。",
            },
            {
                "file": "shot08_first.jpg",
                "label": "📷 镜08首帧:灯笼B垂眼笑",
                "refs": ["char_B_half.jpg"],
                "prompt": "Keep the exact facial identity of the man in <Picture 1>. 写实古装剧电影级面部特写镜头。男主B（来自 <Picture 1>）站在市集摊位前，眼帘微垂后刚抬起眼望向对面的A，嘴角缓缓扬起温柔浅笑，眼神里带着满满的纵容与宠溺。背景是浅景深虚化的红色与橙色市集灯笼温暖光斑，肤质细腻真实。",
            },
            {
                "file": "shot09_11_first.jpg",
                "label": "📷 镜09/11首帧:左侧插妥簪",
                "refs": ["shot07_first.jpg", "char_A_closeup.jpg"],
                "prompt": "Based on <Picture 1> and keeping the exact face of <Picture 2>, keep the same over-the-shoulder composition with the blurred dark shoulder of the man on the left foreground. 写实古装剧越肩近景特写镜头，保持与 <Picture 1> 完全相同的机位、背景和前景左侧虚化的男主肩背。画面中的女主A已经把那支「白花发簪」（细簪杆顶端一朵白色五瓣小花）稳稳插进了自己左侧发间（同时她右侧发间原本的小粉花依然保留），双手已自然放下不再举着，她眼睛发亮、嘴角带着俏皮灵动的神情望着对面的B。",
            },
            {
                "file": "shot10_first.jpg",
                "label": "📷 镜10首帧:反打越肩B笑",
                "refs": ["char_A_half.jpg", "char_B_half.jpg"],
                "prompt": "Keep the exact facial identity of the man in <Picture 2>. 写实古装剧反打越肩镜头（Reverse over-the-shoulder shot）。画面左侧前景是女主A（来自 <Picture 1>）虚化的古装发髻后脑（左侧隐约可见白花发簪轮廓）；镜头越过A的后脑清晰对准对面的男主B（来自 <Picture 2>）。B正专注地看着A，面带温柔认真又纵容的浅笑。温暖的阳光自B身后打来，在他脸庞和发丝边缘勾勒出暖融融的金色轮廓光，市集背景柔和虚化。",
            },
        ],
        "video_tasks": [
            {
                "file": "shot01_raw.mp4",
                "label": "🎞️ 镜01(2.2s):溪边歪头笑",
                "engine": "fl2va",
                "duration": 2.2,
                "first_img": "shot01_first.jpg",
                "last_img": "",
                "tts_text": "",
                "tts_voice": "ming:灵小柔",
                "mix_ambience": False,
                "prompt": "固定镜头，机位与焦距完全静止。林间溪边，左侧女主A（右侧发间带小粉花）头向自己右肩微微歪着朝向镜头，她先是双唇微张像刚要说话，随即抿起嘴唇露出俏皮甜美的浅笑，眼神灵动地望向镜头方向；发丝在微风与身后金色阳光轮廓光中轻轻拂动，右后方男主B虚化身影安静不动。纯自然环境音效：清脆的溪水潺潺流动声与林间清幽鸟鸣声，无背景音乐，无人物说话。",
            },
            {
                "file": "shot02_raw.mp4",
                "label": "🎞️ 镜02(3.0s):小鱼跃水转焦",
                "engine": "fl2va",
                "duration": 3.0,
                "first_img": "shot02_first.jpg",
                "last_img": "shot02_last.jpg",
                "tts_text": "",
                "tts_voice": "ming:灵小柔",
                "mix_ambience": False,
                "prompt": "固定低角度贴近水面镜头，机位完全静止。起幅焦点落在前景溪面：一条小鱼自溪水中灵动跃起，在半空划出一道自然弧线后落回水中，溅起水花与一圈涟漪；随后镜头焦点平滑转移到后景蹲在溪边卵石滩上的女主A身上，她裙摆铺开，正笑得眉眼弯弯，手停在水面上方。纯自然环境音效：小鱼跃出水面的清脆扑通水花声与溪水流动声，无背景音乐。",
            },
            {
                "file": "shot03_raw.mp4",
                "label": "🎞️ 镜03(3.7s):A撩水泼洒笑",
                "engine": "fl2va",
                "duration": 3.7,
                "first_img": "shot03_first.jpg",
                "last_img": "",
                "tts_text": "",
                "tts_voice": "ming:灵小柔",
                "mix_ambience": False,
                "prompt": "固定全景镜头，机位完全静止，左右两缘有虚化绿叶前景。画面左中女主A在溪边开心地用双手撩起溪水向镜头方向泼洒出去，泼水的水花方向与手部动作一致，晶莹水花在阳光下飞溅发亮，泼完水她自己先眉眼弯弯地笑出声；画面右侧青苔大石上，男主B安稳坐着侧身看着A，神情温和，嘴角带一点宠溺笑意。溪水在前方缓缓流过。纯自然环境音效：双手撩水泼洒的水花声、少女清脆轻快的浅笑声、溪水流淌与林间鸟鸣声，无背景音乐。",
            },
            {
                "file": "shot04_raw.mp4",
                "label": "🎞️ 镜04(2.1s):B逆光浅笑",
                "engine": "fl2va",
                "duration": 2.1,
                "first_img": "shot04_first.jpg",
                "last_img": "",
                "tts_text": "",
                "tts_voice": "ming:灵信远_中性",
                "mix_ambience": False,
                "prompt": "固定特写镜头，机位与焦距完全静止。林间溪边，男主B的面部特写，目光专注地望向画面左下方A的方向；他先是神情沉静平和，随即嘴角慢慢扬起，露出一抹温柔而克制的浅笑，眼神随之柔和下来；身后阳光勾勒出金色发丝轮廓光。纯自然环境音效：轻柔的溪水流动声与林间微风鸟鸣，无背景音乐，无对白。",
            },
            {
                "file": "shot05_raw.mp4",
                "label": "🎞️ 镜05(2.6s):市集倒退跟拍",
                "engine": "fl2va",
                "duration": 2.6,
                "first_img": "shot05_first.jpg",
                "last_img": "",
                "tts_text": "",
                "tts_voice": "ming:灵小柔",
                "mix_ambience": False,
                "prompt": "镜头在主体前方一边平稳后退一边拍摄（倒退跟拍机位），正对女主A的表情与步伐，A始终保持中景大小。白天的古代市集街道，A走在前面，笑容明朗俏皮，边走边开心地左右张望两旁虚化的摊位，脚步轻快；男主B跟在她身后两三步的距离，神情平静温和地跟着她前行。两旁商贩与路人全部虚化且不看镜头，远处有石拱桥，阳光斑驳。纯自然环境音效：热闹的古代市集人群低语交谈声、轻快的脚步声与摊位杂声，无背景音乐。",
            },
            {
                "file": "shot06_raw.mp4",
                "label": "🎞️ 镜06(5.3s):拿簪招手唤B",
                "engine": "fl2va",
                "duration": 5.3,
                "first_img": "shot06_first.jpg",
                "last_img": "shot06_last.jpg",
                "tts_text": "你快过来。",
                "tts_voice": "ming:灵小柔",
                "mix_ambience": False,
                "prompt": "固定中近景镜头，机位完全静止。市集香囊小摊前，女主A（右侧发间有小粉花）先低头双手捧着那支白花发簪（细簪杆顶端一朵白色五瓣小花）轻轻翻看端详，随后抬手把白花发簪举到自己左侧头边比试；接着她转头望向画面左后方，笑着朝男主B招了招手，动作俏皮可爱，并开口用甜美中文说道：\"你快过来。\"；男主B应声自画面左后方走近，在后段进入虚化背景停在A对面。音效包含市集环境人群杂声与女主清晰的中文台词\"你快过来\"，无背景音乐。",
            },
            {
                "file": "shot07_raw.mp4",
                "label": "🗣️ 镜07(1.7s):A「怎么样？」",
                "engine": "avatar",
                "duration": 1.7,
                "first_img": "shot07_first.jpg",
                "last_img": "",
                "tts_text": "怎么样？",
                "tts_voice": "ming:灵小柔",
                "mix_ambience": True,
                "prompt": "固定越肩镜头，机位完全静止，左侧前景男主肩背保持虚化。画面中的古装少女A把白花发簪举在左侧头边，笑得眉眼弯弯，眼神亮晶晶地望着对面的B自然开口问话，伴随语流有灵动的微眨眼与俏皮期盼的神情，口型与语音精准同步，阳光暖调。",
            },
            {
                "file": "shot08_raw.mp4",
                "label": "🎞️ 镜08(2.3s):B灯笼下浅笑",
                "engine": "fl2va",
                "duration": 2.3,
                "first_img": "shot08_first.jpg",
                "last_img": "",
                "tts_text": "",
                "tts_voice": "ming:灵信远_中性",
                "mix_ambience": False,
                "prompt": "固定特写镜头，机位与焦距完全静止。背景是虚化的红橙色市集灯笼光斑，男主B的面部特写：他先是微微垂下眼帘，像在认真端详对面A头侧比试的白花发簪，随即缓缓抬起眼看向A，嘴角扬起一抹温柔浅笑，眼神里带着满满的纵容。纯自然环境音效：轻柔的市集背景人群环境声，无背景音乐，无对白。",
            },
            {
                "file": "shot09_raw.mp4",
                "label": "🗣️ 镜09(1.7s):A「这发簪好看吗」",
                "engine": "avatar",
                "duration": 1.7,
                "first_img": "shot09_11_first.jpg",
                "last_img": "",
                "tts_text": "这发簪好看吗？",
                "tts_voice": "ming:灵小柔",
                "mix_ambience": True,
                "prompt": "固定越肩镜头，机位完全静止，左侧前景男主肩背虚化。白花发簪稳稳插在女主A左侧发间，右侧小粉花保留。A眼睛发亮，兴奋又期待地望着对面的B开口追问，嘴角压不住甜美俏皮的笑意，伴随说话有自然的微挑眉和眨眼，口型精准同步。",
            },
            {
                "file": "shot10_raw.mp4",
                "label": "🗣️ 镜10(1.6s):B「好看。」",
                "engine": "avatar",
                "duration": 1.6,
                "first_img": "shot10_first.jpg",
                "last_img": "",
                "tts_text": "好看。",
                "tts_voice": "ming:灵信远_中性",
                "mix_ambience": True,
                "prompt": "固定反打越肩镜头，机位完全静止，左侧前景女主后脑虚化。阳光自男主B身后打来形成暖融融的轮廓光，B专注地看着对面的A，眼神温柔纵容，含笑认真地开口回答，伴随说话有轻微温和的点头与眼神交流，口型自然准确。",
            },
            {
                "file": "shot11_raw.mp4",
                "label": "🗣️ 镜11(3.8s):A「哼～问你什么」",
                "engine": "avatar",
                "duration": 3.8,
                "first_img": "shot09_11_first.jpg",
                "last_img": "",
                "tts_text": "[Dissatisfaction-hnn] 问你什么都说好看，男人的嘴，骗人的鬼。",
                "tts_voice": "ming:灵小柔",
                "mix_ambience": True,
                "prompt": "固定越肩面部特写镜头，机位完全静止，左侧前景男主肩背虚化，女主A左侧发间插着白花发簪、右侧小粉花保留。A先是假装生气地板起脸、微微撇嘴佯装瞪了对面的B一眼，随即俏皮灵动地开口数落他，眼神灵动有戏，眉眼与头部伴随说话节奏有自然的微动作，说完自然收住，口型与牙齿清晰自然。",
            },
        ],
    }


def _load_drama_workflow() -> Dict[str, Any]:
    DRAMA_WORK_DIR.mkdir(parents=True, exist_ok=True)
    if DRAMA_WORKFLOW_FILE.is_file():
        try:
            wf = json.loads(DRAMA_WORKFLOW_FILE.read_text(encoding="utf-8"))
            if isinstance(wf, dict) and wf.get("image_tasks") and wf.get("video_tasks"):
                changed = False
                for vt in wf.get("video_tasks") or []:
                    if vt.get("tts_voice") == "ming:灵逍遥":
                        vt["tts_voice"] = "ming:灵信远_中性"
                        changed = True
                if changed:
                    _save_drama_workflow(wf)
                return wf
        except Exception:
            pass
    wf = _get_default_drama_workflow()
    _save_drama_workflow(wf)
    return wf


def _save_drama_workflow(wf: Dict[str, Any]):
    DRAMA_WORK_DIR.mkdir(parents=True, exist_ok=True)
    DRAMA_WORKFLOW_FILE.write_text(json.dumps(wf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_raw_script_to_workflow(
    raw_script: str,
    title: str = "",
    cast_list: Optional[List[Dict[str, Any]]] = None,
    user_uploaded_chars: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    通用分镜剧本智能解析引擎 (不限分镜数量、不限总时长、支持 A/B/C/D/E... 多角色定妆照锁定与音色绑定)：
    1. 支持直接粘贴 JSON 工作流或 ```json ... ``` 代码块；
    2. 支持解析 ChatGPT / Claude / DeepSeek / Gemini / Kimi 生成的【剧名】【全局设定】【分镜】镜头01... 结构化剧本；
    3. 支持解析自由格式的分镜列表 (镜1 / Shot 1 / 第1镜 / 1. ...)，自动提取 A/B/C/D/E 多角色定妆基准卡、首尾关键帧、情绪音色与动态时长。
    """
    txt = raw_script.strip()
    locked_ids = set(str(x).upper() for x in (user_uploaded_chars or []))

    # 0. 若外部 AI 直接输出了 JSON 格式的工作流，优先直接解析 JSON
    json_cand = txt
    m_fenced = re.search(r"```(?:json)?\s*(\{[\s\S]+\})\s*```", txt)
    if m_fenced:
        json_cand = m_fenced.group(1).strip()
    if json_cand.startswith("{") and json_cand.endswith("}"):
        try:
            parsed_j = json.loads(json_cand)
            if isinstance(parsed_j, dict) and parsed_j.get("video_tasks"):
                if title:
                    parsed_j["title"] = title if title.startswith("《") else f"《{title}》"
                elif not parsed_j.get("title"):
                    parsed_j["title"] = f"《AI分镜短剧·{len(parsed_j['video_tasks'])}镜》"
                if not parsed_j.get("image_tasks"):
                    parsed_j["image_tasks"] = []
                if cast_list:
                    parsed_j["cast_list"] = cast_list
                if locked_ids:
                    parsed_j["user_uploaded_chars"] = sorted(list(locked_ids))
                parsed_j["raw_script"] = raw_script
                return parsed_j
        except Exception:
            pass

    # 剥离外部 AI 可能包裹的 Markdown 代码块围栏 ``` 以及在末尾附加的总结段落 ("剧本共 27 个镜头..." / "几处我替你...")
    txt = re.sub(r"```[a-zA-Z]*", "", txt)
    txt = re.split(r"\n\s*(?:剧本共\s*\d+\s*个?镜头|几处我替你|说明\s*[:：]|设计说明\s*[:：])", txt, maxsplit=1)[0].strip()

    m_title = re.search(r"(?:【剧名】|剧名\s*[:：]|^#\s*)\s*([^\n【]+)", txt)
    if m_title:
        extracted_t = m_title.group(1).strip().strip("《》\"' ")
        if extracted_t:
            title = extracted_t

    m_style = re.search(r"画风\s*[:：]\s*([^\n【]+)", txt)
    raw_style = m_style.group(1).strip() if m_style else "写实真人电影级质感，85mm浅景深，自然柔和光影，细节清晰真实。"

    m_scene = re.search(r"场景\s*[:：]\s*([^\n【]+)", txt)
    scene_desc = m_scene.group(1).strip() if m_scene else ""

    # 若外部 AI 把具体室内家具/包厢/道具塞进了【画风】的分号后面，自动将其剥离到 scene_desc，防止污染室外/走廊镜头造成左右分屏卡顿！
    if "；" in raw_style:
        st_parts = [p.strip() for p in raw_style.split("；") if p.strip()]
        pure_style_parts = []
        extra_scene_parts = []
        for sp in st_parts:
            if any(k in sp for k in ("包厢", "桌面", "墙板", "大理石", "冰桶", "水晶杯", "伞铺", "柜台", "沙发")):
                extra_scene_parts.append(sp)
            else:
                pure_style_parts.append(sp)
        global_style = "；".join(pure_style_parts) if pure_style_parts else st_parts[0]
        if extra_scene_parts and not scene_desc:
            scene_desc = "，".join(extra_scene_parts)
    else:
        global_style = raw_style

    m_char_all = re.search(r"人物(?:锁定|设定)?\s*[:：]\s*([^\n【]+)", txt)
    char_desc_all = m_char_all.group(1).strip() if m_char_all else ""

    def _extract_char_name(desc_str: str) -> str:
        m_nm = re.match(r"^([\u4e00-\u9fffA-Za-z0-9]{2,12})\s*[，,、\(（]", desc_str.strip())
        if m_nm:
            cand = m_nm.group(1)
            if cand not in ("年轻", "古装", "现代", "女性", "男性", "少女", "青年", "身穿", "鹅蛋", "一位", "中年"):
                return cand
        return ""

    default_voices = {
        "A": "ming:灵小柔",
        "B": "ming:灵信远_中性",
        "C": "ming:灵朗",
        "D": "ming:灵采薇_喜悦",
        "E": "ming:灵孤影_悲伤",
        "F": "ming:灵楷",
    }

    # 先扫描剧本正文中的角色设定行：支持 女主A / 男主B / 配角C / 角色A..Z / 人物A..Z
    script_roles: Dict[str, Dict[str, Any]] = {}
    for m_role in re.finditer(r"(?m)^\s*(?:女主|男主|配角|反派|角色|人物)\s*([A-Z])?(?:\s*[（(]([^）)]+)[）)])?\s*[:：]\s*([^\n【]+)", txt):
        raw_line = m_role.group(0)
        rid = (m_role.group(1) or "").strip().upper()
        if not rid:
            if "女主" in raw_line:
                rid = "A"
            elif "男主" in raw_line:
                rid = "B"
            elif "配角" in raw_line:
                rid = "C"
            else:
                continue
        r_paren = (m_role.group(2) or "").strip()
        r_desc = (m_role.group(3) or "").strip()
        if r_desc in ("无", "暂无", "none", "None") or "不出场" in r_desc or "无此角色" in r_desc:
            continue
        extracted_nm = _extract_char_name(r_desc) or (r_paren.split("·")[0].strip() if r_paren else "")
        voice_in_paren = ""
        if "ming:" in r_paren:
            m_vp = re.search(r"(ming:[^\s）)]+)", r_paren)
            if m_vp:
                voice_in_paren = m_vp.group(1).strip()
        is_male_desc = any(k in r_desc for k in ("男", "哥", "叔", "爷", "胡茬", "板寸", "油头", "西装", "先生", "公子", "少侠", "总裁"))
        fallback_v = "ming:灵信远_中性" if (is_male_desc and rid in ("A", "B")) else ("ming:灵朗" if is_male_desc else default_voices.get(rid, "ming:灵小柔"))
        script_roles[rid] = {
            "id": rid,
            "name": extracted_nm or f"角色{rid}",
            "voice": voice_in_paren or fallback_v,
            "desc": r_desc,
            "file": f"char_{rid}_half.jpg",
            "user_uploaded": (rid in locked_ids),
        }

    # 构建 A..Z 多角色花名册：若剧本明确定义了角色列表，以剧本定义为准（并合并前端卡槽的上传锁定状态）；否则使用前端卡槽
    roles_dict: Dict[str, Dict[str, Any]] = {}
    if script_roles:
        roles_dict = dict(script_roles)
        if isinstance(cast_list, list):
            for c_item in cast_list:
                if not isinstance(c_item, dict):
                    continue
                cid = re.sub(r"[^A-Z0-9]", "", str(c_item.get("id") or "").upper())
                if cid in roles_dict and (bool(c_item.get("user_uploaded")) or cid in locked_ids):
                    roles_dict[cid]["user_uploaded"] = True
    elif isinstance(cast_list, list):
        for c_item in cast_list:
            if not isinstance(c_item, dict):
                continue
            cid = re.sub(r"[^A-Z0-9]", "", str(c_item.get("id") or "").upper())
            if not cid:
                continue
            c_desc = str(c_item.get("desc") or "").strip()
            if "不出场" in c_desc or "无此角色" in c_desc:
                continue
            c_name = str(c_item.get("name") or "").strip() or _extract_char_name(c_desc) or f"角色{cid}"
            roles_dict[cid] = {
                "id": cid,
                "name": c_name,
                "voice": str(c_item.get("voice") or default_voices.get(cid, "ming:灵小柔")),
                "desc": c_desc,
                "file": f"char_{cid}_half.jpg",
                "user_uploaded": bool(c_item.get("user_uploaded")) or (cid in locked_ids),
            }

    if not roles_dict:
        roles_dict["A"] = {
            "id": "A",
            "name": _extract_char_name(char_desc_all) or "角色A",
            "voice": "ming:灵小柔",
            "desc": char_desc_all or "年轻女性，五官精致立体，真实细腻皮肤纹理，眼神灵动有戏",
            "file": "char_A_half.jpg",
            "user_uploaded": ("A" in locked_ids),
        }

    sorted_role_ids = sorted(roles_dict.keys())
    final_cast_list: List[Dict[str, Any]] = []

    # 判断题材与时代特征 (古装 vs 现代 vs 动漫)
    is_ancient = any(k in txt for k in ("古装", "古代", "市集", "发簪", "仙侠", "公子", "长袍", "衣袂", "汉服", "江湖", "宫廷"))
    is_anime = any(k in global_style for k in ("动漫", "二次元", "anime", "2D", "新海诚"))
    if is_anime:
        era_tag = "高画质唯美动漫电影风格，细腻光影与精致角色设定"
    elif is_ancient:
        era_tag = "中国古装剧质感，精致古装服饰与发髻"
    else:
        era_tag = "现代影视剧写真质感，自然时尚妆造"

    # 1. 自动构建多角色定妆基准图节点 (每个角色仅生成 1 张干净暗调背景的胸口以上半身定妆卡，避免街道全身背景污染后续分镜)
    image_tasks: List[Dict[str, Any]] = []
    circle_nums = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]
    img_idx = 0
    for rid in sorted_role_ids:
        r_info = roles_dict[rid]
        r_desc = r_info["desc"] or (
            "年轻女性，五官精致立体，真实细腻皮肤纹理" if rid == "A" else "气质鲜明的人物角色，五官立体清晰，真实细腻皮肤纹理"
        )
        r_info["desc"] = r_desc
        final_cast_list.append(r_info)
        c_num = circle_nums[img_idx] if img_idx < len(circle_nums) else f"({img_idx + 1})"
        img_idx += 1
        image_tasks.append({
            "file": f"char_{rid}_half.jpg",
            "label": f"{c_num} 图{img_idx}:角色{rid}({r_info['name']})定妆",
            "refs": [],
            "prompt": f"{global_style} {era_tag}，85mm f/1.4人像特写至中近景镜头（Medium Close-up Bust Portrait，仅取胸口以上至头顶，平视镜头）。角色{rid}（{r_info['name']}）：{r_desc}。背景为极简柔和的深暗色纯色大光圈虚化背景（无任何具体建筑或街景干扰），面部受光均匀立体，双唇自然放松闭合，电影摄影机实拍原片。",
        })

    # 2. 切分所有镜头块 (仅匹配行首带【时长】或换行的真实镜头头，彻底避免说明文字中的“镜头 01”被误拆)
    shot_pat = r"(?m)(?=^\s*(?:【?(?:镜头|分镜|镜|Shot\s*)\s*\d+|第\s*\d+\s*镜)\s*(?:【|\(|（|:|：|\n))"
    shot_blocks = re.split(shot_pat, txt, flags=re.IGNORECASE)
    shot_blocks = [
        b.strip()
        for b in shot_blocks
        if re.match(r"^(?:【?(?:镜头|分镜|镜|Shot\s*)\s*\d+|第\s*\d+\s*镜)\s*(?:【|\(|（|:|：|\n)", b.strip(), flags=re.IGNORECASE)
    ]
    if not shot_blocks:
        # 兜底：按编号行或段落切分 (不设 15 镜硬性截断，支持任意长度)
        paras = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip() and not p.strip().startswith("【")]
        shot_blocks = [f"镜头{i:02d}【时长】3.0秒\n【画面】{p}" for i, p in enumerate(paras[:80], start=1)]

    video_tasks: List[Dict[str, Any]] = []
    ambience_src = ""

    for idx, blk in enumerate(shot_blocks, start=1):
        shot_id = f"shot{idx:02d}"
        m_dur = re.search(r"(?:【时长】|时长\s*[:：]?|\(|（)\s*(\d+(?:\.\d+)?)\s*(?:秒|s|S)", blk)
        dur_s = round(float(m_dur.group(1)), 1) if m_dur else 3.0
        dur_s = max(1.0, min(20.0, dur_s))

        m_cam = re.search(r"(?:【运镜】|运镜\s*[:：])\s*([^\n【]+)", blk)
        cam_txt = m_cam.group(1).strip() if m_cam else "固定镜头，机位平稳。"

        m_roles_line = re.search(r"(?:【出场角色】|出场角色\s*[:：])\s*([^\n【]+)", blk)
        roles_line_txt = m_roles_line.group(1).strip() if m_roles_line else ""

        m_first_ov = re.search(r"(?:【首帧】|首帧\s*[:：]|起幅\s*[:：])\s*([^\n【]+)", blk)
        first_ov_txt = m_first_ov.group(1).strip() if m_first_ov else ""

        m_last_ov = re.search(r"(?:【尾帧】|尾帧\s*[:：]|落幅\s*[:：])\s*([^\n【]+)", blk)
        last_ov_txt = m_last_ov.group(1).strip() if m_last_ov else ""

        m_amb = re.search(r"(?:【环境音】|【音效】|环境音\s*[:：]|音效\s*[:：])\s*([^\n【]+)", blk)
        amb_txt = m_amb.group(1).strip() if m_amb else ""
        if amb_txt in ("无", "暂无", "none"):
            amb_txt = ""

        m_vis = re.search(r"(?:【画面】|画面\s*[:：])\s*([\s\S]*?)(?=【|(?:对白|台词|环境音|音效|音色|尾帧|首帧|出场角色)\s*[:：]|$)", blk)
        if m_vis:
            vis_txt = m_vis.group(1).strip().replace("\n", " ")
        else:
            cleaned_blk = re.sub(r"^(?:【?(?:镜头|分镜|镜|Shot\s*)\s*\d+】?|第\s*\d+\s*镜)\s*(?:【时长】[^\n]+)?", "", blk, flags=re.IGNORECASE)
            cleaned_blk = re.sub(r"【(?:时长|运镜|对白|台词|音色|环境音|音效|首帧|尾帧|引擎|出场角色)】[^\n【]*", "", cleaned_blk)
            vis_txt = cleaned_blk.strip().replace("\n", " ") or "电影级精致画面，光影自然柔和。"

        m_dlg = re.search(r"(?:【对白】|【台词】|对白\s*[:：]|台词\s*[:：])\s*([^\n【]+)", blk)
        dlg_raw = m_dlg.group(1).strip() if m_dlg else ""
        if dlg_raw in ("无", "暂无", "无对白", "无台词", "none", "None", "-"):
            dlg_raw = ""

        m_voice = re.search(r"(?:【音色】|音色\s*[:：])\s*([^\n【]+)", blk)
        voice_ov = m_voice.group(1).strip() if m_voice else ""
        if voice_ov in ("无", "默认", "自动", "none"):
            voice_ov = ""

        m_eng = re.search(r"(?:【引擎】|引擎\s*[:：])\s*([^\n【]+)", blk)
        eng_ov = m_eng.group(1).strip().lower() if m_eng else ""

        speaker_role = "A"
        spoken_text = ""
        if dlg_raw:
            m_sp = re.match(r"^\s*(?:角色|人物|女主|男主|配角)?\s*([A-Z])\b", dlg_raw, flags=re.IGNORECASE)
            if m_sp and m_sp.group(1).upper() in roles_dict:
                speaker_role = m_sp.group(1).upper()
            elif re.match(r"^\s*(?:男主|男)\b", dlg_raw) and "B" in roles_dict:
                speaker_role = "B"
            else:
                for rid, r_info in roles_dict.items():
                    r_nm = r_info.get("name") or ""
                    if r_nm and dlg_raw.startswith(r_nm):
                        speaker_role = rid
                        break
            m_q = re.search(r"[“\"「『]([^”\"」』]+)[”\"」』]", dlg_raw)
            spoken_text = m_q.group(1).strip() if m_q else re.sub(r"^[^:：]+[:：]\s*", "", dlg_raw).strip()
            # 自动将中文括号情绪词或语气词转换为 Ming-3.0 副语言气口标签
            if spoken_text.startswith(("哼～", "哼~", "哼，", "（冷哼）", "(冷哼)", "（傲娇）")):
                spoken_text = "[Dissatisfaction-hnn] " + re.sub(r"^(?:哼[～~，]?|[（(][^）)]+[）)])\s*", "", spoken_text)
            elif spoken_text.startswith(("唉～", "唉~", "唉，", "（叹气）", "(叹气)")):
                spoken_text = "[Sigh] " + re.sub(r"^(?:唉[～~，]?|[（(][^）)]+[）)])\s*", "", spoken_text)
            elif spoken_text.startswith(("哈哈", "呵呵", "扑哧", "（轻笑）", "(轻笑)", "（笑）")):
                spoken_text = "[Laughter] " + re.sub(r"^(?:[（(][^）)]+[）)])\s*", "", spoken_text)
            elif spoken_text.startswith(("（喘息）", "(喘息)", "（深呼吸）", "(深呼吸)")):
                spoken_text = "[Breathing] " + re.sub(r"^(?:[（(][^）)]+[）)])\s*", "", spoken_text)

        # 判断该镜出现的角色参考图与 Picture 锁脸指令 (优先读取【出场角色】，并智能剔除越肩前景背影与无脸手脚特写)
        spoken_words_only = re.sub(r"\[[A-Za-z-]+\]", "", spoken_text).strip()
        check_span = f"{cam_txt} {vis_txt} {first_ov_txt}"
        # 剔除“看着她/他”以及“从X肩后拍/前景是X模糊的肩”中的非正脸前景背景角色，防止正反打传错人脸
        span_no_look_obj = re.sub(r"(?:看着|望向|望着|看向|盯着|背对|指向|冲着)[她他]|从[^，,）)]*肩后(?:拍|看|望)?|前景是[^，,；;。]*肩[^，,；;。]*", "", check_span)

        is_no_face_closeup = (
            not spoken_words_only
            and any(k in check_span for k in ("手部特写", "贴地特写", "皮鞋踏入", "细针从她指间滑落", "剪刀剪断", "取下那把挂了七年", "指尖轻轻碰了碰"))
            and not any(k in check_span for k in ("面部", "侧脸", "眼睫", "眼神", "抬头", "低头看"))
        )

        present_roles: List[str] = []
        if is_no_face_closeup:
            present_roles = []
        elif spoken_words_only and speaker_role in roles_dict and not any(k in cam_txt for k in ("双人", "两人", "全景")):
            # 单人说话镜头 (LongCat-Avatar 数字人驱动)：只传当前开口说话角色的定妆照，严禁混入其他角色导致替身张嘴
            present_roles = [speaker_role]
        elif roles_line_txt:
            if not any(k in roles_line_txt for k in ("无", "空镜", "none", "None")):
                for m_rl in re.finditer(r"([A-Z])", roles_line_txt.upper()):
                    cand_r = m_rl.group(1)
                    if cand_r in roles_dict and cand_r not in present_roles:
                        present_roles.append(cand_r)
                for rid, r_info in roles_dict.items():
                    r_nm = r_info.get("name") or ""
                    if r_nm and r_nm in roles_line_txt and rid not in present_roles:
                        present_roles.append(rid)
        else:
            has_both_ab = any(k in span_no_look_obj for k in ("两人", "二人", "双人", "一前一后", "共撑"))
            for rid in sorted_role_ids:
                r_info = roles_dict[rid]
                r_nm = r_info.get("name") or ""
                tokens = [f"角色{rid}", f"人物{rid}"]
                if r_nm and not r_nm.startswith("角色"):
                    tokens.append(r_nm)
                if rid == "A" and len(roles_dict) <= 2:
                    tokens.extend(["女主", "她", "少女", "女孩", "女子"])
                elif rid == "B" and len(roles_dict) <= 2:
                    tokens.extend(["男主", "他", "公子", "男人", "男子", "少侠", "总裁"])
                is_present = (
                    (has_both_ab and rid in ("A", "B"))
                    or any(t in span_no_look_obj for t in tokens if t)
                    or (bool(spoken_text) and speaker_role == rid)
                    or bool(re.search(rf"(?<![A-Za-z]){rid}(?![A-Za-z])", span_no_look_obj))
                )
                if is_present and rid not in present_roles:
                    present_roles.append(rid)

        is_pure_scenery = (
            is_no_face_closeup
            or (roles_line_txt and any(k in roles_line_txt for k in ("无", "空镜", "none", "None")))
            or (not present_roles and not spoken_text)
        )

        refs: List[str] = []
        subj_defs: List[str] = []
        subj_names: List[str] = []
        if not is_pure_scenery and present_roles:
            for p_idx, rid in enumerate(present_roles[:4], start=1):
                refs.append(f"char_{rid}_half.jpg")
                r_nm = roles_dict[rid].get("name") or f"角色{rid}"
                subj_defs.append(f"The person in <Picture {p_idx}> is <Subject {p_idx}> ({r_nm}).")
                subj_names.append(f"<Subject {p_idx}>")
            joined_defs = " ".join(subj_defs)
            joined_subjs = " and ".join(subj_names)
            ref_hint = (
                f"{joined_defs} Generate a brand-new cinematic shot and camera angle showing {joined_subjs} "
                f"(keep only {joined_subjs}'s exact facial identity, hairstyle, and clothing, while completely changing the camera framing, shot scale, pose, action, and background environment to match this scene): "
            )
        else:
            ref_hint = ""

        # 若画面采用“起幅：...；随后：...”结构，首帧只取“起幅”段，尾帧取“随后”段，确保首尾关键帧画面精准分离
        has_split_transition = ("随后" in vis_txt or "接着" in vis_txt)
        if first_ov_txt:
            first_img_prompt = first_ov_txt
        elif has_split_transition:
            first_img_prompt = re.split(r"[；;。]?\s*(?:随后|接着)\s*[:：]?", vis_txt, maxsplit=1)[0].replace("起幅：", "").replace("起幅:", "").strip()
        else:
            first_img_prompt = vis_txt

        first_img_fn = f"{shot_id}_first.jpg"
        short_vis = re.sub(r"[，。、；：！？\"'“”‘’（）()\s]", "", first_img_prompt)[:10] or f"分镜{idx:02d}"
        is_exterior_shot = any(k in first_img_prompt for k in ("夜景", "江畔", "高楼", "外立面", "俯瞰", "航拍", "长廊", "走廊", "街道", "室外")) and ("包厢内" not in first_img_prompt)
        scene_prefix = f"{scene_desc}。" if (scene_desc and not is_exterior_shot) else ""
        image_tasks.append({
            "file": first_img_fn,
            "label": f"📷 镜{idx:02d}首帧:{short_vis}",
            "refs": refs,
            "prompt": f"{ref_hint}{global_style} {scene_prefix}{cam_txt}，{first_img_prompt}",
        })

        # 判断是否需要生成尾帧 (显式提供了【尾帧】，或画面包含“起幅...随后/接着...”)
        last_img_fn = ""
        if last_ov_txt and last_ov_txt not in ("无", "暂无", "none"):
            last_img_fn = f"{shot_id}_last.jpg"
            short_last = re.sub(r"[，。、；：！？\"'“”‘’（）()\s]", "", last_ov_txt)[:10] or "落幅"
            image_tasks.append({
                "file": last_img_fn,
                "label": f"📷 镜{idx:02d}尾帧:{short_last}",
                "refs": refs,
                "prompt": f"{ref_hint}{global_style} {scene_prefix}{cam_txt}，{last_ov_txt}",
            })
        elif has_split_transition and (not spoken_words_only or dur_s >= 4.5):
            last_img_fn = f"{shot_id}_last.jpg"
            after_part = re.split(r"[；;。]?\s*(?:随后|接着)\s*[:：]?", vis_txt, maxsplit=1)[-1].strip()
            short_last = re.sub(r"[，。、；：！？\"'“”‘’（）()\s]", "", after_part)[:10] or "落幅"
            image_tasks.append({
                "file": last_img_fn,
                "label": f"📷 镜{idx:02d}尾帧:{short_last}",
                "refs": refs,
                "prompt": f"{ref_hint}{global_style} {scene_prefix}{cam_txt}，{after_part}",
            })

        # 智能路由引擎：只有包含真实可发音文字台词 (非纯 [Breathing] / [Surprise-ah] 无字气口) 且无首尾双帧转场的镜头才走 LongCat-Avatar
        if "avatar" in eng_ov or "数字人" in eng_ov:
            use_avatar = bool(spoken_words_only)
        elif "fl2va" in eng_ov or "首尾帧" in eng_ov:
            use_avatar = False
        else:
            use_avatar = bool(spoken_words_only) and (not last_img_fn)

        if not use_avatar and not ambience_src:
            ambience_src = f"{shot_id}_raw.mp4"

        # 优先使用剧本显式【音色】（并自动回填到角色卡槽），其次使用角色卡槽绑定的固定音色
        if voice_ov:
            tts_voice = voice_ov if (voice_ov.startswith("ming:") or voice_ov.startswith("zh-")) else f"ming:{voice_ov}"
            if speaker_role in roles_dict and spoken_words_only:
                roles_dict[speaker_role]["voice"] = tts_voice
        elif speaker_role in roles_dict and roles_dict[speaker_role].get("voice"):
            tts_voice = str(roles_dict[speaker_role]["voice"])
        else:
            tts_voice = default_voices.get(speaker_role, "ming:灵小柔")

        clean_spoken_preview = re.sub(r"\[[A-Za-z-]+\]", "", spoken_text).strip()[:7]
        vid_label = (
            f"🗣️ 镜{idx:02d}({dur_s}s):{speaker_role}「{clean_spoken_preview}」"
            if use_avatar
            else f"🎞️ 镜{idx:02d}({dur_s}s):{short_vis}"
        )
        sound_tail = f" 纯自然环境音效：{amb_txt}，无背景音乐。" if amb_txt else ("" if spoken_text else " 纯自然环境音效，无背景音乐，无人物说话。")
        video_tasks.append({
            "file": f"{shot_id}_raw.mp4",
            "label": vid_label,
            "engine": "avatar" if use_avatar else "fl2va",
            "duration": dur_s,
            "first_img": first_img_fn,
            "last_img": last_img_fn,
            "tts_text": spoken_text,
            "tts_voice": tts_voice,
            "mix_ambience": use_avatar,
            "prompt": f"{cam_txt} {vis_txt}{sound_tail}",
        })

    if not title:
        title = f"《AI分镜短剧·{len(video_tasks)}镜》"
    elif not title.startswith("《"):
        title = f"《{title}》"

    return {
        "title": title,
        "global_style": global_style,
        "ambience_source_file": ambience_src or (video_tasks[0]["file"] if video_tasks else "shot01_raw.mp4"),
        "raw_script": raw_script,
        "cast_list": final_cast_list,
        "user_uploaded_chars": sorted(list(locked_ids)),
        "image_tasks": image_tasks,
        "video_tasks": video_tasks,
    }


def _get_drama_status_payload() -> Dict[str, Any]:
    wf = _load_drama_workflow()
    steps_out = []
    done_cnt = 0
    active_fn = _DRAMA_STATE.get("active_file") or ""
    img_tasks = wf.get("image_tasks") or []
    vid_tasks = wf.get("video_tasks") or []
    all_defs = [(it["file"], it["label"]) for it in img_tasks]
    all_defs += [(vt["file"], vt["label"]) for vt in vid_tasks]
    tot_dur = round(sum(float(vt.get("duration") or 3.0) for vt in vid_tasks), 1)
    all_defs.append(("ancient_drama_30s_final.mp4", f"🎬 {tot_dur}s 完整短剧混音合片"))

    for fn, lbl in all_defs:
        p = DRAMA_WORK_DIR / fn
        is_done = p.is_file() and p.stat().st_size > 8192
        if is_done:
            done_cnt += 1
        if _DRAMA_STATE["running"] and fn == active_fn:
            st = "running"
        elif is_done:
            st = "done"
        else:
            st = "pending"
        steps_out.append({"file": fn, "label": lbl, "status": st})
    tot_cnt = len(all_defs)
    pct = _DRAMA_STATE["pct"] if _DRAMA_STATE["running"] else (int(round(done_cnt * 100.0 / max(1, tot_cnt))) if done_cnt > 0 else 0)
    return {
        "running": _DRAMA_STATE["running"],
        "pct": pct,
        "done_count": done_cnt,
        "total_count": tot_cnt,
        "shot_count": len(vid_tasks),
        "image_count": len(img_tasks),
        "total_duration": tot_dur,
        "title": wf.get("title") or "《分镜短剧》",
        "message": _DRAMA_STATE["message"],
        "error": _DRAMA_STATE["error"],
        "final_video": _DRAMA_STATE["final_video"],
        "steps": steps_out,
    }


def _drama_check_cancel():
    if _DRAMA_STATE.get("cancel"):
        raise RuntimeError("🛑 已暂停全自动导演流水线（已生成的定妆照与分镜已保存，随时可继续）")


def _drama_wait_for_engine(target_engine: str, repo: str, folder: str, pattern: str, label: str):
    """确保云端 G4 已加载并就绪指定的推理引擎 (minimax-h3-image / longcat-avatar-1.6 / minimax-h3)"""
    _drama_check_cancel()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    remote_already_loading = False
    try:
        with opener.open("http://127.0.0.1:30000/health", timeout=3) as r:
            if r.status == 200:
                hd = json.loads(r.read().decode("utf-8"))
                cur_eng = str(hd.get("engine") or "")
                if cur_eng == target_engine or (target_engine == "minimax-h3" and cur_eng in ("minimax-h3", "minimax-h3-t4")):
                    return
    except urllib.error.HTTPError as he:
        if he.code == 503:
            try:
                hd_err = json.loads(he.read().decode("utf-8"))
                if hd_err.get("status") == "loading":
                    remote_already_loading = True
            except Exception:
                pass
    except Exception:
        pass

    _DRAMA_STATE["message"] = f"🔄 正在云端 G4 自动加载/切换至 {label}..."
    _append_panel_log(f"🎬 [自动导演·引擎调度] 正在自动加载/切换云端引擎至 {label} ({folder})...")

    with _DEPLOY_LOCK:
        already_deploying = bool(_DEPLOY_STATE.get("running"))
    if not already_deploying and not remote_already_loading:
        _start_deploy_background(
            repo=repo,
            folder=folder,
            pattern=pattern,
            ctx=262144,
            gpu="G4",
            download_only=False,
        )

    for _ in range(900):
        _drama_check_cancel()
        time.sleep(2)
        with _DEPLOY_LOCK:
            d_run = bool(_DEPLOY_STATE.get("running"))
            d_code = _DEPLOY_STATE.get("exit_code")
        if not d_run:
            if d_code not in (0, None):
                raise RuntimeError(f"加载 {label} 失败，请查看下方实时日志")
            break

    for _ in range(180):
        _drama_check_cancel()
        try:
            with opener.open("http://127.0.0.1:30000/health", timeout=3) as r:
                if r.status == 200:
                    hd = json.loads(r.read().decode("utf-8"))
                    cur_eng = str(hd.get("engine") or "")
                    if cur_eng == target_engine or (target_engine == "minimax-h3" and cur_eng in ("minimax-h3", "minimax-h3-t4")):
                        _append_panel_log(f"✅ [自动导演·引擎就绪] {label} 已在云端就绪！")
                        return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"等待 {label} /health 就绪超时")


def _drama_submit_and_wait_media(job_payload: Dict[str, Any], out_path: Path, step_label: str) -> Path:
    """向云端 30000 网关提交单次图片或视频生成任务，轮询进度并高速回传保存至 out_path 与 ~/Movies/EasyColab/"""
    _drama_check_cancel()
    cfg = _load_endpoint_cfg()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.get('api_key', 'sk-colab-ed28e54eb298d0c1')}",
        "User-Agent": "ColabPanel/1.0",
    }
    body = json.dumps(job_payload).encode("utf-8")

    # 记录提交前云端已存在的所有旧文件名 (completed_videos / latest_video / history)，彻底杜绝极速轮询误取上一镜或历史旧视频导致片段重复！
    prev_known_files = set()
    try:
        st_req0 = urllib.request.Request("http://127.0.0.1:30000/v1/video/status", headers=headers, method="GET")
        with urllib.request.urlopen(st_req0, timeout=8) as sr0:
            sd0 = json.loads(sr0.read().decode("utf-8"))
            for f_item in (sd0.get("completed_videos") or []):
                if f_item:
                    prev_known_files.add(os.path.basename(str(f_item)))
            if sd0.get("latest_video"):
                prev_known_files.add(os.path.basename(str(sd0["latest_video"])))
            for f_item in (sd0.get("history") or []):
                if f_item:
                    prev_known_files.add(os.path.basename(str(f_item)))
    except Exception:
        pass

    # 若云端上一任务尚未完全释放 GPU (HTTP 409 或 already_running)，等待其完成后再提交新任务（绝不把上一镜当成本镜！）
    for _ in range(90):
        _drama_check_cancel()
        try:
            req = urllib.request.Request("http://127.0.0.1:30000/v1/video/generations", data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=25) as resp:
                res = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as he:
            if he.code == 409:
                time.sleep(2.5)
                continue
            raise
        if not res.get("ok"):
            if "正在生成" in str(res.get("error") or "") or "busy" in str(res.get("error") or "").lower():
                time.sleep(2.5)
                continue
            raise RuntimeError(f"{step_label} 提交失败: {res.get('error')}")
        if not res.get("already_running"):
            break
        # 若返回 already_running (例如 LongCat-Avatar 上一镜刚在收尾)，把当前 latest_video 加入 prev_known_files 并等待 2.5s 重新提交
        try:
            st_req_w = urllib.request.Request("http://127.0.0.1:30000/v1/video/status", headers=headers, method="GET")
            with urllib.request.urlopen(st_req_w, timeout=8) as sr_w:
                sd_w = json.loads(sr_w.read().decode("utf-8"))
                for f_item in (sd_w.get("completed_videos") or []):
                    if f_item:
                        prev_known_files.add(os.path.basename(str(f_item)))
                if sd_w.get("latest_video"):
                    prev_known_files.add(os.path.basename(str(sd_w["latest_video"])))
        except Exception:
            pass
        time.sleep(2.5)

    time.sleep(1.5)
    remote_fname = ""
    for poll_i in range(600):
        _drama_check_cancel()
        time.sleep(1.5)
        try:
            st_req = urllib.request.Request("http://127.0.0.1:30000/v1/video/status", headers=headers, method="GET")
            with urllib.request.urlopen(st_req, timeout=12) as sr:
                sd = json.loads(sr.read().decode("utf-8"))
        except Exception:
            continue

        if sd.get("error") and not sd.get("running"):
            raise RuntimeError(f"{step_label} 云端出错: {sd.get('error')}")

        msg = str(sd.get("message") or "").replace("总进度", "单镜进度")
        if sd.get("running"):
            if msg:
                _DRAMA_STATE["message"] = f"🎬 【总进度 {_DRAMA_STATE.get('pct', 0)}%】[{step_label}] {msg}"
                _append_panel_log(f"[自动导演·总进度{_DRAMA_STATE.get('pct', 0)}%] {step_label}: {msg}")
            continue

        # 仅从本次任务产出的 completed_videos / latest_video 读取（严禁回退到 history 历史旧列表，且文件名必须不在 prev_known_files 中！）
        vids = sd.get("completed_videos") or ([sd["latest_video"]] if sd.get("latest_video") else [])
        if vids:
            cand_fname = os.path.basename(str(vids[-1]))
            if cand_fname and cand_fname not in prev_known_files:
                remote_fname = cand_fname
                break

    if not remote_fname:
        raise RuntimeError(f"{step_label} 未获取到生成文件名")

    _DRAMA_STATE["message"] = f"⬇️ [{step_label}] 正在 16 线程极速回传 {remote_fname}..."
    local_dl = LOCAL_MOVIES_DIR / remote_fname
    # 若前端 pollVideoStatus 正在并发回传同一个文件 (_VIDEO_DL_LOCK 被占用)，耐心等待其落盘或自动重试
    for _ in range(30):
        _drama_check_cancel()
        if local_dl.is_file() and local_dl.stat().st_size > 4096:
            break
        if _VIDEO_DL_STATE.get("active"):
            time.sleep(1.0)
            continue
        _download_remote_video_to_local(remote_fname, 0.0)
        if local_dl.is_file() and local_dl.stat().st_size > 4096:
            break
        time.sleep(1.2)

    if not (local_dl.is_file() and local_dl.stat().st_size > 4096):
        raise RuntimeError(f"{step_label} 回传文件 {remote_fname} 失败")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_dl, out_path)
    _append_panel_log(f"✅ [自动导演·单步完成] {step_label} 已落盘 → {out_path.name}")
    return out_path


def _drama_img_to_b64(img_path: Path, target_res: str = "") -> str:
    if target_res and "x" in target_res:
        try:
            from PIL import Image, ImageOps
            tw_s, th_s = target_res.lower().split("x", 1)
            tw, th = int(tw_s), int(th_s)
            target_ratio = float(tw) / float(max(1, th))
            im = Image.open(img_path).convert("RGB")
            cur_ratio = float(im.width) / float(max(1, im.height))
            if abs(cur_ratio - target_ratio) > 0.06:
                if target_ratio > cur_ratio:
                    new_w = im.width
                    new_h = max(64, int(round(new_w / target_ratio / 16.0)) * 16)
                else:
                    new_h = im.height
                    new_w = max(64, int(round(new_h * target_ratio / 16.0)) * 16)
                im = ImageOps.fit(im, (new_w, new_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.25))
                im.save(img_path, format="JPEG", quality=96)
        except Exception:
            pass
    ext = img_path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(img_path.read_bytes()).decode("ascii")


def _drama_run_image_step(
    fn: str,
    step_label: str,
    prompt: str,
    ref_files: List[str],
    res_str: str,
    resume: bool,
    step_idx: int,
    tot_steps: int,
) -> Path:
    _drama_check_cancel()
    out_p = DRAMA_WORK_DIR / fn
    _DRAMA_STATE["active_file"] = fn
    _DRAMA_STATE["pct"] = max(2, int(round((step_idx - 1) * 100.0 / tot_steps)))
    if resume and out_p.is_file() and out_p.stat().st_size > 8192:
        _append_panel_log(f"♻️ [自动导演·断点复用] {step_label} 已存在 ({fn})，跳过重绘")
        return out_p

    w_s, h_s = res_str.split("x")
    ref_b64s = []
    for rf in ref_files:
        rp = DRAMA_WORK_DIR / rf
        if rp.is_file():
            ref_b64s.append(_drama_img_to_b64(rp))

    _DRAMA_STATE["message"] = f"📸 [{step_idx}/{tot_steps}] 正在生成 {step_label}..."
    _append_panel_log(f"📸 [自动导演 {step_idx}/{tot_steps}] 开始生成图片 {step_label} ({res_str})...")
    payload = {
        "mode": "image",
        "prompt": prompt,
        "resolution": res_str,
        "width": int(w_s),
        "height": int(h_s),
        "steps": 6,
        "seed": -1,
        "randomize_seed": True,
        "batch_count": 1,
        "ref_images_b64": ref_b64s,
        "edit_outfit": "",
        "edit_scene": "",
        "clean_matte": True,
    }
    return _drama_submit_and_wait_media(payload, out_p, step_label)


def _drama_run_avatar_step(
    fn: str,
    step_label: str,
    ref_img_fn: str,
    tts_text: str,
    tts_voice: str,
    visual_prompt: str,
    res_str: str,
    resume: bool,
    step_idx: int,
    tot_steps: int,
    dur_s: float = 3.0,
) -> Path:
    _drama_check_cancel()
    out_p = DRAMA_WORK_DIR / fn
    _DRAMA_STATE["active_file"] = fn
    _DRAMA_STATE["pct"] = max(5, int(round((step_idx - 1) * 100.0 / tot_steps)))
    if resume and out_p.is_file() and out_p.stat().st_size > 20480:
        _append_panel_log(f"♻️ [自动导演·断点复用] {step_label} 已存在 ({fn})，跳过重新渲染")
        return out_p

    ref_p = DRAMA_WORK_DIR / ref_img_fn
    if not ref_p.is_file():
        raise RuntimeError(f"缺少首帧参考图 {ref_img_fn}")
    w_s, h_s = res_str.split("x")
    num_segs = 1 if dur_s <= 3.75 else max(1, min(20, 1 + int(math.ceil((dur_s - 3.72) / 3.20))))
    clean_vis = visual_prompt.rstrip("。.,， ") + "。电影摄影机实拍原片，纯洁干净画面 (raw unedited cinema camera footage, pure clean frame)。"
    _DRAMA_STATE["message"] = f"🗣️ [{step_idx}/{tot_steps}] 正在生成对白镜头 {step_label} ({dur_s}s · {num_segs}段)..."
    _append_panel_log(f"🗣️ [自动导演 {step_idx}/{tot_steps}] 开始生成数字人对白镜头 {step_label} (时长: {dur_s}s/{num_segs}段, 音色: {tts_voice}, 台词: {tts_text})...")
    payload = {
        "mode": "avatar",
        "prompt": clean_vis,
        "resolution": res_str,
        "width": int(w_s),
        "height": int(h_s),
        "num_segments": num_segs,
        "steps": 8,
        "fps": 25,
        "seed": -1,
        "randomize_seed": True,
        "batch_count": 1,
        "audio_b64": "",
        "audio_ext": ".wav",
        "is_tts_audio": True,
        "tts_text": tts_text,
        "tts_voice": tts_voice,
        "clone_ref_b64": "",
        "clone_prompt_text": "",
        "use_builtin_sample": False,
        "first_frame_b64": _drama_img_to_b64(ref_p, target_res=res_str),
    }
    return _drama_submit_and_wait_media(payload, out_p, step_label)


def _drama_run_fl2va_step(
    fn: str,
    step_label: str,
    first_img_fn: str,
    last_img_fn: Optional[str],
    prompt: str,
    res_str: str,
    resume: bool,
    step_idx: int,
    tot_steps: int,
    dur_s: float = 3.0,
) -> Path:
    _drama_check_cancel()
    out_p = DRAMA_WORK_DIR / fn
    _DRAMA_STATE["active_file"] = fn
    _DRAMA_STATE["pct"] = max(5, int(round((step_idx - 1) * 100.0 / tot_steps)))
    if resume and out_p.is_file() and out_p.stat().st_size > 20480:
        _append_panel_log(f"♻️ [自动导演·断点复用] {step_label} 已存在 ({fn})，跳过重新渲染")
        return out_p

    first_p = DRAMA_WORK_DIR / first_img_fn
    if not first_p.is_file():
        raise RuntimeError(f"缺少起幅首帧图 {first_img_fn}")
    w_s, h_s = res_str.split("x")
    # 根据分镜设定的秒数动态匹配最优采样帧数 (24fps: 90帧≈3.75s, 124帧≈5.16s, 192帧≈8.0s, 243帧≈10.1s, 294帧≈12.2s, 362帧≈15.1s)
    if dur_s <= 3.5:
        target_frames = 90
    elif dur_s <= 5.3:
        target_frames = 124
    elif dur_s <= 8.0:
        target_frames = 192
    elif dur_s <= 10.2:
        target_frames = 243
    elif dur_s <= 12.5:
        target_frames = 294
    else:
        target_frames = 362
    clean_fl_prompt = prompt.rstrip("。.,， ") + "。电影摄影机实拍原片，纯洁干净画面 (raw unedited cinema camera footage, pure clean frame)。"
    _DRAMA_STATE["message"] = f"🎞️ [{step_idx}/{tot_steps}] 正在生成动作与环境音镜头 {step_label} ({dur_s}s · {target_frames}帧)..."
    _append_panel_log(f"🎞️ [自动导演 {step_idx}/{tot_steps}] 开始生成首尾帧有声视频 {step_label} ({dur_s}s · {target_frames}帧)...")
    payload: Dict[str, Any] = {
        "mode": "fl2va",
        "prompt": clean_fl_prompt,
        "negative_prompt": "字幕，底部字幕，台词文字，水印，乱码，画面拉伸，人物变形，subtitles, captions, text on screen, lower thirds, watermark, logo, stretched face, distorted face",
        "resolution": res_str,
        "width": int(w_s),
        "height": int(h_s),
        "num_frames": target_frames,
        "steps": 25,
        "fps": 24,
        "seed": -1,
        "randomize_seed": True,
        "batch_count": 1,
        "first_frame_b64": _drama_img_to_b64(first_p, target_res=res_str),
    }
    if last_img_fn:
        last_p = DRAMA_WORK_DIR / last_img_fn
        if last_p.is_file():
            payload["last_frame_b64"] = _drama_img_to_b64(last_p, target_res=res_str)
    return _drama_submit_and_wait_media(payload, out_p, step_label)


def _drama_assemble_30s_movie(work_dir: Path, aspect: str = "16:9", auto_2k: bool = False) -> Path:
    """
    将工作流中的所有分镜视频按各自配置的 duration 秒数裁切对齐，
    并将环境底噪源自动以 -16dB 铺入数字人对白背景，输出 1280x720 (或 720x1280) 25fps 完整短剧成片。
    """
    wf = _load_drama_workflow()
    vid_tasks = wf.get("video_tasks") or []
    if not vid_tasks:
        raise RuntimeError("当前工作流没有任何视频分镜节点")

    tot_dur = round(sum(float(vt.get("duration") or 3.0) for vt in vid_tasks), 1)
    _DRAMA_STATE["active_file"] = "ancient_drama_30s_final.mp4"
    _DRAMA_STATE["pct"] = 95
    _DRAMA_STATE["message"] = f"🎞️ 正在执行 FFmpeg 0.1 秒级卡点裁切 + 环境底噪混音 + {tot_dur}s 完整短剧封装..."
    _append_panel_log(f"🎞️ [自动导演·阶段四] 开始按工作流配置的 {len(vid_tasks)} 个分镜秒数 (总计 {tot_dur}s) 精剪混音并封装成片...")

    shot_specs = [
        (
            str(vt.get("file") or ""),
            float(vt.get("duration") or 3.0),
            bool(vt.get("mix_ambience")) or (str(vt.get("engine")) == "avatar"),
            str(vt.get("label") or vt.get("file") or ""),
        )
        for vt in vid_tasks
    ]

    missing = [lbl for fn, _, _, lbl in shot_specs if not (work_dir / fn).is_file()]
    if missing:
        raise RuntimeError(f"尚未生成完毕以下分镜视频，无法合片: {', '.join(missing)}")

    out_w, out_h = (720, 1280) if aspect == "9:16" else (1280, 720)
    norm_dir = work_dir / "norm_clips"
    norm_dir.mkdir(parents=True, exist_ok=True)

    # 提取环境底噪音频 (优先使用 workflow 配置的 ambience_source_file，否则取第一个 fl2va 镜头)
    amb_src_fn = str(wf.get("ambience_source_file") or "shot05_raw.mp4")
    amb_src_p = work_dir / amb_src_fn
    if not amb_src_p.is_file():
        fl2va_cands = [work_dir / str(vt.get("file")) for vt in vid_tasks if str(vt.get("engine")) != "avatar" and (work_dir / str(vt.get("file"))).is_file()]
        if fl2va_cands:
            amb_src_p = fl2va_cands[0]

    amb_wav = norm_dir / "market_ambience_bed.wav"
    if amb_src_p.is_file():
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-stream_loop", "3",
                    "-i", str(amb_src_p),
                    "-vn", "-ac", "2", "-ar", "44100",
                    "-t", "12.0",
                    str(amb_wav),
                ],
                check=True,
                timeout=30,
            )
        except Exception:
            amb_wav = Path("/nonexistent_amb.wav")
    else:
        amb_wav = Path("/nonexistent_amb.wav")

    norm_files: List[Path] = []
    scale_w = ((int(round(out_w * 1.14)) + 1) // 2) * 2
    scale_h = ((int(round(out_h * 1.14)) + 1) // 2) * 2
    for idx, (fn, dur_s, is_avatar_dlg, lbl) in enumerate(shot_specs, start=1):
        src_p = work_dir / fn
        dst_p = norm_dir / f"norm_{idx:02d}_{dur_s:.1f}s.mp4"
        # 采用 1.14x 电影画幅上移裁切 (切除顶部 1.2% 与底部 11.1% 潜在字幕区)，物理级 100% 根除任何底部乱码字幕！
        vf_chain = (
            f"fps=25,scale={scale_w}:{scale_h}:flags=lanczos:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h}:(iw-{out_w})/2:(ih-{out_h})*0.10,setsar=1,format=yuv420p,"
            f"tpad=stop_mode=clone:stop_duration=0.8"
        )
        fade_out_st = max(0.05, round(dur_s - 0.05, 2))
        if is_avatar_dlg and amb_wav.is_file():
            af_complex = (
                f"[0:a]aresample=44100,aformat=channel_layouts=stereo,apad[va];"
                f"[1:a]aresample=44100,aformat=channel_layouts=stereo,volume=0.16[amb];"
                f"[va][amb]amix=inputs=2:duration=first:dropout_transition=0,"
                f"afade=t=in:st=0:d=0.04,afade=t=out:st={fade_out_st}:d=0.04[aout]"
            )
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(src_p),
                "-i", str(amb_wav),
                "-filter_complex", f"[0:v]{vf_chain}[vout];{af_complex}",
                "-map", "[vout]", "-map", "[aout]",
                "-t", f"{dur_s:.2f}",
                "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
                str(dst_p),
            ]
        else:
            af_chain = f"aresample=44100,aformat=channel_layouts=stereo,apad,afade=t=in:st=0:d=0.04,afade=t=out:st={fade_out_st}:d=0.04"
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(src_p),
                "-vf", vf_chain,
                "-af", af_chain,
                "-t", f"{dur_s:.2f}",
                "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
                str(dst_p),
            ]
        subprocess.run(cmd, check=True, timeout=60)
        norm_files.append(dst_p)

    concat_txt = norm_dir / "concat_list.txt"
    concat_txt.write_text("\n".join(f"file '{p.resolve()}'" for p in norm_files) + "\n", encoding="utf-8")

    ts = time.strftime("%Y%m%d_%H%M%S")
    final_work_p = work_dir / "ancient_drama_30s_final.mp4"
    final_gallery_p = LOCAL_MOVIES_DIR / f"drama_workflow_{len(vid_tasks)}shots_{int(round(tot_dur))}s_{ts}.mp4"

    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c:v", "libx264", "-crf", "16", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(final_work_p),
        ],
        check=True,
        timeout=90,
    )
    shutil.copy2(final_work_p, final_gallery_p)
    _append_panel_log(f"🎉 [自动导演·合片完成] {tot_dur} 秒 {len(vid_tasks)} 镜完整短剧成片已生成 → {final_gallery_p}")

    if auto_2k:
        try:
            _append_panel_log(f"✨ [自动导演·ReCaVSR] 正在将成片 {final_gallery_p.name} 提交云端 ReCaVSR 2K 超分...")
            cfg = _load_endpoint_cfg()
            req_up = urllib.request.Request(
                f"{_get_recavsr_base_url()}/v1/video/upscale",
                data=json.dumps({
                    "filename": final_gallery_p.name,
                    "target_preset": "2k",
                    "color_fix": "wavelet",
                    "video_b64": base64.b64encode(final_gallery_p.read_bytes()).decode("ascii"),
                }).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {cfg.get('api_key', 'sk-colab-ed28e54eb298d0c1')}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req_up, timeout=30) as _:
                pass
        except Exception as e_up:
            _append_panel_log(f"⚠️ 自动提交 2K 超分提示: {e_up}")

    return final_gallery_p


def _run_single_drama_step_bg(target_file: str, aspect: str = "16:9"):
    """单独重新生成工作流中的某一张定妆/关键帧图片或某一条分镜视频"""
    DRAMA_WORK_DIR.mkdir(parents=True, exist_ok=True)
    wf = _load_drama_workflow()
    img_res = "1152x2048" if aspect == "9:16" else "2048x1152"
    vid_res = "480x832" if aspect == "9:16" else "832x480"
    try:
        for it in (wf.get("image_tasks") or []):
            if it.get("file") == target_file:
                _drama_wait_for_engine(
                    target_engine="minimax-h3-image",
                    repo="Comfy-Org/MiniMax-H3-Image",
                    folder="MiniMax-H3-Image",
                    pattern="*.safetensors",
                    label="📸 MiniMax-H3-Image (33B · 4K锁脸定妆与首尾帧引擎)",
                )
                _drama_run_image_step(
                    it["file"],
                    it.get("label") or it["file"],
                    it.get("prompt") or "",
                    it.get("refs") or [],
                    img_res,
                    resume=False,
                    step_idx=1,
                    tot_steps=1,
                )
                _DRAMA_STATE["active_file"] = ""
                _DRAMA_STATE["message"] = f"✅ 已单独重抽完成图片节点: {it.get('label') or target_file}"
                return

        for vt in (wf.get("video_tasks") or []):
            if vt.get("file") == target_file:
                eng = str(vt.get("engine") or "fl2va")
                if eng == "avatar":
                    _drama_wait_for_engine(
                        target_engine="longcat-avatar-1.6",
                        repo="meituan-longcat/LongCat-Video-Avatar-1.6",
                        folder="LongCat-Video-Avatar-1.6",
                        pattern="base_model/*.safetensors",
                        label="🎭 LongCat-Video-Avatar-1.6 + Ming-3.0 (越肩对白驱动引擎)",
                    )
                    _drama_run_avatar_step(
                        vt["file"],
                        vt.get("label") or vt["file"],
                        vt.get("first_img") or "",
                        vt.get("tts_text") or "",
                        vt.get("tts_voice") or "ming:灵小柔",
                        vt.get("prompt") or "",
                        vid_res,
                        resume=False,
                        step_idx=1,
                        tot_steps=1,
                        dur_s=float(vt.get("duration") or 3.0),
                    )
                else:
                    _drama_wait_for_engine(
                        target_engine="minimax-h3",
                        repo="MiniMaxAI/MiniMax-H3",
                        folder="MiniMax-H3",
                        pattern="FL2VA/*.safetensors",
                        label="🎬 MiniMax-H3 (58B · FL2VA 首尾帧有声视频引擎)",
                    )
                    _drama_run_fl2va_step(
                        vt["file"],
                        vt.get("label") or vt["file"],
                        vt.get("first_img") or "",
                        vt.get("last_img") or None,
                        vt.get("prompt") or "",
                        vid_res,
                        resume=False,
                        step_idx=1,
                        tot_steps=1,
                        dur_s=float(vt.get("duration") or 3.0),
                    )
                _DRAMA_STATE["active_file"] = ""
                _DRAMA_STATE["message"] = f"✅ 已单独重跑完成视频分镜: {vt.get('label') or target_file}（可点「🎞️ 仅用已有分镜重新合成成片」更新全片）"
                return

        raise RuntimeError(f"在当前工作流中未找到节点 {target_file}")
    except Exception as e:
        _DRAMA_STATE["active_file"] = ""
        _DRAMA_STATE["error"] = str(e)
        _DRAMA_STATE["message"] = str(e)
        _append_panel_log(f"⚠️ [单步重抽中断] {e}")
    finally:
        _DRAMA_STATE["running"] = False


def _run_auto_drama_pipeline_bg(aspect: str = "16:9", resume: bool = True, auto_2k: bool = False):
    """三阶段全自动执行当前工作流 (workflow.json) 中的所有定妆图、分镜首尾帧、分流视频生成与剪辑合片"""
    DRAMA_WORK_DIR.mkdir(parents=True, exist_ok=True)
    wf = _load_drama_workflow()
    img_tasks = wf.get("image_tasks") or []
    vid_tasks = wf.get("video_tasks") or []
    avatar_tasks = [vt for vt in vid_tasks if str(vt.get("engine")) == "avatar"]
    fl2va_tasks = [vt for vt in vid_tasks if str(vt.get("engine")) != "avatar"]
    tot_steps = len(img_tasks) + len(vid_tasks) + 1
    img_res = "1152x2048" if aspect == "9:16" else "2048x1152"
    vid_res = "480x832" if aspect == "9:16" else "832x480"
    locked_char_ids = set(str(x).upper() for x in (wf.get("user_uploaded_chars") or []))
    for c_item in (wf.get("cast_list") or []):
        if isinstance(c_item, dict) and c_item.get("user_uploaded"):
            locked_char_ids.add(str(c_item.get("id") or "").upper())
    locked_char_files = {f"char_{cid}_half.jpg" for cid in locked_char_ids}
    if "A" in locked_char_ids:
        locked_char_files.add("char_A_closeup.jpg")

    try:
        # ==================== 阶段一 & 阶段二：MiniMax-H3-Image 生成所有定妆照 + 分镜首尾关键帧 ====================
        need_any_img = any(
            not ((resume or it["file"] in locked_char_files) and (DRAMA_WORK_DIR / it["file"]).is_file() and (DRAMA_WORK_DIR / it["file"]).stat().st_size > 8192)
            for it in img_tasks
        )
        if need_any_img:
            _drama_wait_for_engine(
                target_engine="minimax-h3-image",
                repo="Comfy-Org/MiniMax-H3-Image",
                folder="MiniMax-H3-Image",
                pattern="*.safetensors",
                label="📸 MiniMax-H3-Image (33B · 4K锁脸定妆与首尾帧引擎)",
            )

        step_cursor = 1
        for it in img_tasks:
            use_resume_for_img = resume or (it["file"] in locked_char_files)
            _drama_run_image_step(
                it["file"],
                it.get("label") or it["file"],
                it.get("prompt") or "",
                it.get("refs") or [],
                img_res,
                use_resume_for_img,
                step_cursor,
                tot_steps,
            )
            step_cursor += 1

        # ==================== 阶段三-A：LongCat-Video-Avatar-1.6 + Ming-3.0 集中生成所有数字人对白镜头 ====================
        need_any_avatar = any(
            not (resume and (DRAMA_WORK_DIR / vt["file"]).is_file() and (DRAMA_WORK_DIR / vt["file"]).stat().st_size > 20480)
            for vt in avatar_tasks
        )
        if need_any_avatar:
            _drama_wait_for_engine(
                target_engine="longcat-avatar-1.6",
                repo="meituan-longcat/LongCat-Video-Avatar-1.6",
                folder="LongCat-Video-Avatar-1.6",
                pattern="base_model/*.safetensors",
                label="🎭 LongCat-Video-Avatar-1.6 + Ming-3.0 (越肩对白驱动引擎)",
            )

        for vt in avatar_tasks:
            _drama_run_avatar_step(
                vt["file"],
                vt.get("label") or vt["file"],
                vt.get("first_img") or "",
                vt.get("tts_text") or "",
                vt.get("tts_voice") or "ming:灵小柔",
                vt.get("prompt") or "",
                vid_res,
                resume,
                step_cursor,
                tot_steps,
                dur_s=float(vt.get("duration") or 3.0),
            )
            step_cursor += 1

        # ==================== 阶段三-B：MiniMax-H3 (FL2VA) 集中生成所有动作与原生环境音镜头 ====================
        need_any_fl2va = any(
            not (resume and (DRAMA_WORK_DIR / vt["file"]).is_file() and (DRAMA_WORK_DIR / vt["file"]).stat().st_size > 20480)
            for vt in fl2va_tasks
        )
        if need_any_fl2va:
            _drama_wait_for_engine(
                target_engine="minimax-h3",
                repo="MiniMaxAI/MiniMax-H3",
                folder="MiniMax-H3",
                pattern="FL2VA/*.safetensors",
                label="🎬 MiniMax-H3 (58B · FL2VA 首尾帧有声视频引擎)",
            )

        for vt in fl2va_tasks:
            _drama_run_fl2va_step(
                vt["file"],
                vt.get("label") or vt["file"],
                vt.get("first_img") or "",
                vt.get("last_img") or None,
                vt.get("prompt") or "",
                vid_res,
                resume,
                step_cursor,
                tot_steps,
                dur_s=float(vt.get("duration") or 3.0),
            )
            step_cursor += 1

        # ==================== 阶段四：FFmpeg 0.1s 精准卡点 + 环境底噪混音 + 成片封装 ====================
        final_p = _drama_assemble_30s_movie(DRAMA_WORK_DIR, aspect=aspect, auto_2k=auto_2k)
        _DRAMA_STATE["active_file"] = ""
        _DRAMA_STATE["final_video"] = final_p.name
        _DRAMA_STATE["pct"] = 100
        _DRAMA_STATE["error"] = ""
        _DRAMA_STATE["message"] = f"🎉 全自动工作流拍摄与精剪完成！完整短剧成片已保存至 ~/Movies/EasyColab/{final_p.name}"
    except Exception as e:
        _DRAMA_STATE["active_file"] = ""
        _DRAMA_STATE["error"] = str(e)
        _DRAMA_STATE["message"] = str(e)
        _append_panel_log(f"⚠️ [自动导演中断] {e}")
    finally:
        _DRAMA_STATE["running"] = False


def _ensure_macos_launch_agent():
    """在 macOS 上自动写入 LaunchAgent 配置，确保重启电脑后 18080 控制台与 30000 网关开机自启"""
    if sys.platform != "darwin":
        return
    try:
        plist_dir = Path.home() / "Library" / "LaunchAgents"
        plist_path = plist_dir / "com.easycolab.panel.plist"
        if plist_path.is_file():
            return
        plist_dir.mkdir(parents=True, exist_ok=True)
        py_bin = str(Path(sys.executable).resolve())
        panel_py = str(Path(__file__).resolve())
        work_dir = str(Path(__file__).resolve().parent)
        log_path = str(Path(__file__).resolve().parent / ".colab_panel_daemon.log")
        plist_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.easycolab.panel</string>
    <key>ProgramArguments</key>
    <array>
        <string>{py_bin}</string>
        <string>{panel_py}</string>
        <string>--no-browser</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{work_dir}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_path}</string>
    <key>StandardErrorPath</key>
    <string>{log_path}</string>
</dict>
</plist>
"""
        plist_path.write_text(plist_xml, encoding="utf-8")
    except Exception:
        pass


def main():
    _ensure_macos_launch_agent()
    # 确保 127.0.0.1:30000 本地固定转发网关已启动
    from deploy_colab_model import ensure_local_proxy_running
    ensure_local_proxy_running(30000)
    _query_local_gdrive_info(force=True)

    url = f"http://127.0.0.1:{PANEL_PORT}"
    print("=" * 68)
    print(f" 🚀 Colab G4 可视化网页控制台已启动: {url}")
    print("    • 本机永久固定 OpenAI API 地址: http://127.0.0.1:30000/v1")
    print("=" * 68, flush=True)

    if "--no-browser" not in sys.argv:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("127.0.0.1", PANEL_PORT), PanelRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n控制台已退出。")


if __name__ == "__main__":
    main()
