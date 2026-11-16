#!/usr/bin/env python3
"""
Google Colab G4 本地可视化网页控制台 (http://127.0.0.1:18080)
提供免记命令的图形化界面：
  - 一键启动 / 切换模型 / 挂载 Google Drive (MyDrive/models/<文件夹>)
  - 一键关停 G4 实例省算力 (colab stop)
  - 显存 (VRAM)、系统内存 (RAM)、内存盘 (/dev/shm)、NVMe 硬盘、Google Drive 实时进度条监控
  - 一键复制永久固定 API 地址 (http://127.0.0.1:30000/v1) 与在线快速对话测试
"""

import json
import os
import re
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
from typing import Any, Dict, List

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
            m_dl = re.search(r"\[视频回传\]\s+([^:]+):", s)
            if m_dl and f"[视频回传] {m_dl.group(1)}:" in prev:
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
    if "[提交视频生成任务]" in clean or "正在使用 32B 无审查多模态编码器解析提示词" in clean:
        _VIDEO_STEP_TIMING["first_step_ts"] = 0.0
        _VIDEO_STEP_TIMING["first_step_num"] = 0.0
    elif "正在扩散采样生成有声视频:" in clean:
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
            m_dl = re.match(r"^\[视频回传\]\s+([^:]+):", clean)
            for idx in range(len(logs) - 1, max(-1, len(logs) - 7), -1):
                cand_ln = logs[idx]
                if m_dl and f"[视频回传] {m_dl.group(1)}:" in cand_ln:
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
        if "[提交视频生成任务]" in ln:
            m_t = re.match(r"^\[(\d{2}:\d{2}:\d{2})\]", ln)
            start_ts = _parse_log_hms_to_epoch(m_t.group(1), now_ts) if m_t else now_ts
            start_idx = i
            break
    if start_idx < 0:
        for i in range(len(lines) - 1, -1, -1):
            ln = lines[i]
            if "[视频生成]" in ln:
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
        if "正在使用 32B 无审查多模态编码器解析提示词" in ln:
            m_p = re.match(r"^\[(\d{2}:\d{2}:\d{2})\]", ln)
            if m_p:
                prompt_enc_ts = _parse_log_hms_to_epoch(m_p.group(1), now_ts)
        elif "正在扩散采样生成有声视频" in ln:
            m_s = re.search(r"^\[(\d{2}:\d{2}:\d{2})\].*第\s*(\d+)/(\d+)\s*步", ln)
            if m_s:
                ts_val = _parse_log_hms_to_epoch(m_s.group(1), now_ts)
                c_s = int(m_s.group(2))
                t_s = int(m_s.group(3))
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
        url = f"http://127.0.0.1:30000/v1/video/file/{urllib.parse.quote(safe_name)}"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {cfg.get('api_key', 'sk-colab-ed28e54eb298d0c1')}"},
            method="GET",
        )
        tmp_path = LOCAL_MOVIES_DIR / (safe_name + ".part")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp, open(tmp_path, "wb") as wf:
                tot_bytes = int(resp.headers.get("Content-Length", "0") or "0")
                tot_mb = tot_bytes / (1024 * 1024) if tot_bytes > 0 else 12.0
                _VIDEO_DL_STATE["tot_mb"] = tot_mb
                done_bytes = 0
                last_log_t = 0.0
                while True:
                    chunk = resp.read(131072)
                    if not chunk:
                        break
                    wf.write(chunk)
                    done_bytes += len(chunk)
                    done_mb = done_bytes / (1024 * 1024)
                    pct = min(99, int(done_bytes * 100 // max(1, tot_bytes))) if tot_bytes > 0 else 50
                    _VIDEO_DL_STATE["done_mb"] = done_mb
                    _VIDEO_DL_STATE["pct"] = pct
                    now = time.time()
                    if now - last_log_t >= 2.0:
                        last_log_t = now
                        _append_panel_log(f"[视频回传] {safe_name}: {pct}% ({done_mb:.1f}/{tot_mb:.1f} MB, 保存至 ~/Movies/EasyColab)")
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
            _append_panel_log(f"✅ [视频已落盘] {safe_name} 已完整保存至 {LOCAL_MOVIES_DIR}/{safe_name}")
            return True
        except Exception as ex:
            _append_panel_log(f"⚠️ [视频回传重试] {safe_name}: {ex}")
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
            _DEPLOY_STATE["logs"] = _compact_progress_log_lines(_init_logs_obj["logs"])[-400:]
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
    cfg = _load_endpoint_cfg()
    rc_url = str(cfg.get("recavsr_tunnel_url") or "").strip().rstrip("/")
    if rc_url:
        return rc_url
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


def _get_accounts_status(force: bool = False, cu_balance: Any = None) -> Dict[str, Any]:
    now = time.time()
    if not force and _ACCOUNT_INFO_CACHE["data"] and (now - _ACCOUNT_INFO_CACHE["timestamp"] < 45):
        cached = dict(_ACCOUNT_INFO_CACHE["data"])
        if cu_balance is not None and cached.get("active_email"):
            for it in cached.get("accounts", []):
                if it.get("email") == cached["active_email"]:
                    it["cu_balance"] = cu_balance
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
                    acc_map[active_email] = {
                        "email": active_email,
                        "name": active_name,
                        "picture": active_pic,
                        "token_json": token_json,
                        "rclone_conf": rclone_txt,
                        "cu_balance": cu_balance if cu_balance is not None else prev.get("cu_balance"),
                        "updated_at": time.strftime("%Y-%m-%d %H:%M"),
                    }
                    _save_saved_accounts(acc_map)
    except Exception:
        pass

    public_list = []
    for em, info in acc_map.items():
        public_list.append({
            "email": em,
            "name": info.get("name") or em,
            "picture": info.get("picture") or "",
            "cu_balance": info.get("cu_balance"),
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
        from colab_cli.common import SessionState, state
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
llama_proc = subprocess.getoutput("pgrep -af '[l]lama-server|minimax_server\\.py'").strip()
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
        "MiniMax-H3-Turbo",
        "Qwen3.8-27B-GGUF",
        "Qwen-Image-2.1",
        "Z-Image-Turbo",
    }
    auto_switched_note = ""
    if not download_only:
        try:
            cu_check = _query_cu_usage(force=False)
            bal_val = cu_check.get("paid_balance")
            if bal_val is not None and float(bal_val) <= 0.0:
                gpu = "T4"
                if folder in ("MiniMax-H3", "MiniMax-H3-Turbo"):
                    old_folder = folder
                    repo = "unsloth/MiniMax-H3-GGUF"
                    folder = "MiniMax-H3-T4-GGUF"
                    pattern = "minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf"
                    auto_switched_note = f"⚡ [T4 智能适配] 当前为 T4 (16GB) 账号，已自动将 {old_folder} 切换为 T4 专享量化版 {folder} (UD-Q2_K_XL · 8步有声视频)"
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
        except Exception as e:
            with _DEPLOY_LOCK:
                _DEPLOY_STATE["logs"].append(f"[错误] {e}")
                _DEPLOY_STATE["exit_code"] = 1
                _DEPLOY_STATE["running"] = False
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

      <div class="folder-tag preset-card" id="preset_minimax_turbo" onclick="selectPresetModel('preset_minimax_turbo', 'Comfy-Org/MiniMax-H3', 'MiniMax-H3-Turbo', '*.safetensors')" style="margin:0;padding:8px 10px;justify-content:space-between;">
        <div>
          <div style="font-weight:700;color:#f8fafc;font-size:13px;">⚡ MiniMax-H3 Turbo</div>
          <div style="color:var(--muted);font-size:11px;">INT8+FP4 · 37.3 GB (8步极速)</div>
        </div>
        <span class="preset-check" style="color:#3b82f6;font-weight:700;display:none;">✔</span>
      </div>

      <div class="folder-tag preset-card" id="preset_qwen_q4" onclick="selectPresetModel('preset_qwen_q4', 'unsloth/Qwen3.8-27B-GGUF', 'Qwen3.8-27B-GGUF', 'Qwen3.8-27B-UD-Q4_K_M.gguf')" style="margin:0;padding:8px 10px;justify-content:space-between;">
        <div>
          <div style="font-weight:700;color:#f8fafc;font-size:13px;">🚀 Qwen3.8-27B-GGUF</div>
          <div style="color:var(--muted);font-size:11px;">UD-Q4_K_M + MTP · 18.7 GB</div>
        </div>
        <span class="preset-check" style="color:#3b82f6;font-weight:700;display:none;">✔</span>
      </div>

      <div class="folder-tag preset-card" id="preset_qwen_img" onclick="selectPresetModel('preset_qwen_img', 'Qwen/Qwen-Image-2.1', 'Qwen-Image-2.1', '*.safetensors')" style="margin:0;padding:8px 10px;justify-content:space-between;">
        <div>
          <div style="font-weight:700;color:#f8fafc;font-size:13px;">🎨 Qwen-Image-2.1</div>
          <div style="color:var(--muted);font-size:11px;">BF16 原生 · 30.9 GB (文生图/修图)</div>
        </div>
        <span class="preset-check" style="color:#3b82f6;font-weight:700;display:none;">✔</span>
      </div>

      <div class="folder-tag preset-card" id="preset_z_img" onclick="selectPresetModel('preset_z_img', 'Tongyi-MAI/Z-Image-Turbo', 'Z-Image-Turbo', '*.safetensors')" style="margin:0;padding:8px 10px;justify-content:space-between;">
        <div>
          <div style="font-weight:700;color:#f8fafc;font-size:13px;">⚡ Z-Image-Turbo</div>
          <div style="color:var(--muted);font-size:11px;">8步极速生图 · 30.6 GB</div>
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
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--border);">
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
      <span style="font-size:16px;font-weight:800;color:#f8fafc;">🎬 MiniMax-H3 (33B · 原生BF16 · 无审查) 多模态有声视频创作工作台</span>
      <span style="font-size:11.5px;background:rgba(16,185,129,0.16);color:#34d399;padding:3px 10px;border-radius:999px;font-weight:600;">32kHz 原生立体声音轨 + 破限 32B 视觉编码器</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      <button type="button" id="tabBtnT2va" class="btn-primary" style="padding:5px 14px;font-size:12.5px;" onclick="switchMiniMaxMode('t2va')">✨ 文生音视频 (T2VA)</button>
      <button type="button" id="tabBtnFl2va" class="btn-outline" style="padding:5px 14px;font-size:12.5px;" onclick="switchMiniMaxMode('fl2va')">🖼️ 首尾帧生音视频 (FL2VA)</button>
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
  <div style="padding:18px;margin-bottom:16px;background:#0f172a;border:1px solid var(--border);border-radius:14px;">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:14px;">
      <div style="display:flex;align-items:center;gap:8px;font-size:15.5px;font-weight:800;color:#f8fafc;">
        <span style="color:#3b82f6;">☰</span>
        <span>镜头与多模态提示词</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <button type="button" class="btn-outline" style="padding:4px 12px;font-size:12px;border-color:#3b82f6;color:#93c5fd;" onclick="copyChatToVideoPrompt()">📋 从上方 Qwen3.8 结果填入四框</button>
        <button type="button" class="btn-outline" style="padding:4px 12px;font-size:12px;border-color:rgba(239,68,68,0.45);color:#fca5a5;" onclick="clearAllVideoPrompts()">🗑️ 一键清空四框</button>
      </div>
    </div>

    <!-- 框 1 (单独一行)：主体、构图、光影与风格描述 -->
    <div style="margin-bottom:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <label style="font-size:12.5px;font-weight:700;color:#94a3b8;margin:0;">1. 主体、构图、光影与风格描述</label>
        <button type="button" onclick="clearSingleVideoPrompt('videoPrompt')" style="padding:2px 8px;font-size:11px;border-radius:5px;border:1px solid var(--border);background:transparent;color:#94a3b8;cursor:pointer;" onmouseover="this.style.color='#fca5a5';this.style.borderColor='rgba(239,68,68,0.45)'" onmouseout="this.style.color='#94a3b8';this.style.borderColor='var(--border)'">✕ 清空</button>
      </div>
      <textarea id="videoPrompt" rows="4" placeholder="描述画面主体、构图、运镜方式、光影与风格..." style="width:100%;min-height:110px;padding:10px 14px;border-radius:10px;border:1px solid var(--border);background:#1e293b;color:#f8fafc;font-size:13.5px;line-height:1.55;resize:vertical;">Masterpiece Makoto Shinkai anime style, 2D cel-shaded animation. A teenage girl in a Japanese sailor school uniform standing at a sunlit railway crossing in late spring. A green commuter train speeds past in front of her, creating a strong gust of wind that blows her long black hair and a flurry of pink cherry blossom petals across the frame. Dramatic lens flare, glowing cumulous clouds in a deep blue sky, vibrant cinematic lighting, smooth 24fps keyframe animation.</textarea>
    </div>

    <!-- 框 2 (单独一行)：环境音效与台词对白（可选） -->
    <div style="margin-bottom:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <label style="font-size:12.5px;font-weight:700;color:#94a3b8;margin:0;">2. 环境音效与台词对白（可选）</label>
        <button type="button" onclick="clearSingleVideoPrompt('videoSoundPrompt')" style="padding:2px 8px;font-size:11px;border-radius:5px;border:1px solid var(--border);background:transparent;color:#94a3b8;cursor:pointer;" onmouseover="this.style.color='#fca5a5';this.style.borderColor='rgba(239,68,68,0.45)'" onmouseout="this.style.color='#94a3b8';this.style.borderColor='var(--border)'">✕ 清空</button>
      </div>
      <textarea id="videoSoundPrompt" rows="2" placeholder="例如：雨滴敲击声、脚步踩水声、角色台词对白..." style="width:100%;min-height:68px;padding:10px 14px;border-radius:10px;border:1px solid var(--border);background:#1e293b;color:#f8fafc;font-size:13px;line-height:1.5;resize:vertical;">Metallic clacking of train wheels on railway tracks, rhythmic dinging of the railroad crossing warning bell, warm spring wind rustling through leaves, a soft gasp from the girl.</textarea>
    </div>

    <!-- 框 3 (单独一行)：背景音乐风格（可选） -->
    <div style="margin-bottom:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <label style="font-size:12.5px;font-weight:700;color:#94a3b8;margin:0;">3. 背景音乐风格（可选）</label>
        <button type="button" onclick="clearSingleVideoPrompt('videoMusicPrompt')" style="padding:2px 8px;font-size:11px;border-radius:5px;border:1px solid var(--border);background:transparent;color:#94a3b8;cursor:pointer;" onmouseover="this.style.color='#fca5a5';this.style.borderColor='rgba(239,68,68,0.45)'" onmouseout="this.style.color='#94a3b8';this.style.borderColor='var(--border)'">✕ 清空</button>
      </div>
      <textarea id="videoMusicPrompt" rows="2" placeholder="例如：抒情独奏钢琴、赛博朋克合成器电子乐、恢弘管弦乐..." style="width:100%;min-height:68px;padding:10px 14px;border-radius:10px;border:1px solid var(--border);background:#1e293b;color:#f8fafc;font-size:13px;line-height:1.5;resize:vertical;">Emotional and nostalgic solo piano melody joined by warm acoustic guitar and soaring cinematic strings.</textarea>
    </div>

    <!-- 框 4 (单独一行)：反向提示词 Negative Prompt（可选，排除低质或多余元素） -->
    <div>
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
  <div style="padding:12px 16px;margin-bottom:14px;background:#0f172a;border:1px solid var(--border);border-radius:14px;">
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
    <div style="display:grid;grid-template-columns: 1.3fr 1.15fr 1fr 1.15fr;gap:12px;align-items:center;margin-bottom:10px;">
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
      <div style="display:flex;align-items:center;gap:8px;background:#1e293b;padding:6px 12px;border-radius:9px;border:1px solid var(--border);">
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
      <div style="display:flex;align-items:center;gap:8px;background:#1e293b;padding:4px 8px;border-radius:9px;border:1px solid var(--border);">
        <span style="font-size:12px;color:#94a3b8;white-space:nowrap;font-weight:600;padding-left:2px;">连抽</span>
        <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:4px;flex:1;" id="batchCountGroup">
          <button type="button" class="batch-pill" data-batch="1" onclick="selectBatchCount(1)" style="padding:4px 0;border-radius:6px;border:none;background:#3b82f6;color:#ffffff;font-size:12px;font-weight:700;cursor:pointer;">1条</button>
          <button type="button" class="batch-pill" data-batch="2" onclick="selectBatchCount(2)" style="padding:4px 0;border-radius:6px;border:none;background:transparent;color:#cbd5e1;font-size:12px;font-weight:600;cursor:pointer;">2条</button>
          <button type="button" class="batch-pill" data-batch="3" onclick="selectBatchCount(3)" style="padding:4px 0;border-radius:6px;border:none;background:transparent;color:#cbd5e1;font-size:12px;font-weight:600;cursor:pointer;">3条</button>
          <button type="button" class="batch-pill" data-batch="4" onclick="selectBatchCount(4)" style="padding:4px 0;border-radius:6px;border:none;background:transparent;color:#cbd5e1;font-size:12px;font-weight:600;cursor:pointer;">4条</button>
        </div>
      </div>
    </div>

    <!-- 第 2 排横条：左侧「视频总帧数滑块 + 4s~20s 快捷按钮 + 自动 2K 超分开关」，右侧「🎬 开始生成音视频」按钮 -->
    <div style="display:grid;grid-template-columns: 1fr 290px;gap:12px;align-items:center;">
      <div style="display:flex;align-items:center;gap:10px;background:#1e293b;padding:6px 12px;border-radius:9px;border:1px solid var(--border);flex-wrap:wrap;">
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
        <label title="勾选后：MiniMax-H3 生成完 480p 视频后，自动接力调用 ReCaVSR (Wan2.2-5B) 单步流式超分至 2K 超清并保留原生立体声" style="display:inline-flex;align-items:center;gap:5px;padding:3px 9px;border-radius:7px;background:rgba(139,92,246,0.16);border:1px solid rgba(139,92,246,0.42);color:#c4b5fd;font-size:11.5px;font-weight:700;cursor:pointer;white-space:nowrap;user-select:none;">
          <input id="autoUpscale2kChk" type="checkbox" style="width:13.5px;height:13.5px;accent-color:#8b5cf6;cursor:pointer;">
          <span>✨ 生成后自动 2K 超分</span>
        </label>
      </div>

      <button class="btn-primary" style="width:100%;height:100%;min-height:38px;justify-content:center;padding:8px 14px;font-size:14px;font-weight:800;border-radius:9px;box-shadow:0 6px 16px -4px rgba(59,130,246,0.5);" onclick="startVideoGenerate()" id="btnVideoGen">
        🎬 开始生成音视频
      </button>
    </div>

    <div id="videoProgressWrap" style="display:none;margin-top:8px;">
      <div class="progress-bar" style="height:7px;"><div class="progress-fill green" id="videoProgressBar" style="width:5%;"></div></div>
    </div>
  </div>

  <!-- ==================== 第 3 行 (单独一行 100% 宽度，1:1 复刻原版 EasyColab 预览图)：▶️ 生成结果预览 (左大屏) + 🕒 本地历史 (右列表) ==================== -->
  <div style="display:grid;grid-template-columns: 3.1fr 1fr;gap:16px;align-items:stretch;">
    <!-- 左卡片：▶️ 生成结果预览 (超大屏 16:9 画面) -->
    <div style="background:#0f172a;border:1px solid var(--border);border-radius:14px;padding:18px;display:flex;flex-direction:column;justify-content:space-between;">
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
        <video id="mmVideoPlayer" controls autoplay loop style="width:100%;height:520px;border-radius:12px;background:#000000;object-fit:contain;display:block;"></video>
      </div>

      <!-- 底部文件信息栏：单行紧凑对齐 -->
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:nowrap;gap:10px;margin-top:12px;padding-top:4px;">
        <div style="display:flex;align-items:center;gap:6px;min-width:0;flex:1;overflow:hidden;">
          <span style="color:#94a3b8;font-size:14px;flex-shrink:0;">🎞️</span>
          <span id="mmVideoMeta" title="" style="font-size:12.5px;color:#cbd5e1;font-family:'SF Mono',Menlo,monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">尚未选择视频</span>
          <span id="previewBottomShortBadge" style="display:none;background:#1e293b;color:#94a3b8;border:1px solid var(--border);padding:1px 7px;border-radius:999px;font-size:11px;font-weight:700;flex-shrink:0;">--</span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;flex-shrink:0;white-space:nowrap;">
          <div style="display:inline-flex;align-items:center;gap:4px;background:rgba(139,92,246,0.14);border:1px solid rgba(139,92,246,0.45);border-radius:8px;padding:2px 5px;">
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
          <button type="button" class="btn-outline" title="截取当前视频最后一帧作为「首帧参考图」，无缝续写下一段视频" style="padding:5px 10px;font-size:12px;font-weight:700;background:rgba(59,130,246,0.14);border-color:rgba(59,130,246,0.45);color:#93c5fd;" onclick="continueFromCurrentVideoLastFrame()">🔗 尾帧续写</button>
          <button type="button" class="btn-outline" style="padding:5px 10px;font-size:12px;font-weight:700;background:#1e293b;border-color:#334155;color:#e2e8f0;" onclick="replayCurrentVideo()">重播</button>
          <button type="button" class="btn-outline" style="padding:5px 10px;font-size:12px;font-weight:700;background:#1e293b;border-color:#334155;color:#e2e8f0;" onclick="revealInFinder(false)">Finder</button>
          <a id="mmVideoDownload" href="#" download="minimax_h3_output.mp4" class="btn-outline" style="display:none;padding:5px 10px;font-size:12px;text-decoration:none;border-color:#10b981;color:#10b981;">⬇️ 另存为</a>
        </div>
      </div>
    </div>

    <!-- 右卡片：🕒 本地历史 (~/Movies/EasyColab) -->
    <div style="background:#0f172a;border:1px solid var(--border);border-radius:14px;padding:16px;display:flex;flex-direction:column;">
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
      <button type="button" id="btnStopTaskBottom" onclick="stopActiveVideoTask()" style="display:none;padding:3px 10px;font-size:11.5px;font-weight:800;border-radius:6px;border:1px solid rgba(239,68,68,0.65);background:rgba(239,68,68,0.18);color:#fca5a5;cursor:pointer;">🛑 停止当前任务</button>
    </div>
  </h2>
  <div id="taskProgressArea" style="margin-bottom:14px;padding:12px 14px;background:#0f172a;border:1px solid var(--border);border-radius:10px;">
    <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px;">
      <span id="taskProgressLabel" style="font-weight:600;color:#93c5fd;">当前进度：已就绪 (100%)</span>
      <span id="taskProgressSpeed" style="font-family:'SF Mono',monospace;color:var(--green);">50.90 / 50.90 GB · 服务运行中</span>
    </div>
    <div class="progress-bar" style="height:10px;">
      <div class="progress-fill green" id="taskProgressBar" style="width:100%;"></div>
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
      <button class="btn-outline" style="padding:3px 9px;font-size:13px;" onclick="closeAccountModal()">✕</button>
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
    const hasG4 = cuNum !== null && cuNum > 0;
    const cuBadge = cuNum === null
      ? `<span style="font-size:11px;padding:1px 7px;border-radius:999px;background:rgba(148,163,184,0.18);color:#cbd5e1;font-weight:700;">-- CU</span>`
      : (hasG4
          ? `<span style="font-size:11px;padding:1px 7px;border-radius:999px;background:rgba(16,185,129,0.2);color:#34d399;font-weight:700;">⚡ ${cuNum.toFixed(2)} CU · RTX 6000</span>`
          : `<span style="font-size:11px;padding:1px 7px;border-radius:999px;background:rgba(59,130,246,0.18);color:#93c5fd;font-weight:700;" title="T4 (16GB)，单号每日可用约 3~5 小时">⚡ T4 (16GB · ≈3~5h/天)</span>`);
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
      if (!curF.includes('-T4-') && curF !== 'ReCaVSR') {
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
      if (curF.includes('-T4-') || curF === 'ReCaVSR') {
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
    const isT4Named = item.folder.includes('-T4-') || item.folder === 'ReCaVSR';
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
    'MiniMax-H3-Turbo': 'MiniMax-H3 Turbo',
    'Qwen3.8-27B-GGUF': 'Qwen3.8-27B-Q4',
    'Qwen-Image-2.1': 'Qwen-Image-2.1',
    'Z-Image-Turbo': 'Z-Image-Turbo',
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

function selectPresetModel(cardId, repo, folder, pattern) {
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
    'MiniMax-H3-Turbo': 'Comfy-Org/MiniMax-H3',
    'Qwen3.8-27B-GGUF': 'unsloth/Qwen3.8-27B-GGUF',
    'Qwen-Image-2.1': 'Qwen/Qwen-Image-2.1',
    'Z-Image-Turbo': 'Tongyi-MAI/Z-Image-Turbo',
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
    'MiniMax-H3-Turbo': 'preset_minimax_turbo',
    'Qwen3.8-27B-GGUF': 'preset_qwen_q4',
    'Qwen-Image-2.1': 'preset_qwen_img',
    'Z-Image-Turbo': 'preset_z_img',
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
        basePaidBalance = bal;
        baseBalanceSyncTs = Date.now();
        isFreeT4Available = d.cu_info.t4_available !== false;
        if (bal > 0) {
          balEl.innerText = `${bal.toFixed(2)} CU`;
          balEl.style.color = '#f8fafc';
          if (hrsEl) {
            const refRate = br > 0 ? br : 10.05;
            const hrs = bal / refRate;
            hrsEl.innerText = br > 0
              ? `当前速率可跑 ≈ ${hrs.toFixed(1)} 小时`
              : `按 RTX PRO 6000 (10.05/h) ≈ ${hrs.toFixed(1)} 小时`;
          }
        } else if (isFreeT4Available) {
          balEl.innerText = 'T4 (16GB)';
          balEl.style.color = '#34d399';
          if (hrsEl) {
            hrsEl.innerText = '🟢 今日配额可用 · 单次约 3~5 小时';
          }
          const curF = (document.getElementById('inpFolder') || {}).value || '';
          const g4OnlySet = new Set(['Huihui-Qwen3.8-27B-abliterated-GGUF', 'MiniMax-H3', 'MiniMax-H3-Turbo', 'Qwen3.8-27B-GGUF', 'Qwen-Image-2.1', 'Z-Image-Turbo']);
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
      'Qwen3.8-27B-GGUF': 'Qwen3.8-27B(极速版)',
      'unsloth/Qwen3.8-27B-GGUF': 'Qwen3.8-27B(极速版)',
      'MiniMax-H3': 'MiniMax-H3(33B)',
      'MiniMaxAI/MiniMax-H3': 'MiniMax-H3(33B)',
      'MiniMax-H3-Turbo': 'MiniMax-H3-Turbo',
      'Comfy-Org/MiniMax-H3': 'MiniMax-H3-Turbo',
      'Qwen-Image-2.1': 'Qwen-Image-2.1',
      'Qwen/Qwen-Image-2.1': 'Qwen-Image-2.1',
      'Z-Image-Turbo': 'Z-Image-Turbo',
      'Tongyi-MAI/Z-Image-Turbo': 'Z-Image-Turbo',
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
      } else {
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
        document.getElementById('driveVal').innerText = `${gd.quota.used_gb} / ${gd.quota.total_gb} GB`;
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

async function pollLogs() {
  try {
    const r = await fetch('/api/logs');
    const d = await r.json();
    const box = document.getElementById('logBox');
    if (d.logs && d.logs.length > 0) {
      const fullTxt = d.logs.join('\n');
      box.innerText = fullTxt;
      box.scrollTop = box.scrollHeight;

      // 解析最新一条任务状态：优先匹配最新的视频生成/视频超分/视频回传/视频完成，或下载/云盘预热/显存装载进度
      let latestDeployMatch = null;
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
        const mvAct = line.match(/\[(视频生成|视频回传|自动切换引擎)\]\s+(.+)$/);
        if (mvAct) {
          if (mvAct[1] === '视频回传' && mvAct[2].includes('_ReCaVSR_')) {
            latestUpscaleActive = { stage: mvAct[1], msg: mvAct[2] };
          } else {
            latestVideoActive = { stage: mvAct[1], msg: mvAct[2] };
          }
          break;
        }
        const mvDone = line.match(/\[(视频完成|视频已落盘)\]\s+(.+)$/);
        if (mvDone) {
          if (mvDone[2].includes('_ReCaVSR_')) {
            latestUpscaleDone = mvDone[2];
          } else {
            latestVideoDone = mvDone[2];
          }
          break;
        }
        const m = line.match(/\[(HF 下载|云盘极速预热|云盘同步|显存装载)\]\s+([^:]+):\s+(\d+)%\s+\(([^,]+),\s*([^)]+)\)/);
        if (m) {
          latestDeployMatch = m;
          break;
        }
      }
      const pLabel = document.getElementById('taskProgressLabel');
      const pSpeed = document.getElementById('taskProgressSpeed');
      const pBar = document.getElementById('taskProgressBar');
      const st = document.getElementById('deployStateText');

      if (!d.running && (videoPollTimer || upscalePollTimer)) {
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
        const vPct = pctM ? Math.max(5, Math.min(99, parseInt(pctM[1], 10))) : 8;
        pLabel.innerText = `⏳ ${vMsg}`;
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
        pBar.className = 'progress-fill green';
        pBar.style.width = '100%';
      } else if (latestDeployMatch) {
        const stage = latestDeployMatch[1];
        const fname = latestDeployMatch[2];
        const pct = parseInt(latestDeployMatch[3], 10);
        const gbStr = latestDeployMatch[4];
        const spdStr = latestDeployMatch[5];
        const icon = stage === '显存装载' ? '🎮' : '⏳';
        pLabel.innerText = `${icon} [${stage}] ${fname}: ${pct}%`;
        pSpeed.innerText = `${gbStr} · 速度 ${spdStr}`;
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

function switchMiniMaxMode(mode) {
  currentMiniMaxMode = mode;
  const isFl = mode === 'fl2va';
  document.getElementById('fl2vaImageRow').style.display = isFl ? 'grid' : 'none';
  document.getElementById('tabBtnT2va').className = isFl ? 'btn-outline' : 'btn-primary';
  document.getElementById('tabBtnFl2va').className = isFl ? 'btn-primary' : 'btn-outline';
}

function handleFrameFileSelect(inputEl, previewId) {
  const file = inputEl.files && inputEl.files[0];
  const imgEl = document.getElementById(previewId);
  if (!file) {
    if (previewId === 'firstFramePreview') firstFrameB64 = '';
    if (previewId === 'lastFramePreview') lastFrameB64 = '';
    if (imgEl) imgEl.style.display = 'none';
    return;
  }
  const reader = new FileReader();
  reader.onload = function(ev) {
    const dataUrl = ev.target.result;
    if (previewId === 'firstFramePreview') firstFrameB64 = dataUrl;
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
    const inp = document.getElementById('inpFirstFrame');
    const img = document.getElementById('firstFramePreview');
    if (inp) inp.value = '';
    if (img) { img.src = ''; img.style.display = 'none'; }
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
    const isSr = it.name.includes('_ReCaVSR_');
    const rowBg = active ? 'rgba(59,130,246,0.20)' : 'transparent';
    const rowBorder = active ? '1px solid rgba(59,130,246,0.5)' : '1px solid transparent';
    const iconColor = active ? '#60a5fa' : (isSr ? '#a78bfa' : '#94a3b8');
    const badgeBg = active ? 'rgba(59,130,246,0.28)' : '#1e293b';
    const badgeColor = active ? '#60a5fa' : '#94a3b8';
    const shortBadge = it.elapsed_short
      ? `<span style="background:${badgeBg};color:${badgeColor};padding:2px 7px;border-radius:999px;font-size:11px;font-weight:700;flex-shrink:0;">${escapeHtml(it.elapsed_short)}</span>`
      : '';
    const srAction = isSr
      ? `<span style="background:rgba(139,92,246,0.22);color:#c4b5fd;border:1px solid rgba(139,92,246,0.45);padding:1px 6px;border-radius:6px;font-size:10.5px;font-weight:800;flex-shrink:0;">✨超清</span>`
      : `<button type="button" title="使用 ReCaVSR 将该视频超分至 2K 超清" onclick="event.stopPropagation();upscaleVideoByName('${escapeHtml(it.name)}')" style="background:rgba(139,92,246,0.16);color:#c4b5fd;border:1px solid rgba(139,92,246,0.42);padding:2px 7px;border-radius:6px;font-size:10.5px;font-weight:700;cursor:pointer;flex-shrink:0;">✨2K</button>`;
    return `
      <div onclick="selectLocalHistoryItem(${idx})" style="display:flex;justify-content:space-between;align-items:center;gap:6px;padding:9px 10px;border-radius:10px;background:${rowBg};border:${rowBorder};cursor:pointer;transition:all 0.12s;">
        <div style="display:flex;align-items:center;gap:8px;min-width:0;flex:1;">
          <span style="color:${iconColor};font-size:14px;flex-shrink:0;">🎞️</span>
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

function selectLocalHistoryItem(idx) {
  const it = localHistoryItems[idx];
  if (!it) return;
  currentSelectedVideoName = it.name;
  renderLocalHistoryList();
  const url = `/api/local_video_file?name=${encodeURIComponent(it.name)}`;
  const player = document.getElementById('mmVideoPlayer');
  const dl = document.getElementById('mmVideoDownload');
  const emptyEl = document.getElementById('videoEmptyPlaceholder');
  const rWrap = document.getElementById('videoResultWrap');
  const topBadge = document.getElementById('previewElapsedBadge');
  const botBadge = document.getElementById('previewBottomShortBadge');
  if (emptyEl) emptyEl.style.display = 'none';
  if (rWrap) rWrap.style.display = 'block';
  if (player.getAttribute('data-fname') !== it.name) {
    player.src = url;
    player.setAttribute('data-fname', it.name);
  }
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
  const player = document.getElementById('mmVideoPlayer');
  const dl = document.getElementById('mmVideoDownload');
  const emptyEl = document.getElementById('videoEmptyPlaceholder');
  const rWrap = document.getElementById('videoResultWrap');
  if (emptyEl) emptyEl.style.display = 'none';
  if (rWrap) rWrap.style.display = 'block';
  player.src = url;
  player.setAttribute('data-fname', fname);
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
  if (liveVideoElapsedSec === null) return;
  const tSpeed = document.getElementById('taskProgressSpeed');
  const st = document.getElementById('videoGenStatus');
  const elStr = `已耗时 ${formatCnDur(liveVideoElapsedSec)}`;
  const etaStr = (liveVideoEtaSec && liveVideoEtaSec > 0) ? ` · 预计剩余 ≈ ${formatCnDur(liveVideoEtaSec)}` : '';
  const pctStr = liveVideoPct !== null ? ` · 总进度 ${liveVideoPct}%` : '';
  if (tSpeed) {
    tSpeed.innerText = `${elStr}${etaStr}${pctStr}`;
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
  const btn = document.getElementById('btnVideoGen');
  if (!btn) return;
  btn.disabled = false;
  if (running) {
    btn.setAttribute('data-running', '1');
    btn.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
    btn.style.boxShadow = '0 6px 16px -4px rgba(239,68,68,0.55)';
    btn.innerText = customLabel || '🛑 正在合成有声视频... (点击停止)';
    setStopButtonsVisible(true);
  } else {
    btn.setAttribute('data-running', '0');
    btn.style.background = '';
    btn.style.boxShadow = '0 6px 16px -4px rgba(59,130,246,0.5)';
    btn.innerText = '🎬 开始生成音视频';
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
    st.innerText = isUpscaling ? '🛑 正在停止 ReCaVSR 超分任务...' : '🛑 正在停止有声视频生成任务...';
    st.style.color = '#f59e0b';
  }
  if (tLabel) tLabel.innerText = isUpscaling ? '🛑 正在安全中断 ReCaVSR 超分任务...' : '🛑 正在安全中断有声视频采样 (保留 134GB 模型显存常驻)...';
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

async function startVideoGenerate() {
  const btn = document.getElementById('btnVideoGen');
  if (btn && btn.getAttribute('data-running') === '1') {
    await stopActiveVideoTask('video');
    return;
  }
  const prompt = document.getElementById('videoPrompt').value.trim();
  if (!prompt) {
    alert('请输入有声视频画面提示词');
    return;
  }
  const resVal = document.getElementById('videoRes').value || '832x480';
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
    steps: parseInt(document.getElementById('videoSteps').value || '30', 10),
    fps: parseInt(document.getElementById('videoFps').value || '24', 10),
    seed: seedVal,
    randomize_seed: randSeed,
    batch_count: selectedBatchCount,
  };
  if (currentMiniMaxMode === 'fl2va' && firstFrameB64) {
    payload.first_frame_b64 = firstFrameB64;
    if (lastFrameB64) payload.last_frame_b64 = lastFrameB64;
  }
  const st = document.getElementById('videoGenStatus');
  const pWrap = document.getElementById('videoProgressWrap');
  const pBar = document.getElementById('videoProgressBar');
  setVideoBtnRunningState(true, selectedBatchCount > 1 ? `🛑 正在连抽 ${selectedBatchCount} 条... (点击停止)` : '🛑 正在合成有声视频... (点击停止)');
  pWrap.style.display = 'block';
  pBar.style.width = '8%';
  st.innerText = `⏳ 已提交至 RTX PRO 6000 MiniMax-H3 引擎 (连抽 ${selectedBatchCount} 条)，正在调用 32B 无审查编码器...`;
  st.style.color = '#f59e0b';
  try {
    const r = await fetch('/api/video_generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const d = await r.json();
    if (!d.ok) {
      st.innerText = '❌ ' + (d.error || '启动失败（请确认当前已加载运行 MiniMax-H3）');
      st.style.color = '#ef4444';
      setVideoBtnRunningState(false);
      return;
    }
    if (d.auto_switching) {
      selectPresetModel('preset_minimax_h3', 'MiniMaxAI/MiniMax-H3', 'MiniMax-H3', 'FL2VA/*.safetensors');
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
      setVideoBtnRunningState(true, '🛑 正在合成有声视频... (点击停止)');
      if (pWrap) pWrap.style.display = 'block';
      if (!videoPollTimer) {
        videoPollTimer = setInterval(pollVideoStatus, 2000);
      }
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
        if (tLabel) {
          tLabel.innerText = `⏳ ${d.message}`;
        }
        renderLiveVideoTimingText();
      } else {
        const doneDur = (typeof d.elapsed_s === 'number' && d.elapsed_s > 0) ? ` (总耗时 ${formatCnDur(d.elapsed_s)})` : '';
        st.innerText = `${d.message}`;
        st.style.color = '#10b981';
        if (tLabel) {
          tLabel.innerText = `✅ ${d.message}`;
        }
        if (tSpeed) {
          tSpeed.innerText = `已完成并保存至 ~/Movies/EasyColab${doneDur}`;
        }
      }
      if (dState) {
        dState.innerText = d.running ? '⏳ 视频正在生成中...' : '✅ 视频生成已完成';
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
      if (autoSrChk && autoSrChk.checked && d.latest_video && !d.latest_video.includes('_ReCaVSR_') && lastAutoUpscaledVideo !== d.latest_video) {
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
    }
  } catch (e) {}
}

refreshStatus(false);
loadLocalVideoHistory(true);
(async () => {
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
            acc_info = _get_accounts_status(force=force, cu_balance=(cu_info or {}).get("paid_balance"))
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
                    self.send_header("Content-Type", "video/mp4")
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
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path.startswith("/api/video_status"):
            if _PENDING_VIDEO_JOB.get("active"):
                pct_val = 4
                stage_msg = "🔄 正在自动从 Google Drive 云盘切换并装载 MiniMax-H3 至显存，加载完成后将立即自动生成视频..."
                with _DEPLOY_LOCK:
                    logs_snap = list(_DEPLOY_STATE.get("logs") or [])
                for line in reversed(logs_snap):
                    m = re.search(r"\[(HF 下载|云盘极速预热|云盘同步|显存装载)\]\s+([^:]+):\s+(\d+)%", line)
                    if m:
                        st_type, _, st_pct_s = m.group(1), m.group(2), m.group(3)
                        st_pct = int(st_pct_s)
                        if st_type == "显存装载":
                            pct_val = max(8, min(15, 8 + int(st_pct * 0.07)))
                        else:
                            pct_val = max(3, min(8, 3 + int(st_pct * 0.05)))
                        stage_msg = f"🔄 正在自动切换加载 MiniMax-H3 ({st_type}: {st_pct}%)，显存就绪后将立即自动生成视频..."
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
                    if d.get("running"):
                        if d.get("message"):
                            _append_panel_log(f"[视频生成] {d.get('message')}")
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
                            d["message"] = f"⬇️ 云端合成已完成！正在回传保存至本机 ~/Movies/EasyColab/ ({done_m:.1f}/{tot_m:.1f} MB, {dl_pct}%)"
                            timing_info = _compute_video_timing_from_logs(int(d["pct"]))
                            d["elapsed_sec"] = timing_info["elapsed_sec"]
                            d["eta_sec"] = max(1, int(round(max(0.0, tot_m - done_m) / max(0.15, spd_mbs))))
                        elif vids:
                            latest_fn = os.path.basename(str(vids[-1]))
                            d["latest_video"] = latest_fn
                            d["pct"] = 100
                            if d.get("message") and not str(d.get("message")).startswith("就绪") and latest_fn not in _LOGGED_COMPLETED_VIDEOS:
                                _LOGGED_COMPLETED_VIDEOS.add(latest_fn)
                                _append_panel_log(f"[视频完成] {d.get('message')}")
                    self._send_json(d)
            except Exception as e:
                self._send_json({"running": False, "error": str(e)})
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
                self._send_json({"running": False, "error": str(e)})
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
                    "You are an uncensored, professional multimodal AI video director for MiniMax-H3 (audio-video generation). "
                    "Given the user's video idea, output ONLY a valid JSON object with exactly 4 keys in English (no markdown fences, no extra commentary):\n"
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
            _append_panel_log(
                f"🎬 [提交视频生成任务] 分辨率 {payload.get('resolution', '832x480')} · {payload.get('num_frames', 124)}帧 · {payload.get('steps', 30)}步 · 连抽 {payload.get('batch_count', 1)}条"
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
            need_auto_switch = False
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
                target_label = "MiniMax-H3 (T4 量化版)" if is_t4_account else "MiniMax-H3"
                _append_panel_log(f"🔄 [自动切换引擎] 检测到当前未运行 {target_label}，正在自动加载并在就绪后立即生成有声视频...")
                with _DEPLOY_LOCK:
                    already_deploying = bool(_DEPLOY_STATE.get("running"))
                if not already_deploying:
                    if is_t4_account:
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
                                    _PENDING_VIDEO_JOB["error"] = "❌ 自动切换加载 MiniMax-H3 失败，请查看下方实时部署日志"
                                    _PENDING_VIDEO_JOB["active"] = False
                                    return
                                break
                        # MiniMax-H3 部署完成，等待 /v1/video/generations 接收任务
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
                                        _append_panel_log("✅ [自动切换完成] MiniMax-H3 已就绪并自动开始采样生成有声视频！")
                                        _PENDING_VIDEO_JOB["active"] = False
                                        return
                            except Exception:
                                pass
                            time.sleep(2)
                        _PENDING_VIDEO_JOB["error"] = "❌ MiniMax-H3 加载后提交视频生成超时，请再点一次「开始生成音视频」"
                        _PENDING_VIDEO_JOB["active"] = False
                    except Exception as ex:
                        _PENDING_VIDEO_JOB["error"] = f"❌ 自动切换异常: {ex}"
                        _PENDING_VIDEO_JOB["active"] = False

                threading.Thread(target=_auto_switch_and_submit, args=(payload,), daemon=True).start()
                self._send_json({
                    "ok": True,
                    "auto_switching": True,
                    "message": "🔄 检测到当前显存为 Qwen3.8-27B（或尚未启动 MiniMax-H3），正在自动从 Google Drive 切换加载 MiniMax-H3，就绪后将自动开始生成视频...",
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
            _append_panel_log("🛑 [手动停止] 已发送中断信号停止当前有声视频生成任务 (134GB 模型继续常驻显存)")
            self._send_json({"ok": True, "message": "🛑 已手动停止有声视频生成任务 (134GB 模型继续常驻显存)"})
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

        self.send_error(404)


def main():
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

    server = ThreadingHTTPServer(("127.0.0.1", PANEL_PORT), PanelRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n控制台已退出。")


if __name__ == "__main__":
    main()
