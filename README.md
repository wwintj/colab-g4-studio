# 🎬 Colab G4 Pro 视频与无审查大模型全能工作台 (`colab-g4-studio`)

专为 **Google Colab NVIDIA RTX PRO 6000 Blackwell (`G4` · `96 GB` 显存)** 打造的一站式本地可视化网页控制台（`http://127.0.0.1:18080`）与永久固定 OpenAI API 网关（`http://127.0.0.1:30000/v1`）。

Mac / PC 本地 **零显卡占用、零机身发热**，所有百亿参数重算力全部在云端 96GB 显存上满血运行，生成的有声视频与超分作品自动回传落盘至本机 `~/Movies/EasyColab/` 并双备份至 Google Drive。

---

## ✨ 核心特性

1. **🎬 MiniMax-H3 (33B · 原生 BF16 · 无审查) 有声视频创作工作台**
   * 原生直出 **32kHz 双声道立体声**（环境音效 + 角色台词对白 + 背景音乐）与电影级画面。
   * 内置 **Blackwell 专属 5 重满血加速**：`SageAttention 2 (NHD)` + `TeaCache 自适应残差缓存` + `TorchInductor 融合算子` + `96GB 显存 + 177GB DDR5 双内存池零换页常驻` + `提示词编码缓存`。5 秒有声视频仅需 **~48 秒**，15 秒有声视频仅需 **~3 分半**！
   * 支持 **文生音视频 (`T2VA`)**、**首尾帧控制 (`FL2VA`)**、**单次最高 20 秒 (`481 帧`)**，以及 **一键「🔗 提取尾帧续写」** 无限延长视频（30s / 45s / 60s 满血画质接力）。
   * 支持 **随时中断（二次点击主按钮或点击「🛑 停止」）**：1~2 秒内安全停止当前采样，134GB 模型继续常驻显存无需重载。

2. **✨ ReCaVSR (`Wan2.2-5B` + `FlashDecoder`) 单步流式 2K / 4K 极清放大**
   * 与 `MiniMax-H3` 同驻 96GB 显存（闲置 `0 MB` 显存占用），针对 4K 模式特别优化 **Pinned DDR5 内存 KV Cache 流式传输 + 算子原位融合**，彻底杜绝爆显存（OOM）。
   * 支持 **1080P 全高清**、**2K 超清 (`2496×1440` / `1440×2496`，约 45 秒)**、**4× 极清放大 (`3328×1920` / `1920×3328` 4K，约 1 分 40 秒)**，并支持勾选 **「✨ 生成后自动超分」** 全自动流水线。

3. **🧠 Qwen3.8-27B (无审查原生 BF16 · 262K 上下文 + MTP 投机解码)**
   * 满血 `130+ tokens/s` 极速生成，支持在控制台输入一句中文视频想法，一键自动扩写并填入【主体画面 / 环境音效 / 背景音乐 / 反向排除项】四个专业提示词框。
   * 提供本机永久固定 OpenAI 兼容接口：`http://127.0.0.1:30000/v1`（可直连 Cursor、NextChat、OpenWebUI 等任意客户端）。

4. **☁️ Google Drive 永久零弹窗持久化 + 算力防忘关自动保护**
   * 模型只需从 HuggingFace 下载一次即永久保存在您自己的 `Google Drive (MyDrive/models/)`，后续开机直接通过 16 线程并发极速预热。
   * 内置实时算力（CU）监控、账单重置日倒计时与闲置自动关机保护，关机即彻底归零（`0.00 CU/h`）。

---

## 🚀 一键安装 / 一键升级 / 一键卸载

打开 Mac（或 Linux）终端，复制粘贴以下单行命令即可：

### 1️⃣ 一键安装（首次使用）

```bash
curl -fsSL https://raw.githubusercontent.com/wwintj/colab-g4-studio/main/install.sh | bash
```

> 安装完成后：
> * 会自动在浏览器打开可视化控制台：**[http://127.0.0.1:18080](http://127.0.0.1:18080)**
> * 会在 Mac 桌面生成双击快捷方式：**`打开 G4 控制台.command`**
> * 首次使用时，在终端运行一次 `g4-studio auth` 即可在浏览器登录绑定您自己的 Google Colab Pro 账号与 Google Drive 云盘。

---

### 2️⃣ 一键升级（获取最新加速补丁与新功能）

无论何时有新功能更新，只需运行以下任一命令（**不会丢失**您的账号授权、自定义模型列表与本地已生成的视频）：

```bash
curl -fsSL https://raw.githubusercontent.com/wwintj/colab-g4-studio/main/install.sh | bash -s -- --upgrade
```

*(或者直接双击项目文件夹内的 **`4-一键升级最新版.command`** / 在终端输入 **`g4-studio upgrade`**)*

---

### 3️⃣ 一键卸载（安全关停云端实例并清理本机文件）

```bash
curl -fsSL https://raw.githubusercontent.com/wwintj/colab-g4-studio/main/uninstall.sh | bash
```

*(或者在终端输入 **`g4-studio uninstall`**。卸载脚本会自动先关闭云端正在计费的 Colab 实例，并保留您本机 `~/Movies/EasyColab/` 下已生成的视频作品。)*

---

## 🛠️ 常用终端快捷指令 (`g4-studio`)

| 命令 | 功能说明 |
| :--- | :--- |
| `g4-studio start` | 启动后台服务并在浏览器打开 `http://127.0.0.1:18080` 网页控制台 |
| `g4-studio auth` | 首次登录 Google Colab 账号并授权绑定 Google Drive (`rclone`) |
| `g4-studio stop` | 立即释放并关闭云端 Colab G4 实例（扣费速率归零 `0.00 CU/h`） |
| `g4-studio upgrade` | 从 GitHub 一键无损拉取最新版本并自动重启控制台 |
| `g4-studio uninstall` | 一键安全卸载 |
