"""
下载运行 mediapipe2 (EdgeSAM 攀岩线路色识别) 所需的全部权重。

使用:
  python download_weights.py                 # 下载全部
  python download_weights.py edgesam yolo26 # 下载指定

支持的权重 (按需):
  edgesam       — EdgeSAM 官方权重 (~40MB)
  yolo26        — YOLO26m-seg 攀岩者人体分割 (~53MB)
  mp_pose       — MediaPipe Pose landmarker (lite 5.5MB / heavy 29MB)
  dwpose        — DWPose ONNX (yolox_l 206MB + dw-ll_ucoco 128MB)

国内下载建议用 ghfast.top 代理（脚本默认尝试直连，失败回退代理）。
"""
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (target_filename, official_url, mirror_url, description)
WEIGHTS = {
    'edgesam': (
        'weights/edge_sam.pth',
        'https://github.com/spaceship99999/edgesam-portable/releases/download/v1.0/edge_sam.pth',
        'https://ghfast.top/https://github.com/spaceship99999/edgesam-portable/releases/download/v1.0/edge_sam.pth',
        'EdgeSAM 官方权重 (~40MB)'
    ),
    'yolo26': (
        'weights/yolo26m-seg.pt',
        'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt',
        'https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt',
        'YOLO26m-seg 攀岩者人体分割 (~53MB)'
    ),
    'mp_pose_lite': (
        'weights/pose_landmarker_lite.task',
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
        None,
        'MediaPipe Pose landmarker lite (~5.5MB)'
    ),
    'dwpose': (
        # DWPose 需要两个文件: yolox_l.onnx (人体检测) + dw-ll_ucoco_384.onnx (133 点 pose)
        # 这里只声明一个标记，实际下载两个文件
        'dwpose_pair',
        'https://github.com/TH14GZI-J/DWPose-onnx/releases/download/dwpose-ll/dw-ll_ucoco_384.onnx',
        None,
        'DWPose ONNX 模型 (yolox_l 206MB + dw-ll_ucoco 128MB)'
    ),
    'dinov2': (
        # DINOv2-S/14 全局特征(检索器 Stage 1)。下载整个目录到 weights/dinov2-small/
        # 需要 config.json + preprocessor_config.json + model.safetensors
        'dinov2_dir',
        'https://huggingface.co/facebook/dinov2-small/resolve/main/model.safetensors',
        'https://ghfast.top/https://huggingface.co/facebook/dinov2-small/resolve/main/model.safetensors',
        'DINOv2-S/14 全局特征权重 (~88MB,检索器)'
    ),
}


def download(url: str, dest: Path) -> bool:
    """下载文件到目标路径。返回 True 成功, False 失败。"""
    if dest.exists():
        print(f"  ✓ 已存在: {dest.name} ({dest.stat().st_size // 1024 // 1024}MB)")
        return True
    print(f"  ↓ 下载: {dest.name}")
    print(f"    URL: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  ✓ 完成: {dest.name} ({dest.stat().st_size // 1024 // 1024}MB)")
        return True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_one(name: str) -> bool:
    if name not in WEIGHTS:
        print(f"未知权重: {name} (可选: {', '.join(WEIGHTS.keys())})")
        return False
    target, url, mirror, desc = WEIGHTS[name]
    print(f"\n[{name}] {desc}")
    dest = ROOT / target
    dest.parent.mkdir(parents=True, exist_ok=True)

    if name == 'dwpose':
        # 特殊: yolox_l.onnx + dw-ll_ucoco_384.onnx 一对
        ok1 = download(
            'https://github.com/TH14GZI-J/DWPose-onnx/releases/download/dwpose-ll/yolox_l.onnx',
            ROOT / 'weights/yolox_l.onnx'
        )
        ok2 = download(url, ROOT / 'weights/dw-ll_ucoco_384.onnx')
        return ok1 and ok2

    if name == 'dinov2':
        # 特殊: 需要整个目录(3 个文件)到 weights/dinov2-small/
        dest_dir = ROOT / 'weights/dinov2-small'
        dest_dir.mkdir(parents=True, exist_ok=True)
        files = [
            ('config.json',
             'https://huggingface.co/facebook/dinov2-small/resolve/main/config.json', None),
            ('preprocessor_config.json',
             'https://huggingface.co/facebook/dinov2-small/resolve/main/preprocessor_config.json', None),
            ('model.safetensors',
             'https://huggingface.co/facebook/dinov2-small/resolve/main/model.safetensors',
             'https://ghfast.top/https://huggingface.co/facebook/dinov2-small/resolve/main/model.safetensors'),
        ]
        ok = True
        for fname, f_url, f_mirror in files:
            fdest = dest_dir / fname
            if fdest.exists():
                print(f"  ✓ 已存在: {fname} ({fdest.stat().st_size // 1024 // 1024}MB)")
                continue
            print(f"  ↓ 下载: {fname}")
            try:
                urllib.request.urlretrieve(f_url, fdest)
                print(f"  ✓ 完成: {fname} ({fdest.stat().st_size // 1024 // 1024}MB)")
            except Exception as e:
                print(f"  ✗ 失败: {e}")
                if f_mirror:
                    print(f"  ↻ 回退到镜像...")
                    try:
                        urllib.request.urlretrieve(f_mirror, fdest)
                        print(f"  ✓ 完成: {fname} ({fdest.stat().st_size // 1024 // 1024}MB)")
                    except Exception as e2:
                        print(f"  ✗ 镜像也失败: {e2}")
                        ok = False
                        if fdest.exists():
                            fdest.unlink()
                else:
                    ok = False
        return ok

    # 优先尝试直连，失败回退 mirror
    if download(url, dest):
        return True
    if mirror:
        print(f"  ↻ 回退到镜像...")
        return download(mirror, dest)
    return False


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(WEIGHTS.keys())
    print(f"将下载: {targets}")
    results = {t: download_one(t) for t in targets}
    failed = [t for t, ok in results.items() if not ok]
    if failed:
        print(f"\n失败: {failed}")
        print("提示: 国内网络可手动设置代理或下载到本地后放到 weights/")
        sys.exit(1)
    print("\n全部下载完成。")


if __name__ == '__main__':
    main()
