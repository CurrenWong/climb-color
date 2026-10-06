"""加密采样验证实验：1s 间隔帧 vs 原 5s 间隔帧的稳定性对比。

假设：5s 采样太稀 → 单帧噪声（端点漂移）未被平均掉 → margin 小、bootstrap 低。
      按 1s 重抽（4.4x 帧数）应抬高 margin 与 bootstrap 期望。

只跑 3 面最脆弱的墙（bootstrap 最低那三面）。
输出格式与 exp_dwpose_full.py 一致，便于 paper/bootstrap_margin.py 复用。

用法: python exp_dense_sampling.py
输出: dense_sampling.log  (+ 终端)

帧图不在 git 中（~12MB）；按下列命令可从源视频重建：
  cd mediapipe2
  ffmpeg -v error -ss 0   -t 36 -i "..\\videos\\小红书_3OHz6rZ5opG.mp4" -vf fps=1 -q:v 2 -start_number 0   "img_xhs_walls_1s\\3OHz6rZ5opG_wall_01_unknown\\frame_%04ds.jpg"
  ffmpeg -v error -ss 125 -t 16 -i "..\\videos\\小红书_3OHz6rZ5opG.mp4" -vf fps=1 -q:v 2 -start_number 125 "img_xhs_walls_1s\\3OHz6rZ5opG_wall_04_line2_pink\\frame_%04ds.jpg"
  ffmpeg -v error -ss 145 -t 31 -i "..\\videos\\小红书_3OHz6rZ5opG.mp4" -vf fps=1 -q:v 2 -start_number 145 "img_xhs_walls_1s\\3OHz6rZ5opG_wall_05_known_IMG_20260808_193650\\frame_%04ds.jpg"
（cmd 中用 %04d；在 .bat 里需写成 %%04d）
"""
import os
import sys
import time
from collections import Counter
from pathlib import Path

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BASE)
sys.path[0:0] = [_BASE, _ROOT, os.path.join(_ROOT, "EdgeSAM")]

import cv2
import edgesam_stub  # noqa: F401

import exp_dwpose_full as E          # 复用 analyze_frame / SKIP_COLORS
from dwpose_onnx import DWPoseONNX
from edgesam_holds import EdgeSamHoldSegmenter
from yolo_person_mask import PersonMasker

WALLS_DIR = Path(_BASE) / ".." / "img_xhs_walls_1s"

TRUTH = {
    '3OHz6rZ5opG_wall_01_unknown': '蓝',
    '3OHz6rZ5opG_wall_04_line2_pink': '粉红',
    '3OHz6rZ5opG_wall_05_known_IMG_20260808_193650': '绿',
}


class Tee:
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def main():
    walls = sorted([d for d in WALLS_DIR.iterdir() if d.is_dir()])
    if not walls:
        print(f"no walls under {WALLS_DIR}")
        return 1
    print(f"dense-sampling walls: {len(walls)}")
    for w in walls:
        n = len(list(w.glob('*.jpg')))
        print(f"  {w.name}  ({n} frames @1s)")

    print("\nloading models ...")
    dw = DWPoseONNX('../yolox_l.onnx', '../dw-ll_ucoco_384.onnx')
    segmenter = EdgeSamHoldSegmenter('../weights/edge_sam.pth', device='cpu')
    pm = PersonMasker(model_name="../yolo26m-seg.pt", device='cpu')

    tee = Tee(str(Path(_BASE) / "dense_sampling.log"))
    old = sys.stdout
    sys.stdout = tee
    lines, all_results = [], []
    t_start = time.time()
    total_frames = 0

    for wall_dir in walls:
        wall_name = wall_dir.name
        files = sorted(wall_dir.glob('*.jpg'))
        print(f"\n=== {wall_name} ({len(files)} 张) ===")
        all_colors = Counter()
        t_wall = time.time()
        for i, fp in enumerate(files):
            img = cv2.imread(str(fp))
            if img is None:
                print(f"  [!] cannot read {fp.name}")
                continue
            res = E.analyze_frame(img, dw, segmenter, pm)
            total_frames += 1
            print(f"  >>> 线路色 = {res['route_color']}  置信度 = {res['confidence']:.0%}"
                  f"  票数 = {res['votes']}")
            for s in res['samples']:
                c = s.get('name', '')
                if c in E.SKIP_COLORS:
                    continue
                all_colors[c] += 1
            print(f"  ⏱️  {fp.name}: {time.time() - t_wall:.2f}s cumulative")
        print(f"\n  🕐 wall 总耗时: {time.time() - t_wall:.1f}s  ({len(files)} 帧)")

        total = sum(all_colors.values())
        print("=" * 60)
        print(f"  📊 跨帧聚合: {len(files)} 帧, {total} 有效端点票")
        print("=" * 60)
        if total:
            for color, n in all_colors.most_common():
                print(f"    {color:5s}: {n:3d}  ({n / total * 100:5.1f}%)")
            top_color, top_n = all_colors.most_common(1)[0]
            print(f"  >>> 线路色 = {top_color}  ({top_n}/{total} = {top_n / total * 100:.0f}%)")
        else:
            top_color, top_n = '?', 0
        truth = TRUTH.get(wall_name)
        hit = '?' if truth is None else ('✅' if top_color == truth else '❌')
        print(f"  真值={truth or '?'}  {hit}")
        lines.append(f"{wall_name[:55]:55s}  判={top_color:4s} ({top_n}/{total})"
                     f"  真值={truth or '?':6s}  {hit}")
        all_results.append((wall_name, top_color, top_n, total, truth, hit))
        print()

    print("\n=== 汇总 (dense sampling @1s) ===")
    hits = sum(1 for r in all_results if r[5] == '✅')
    known = sum(1 for r in all_results if r[4] is not None)
    print(f"命中: {hits}/{known}")
    for l in lines:
        print(f"  {l}")
    t_total = time.time() - t_start
    print(f"\n=== 计时 ===\n总耗时: {t_total:.1f}s ({t_total / 60:.1f}min)\n"
          f"总帧数: {total_frames}  ({len(walls)} walls)\n"
          f"每帧均: {t_total / max(total_frames, 1):.2f}s")

    sys.stdout = old
    print(f"done: {hits}/{known} hits, {total_frames} frames, {t_total:.0f}s")
    tee.f.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
