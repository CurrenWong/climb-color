"""项目改名辅助：把项目内文本里的 'mediapipe2' 批量替换为新目录名。

背景：本仓库原名 mediapipe2，但项目核心是 EdgeSAM + 人体掩码提示
（姿态源可换 MediaPipe / DWPose），叫 mediapipe 有误导。

用法（**先把目录重命名**，再在项目根运行）：
    cd /d D:\\Project\\climbing_wall_search
    ren mediapipe2 climb-color          # 注意：需先停止 QwenPaw 等占用进程
    cd climb-color
    python debug\\_rename_project.py            # dry-run，列文件
    python debug\\_rename_project.py --apply    # 实际写入

设计：OLD 固定为 'mediapipe2'，NEW 取**当前目录名**——所以在改名后的目录里
运行即可自动对齐；若在未改名的目录里运行，NEW == OLD，为空操作（安全）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # 项目根（脚本在 debug/ 下）
OLD = 'mediapipe2'
NEW = ROOT.name                                    # 自动取当前目录名
SKIP_DIRS = {'.git', 'EdgeSAM', '__pycache__', '.venv', 'node_modules', 'weights'}
EXTS = {'.py', '.md', '.json', '.txt', '.bat', '.ps1', '.yaml', '.yml',
        '.cfg', '.toml', '.cff'}

APPLY = '--apply' in sys.argv


def main():
    if OLD == NEW:
        print(f"directory is still named '{OLD}' — nothing to align.")
        print("rename the folder first (stop QwenPaw / editors holding it), then re-run.")
        return 0

    hits = []
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        if p.suffix.lower() not in EXTS and p.name != '.gitignore':
            continue
        try:
            text = p.read_text(encoding='utf-8')
        except Exception:
            continue
        n = text.count(OLD)
        if n:
            hits.append((rel, n, text))

    total = sum(n for _, n, _ in hits)
    print(f"'{OLD}' -> '{NEW}':  {len(hits)} files, {total} occurrences\n")
    for rel, n, _ in hits:
        print(f"  {n:>3}  {rel}")

    if not APPLY:
        print("\n(dry-run; pass --apply to write)")
        return 0

    for rel, _, text in hits:
        (ROOT / rel).write_text(text.replace(OLD, NEW), encoding='utf-8')
    print(f"\nupdated {len(hits)} files, {total} occurrences")
    return 0


if __name__ == '__main__':
    sys.exit(main())
