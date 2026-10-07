"""权重路径批量改写：把根目录模型文件名改成 weights/ 前缀。

处理三类引用形态：
  1. '../xxx' / '..\\xxx'   -> '../weights/xxx'   （dwpose_exp/、debug/ 的子脚本）
  2. 'xxx' / "xxx"          -> 'weights/xxx'      （根目录脚本，cwd = 项目根）
  3. 绝对路径 ...\\mediapipe2\\xxx  ->  ...\\mediapipe2\\weights\\xxx

dry-run 默认；--apply 实际写入。
"""
import re
import sys
from pathlib import Path

ROOT = Path(r'D:\Project\climbing_wall_search\mediapipe2')
SKIP_DIRS = {'.git', 'EdgeSAM', '__pycache__', '.venv', 'node_modules', 'weights'}
EXTS = {'.py', '.md', '.bat', '.ps1', '.txt'}

MODELS = [
    'yolox_l.onnx',
    'dw-ll_ucoco_384.onnx',
    'yolo26m-seg.pt',
    'yolo26n-seg.pt',
    'yolo26s-seg.pt',
    'yolo26l-seg.pt',
    'yolo26n-pose.pt',
    'yolo26s-pose.pt',
    'pose_landmarker_full.task',
    'pose_landmarker_heavy.task',
    'pose_landmarker_lite.task',
    'hand_landmarker.task',
]
ALT = '|'.join(re.escape(m) for m in MODELS)

# 1/2) 引号内的相对名（可选 ../ 前缀且尚未带 weights/）
REL_RE = re.compile(r"""(['"])(\.\.[/\\])?((?!weights[/\\])""" + f'({ALT})' + r""")\1""")

# 3) 绝对路径片段
ABS_RE = re.compile(r'(mediapipe2[/\\])(' + ALT + r')')

APPLY = '--apply' in sys.argv


def fix(text):
    hits = []

    def repl_rel(m):
        q, prefix, name = m.group(1), m.group(2) or '', m.group(3)
        sep = '\\' if '\\' in prefix else '/'
        hits.append(f'{prefix}{name} -> {prefix}weights{sep}{name}')
        return f'{q}{prefix}weights{sep}{name}{q}'

    text = REL_RE.sub(repl_rel, text)

    def repl_abs(m):
        hits.append(f'{m.group(0)} -> {m.group(1)}weights\\{m.group(2)}')
        return f'{m.group(1)}weights\\{m.group(2)}'

    text = ABS_RE.sub(repl_abs, text)
    return text, hits


def main():
    total_files, total_hits = 0, 0
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        if p.name == '_reorg_weights.py':        # 本脚本自身含模型名常量，跳过
            continue
        if p.suffix.lower() not in EXTS:
            continue
        try:
            text = p.read_text(encoding='utf-8')
        except Exception:
            continue
        new, hits = fix(text)
        if not hits:
            continue
        total_files += 1
        total_hits += len(hits)
        print(f'  {len(hits):>3}  {rel}')
        if APPLY:
            p.write_text(new, encoding='utf-8')

    print(f'\n{total_files} files, {total_hits} rewrites'
          + ('  [APPLIED]' if APPLY else '  (dry-run; use --apply)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
