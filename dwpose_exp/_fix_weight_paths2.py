"""批量把 dwpose_exp 内脚本的权重引用改为 ../ 相对根目录（第二轮）。"""
import os

EXP_DIR = r"D:\Project\climbing_wall_search\mediapipe2\dwpose_exp"

REPL = [
    ('PersonMasker(model_name="../weights/yolo26m-seg.pt"', 'PersonMasker(model_name="../weights/yolo26m-seg.pt"'),
    ("YOLO('../weights/yolo26n-pose.pt')", "YOLO('../weights/yolo26n-pose.pt')"),
    ("EdgeSamHoldSegmenter('../weights/edge_sam.pth'", "EdgeSamHoldSegmenter('../weights/edge_sam.pth'"),
]

for f in os.listdir(EXP_DIR):
    if not f.endswith('.py'):
        continue
    p = os.path.join(EXP_DIR, f)
    with open(p, encoding='utf-8') as fh:
        c = fh.read()
    orig = c
    for old, new in REPL:
        c = c.replace(old, new)
    if c != orig:
        with open(p, 'w', encoding='utf-8') as fh:
            fh.write(c)
        print(f"OK {f}")
