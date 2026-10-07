"""批量把 dwpose_exp 内脚本的权重引用改为 ../ 相对根目录。"""
import os

EXP_DIR = r"D:\Project\climbing_wall_search\mediapipe2\dwpose_exp"

REPL = [
    ("DWPoseONNX('../weights/yolox_l.onnx', '../weights/dw-ll_ucoco_384.onnx')",
     "DWPoseONNX('../weights/yolox_l.onnx', '../weights/dw-ll_ucoco_384.onnx')"),
    ("'../weights/yolox_l.onnx'", "'../weights/yolox_l.onnx'"),
    ("'../weights/dw-ll_ucoco_384.onnx'", "'../weights/dw-ll_ucoco_384.onnx'"),
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
