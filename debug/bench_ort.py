"""测 onnxruntime 线程配置对 yolox_l 推理速度的影响。"""
import time
import numpy as np
import onnxruntime as ort

MODEL = "../weights/yolox_l.onnx"
inp = np.random.rand(1, 3, 640, 640).astype(np.float32)

# 默认
t = time.time()
s1 = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
print(f"session create (default): {time.time()-t:.2f}s")
name = s1.get_inputs()[0].name
t = time.time(); s1.run(None, {name: inp}); print(f"  run 1 (default)   : {time.time()-t:.2f}s")
t = time.time(); s1.run(None, {name: inp}); print(f"  run 2 (default)   : {time.time()-t:.2f}s")

for nt in (4, 8, 12):
    so = ort.SessionOptions()
    so.intra_op_num_threads = nt
    so.inter_op_num_threads = 1
    s2 = ort.InferenceSession(MODEL, so, providers=['CPUExecutionProvider'])
    s2.run(None, {name: inp})  # warmup
    t = time.time(); s2.run(None, {name: inp})
    print(f"  run ({nt} threads)  : {time.time()-t:.2f}s")

print(f"\nort version: {ort.__version__}")
