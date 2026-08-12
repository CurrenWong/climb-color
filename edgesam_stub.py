"""
EdgeSAM 加载垫片：绕过 EdgeSAM 顶层不必要的 training-only import。

问题：`edge_sam/modeling/sam.py` 顶层无条件 import：
  - mmdet.models.dense_heads / necks
  - mmengine
  - projects.EfficientDet （仓库内 git submodule，--depth 1 clone 不到）
  - loralib （仅在 LoRA 启用时需要，否则 inference 不实例化）

我们只做 inference（不会构造 RPNHead 实例），这些 import 都用 stub 替代。
在所有 edge_sam.* import 之前 import 本模块即可。
"""
import sys
import types

import torch


class _Stub:
    """被实例化也无所谓（我们 inference 路径不会触发）。"""
    def __init__(self, *a, **kw):
        self.cfg = kw


def _install_stubs() -> None:
    """安装 mmdet / mmengine / projects.EfficientDet 的最小 stub。"""
    if "edge_sam" in sys.modules:
        return

    # ---- mmdet.* stub tree ----
    mmdet = types.ModuleType("mmdet")
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models_dense_heads = types.ModuleType("mmdet.models.dense_heads")
    mmdet_models_necks = types.ModuleType("mmdet.models.necks")

    mmdet_models_dense_heads.RPNHead = _Stub
    mmdet_models_dense_heads.CenterNetUpdateHead = _Stub
    mmdet_models_necks.FPN = _Stub
    mmdet_models.dense_heads = mmdet_models_dense_heads
    mmdet_models.necks = mmdet_models_necks
    mmdet.models = mmdet_models

    sys.modules["mmdet"] = mmdet
    sys.modules["mmdet.models"] = mmdet_models
    sys.modules["mmdet.models.dense_heads"] = mmdet_models_dense_heads
    sys.modules["mmdet.models.necks"] = mmdet_models_necks

    # ---- mmengine stub ----
    mmengine = types.ModuleType("mmengine")

    class _ConfigDict(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

        def __setattr__(self, k, v):
            self[k] = v

    mmengine.ConfigDict = _ConfigDict
    sys.modules["mmengine"] = mmengine

    # ---- projects.EfficientDet (git submodule) stub ----
    projects = types.ModuleType("projects")
    efficient_det = types.ModuleType("projects.EfficientDet")
    efficient_det.efficientdet = _Stub
    projects.EfficientDet = efficient_det
    sys.modules["projects"] = projects
    sys.modules["projects.EfficientDet"] = efficient_det


_install_stubs()


# loralib 仅 inference（无 LoRA）路径用不到，先尝试真 import，没有就 stub
try:
    import loralib  # noqa: F401
except ImportError:
    _loralib = types.ModuleType("loralib")
    _loralib.Linear = _Stub
    _loralib.MergedLinear = _Stub
    _loralib.LoRAConv1d = _Stub
    _loralib.LoRAConv2d = _Stub
    _loralib.Conv1d = _Stub
    _loralib.Conv2d = _Stub
    _loralib.mark_only_lora_trainable = lambda *a, **kw: None
    _loralib.merge_and_unload = lambda *a, **kw: None
    sys.modules["loralib"] = _loralib


# EdgeSAM 的 .pth 是用 CUDA 训练的，state_dict 的 key 包含 'cuda:0'。
# EdgeSAM 仓库 build_sam.py 第 143 行：state_dict = torch.load(f)
# —— 没用 map_location，所以 CPU 机器直接 deserial 失败。
# 我们把 torch.load 默认 map_location 设为 'cpu'，避免改 EdgeSAM 源码。
import functools as _ft
_orig_load = torch.load
if getattr(_orig_load, "__wrapped__", None) is None or not hasattr(_orig_load, "_edgesam_cpu_safe"):
    torch.load = _ft.partial(_orig_load, map_location=torch.device("cpu"))
    torch.load._edgesam_cpu_safe = True
