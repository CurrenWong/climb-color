# paper/ — 论文与实验材料

**ClimbColor: Pose-Anchored Hold Segmentation for Climbing Route Color Recognition**

中英双语正文 + 图 + 数据 + 导出脚本。目标投稿：CVPR/ICCV/ECCV 运动分析 workshop
或 ACM MM。

## 目录

| 路径 | 内容 |
|---|---|
| `main.md` | 论文正文（英文，约 1240 行） |
| `main_zh.md` | 中文版（全文翻译，术语表见 `../PAPER_ZH_TODO.md`） |
| `method_3_5_ring_black.md` | 3.5 节详版：环形带黑票修正的算法、阈值推导、成本、失效模式 |
| `tables_data.md` | 各表原始数据 + 复现命令 |
| `figures/` | 图 1–5 |
| `scripts/` | 图 / DOCX 生成脚本 + bootstrap 分析脚本 |
| `data/` | bootstrap 结果 JSON |
| `*.docx` | DOCX 导出（可再生产物，已 gitignore） |

## 图

| 编号 | 文件 | 内容 |
|---|---|---|
| Fig 1 | `figures/fig1_joint.png` | 联合系统总览（照片 → 检索 → 识别） |
| Fig 1b | `figures/fig1_pipeline.png` | 识别流水线（六阶段） |
| Fig 2 | `figures/fig2_prompt_design.png` | 提示几何：1 正点 + 14 负点 + 自适应框 |
| Fig 3 | `figures/fig3_ring_black.png` | 环形带黑票修正：几何 + 决策规则 + 实测阈值 |
| Fig 4 | `figures/fig4_box_adaptive.jpg` | 自适应框前后对比（box=160 选远处红点 vs box=128 选白点） |
| Fig 5 | `figures/fig5_failure_cases.jpg` | 两个失败案例的机理（小目标 / 遮挡） |

## 表

| 编号 | 内容 | 数据来源 |
|---|---|---|
| Table 1 | 主结果（4 → 6 → 8 → 9 → 10） | `../baselines/*.txt` + `../dwpose_exp/dwpose_full_summary.txt` |
| Table 2 | 单图测试集 | `tables_data.md` |
| Table 3 | 消融 | `tables_data.md` |
| Table 4 | 姿态模型对比 | `tables_data.md` |
| Table 5 | 逐帧延迟分解 | `../dwpose_exp/profile_stages.json` |
| Table 6 | 逐墙稳定性 + bootstrap | `data/bootstrap_results.json` |
| Table 7 | 采样密度（5 s vs 1 s） | `data/bootstrap_compare.json` |
| Table 8 | Qwen-VL 对比 | `../vlm_exp/vlm_results.json` |
| Table 9 | 联合墙与路线检索（34 照片 LOO / 7 链路 / 延迟） | `../retrieval_exp/joint_eval.py` + `demo_search.py` + `loo_hybrid.py` |

## 重新生成

```bash
cd paper/scripts

# 图（需要 matplotlib）
python make_fig1_joint.py ; python make_fig1.py ; python make_fig2.py ; python make_fig3.py
# Fig 4 的素材需先跑两次候选诊断（FIXED_BOX=0 / 1）：
#   cd ../../dwpose_exp && python dbg_line1_frame02_candidates.py 1
python make_fig4.py ; python make_fig_failcases.py

# bootstrap 分析（依赖 dwpose_exp 里的 log）
python bootstrap_margin.py      # -> data/bootstrap_results.json
python bootstrap_compare.py     # -> data/bootstrap_compare.json

# 联合检索（依赖 retrieval_exp 环境：faiss/kornia/transformers + weights/dinov2-small）
cd ../../retrieval_exp
python joint_eval.py            # 7 张照片→视频链路 + 联合准确率（any-hit 7/7, top-1 6/7）
python demo_search.py           # 34 张照片 LOO（8 对标注 GT: TP=6 FP=0）
python loo_hybrid.py            # 34 照片混合索引 LOO（物理墙语义判定）

# DOCX 导出（默认输出到 paper/）
cd ../scripts
python make_docx.py climbing_route_color_paper.docx main.md
python make_docx.py climbing_route_color_paper_zh.docx main_zh.md
```
