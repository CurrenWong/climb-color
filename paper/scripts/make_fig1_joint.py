"""画 paper Fig 1: Joint wall-and-route retrieval system overview.

两阶段:Stage 1 检索(DINOv2 召回 + ALIKED/RANSAC 验证 + 自动墙聚类)
         → 墙 ID + 时间戳
        Stage 2 识别(EdgeSAM 身体锚定管线) → 路线颜色
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis('off')

# Palette
C_QUERY = '#4A90E2'     # blue
C_RECALL = '#50E3C2'    # teal
C_VERIFY = '#F5A623'    # orange
C_WALL = '#7ED321'      # green
C_EDGESAM = '#BD10E0'   # purple
C_VOTE = '#D0021B'      # red
C_OUT = '#9013FE'       # dark purple

def box(x, y, w, h, text, color, fontsize=9, textcolor='white', ls='-', lw=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.08,rounding_size=0.15',
                       linewidth=lw, edgecolor='black', facecolor=color,
                       alpha=0.85, linestyle=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, color=textcolor, weight='bold')

def arrow(x1, y1, x2, y2, label='', color='black', ls='-', lw=2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='->,head_width=0.25,head_length=0.3',
                        linewidth=lw, color=color, linestyle=ls)
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.25, label,
                ha='center', fontsize=8, color='black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))

# ── Stage 1 label ──
ax.text(3.5, 6.6, 'Stage 1 — Wall retrieval', fontsize=13, weight='bold', ha='center', color='#1B5E20')
ax.text(11.8, 6.6, 'Stage 2 — Route color recognition', fontsize=13, weight='bold', ha='center', color='#4A148C')

# Query photo
box(0.4, 3.0, 1.6, 1.6, 'Wall photo\n(query q)', C_QUERY, fontsize=10)

# DINOv2 recall
box(2.8, 4.6, 2.4, 1.2, 'DINOv2-S/14\nglobal descriptor\n(cosine ≥ 0.70)', C_RECALL, fontsize=8)
arrow(2.0, 3.8, 2.8, 5.0, label='g(q)')

# FAISS
box(2.8, 2.3, 2.4, 1.2, 'FAISS IndexFlatIP\n(video frames 5s)', C_RECALL, fontsize=8)
arrow(2.0, 3.8, 2.8, 2.9)

# recall candidates
box(5.9, 3.4, 2.0, 1.6, 'Candidates\nC(q)', '#FFFFFF', fontsize=9, textcolor='black')
arrow(5.2, 3.6, 5.9, 4.0, label='recall')
arrow(5.2, 3.2, 5.9, 3.8)

# ALIKED + RANSAC verification
box(8.7, 4.6, 2.6, 1.2, 'ALIKED local features\n+ Lowe ratio', C_VERIFY, fontsize=8)
arrow(7.9, 4.6, 8.7, 5.2)
box(8.7, 2.3, 2.6, 1.2, 'RANSAC homography\n(inliers ≥ 20, ratio ≥ 0.40)', C_VERIFY, fontsize=8)
arrow(7.9, 3.8, 8.7, 2.9)

# Verified frames → wall ID + timestamp
box(12.1, 3.4, 2.0, 1.6, 'Verified frames\n→ Wall ID\n+ timestamp', C_WALL, fontsize=9)
arrow(11.3, 4.2, 12.1, 4.2, label='verify')
arrow(11.3, 3.2, 12.1, 3.6)

# ── Stage 2 (recognition detail, compact) ──
box(12.1, 0.7, 3.4, 1.3,
    'EdgeSAM body-anchored\nperson mask + pose → prompt\npixel voting + ring black fix',
    C_EDGESAM, fontsize=8)
arrow(13.1, 3.4, 13.1, 2.0, label='retrieved frames')

# Final output
box(15.2, 0.4, 0.0, 0.0, '', C_OUT)  # placeholder removed below

box(14.9, 0.4, 0.0, 0.0, '', C_OUT)

# Final output box (right)
box(15.2, 2.0, 0.6, 0.6, '', C_OUT)
ax.text(15.5, 2.3, 'Route\ncolor', ha='center', va='center', fontsize=9, weight='bold', color='white')

# Auto wall clustering (dashed, bottom of Stage 1)
ax.add_patch(FancyBboxPatch((5.9, 0.6), 2.6, 1.0,
                             boxstyle='round,pad=0.08,rounding_size=0.15',
                             linewidth=1.5, edgecolor='black', facecolor='#FFF3CD',
                             alpha=0.8, linestyle='--'))
ax.text(7.2, 1.1, 'Union-find clustering\n→ automatic wall split\n(= manual 10-route split)',
        ha='center', va='center', fontsize=8, weight='bold')
arrow(7.2, 3.4, 7.2, 1.6, color='#8A6D3B', ls='--', lw=1.5)

# Title
ax.text(8.0, 6.9, 'Joint Wall & Route Retrieval: photo → wall ID + timestamp → route color',
        ha='center', fontsize=14, weight='bold')
ax.text(8.0, 6.35, '(solid arrows = main flow, dashed = automatic wall clustering)',
        ha='center', fontsize=9, style='italic', color='gray')

# Legend
lg_y = 0.15
ax.text(0.3, lg_y, '● query photo', fontsize=8, color=C_QUERY)
ax.text(2.0, lg_y, '● global recall', fontsize=8, color=C_RECALL)
ax.text(4.4, lg_y, '● geometric verify', fontsize=8, color=C_VERIFY)
ax.text(7.0, lg_y, '● wall ID / frames', fontsize=8, color=C_WALL)
ax.text(10.0, lg_y, '● EdgeSAM recognition', fontsize=8, color=C_EDGESAM)
ax.text(13.2, lg_y, '● output color', fontsize=8, color=C_OUT)

plt.tight_layout()
plt.savefig(r'D:\Project\climbing_wall_search\mediapipe2\paper\figures\fig1_joint.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved fig1_joint.png")
