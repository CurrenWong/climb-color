"""画 paper Fig 2: body-anchored visual prompting 几何示意图。

展示 1 个正点 + 14 个负点的布局:
  - 端点 (px, py) = 正点 (绿 +)
  - 身体方向 1 个 (红 -)
  - 端点环 5 个 (红 -)
  - person mask 边界 8 个 (橙 -)
  外加 box prompt 虚线框 (自适应 half)
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, FancyArrow, Ellipse

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.6, 1.6)
ax.set_aspect('equal')
ax.axis('off')

# --- person mask 区域 (下半部不规则 blob) ---
theta = np.linspace(0, 2 * np.pi, 200)
person_r = 0.85 + 0.12 * np.sin(3 * theta)
person_x = person_r * np.cos(theta) * 0.95
person_y = person_r * np.sin(theta) * 0.95 - 0.35
ax.fill(person_x, person_y, color='#B39DDB', alpha=0.35, zorder=1,
        label='person mask (YOLO26m)')
ax.plot(person_x, person_y, color='#7E57C2', linewidth=1.5, linestyle='--', zorder=2)

# --- hold (端点所在，椭圆) ---
hold = Ellipse((0, 0), 0.62, 0.42, angle=20, facecolor='#FF8A65',
               edgecolor='#BF360C', linewidth=2, alpha=0.7, zorder=3)
ax.add_patch(hold)
ax.text(0, 0.0, 'hold', ha='center', va='center', fontsize=11,
        weight='bold', color='#3E2723', zorder=4)

# --- box prompt (自适应 half) ---
box_half = 0.62
ax.add_patch(Rectangle((-box_half, -box_half), 2 * box_half, 2 * box_half,
                       fill=False, edgecolor='#1976D2', linewidth=2.5,
                       linestyle='-.', zorder=5, label=f'box prompt (half={box_half})'))

# --- 端点 = 正点 ---
ax.plot(0, 0.02, marker='+', markersize=22, markeredgewidth=4,
        color='#00C853', zorder=10, label='positive: endpoint (px, py)')

# --- 身体方向负点 ---
body_cx, body_cy = 0.0, -0.55
ux, uy = body_cx - 0, body_cy - 0.02
norm = np.hypot(ux, uy)
ux, uy = ux / norm, uy / norm
ax.annotate('', xy=(0.18 * ux, 0.02 + 0.18 * uy), xytext=(0, 0.02),
            arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2), zorder=8)
ax.plot(0.18 * ux, 0.02 + 0.18 * uy, marker='o', markersize=11,
        color='#D32F2F', zorder=9, label='negative: 1 toward body')

# --- 端点环 5 负点 ---
ring_r = 0.22
for t in np.linspace(0, 2 * np.pi, 5, endpoint=False):
    ax.plot(ring_r * np.cos(t), 0.02 + ring_r * np.sin(t), marker='o',
            markersize=9, color='#E53935', zorder=9)
_ = ax.plot([], [], marker='o', markersize=9, color='#E53935', linestyle='none',
            label='negative: 5 around endpoint')

# --- person mask 边界 8 负点 ---
edge_pts = []
for t in np.linspace(0, 2 * np.pi, 8, endpoint=False):
    # 采样 person blob 边界附近
    x = 0.9 * np.cos(t) * 0.95
    y = (0.85 + 0.12 * np.sin(3 * t)) * np.sin(t) * 0.95 - 0.35
    # 只保留靠近端点 (<0.6) 的
    if np.hypot(x - 0, y - 0.02) <= 0.75:
        edge_pts.append((x, y))
for x, y in edge_pts:
    ax.plot(x, y, marker='o', markersize=9, color='#FB8C00', zorder=9)
_ = ax.plot([], [], marker='o', markersize=9, color='#FB8C00', linestyle='none',
            label=f'negative: {len(edge_pts)} on person edge')

# --- 标注 ---
ax.annotate('endpoint drift\n(≤49 px)', xy=(0, 0.02), xytext=(0.75, 0.95),
            fontsize=9, color='#33691E',
            arrowprops=dict(arrowstyle='->', color='#33691E', lw=1.2,
                            connectionstyle='arc3,rad=-0.2'))

ax.set_title('Body-Anchored Visual Prompting\n'
             '1 positive point + 14 negative points (1 body + 5 ring + 8 person-edge)',
             fontsize=12, weight='bold', pad=14)

ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig(r'D:\Project\climbing_wall_search\mediapipe2\paper\fig2_prompt_design.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved fig2_prompt_design.png")
