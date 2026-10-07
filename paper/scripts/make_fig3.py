"""画 paper Fig 3: ring-based black vote correction 示意图（抽象 + 决策规则）。"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Circle, FancyBboxPatch, FancyArrowPatch

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6),
                               gridspec_kw={'width_ratios': [1, 1.15]})

# ============ Left: geometry ============
ax = axL
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal')
ax.axis('off')

# ring annulus: d1=0.9, d2=1.9 (scaled)
theta = np.linspace(0, 2 * np.pi, 200)
for r, style, lw in ((0.9, '--', 2.0), (1.9, '--', 2.0)):
    ax.plot(r * np.cos(theta), r * np.sin(theta), color='#F9A825',
            linestyle=style, linewidth=lw, zorder=3)
# light fill for the annulus
tt = np.linspace(0, 2 * np.pi, 100)
for r_in, r_out in ((0.9, 1.9),):
    xs = np.concatenate([r_in * np.cos(tt), r_out * np.cos(tt[::-1])])
    ys = np.concatenate([r_in * np.sin(tt), r_out * np.sin(tt[::-1])])
    ax.fill(xs, ys, color='#FFF59D', alpha=0.20, zorder=1)

# mask (EdgeSAM output) = the orange hold the endpoint sits on
mask = Ellipse((0, 0), 1.15, 0.78, angle=15, facecolor='#FF7043',
               edgecolor='#BF360C', linewidth=2.4, alpha=0.85, zorder=4)
ax.add_patch(mask)
ax.text(0, 0.16, 'mask\n(EdgeSAM)', ha='center', va='center', fontsize=8,
        weight='bold', color='white', zorder=5)

# endpoint
ax.plot(0, 0, marker='+', markersize=20, markeredgewidth=3.5,
        color='#00E676', zorder=10)
ax.annotate('endpoint\n(drifted 49 px)', xy=(0, 0), xytext=(-1.55, 1.5),
            fontsize=8.5, color='#1B5E20', ha='left',
            arrowprops=dict(arrowstyle='->', color='#1B5E20', lw=1.2,
                            connectionstyle='arc3,rad=0.25'))
_ = ax.plot([], [], marker='+', markersize=12, markeredgewidth=3,
            color='#00E676', linestyle='none', label='endpoint (positive prompt)')

# the black hold missed by the mask, sitting inside the ring
blk = Ellipse((-1.25, -0.75), 0.62, 0.5, angle=-20, facecolor='#212121',
              edgecolor='#000000', linewidth=2, zorder=6)
ax.add_patch(blk)
ax.text(-1.25, -0.75, 'black\nhold', ha='center', va='center', fontsize=8,
        weight='bold', color='white', zorder=7)
ax.annotate('missed by mask\n(low-sat pixels\ndropped by HSV)',
            xy=(-1.25, -0.75), xytext=(0.35, -1.75),
            fontsize=8, color='#B71C1C',
            arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.2,
                            connectionstyle='arc3,rad=-0.2'))

ax.text(-2.05, 1.95, 'ring d1=20 px\nd2=70 px', fontsize=8.5, color='#F57F17')
ax.set_title('(a) Mask misses the black hold;\nring captures it', fontsize=11, weight='bold')

# ============ Right: decision rule ============
ax = axR
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

def box(x, y, w, h, text, fc, ec='black', fs=10, tc='black', weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.12,rounding_size=0.2',
                                linewidth=1.8, edgecolor=ec, facecolor=fc, alpha=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color=tc, weight=weight, linespacing=1.5)

def arrow(x1, y1, x2, y2, label='', color='black', ls='-'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle='->,head_width=0.22,head_length=0.28',
                                 linewidth=1.8, color=color, linestyle=ls))
    if label:
        ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, label, fontsize=8.5,
                color=color, ha='left',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))

box(0.6, 8.3, 8.8, 1.3,
    'Input: mask color = non-black, ring = 20-70 px annulus',
    '#E3F2FD', fs=10.5, weight='bold')
arrow(5.0, 8.3, 5.0, 7.5)

box(0.6, 6.2, 8.8, 1.3,
    'Count ring pixels with  V < 60  OR  S < 35',
    '#FFF9C4', fs=10.5)
arrow(5.0, 6.2, 5.0, 5.4)

box(0.6, 4.1, 8.8, 1.3,
    'black_ratio = black / |ring|      (require |ring| >= 100 px)',
    '#FFF9C4', fs=10.5)
arrow(5.0, 4.1, 5.0, 3.3)

ax.add_patch(FancyBboxPatch((0.6, 1.7), 8.8, 1.6,
                            boxstyle='round,pad=0.12,rounding_size=0.2',
                            linewidth=2.2, edgecolor='#2E7D32',
                            facecolor='#C8E6C9', alpha=0.95))
ax.text(5.0, 2.5, 'if black_ratio >= 0.18:   correct  "non-black" -> "black"',
        ha='center', va='center', fontsize=11, weight='bold', color='#1B5E20')

# measured numbers box
ax.text(5.0, 0.95,
        'measured:  R_hand (needs fix) 29.3%   vs   L_hand (correct) 7.8%   ->   threshold 18%',
        ha='center', fontsize=9, style='italic', color='#37474F',
        bbox=dict(boxstyle='round,pad=0.35', fc='#ECEFF1', ec='#90A4AE'))

ax.text(5.0, 0.15,
        'cost: one numpy pass over ~13k pixels,  zero extra EdgeSAM forward',
        ha='center', fontsize=9, color='#6A1B9A')

ax.set_title('(b) Decision rule and measured threshold', fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig(r'D:\Project\climbing_wall_search\mediapipe2\paper\figures\fig3_ring_black.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved fig3_ring_black.png")
