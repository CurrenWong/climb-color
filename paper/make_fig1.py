"""画 paper Fig 1: Pipeline overview."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Color palette
COLOR_INPUT = '#4A90E2'    # Blue
COLOR_PERSON = '#7ED321'    # Green
COLOR_POSE = '#F5A623'     # Orange
COLOR_EDGESAM = '#BD10E0'   # Purple
COLOR_VOTE = '#D0021B'      # Red
COLOR_OUTPUT = '#9013FE'    # Dark purple

def box(x, y, w, h, text, color, fontsize=10, textcolor='white'):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.08,rounding_size=0.15',
                       linewidth=2, edgecolor='black', facecolor=color, alpha=0.85)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, color=textcolor, weight='bold')

def arrow(x1, y1, x2, y2, label=''):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='->,head_width=0.25,head_length=0.3',
                        linewidth=2, color='black')
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label,
                ha='center', fontsize=8, color='black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))

# Input image
box(0.3, 2.3, 1.5, 1.4, 'Input\nimage\n(WxH)', COLOR_INPUT, fontsize=10)

# Person mask branch
box(2.5, 4.2, 2.0, 1.0, 'YOLO26m\nPerson mask', COLOR_PERSON, fontsize=9)
arrow(1.8, 3.2, 2.5, 4.5, label='image')

# Pose branch
box(2.5, 2.3, 2.0, 1.0, 'MediaPipe Pose\nor DWPose', COLOR_POSE, fontsize=9)
arrow(1.8, 3.0, 2.5, 2.8, label='image')

# Person mask + Pose → Endpoints + Box prompt
box(5.2, 3.8, 2.2, 1.4,
    '4 Endpoints\n(L/R hand, L/R foot)\n+\nAdaptive box prompt',
    COLOR_PERSON, fontsize=8)

arrow(4.5, 4.7, 5.2, 4.7)
arrow(4.5, 2.8, 5.2, 4.4)

# EdgeSAM 4 candidates
box(5.2, 1.4, 2.2, 1.4,
    'EdgeSAM\n(box+1+14 prompts)\n→ 4 candidate masks',
    COLOR_EDGESAM, fontsize=8)

arrow(6.3, 3.8, 6.3, 2.8, label='per endpoint')

# Pixel-level voting
box(8.2, 2.3, 2.2, 1.4,
    'Pixel-level\ncolor voting\n+ main component',
    COLOR_VOTE, fontsize=9)

arrow(7.4, 2.1, 8.2, 2.8)

# Ring black fix (旁路，标虚线)
ax.add_patch(FancyBboxPatch((8.0, 0.4), 2.6, 1.0,
                             boxstyle='round,pad=0.08,rounding_size=0.15',
                             linewidth=1.5, edgecolor='black', facecolor='#FFE082',
                             alpha=0.7, linestyle='--'))
ax.text(9.3, 0.9, 'Ring black vote\ncorrection\n(if non-black)',
        ha='center', va='center', fontsize=8, weight='bold')

arrow(9.3, 2.3, 9.3, 1.4, label='')

# Per-endpoint color (4 端点 → 4 票)
box(11.0, 2.3, 1.4, 1.4,
    '4 endpoint\ncolors\n(1 vote each)',
    '#50E3C2', fontsize=9)

arrow(10.4, 3.0, 11.0, 3.0)

# Cross-frame aggregation
box(11.0, 0.3, 1.4, 1.4,
    'Cross-frame\naggregation\n(per wall)',
    '#9013FE', fontsize=8)

arrow(11.7, 2.3, 11.7, 1.7, label='')

# Final output
box(12.8, 2.3, 1.0, 1.4,
    'Route\ncolor',
    '#BD10E0', fontsize=10)

arrow(12.4, 3.0, 12.8, 3.0)

# Title
ax.text(7.0, 5.7, 'EdgeSAM Body-Anchored Pipeline for Route Color Recognition',
        ha='center', fontsize=14, weight='bold')
ax.text(7.0, 5.35,
        '(solid arrows = main flow, dashed = optional post-correction)',
        ha='center', fontsize=9, style='italic', color='gray')

# Legend for color coding
legend_y = 0.2
ax.text(0.3, legend_y, '● Pose input', fontsize=8, color=COLOR_POSE)
ax.text(2.5, legend_y, '● Body context', fontsize=8, color=COLOR_PERSON)
ax.text(5.0, legend_y, '● EdgeSAM', fontsize=8, color=COLOR_EDGESAM)
ax.text(7.5, legend_y, '● Voting/correction', fontsize=8, color=COLOR_VOTE)
ax.text(10.5, legend_y, '● Aggregation', fontsize=8, color=COLOR_OUTPUT)

plt.tight_layout()
plt.savefig(r'D:\Project\climbing_wall_search\mediapipe2\paper\fig1_pipeline.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved fig1_pipeline.png")
