"""paper/ 重组后的路径修正。

处理：
  A. 通用：`paper/figN_*` -> `paper/figures/figN_*`（md 引用 + 脚本输出路径）
     `paper/bootstrap_*.json` -> `paper/data/bootstrap_*.json`
  B. make_fig4.py：素材由 dwpose_exp/ 改为 debug/（此前已移动并改名），输出进 figures/
  C. make_fig_failcases.py：输入改为 debug/failcase_*.jpg，输出进 figures/
  D. make_docx.py：BASE 上移一层（脚本现在在 paper/scripts/），图片查找改 paper/figures/
"""
import re
import sys
from pathlib import Path

PAPER = Path(r'D:\Project\climbing_wall_search\mediapipe2\paper')
APPLY = '--apply' in sys.argv


def sub_file(path: Path, pairs, regex_pairs=()):
    t = path.read_text(encoding='utf-8')
    orig = t
    for a, b in pairs:
        t = t.replace(a, b)
    for pat, rep in regex_pairs:
        t = re.sub(pat, rep, t)
    if t != orig:
        n = sum(1 for a, _ in pairs if a in orig)
        print(f'  changed: {path.relative_to(PAPER.parent)}')
        if APPLY:
            path.write_text(t, encoding='utf-8')
        return True
    return False


def main():
    # --- A. 通用路径 ---
    gen_pairs = [
        ('paper\\bootstrap_compare.json', 'paper\\data\\bootstrap_compare.json'),
        ('paper/bootstrap_compare.json', 'paper/data/bootstrap_compare.json'),
        ('paper\\bootstrap_results.json', 'paper\\data\\bootstrap_results.json'),
        ('paper/bootstrap_results.json', 'paper/data/bootstrap_results.json'),
    ]
    gen_rx = [
        (r'paper[\\/]fig(\d)', r'paper\1'.replace('\\1', r'\g<0>') and r'paper/figures/fig\1'),
    ]
    # 上面这行太绕，改成两条明确规则
    gen_rx = [
        (r'paper\\(fig\d)', r'paper\\figures\\\1'),
        (r'paper/(fig\d)', r'paper/figures/\1'),
    ]

    targets = list(PAPER.glob('*.md')) + list((PAPER / 'scripts').glob('*.py'))
    for p in targets:
        if p.name == '_reorg_paper.py':
            continue
        sub_file(p, gen_pairs, gen_rx)

    # --- B. make_fig4.py ---
    f4 = PAPER / 'scripts' / 'make_fig4.py'
    sub_file(f4, [
        ('BASE = Path(r"D:\\Project\\climbing_wall_search\\mediapipe2")',
         'BASE = Path(__file__).resolve().parents[2]      # 项目根'),
        ('EXP = BASE / "dwpose_exp"', 'EXP = BASE / "debug"      # 素材已移到 debug/'),
        ("('_fig4_before.jpg',", "('fig4_src_box160.jpg',"),
        ("('_fig4_after.jpg',", "('fig4_src_box128.jpg',"),
        ('素材: dwpose_exp/_fig4_before.jpg / _fig4_after.jpg',
         '素材: debug/fig4_src_box160.jpg / debug/fig4_src_box128.jpg（FIXED_BOX=0/1 两次运行）'),
        ('out = BASE / "paper" / "fig4_box_adaptive.jpg"',
         'out = BASE / "paper" / "figures" / "fig4_box_adaptive.jpg"'),
    ])

    # --- C. make_fig_failcases.py ---
    f5 = PAPER / 'scripts' / 'make_fig_failcases.py'
    sub_file(f5, [
        ('BASE = Path(r"D:\\Project\\climbing_wall_search\\mediapipe2")',
         'BASE = Path(__file__).resolve().parents[2]      # 项目根'),
        ("('failcase_1.jpg',", "('debug/failcase_img6_pred_white.jpg',"),
        ("('failcase_2.jpg',", "('debug/failcase_img8_pred_purple.jpg',"),
        ('out = BASE / "paper" / "fig5_failure_cases.jpg"',
         'out = BASE / "paper" / "figures" / "fig5_failure_cases.jpg"'),
    ])

    # --- D. make_docx.py ---
    md = PAPER / 'scripts' / 'make_docx.py'
    sub_file(md, [
        ('BASE = Path(__file__).parent', 'BASE = Path(__file__).resolve().parents[1]   # paper/'),
        ("img_path = BASE.parent / 'paper' / img", "img_path = BASE / img"),
        (r"IMG_RE = re.compile(r'paper[/\\](fig[\w]+_[\w]+\.(?:png|jpg))')",
         r"IMG_RE = re.compile(r'paper[/\\]figures[/\\](fig[\w]+_[\w]+\.(?:png|jpg))')"),
        ("'paper/figures/figN_*.png|jpg'", "'paper/figures/figN_*.png|jpg'"),
    ])

    print('  (dry-run; use --apply)' if not APPLY else '  [APPLIED]')
    return 0


if __name__ == '__main__':
    sys.exit(main())
