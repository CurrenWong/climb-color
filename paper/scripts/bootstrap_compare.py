"""5s vs 1s 采样间隔的稳定性对比（验证"采样太稀"假设）。

数据源：
  5s: dwpose_exp/_verify_full.log        (原论文基准, 65 帧 / 10 墙, 每墙 4-9 帧)
  1s: dwpose_exp/dense_sampling.log      (加密采样, 3 面脆弱墙, 每墙 16-36 帧)

解析技巧：两个 log 都以 "⏱️ frame_XXXXs.jpg" 行标记每帧结束，
取其前面最近的 ">>> 线路色 = ... 票数 = {...}" 行作为该帧的票。
（dense log 因 Tee 会把同一行打印两次，用帧标记切分可自然去重。）

输出：paper/data/bootstrap_compare.json + 终端对比表
"""
import re
import json
import random
from pathlib import Path
from collections import Counter

BASE = Path(r"D:\Project\climbing_wall_search\mediapipe2")
LOGS = {
    '5s': BASE / "dwpose_exp" / "_verify_full.log",
    '1s': BASE / "dwpose_exp" / "dense_sampling.log",
}

TRUTH = {
    '3OHz6rZ5opG_wall_01_unknown': '蓝',
    '3OHz6rZ5opG_wall_04_line2_pink': '粉红',
    '3OHz6rZ5opG_wall_05_known_IMG_20260808_193650': '绿',
}
SHORT = {
    '3OHz6rZ5opG_wall_01_unknown': '3O_wall_01',
    '3OHz6rZ5opG_wall_04_line2_pink': '3O_wall_04_line2_pink',
    '3OHz6rZ5opG_wall_05_known_IMG_20260808_193650': '3O_wall_05',
}

VOTE_RE = re.compile(r'>>> 线路色 = (\S+)\s+置信度 = \d+%\s+票数 = (\{[^}]*\})')
WALL_RE = re.compile(r'^=== (\S+) \((\d+) 张\) ===$', re.M)


def parse_wall_frames(wall_name):
    """从两个 log 中找出该墙的逐帧票（用 '⏱️ frame_...' 行切帧）。"""
    out = {}
    for tag, path in LOGS.items():
        if not path.exists():
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        text = text.replace('\r\n', '\n')   # CRLF -> LF (re.M 的 $ 需要)
        # 定位该墙的 block
        blocks = re.split(r'^=== (.+?) \(\d+ 张\) ===$', text, flags=re.M)
        body = None
        for i in range(1, len(blocks), 2):
            if blocks[i].strip() == wall_name:
                body = blocks[i + 1]
                break
        if body is None:
            continue
        # 去掉块尾（下一墙/汇总），按帧标记切分
        frames = []
        # 每帧: [分析输出..., >>> 票数行, ⏱️ frame_XXXs.jpg: ...]
        chunks = re.split(r'frame_\d+s\.jpg: [\d.]+s', body)
        for ch in chunks[:-1]:
            ms = VOTE_RE.findall(ch)
            if not ms:
                continue
            votes = eval(ms[-1][1])
            if votes:
                frames.append(votes)
        out[tag] = frames
    return out


def analyze(frames, truth, B=2000, seed=0):
    total = Counter()
    for fv in frames:
        total.update(fv)
    pred, n = total.most_common(1)[0]
    overall = sum(total.values())
    second = total.most_common(2)[1] if len(total) > 1 else ('-', 0)
    rnd = random.Random(seed)
    hit = 0
    for _ in range(B):
        c = Counter()
        for _ in range(len(frames)):
            c.update(rnd.choice(frames))
        if c.most_common(1)[0][0] == truth:
            hit += 1
    return {
        'n_frames': len(frames),
        'votes_total': overall,
        'pred': pred,
        'top': f"{n}/{overall}",
        'share': n / overall if overall else 0.0,
        'runner_up': second[0],
        'runner_up_n': second[1],
        'margin': n - second[1],
        'bootstrap_acc': hit / B,
        'dist': dict(total.most_common(5)),
    }


def main():
    print(f"{'wall':<26} {'sample':<7} {'frames':>6} {'votes':>6} {'margin':>7} "
          f"{'boot acc':>9}  top-3")
    print('-' * 110)
    results = {}
    for wall, truth in TRUTH.items():
        per = parse_wall_frames(wall)
        results[SHORT[wall]] = {}
        for tag in ('5s', '1s'):
            if tag not in per or not per[tag]:
                print(f"{SHORT[wall]:<26} {tag:<7} {'-- no data --'}")
                continue
            r = analyze(per[tag], truth)
            results[SHORT[wall]][tag] = r
            top3 = '  '.join(f"{k}:{v}" for k, v in list(r['dist'].items())[:3])
            print(f"{SHORT[wall]:<26} {tag:<7} {r['n_frames']:>6} {r['votes_total']:>6} "
                  f"{r['margin']:>+7} {r['bootstrap_acc']:>8.1%}  {top3}")

    # joint: 3 walls all correct in one resample
    joint_out = {}
    print(f"\n=== joint (3 walls, one resample pass, B=2000) ===")
    for tag in ('5s', '1s'):
        if all(tag in results[w] for w in results if w != '_joint'):
            rnd = random.Random(0)
            per = {w: parse_wall_frames(k)[tag] for w, k in
                   zip([SHORT[k] for k in TRUTH], TRUTH)}
            ok = 0
            for _ in range(2000):
                good = True
                for wall, truth in TRUTH.items():
                    fr = per[SHORT[wall]]
                    c = Counter()
                    for _ in range(len(fr)):
                        c.update(rnd.choice(fr))
                    if c.most_common(1)[0][0] != truth:
                        good = False
                        break
                if good:
                    ok += 1
            print(f"  {tag}: P(all 3 walls correct) = {ok / 2000:.1%}")
            joint_out[tag] = ok / 2000
    results['_joint'] = joint_out

    # summary
    print(f"\n=== summary (mean over 3 walls) ===")
    for tag in ('5s', '1s'):
        accs = [results[w][tag]['bootstrap_acc'] for w in results
                if w != '_joint' and tag in results.get(w, {})]
        margins = [results[w][tag]['margin'] for w in results
                   if w != '_joint' and tag in results.get(w, {})]
        frames = [results[w][tag]['n_frames'] for w in results
                  if w != '_joint' and tag in results.get(w, {})]
        if accs:
            print(f"  {tag}: mean bootstrap acc = {sum(accs) / len(accs):.1%}, "
                  f"mean margin = {sum(margins) / len(margins):+.1f} votes, "
                  f"mean frames/wall = {sum(frames) / len(frames):.1f}")

    out = BASE / "paper" / "bootstrap_compare.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nsaved {out}")


if __name__ == '__main__':
    main()
