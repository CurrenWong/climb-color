"""VLM 基线实验：直接把帧交给视觉大模型（Qwen-VL）判断线路颜色。

目的：论文需要"通用视觉大模型能直接做这件事吗"的对照。
      VLM 是单帧、零领域先验的端到端答案；我们的方法用姿态锚定 + 跨帧聚合。

用法:
  python exp_vlm_baseline.py --test            # 1 张图冒烟（成本 <0.01 元）
  python exp_vlm_baseline.py --images          # 15 张单图测试集
  python exp_vlm_baseline.py --walls [每墙帧数] # 10 墙段（默认每墙 3 帧）
  python exp_vlm_baseline.py --walls 1         # 每墙 1 帧（最省）

输出: vlm_results.json + 终端表

注意: 需要环境变量 DASHSCOPE_API_KEY（百炼平台）。模型可用性见 --models。
"""
import os
import sys
import json
import time
import base64
from pathlib import Path

from openai import OpenAI

BASE = Path(__file__).resolve().parent.parent          # mediapipe2/
IMG_DIR = BASE / "img"
WALLS_DIR = BASE / "img_xhs_walls"
OUT_JSON = Path(__file__).resolve().parent / "vlm_results.json"

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-vl-max-latest"
CANDIDATE_MODELS = [
    "qwen-vl-max-latest",
    "qwen2.5-vl-72b-instruct",
    "qwen-vl-plus-latest",
    "qwen3-vl-plus",
]

COLORS = ['紫', '黄', '绿', '蓝', '红', '粉红', '白', '黑', '青', '橙']

PROMPT = """你是一位攀岩线路识别助手。图中有一位攀岩者正在攀爬一面岩壁。
请判断他此刻正在攀爬的线路是什么颜色。

候选颜色（必须从中选一个，只回答颜色名，不要任何解释）：
紫、黄、绿、蓝、红、粉红、白、黑、青、橙

若确实无法判断，回答：无法判断

只输出颜色名。"""

# 单图测试集真值（与论文表 2 一致）
IMG_TRUTH = {
    1: '紫', 2: '?', 3: '?', 4: '粉红', 5: '?', 6: '黄',
    7: '?', 8: '蓝', 9: '白',
}

# 10 墙段真值
WALL_TRUTH = {
    '2XRo7VbPgTe_wall_01_known_IMG_20260808_193753': '黑',
    '2XRo7VbPgTe_wall_02_known_IMG_20260808_193745': '蓝',
    '2XRo7VbPgTe_wall_03_known_IMG_20260808_193754': '粉红',
    '2XRo7VbPgTe_wall_04_known_IMG_20260808_193724': '青',
    '3OHz6rZ5opG_wall_01_unknown': '蓝',
    '3OHz6rZ5opG_wall_02_unknown': '红',
    '3OHz6rZ5opG_wall_03_known_IMG_20260808_193612': '黄',
    '3OHz6rZ5opG_wall_04_line1_white': '白',
    '3OHz6rZ5opG_wall_04_line2_pink': '粉红',
    '3OHz6rZ5opG_wall_05_known_IMG_20260808_193650': '绿',
}


def make_client():
    key = os.environ.get('DASHSCOPE_API_KEY')
    if not key:
        raise SystemExit("DASHSCOPE_API_KEY not set")
    return OpenAI(api_key=key, base_url=BASE_URL)


def encode(img_path):
    return base64.b64encode(Path(img_path).read_bytes()).decode()


def ask(client, img_path, model, retries=5, base_sleep=2.5):
    """调用一次；对 403 access_denied（限流）做指数退避重试。"""
    b64 = encode(img_path)
    last = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=32,
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'image_url',
                         'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                        {'type': 'text', 'text': PROMPT},
                    ],
                }],
            )
            txt = (r.choices[0].message.content or '').strip()
            time.sleep(0.8)          # 主动节流，避免触发限流
            return txt, time.time() - t0, None
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            if attempt == retries:
                return '', time.time() - t0, last
            wait = base_sleep * (attempt + 1)
            if '403' in str(e) or 'access_denied' in str(e):
                wait = max(wait, 5.0 * (attempt + 1))
            time.sleep(wait)
    return '', 0.0, last


def normalize(txt):
    """从模型输出中抽取颜色名（容忍多余文字）。"""
    for c in COLORS:
        if c in txt:
            return c
    if '无法' in txt or '不能' in txt:
        return '无法判断'
    return txt[:12] if txt else '空'


def run_images(client, model):
    rows = []
    files = sorted(IMG_DIR.glob('*.jpg'), key=lambda p: int(p.stem) if p.stem.isdigit() else 999)
    for p in files:
        n = int(p.stem) if p.stem.isdigit() else None
        if n is None or n not in IMG_TRUTH:
            continue
        truth = IMG_TRUTH[n]
        raw, dt, err = ask(client, p, model)
        pred = normalize(raw) if not err else 'ERROR'
        ok = (pred == truth) if truth != '?' else None
        rows.append({'img': p.name, 'truth': truth, 'raw': raw, 'pred': pred,
                     'ok': ok, 'sec': round(dt, 2), 'error': err})
        flag = {True: '✅', False: '❌', None: '?'}[ok]
        print(f"  {p.name:>9}  truth={truth:<4} pred={pred:<6} {flag}  ({dt:.1f}s)"
              + (f"  [{err[:60]}]" if err else ''))
    return rows


def run_walls(client, model, n_frames=3):
    rows = []
    walls = sorted([d for d in WALLS_DIR.iterdir() if d.is_dir()])
    for w in walls:
        truth = WALL_TRUTH.get(w.name)
        if truth is None:
            continue
        frames = sorted(w.glob('*.jpg'))
        if n_frames and n_frames < len(frames):
            step = max(1, len(frames) // n_frames)
            frames = frames[::step][:n_frames]
        print(f"\n=== {w.name}  ({len(frames)} 帧, truth={truth}) ===")
        per = []
        for fp in frames:
            raw, dt, err = ask(client, fp, model)
            pred = normalize(raw) if not err else 'ERROR'
            per.append(pred)
            print(f"  {fp.name:>16}  pred={pred:<6} ({dt:.1f}s)"
                  + (f"  [{err[:60]}]" if err else ''))
        votes = {}
        for p in per:
            if p in COLORS:
                votes[p] = votes.get(p, 0) + 1
        if votes:
            agg = max(votes.items(), key=lambda kv: kv[1])[0]
        else:
            agg = '无法判断'
        ok = (agg == truth)
        print(f"  >>> 单帧多数票 = {agg}  {votes}  真值={truth}  {'✅' if ok else '❌'}")
        rows.append({'wall': w.name, 'truth': truth, 'frames': n_frames,
                     'per_frame': per, 'votes': votes, 'pred': agg, 'ok': ok})
    return rows


def main():
    args = sys.argv[1:]
    model = DEFAULT_MODEL
    if '--model' in args:
        model = args[args.index('--model') + 1]
    client = make_client()

    if '--models' in args:
        for m in CANDIDATE_MODELS:
            raw, dt, err = ask(client, next(IMG_DIR.glob('1.jpg')), m)
            status = 'OK' if not err else err[:80]
            print(f"  {m:<28} {status}  -> {normalize(raw) if not err else ''}  ({dt:.1f}s)")
        return 0

    if '--test' in args:
        p = next(IMG_DIR.glob('1.jpg'))
        print(f"smoke test: {p.name} with {model}")
        raw, dt, err = ask(client, p, model)
        print(f"  raw  : {raw!r}")
        print(f"  pred : {normalize(raw)}   ({dt:.1f}s)")
        if err:
            print(f"  error: {err}")
        return 0

    result = {'model': model, 'mode': None, 'rows': []}
    t0 = time.time()
    if '--images' in args:
        result['mode'] = 'images'
        result['rows'] = run_images(client, model)
    elif '--walls' in args:
        n = 3
        for a in args:
            if a.isdigit():
                n = int(a)
        result['mode'] = f'walls_{n}f'
        result['rows'] = run_walls(client, model, n)
    else:
        print(__doc__)
        return 1

    # summary
    graded = [r for r in result['rows'] if r.get('ok') is not None]
    hits = sum(1 for r in graded if r['ok'])
    result['hits'] = hits
    result['total'] = len(graded)
    result['elapsed_sec'] = round(time.time() - t0, 1)
    print(f"\n=== summary ===\n  {hits}/{len(graded)} correct")
    print(f"  elapsed {result['elapsed_sec']}s")

    hist = []
    if OUT_JSON.exists():
        try:
            hist = json.loads(OUT_JSON.read_text(encoding='utf-8'))
            if isinstance(hist, dict):
                hist = [hist]
        except Exception:
            hist = []
    hist.append(result)
    OUT_JSON.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"  appended to {OUT_JSON.name}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
