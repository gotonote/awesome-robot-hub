#!/usr/bin/env python3
"""Fetch the latest star count of this repo and render the README star chart.

- Queries the GitHub API for gotonote/awesome-robot-hub's stargazers_count
- Records it into docs/star-history.json (same-day records are overwritten)
- Regenerates docs/star-chart.svg from the history (self-hosted, served by
  GitHub Pages so the README image never depends on third-party services)

Intended to run inside GitHub Actions (GITHUB_TOKEN gives 5000 req/h);
also works locally with the anonymous limit of 60 req/h.
"""

import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = "gotonote/awesome-robot-hub"
API = "https://api.github.com/repos/"
TOKEN = os.environ.get("GITHUB_TOKEN", "")
USER_AGENT = "awesome-robot-hub star-updater"

ROOT = Path(__file__).resolve().parent.parent
HIST_PATH = ROOT / "docs" / "star-history.json"
CHART_PATH = ROOT / "docs" / "star-chart.svg"

# Chart layout (mirrors awesome-agent-boom's star chart)
W, H, pad_l, pad_r, pad_t, pad_b = 800, 260, 70, 20, 30, 40
COLOR = "#6c8cff"


def fetch_stars(repo: str = REPO) -> int:
    req = urllib.request.Request(API + repo)
    req.add_header("User-Agent", USER_AGENT)
    req.add_header("Accept", "application/vnd.github+json")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return int(data["stargazers_count"])


def render_star_chart(hist: dict) -> str:
    """根据 docs/star-history.json 生成 SVG 折线图（README 展示，托管在 GitHub Pages）。"""
    dates = sorted(hist)
    if len(dates) < 2:
        msg = "📈 增长曲线数据积累中，每日自动更新…" if dates else "📈 Star 增长曲线（每日自动更新）"
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'style="font-family:Segoe UI,PingFang SC,sans-serif">'
            f'<rect width="{W}" height="{H}" rx="12" fill="#141a2b"/>'
            f'<text x="{W / 2}" y="{H / 2}" fill="#8b93a7" font-size="16" text-anchor="middle">{msg}</text>'
            f"</svg>"
        )
    vals = [hist[d] for d in dates]
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        vmax = vmin + 1
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b

    def px(d: str, v: int) -> tuple:
        x = pad_l + (dates.index(d) / (len(dates) - 1)) * plot_w
        y = pad_t + (1 - (v - vmin) / (vmax - vmin)) * plot_h
        return x, y

    pts = " ".join(f"{px(d, v)[0]:.1f},{px(d, v)[1]:.1f}" for d, v in zip(dates, vals))
    # 网格与 Y 轴刻度（4 档）
    grid = []
    for i in range(4):
        v = vmin + (vmax - vmin) * i / 3
        y = pad_t + plot_h * (1 - i / 3)
        grid.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W - pad_r}" y2="{y:.1f}" stroke="#232b40"/>')
        grid.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" fill="#8b93a7" font-size="11" text-anchor="end">{int(round(v))}</text>')
    # X 轴日期标签（首 / 中 / 尾）
    xl = []
    for i in (0, len(dates) // 2, len(dates) - 1):
        d = dates[i]
        xl.append(f'<text x="{px(d, vals[i])[0]:.1f}" y="{H - 14}" fill="#8b93a7" font-size="11" text-anchor="middle">{d}</text>')
    latest = vals[-1]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'style="font-family:Segoe UI,PingFang SC,sans-serif">'
        f'<rect width="{W}" height="{H}" rx="12" fill="#141a2b"/>'
        f'<text x="{pad_l}" y="18" fill="#e6e9f2" font-size="14" font-weight="bold">⭐ Star 增长趋势 · 当前 {latest}</text>'
        + "".join(grid)
        + f'<polyline points="{pts}" fill="none" stroke="{COLOR}" stroke-width="2.5"/>'
        + "".join(
            f'<circle cx="{px(d, v)[0]:.1f}" cy="{px(d, v)[1]:.1f}" r="3.5" fill="{COLOR}"/>'
            for d, v in zip(dates, vals)
        )
        + "".join(xl)
        + f'<text x="{W - pad_r}" y="{H - 14}" fill="#8b93a7" font-size="11" text-anchor="end">由 GitHub Actions 每日自动更新</text>'
        + "</svg>"
    )


def main() -> int:
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")

    try:
        stars = fetch_stars()
        print(f"{REPO}: {stars} stars")
    except Exception as exc:
        print(f"!! fetch failed: {exc}", file=sys.stderr)
        return 1

    # 记录本仓库自身的 Star 历史（同一天覆盖，用于 README 增长图）
    HIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    hist = json.loads(HIST_PATH.read_text(encoding="utf-8")) if HIST_PATH.exists() else {}
    hist[now] = stars
    HIST_PATH.write_text(json.dumps(hist, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"star history: {now} -> {stars}")

    # 重新生成 SVG 图表
    svg = render_star_chart(hist)
    CHART_PATH.write_text(svg, encoding="utf-8")
    print(f"chart: {CHART_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
