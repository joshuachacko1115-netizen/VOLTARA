"""Look and feel of the dashboard: fonts, cards, header and small HTML components.

Styles Streamlit's own widgets with CSS (by their data-testid) instead of replacing
them, so behaviour, accessibility and the AppTest tests are unchanged.
Colours come from src/visualization/style.py, the same palette as the report figures.
"""
from __future__ import annotations

import html

from src.config import EOL_SOH
from src.visualization.style import CATEGORICAL, INK, INK_2, MUTED, STATUS

PRIMARY = CATEGORICAL[0]
PRIMARY_DARK = "#184f95"
BORDER = "#e6e5df"
CARD = "#ffffff"
PAGE = "#f6f6f3"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* Text font only: Streamlit's icons are a ligature font set on their own element, so never use a
   broad selector such as [class*="st-"] here (it turns icons into words like "arrow_right"). */
html, body, .stApp, .stMarkdown p, .stMarkdown li, label, input, textarea, [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"] p, .stButton button p, [data-testid="stTabs"] button p {{
  font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}}
.stApp {{ background: {PAGE}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1480px; }}
h1, h2, h3, h4, h5 {{ font-family: 'Space Grotesk', 'Inter', sans-serif !important; letter-spacing: -0.01em; color: {INK}; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* Header */
.v-head {{ display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; margin: 0 0 18px 2px; }}
.v-head .logo {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.9rem; color: {INK};
  letter-spacing: -0.02em; }}
.v-head .logo span {{ color: {PRIMARY}; }}
.v-head .tag {{ color: {INK_2}; font-size: 0.95rem; }}

/* Section titles with step numbers */
.v-step {{ display: flex; align-items: center; gap: 10px; margin: 2px 0 12px 0; }}
.v-step .n {{
  width: 28px; height: 28px; border-radius: 50%; display: grid; place-items: center; font-weight: 700;
  font-family: 'Space Grotesk', sans-serif; color: #fff; background: {PRIMARY}; font-size: 0.9rem;
}}
.v-step .t {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.32rem; font-weight: 600; color: {INK}; }}
.v-sub {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.02rem; font-weight: 600; color: {INK};
          margin: 18px 0 6px 0; display: flex; align-items: center; gap: 8px; }}
.v-sub:first-child {{ margin-top: 4px; }}
.v-sub .dot {{ width: 8px; height: 8px; border-radius: 50%; background: {PRIMARY}; }}

/* Bordered containers become cards */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
  background: {CARD}; border: 1px solid {BORDER} !important; border-radius: 16px !important;
  box-shadow: 0 1px 2px rgba(16,24,40,0.04), 0 6px 18px rgba(16,24,40,0.05);
}}

/* Metrics as tiles */
[data-testid="stMetric"] {{
  background: #fafaf8; border: 1px solid {BORDER}; border-radius: 12px; padding: 12px 14px 10px;
}}
[data-testid="stMetricLabel"] p {{ font-size: 0.8rem !important; color: {INK_2} !important; font-weight: 500;
  text-transform: uppercase; letter-spacing: 0.04em; }}
[data-testid="stMetricValue"] {{ font-family: 'Space Grotesk', sans-serif !important; font-weight: 600;
  color: {INK}; font-size: 2.0rem !important; }}

/* Predict button */
[data-testid="stBaseButton-primary"], .stButton button[kind="primary"] {{
  background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%) !important; border: 0 !important;
  border-radius: 12px !important; min-height: 3.1rem; font-size: 1.05rem !important; font-weight: 600 !important;
  box-shadow: 0 8px 20px rgba(42,120,214,0.30); transition: transform .08s ease, box-shadow .15s ease;
}}
[data-testid="stBaseButton-primary"]:hover, .stButton button[kind="primary"]:hover {{
  transform: translateY(-1px); box-shadow: 0 10px 24px rgba(42,120,214,0.38);
}}
[data-testid="stBaseButton-primary"]:disabled {{ background: #c3c2b7 !important; box-shadow: none; }}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {{ gap: 6px; border-bottom: 1px solid {BORDER}; }}
[data-testid="stTabs"] button[role="tab"] {{ padding: 8px 14px; border-radius: 10px 10px 0 0; font-weight: 500; }}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{ background: {CARD}; color: {PRIMARY}; }}

/* Alerts a bit softer */
[data-testid="stAlert"] {{ border-radius: 12px; }}
[data-testid="stExpander"] details {{ border-radius: 12px; border-color: {BORDER}; background: #fafaf8; }}

/* SoH health bar */
.v-bar {{ margin: 8px 0 14px 0; }}
.v-bar .track {{ position: relative; height: 12px; border-radius: 999px; background: #ecebe6; overflow: visible; }}
.v-bar .fill {{ height: 100%; border-radius: 999px; }}
.v-bar .eol {{ position: absolute; top: -4px; width: 2px; height: 20px; background: {INK_2}; }}
.v-bar .scale {{ position: relative; height: 16px; font-size: 0.72rem; color: {MUTED}; margin-top: 4px; }}
.v-bar .scale span {{ position: absolute; top: 0; white-space: nowrap; }}
.v-bar .lbl {{ font-size: 0.82rem; color: {INK_2}; margin-top: 2px; }}

/* EV page: insight list */
.v-insights {{ margin: 0; padding-left: 1.1rem; }}
.v-insights li {{ margin: 0 0 8px 0; line-height: 1.45; color: {INK}; }}

/* Risk pill */
.v-pill {{ display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: 999px;
  font-weight: 600; font-size: 1.0rem; background: #fff; }}
.v-muted {{ color: {INK_2}; font-size: 0.86rem; }}
.v-empty {{ text-align: center; padding: 38px 10px; color: {INK_2}; }}
.v-empty .big {{ font-size: 2.4rem; }}
</style>
"""


def header(tag: str = "Battery health, remaining life and risk · NASA laboratory cells") -> str:
    return ('<div class="v-head"><div class="logo">⚡ VOLT<span>ARA</span></div>'
            f'<div class="tag">{html.escape(tag)}</div></div>')


def step(n: int, title: str) -> str:
    return f'<div class="v-step"><div class="n">{n}</div><div class="t">{html.escape(title)}</div></div>'


def sub(title: str) -> str:
    return f'<div class="v-sub"><span class="dot"></span>{html.escape(title)}</div>'


def health_color(soh: float) -> tuple[str, str]:
    if soh <= EOL_SOH:
        return STATUS["critical"], "at or past end of life"
    if soh <= EOL_SOH + 0.10:
        return STATUS["warning"], "within 10 points of end of life"
    return STATUS["good"], "healthy"


def soh_bar(soh: float, label: str) -> str:
    """Horizontal SoH gauge 50-100 % with the 80 % end-of-life mark; colour + words, not colour alone."""
    lo, hi = 0.5, 1.05
    pos = lambda v: max(0.0, min(1.0, (v - lo) / (hi - lo))) * 100
    color, words = health_color(soh)
    return (
        '<div class="v-bar"><div class="track">'
        f'<div class="fill" style="width:{pos(soh):.1f}%;background:{color}"></div>'
        f'<div class="eol" style="left:{pos(EOL_SOH):.1f}%" title="End of life (80 %)"></div></div>'
        '<div class="scale"><span style="left:0">50 %</span>'
        f'<span style="left:{pos(EOL_SOH):.1f}%;transform:translateX(-50%)">EOL 80 %</span>'
        f'<span style="left:{pos(1.0):.1f}%;transform:translateX(-50%)">100 %</span></div>'
        f'<div class="lbl">{html.escape(label)}: <b>{soh:.1%}</b> · {words}</div></div>')


def empty_state() -> str:
    return ('<div class="v-empty"><div class="big">🔋</div>'
            "<div>Pick a battery, then press <b>Predict</b>.</div></div>")

