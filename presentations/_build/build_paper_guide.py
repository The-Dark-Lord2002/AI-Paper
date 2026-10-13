"""
Builds docs/paper_guide_fa.pdf from docs/paper_guide_fa.md (Persian, right-to-left).

  pip install markdown && npm install @fontsource/vazirmatn
  python3 build_paper_guide.py ../../docs/paper_guide_fa.md paper_guide.html
  chromium --headless --no-pdf-header-footer --print-to-pdf=../../docs/paper_guide_fa.pdf paper_guide.html
"""
import html
import re
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).parent
FONTS = HERE / "node_modules/@fontsource/vazirmatn/files"
SOURCE, OUT = Path(sys.argv[1]), Path(sys.argv[2])

LATIN_RUN = re.compile(r"[A-Za-z0-9\[(+−\-±&γβα√][A-Za-z0-9 _+−\-±.,:;/=×÷%&\[\]()'^γβαΣ·|<>√≈→←↔∝²…*]*[A-Za-z0-9\])%γβα²*]"
                       r"|[A-Za-z0-9]")


def isolate(text):
    """Wrap every Latin/number run in an LTR isolate so the right-to-left paragraph never reorders it."""
    out, pos = [], 0
    for m in LATIN_RUN.finditer(text):
        a, b = m.start(), m.end()
        while a < b and text[a] in "([" and text[a:b].count(text[a]) > text[a:b].count(")]"["([".index(text[a])]):
            a += 1
        while b > a and text[b - 1] in ")]" and text[a:b].count(text[b - 1]) > text[a:b].count("(["[")]".index(text[b - 1])]):
            b -= 1
        if a == b:
            continue
        out.append(text[pos:a])
        nowrap = " nw" if " " not in text[a:b] else ""
        out.append(f'<span class="ltr{nowrap}" dir="ltr">{text[a:b]}</span>')
        pos = b
    out.append(text[pos:])
    return "".join(out)


def process(body):
    """Isolate Latin runs in text nodes (not inside code/pre); make mostly-English paragraphs left-to-right."""
    parts = re.split(r"(<[^>]+>)", body)
    skip = 0
    for i, part in enumerate(parts):
        if part.startswith("<"):
            tag = re.match(r"</?(\w+)", part)
            if tag and tag.group(1) in ("code", "pre"):
                skip += -1 if part.startswith("</") else 1
            continue
        if skip == 0 and part.strip():
            parts[i] = isolate(part)
    body = "".join(parts)

    def direction(m):
        text = re.sub(r"<[^>]+>", "", m.group(2))
        latin = len(re.findall(r"[A-Za-z]", text))
        persian = len(re.findall(r"[؀-ۿ]", text))
        if latin > 3 * max(persian, 1) and len(text) > 70:     # a real English sentence, not a name
            return f'<{m.group(1)} dir="ltr" class="en">{m.group(2)}</{m.group(1)}>'
        return m.group(0)
    return re.sub(r"<(p|li|td)>(.*?)</\1>", direction, body, flags=re.S)


md = SOURCE.read_text(encoding="utf-8")
body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
body = body.replace("[ ]", "☐")
body = process(body)

CSS = f"""
@font-face {{ font-family: Vazirmatn; font-weight: 400; src: url({(FONTS / 'vazirmatn-arabic-400-normal.woff2').as_uri()}) format('woff2'); unicode-range: U+0600-06FF, U+0750-077F, U+200C-200E, U+FB50-FDFF, U+FE70-FEFF; }}
@font-face {{ font-family: Vazirmatn; font-weight: 700; src: url({(FONTS / 'vazirmatn-arabic-700-normal.woff2').as_uri()}) format('woff2'); unicode-range: U+0600-06FF, U+0750-077F, U+200C-200E, U+FB50-FDFF, U+FE70-FEFF; }}
@font-face {{ font-family: Vazirmatn; font-weight: 400; src: url({(FONTS / 'vazirmatn-latin-400-normal.woff2').as_uri()}) format('woff2'); }}
@font-face {{ font-family: Vazirmatn; font-weight: 700; src: url({(FONTS / 'vazirmatn-latin-700-normal.woff2').as_uri()}) format('woff2'); }}
@page {{ size: A4; margin: 17mm 16mm 18mm 16mm;
         @bottom-center {{ content: counter(page, persian); font-family: Vazirmatn; font-size: 9pt; color: #8a92a3; }} }}
html {{ direction: rtl; }}
body {{ font-family: Vazirmatn, 'DejaVu Sans', sans-serif; font-size: 10.6pt; line-height: 1.85; color: #1d2433; background: #fff; margin: 0; }}
.ltr {{ direction: ltr; unicode-bidi: isolate; }} .nw {{ white-space: nowrap; }}
.en {{ text-align: left; font-style: italic; color: #24324d; }}
code {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 8.8pt; background: #f1f3f7; padding: 0 4px; border-radius: 4px;
        direction: ltr; unicode-bidi: isolate; white-space: nowrap; }}
pre {{ direction: ltr; text-align: left; background: #f4f6fa; border: 1px solid #d9dee7; border-radius: 8px; padding: 9px 12px;
       font-size: 8.4pt; line-height: 1.5; white-space: pre-wrap; break-inside: avoid; }}
pre code {{ background: none; padding: 0; white-space: pre-wrap; }}
h1 {{ font-size: 20pt; color: #1f3864; margin: 40mm 0 4px; }}
h2 {{ font-size: 15pt; color: #1f3864; margin: 22px 0 8px; padding-bottom: 5px; border-bottom: 2px solid #1f3864; break-after: avoid; }}
h3 {{ font-size: 12pt; color: #0f766e; margin: 16px 0 5px; break-after: avoid; }}
h4 {{ font-size: 11pt; color: #1e4fa3; margin: 12px 0 4px; break-after: avoid; }}
h1 + h2 {{ border: none; font-size: 13pt; color: #5b6475; margin-top: 4px; }}
p {{ margin: 5px 0 7px; text-align: justify; }}
ul, ol {{ margin: 4px 0 8px; padding-right: 22px; padding-left: 0; }}
ol {{ list-style-type: persian; }}
li {{ margin: 2px 0; }}
blockquote {{ margin: 8px 0; padding: 6px 14px; background: #ecf8f6; border-right: 4px solid #0f766e; border-radius: 6px; direction: ltr; text-align: left; }}
blockquote p {{ text-align: left; font-style: italic; }}
table {{ width: 100%; border-collapse: collapse; margin: 8px 0 12px; font-size: 9.4pt; line-height: 1.55; }}
thead {{ display: table-header-group; }} tr {{ break-inside: avoid; }}
th {{ background: #1f3864; color: #fff; font-weight: 700; padding: 5px 7px; text-align: right; }}
td {{ padding: 4px 7px; border-bottom: 1px solid #d9dee7; vertical-align: top; text-align: right; }}
tbody tr:nth-child(even) td {{ background: #f6f8fb; }}
hr {{ border: none; border-top: 1px solid #d9dee7; margin: 14px 0; }}
h2 {{ break-before: page; }}
h1 + h2, hr + h2:first-of-type {{ break-before: auto; }}
"""
OUT.write_text(f"""<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8">
<title>راهنمای نوشتن مقاله</title><style>{CSS}</style></head><body>{body}</body></html>""", encoding="utf-8")
print("wrote", OUT)
