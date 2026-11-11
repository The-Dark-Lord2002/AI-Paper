"""Format pandoc's body.docx as an IEEE conference paper (letter, two columns, Times New Roman)."""
import copy
import re
import sys
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

SRC = Path(sys.argv[1])          # paper folder (main.tex)
OUT = sys.argv[2]
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
FONT = "Times New Roman"
COL_TW = 5040                    # one column: 3.5 in (IEEE Word template: 2 x 3.5 in, 0.25 in apart)

doc = docx.Document("body.docx")
body = doc.element.body


# ----------------------------------------------------------------------------- helpers
def set_rfonts(rpr, name=FONT):
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.insert(0, rf)
    for a in list(rf.attrib):
        del rf.attrib[a]
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), name)


def style_font(style, size=None, bold=None, italic=None, small_caps=None, name=FONT):
    f = style.font
    set_rfonts(style.element.get_or_add_rPr(), name)
    if size: f.size = Pt(size)
    if bold is not None: f.bold = bold
    if italic is not None: f.italic = italic
    if small_caps is not None: f.small_caps = small_caps
    f.color.rgb = docx.shared.RGBColor(0, 0, 0)


def para_format(pf, align=None, before=0, after=0, first=None, keep=None):
    if align is not None: pf.alignment = align
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing = 1.0
    if first is not None: pf.first_line_indent = Pt(first)
    if keep is not None: pf.keep_with_next = keep


def run(p, text, size=None, bold=None, italic=None, small_caps=None, superscript=None):
    r = p.add_run(text)
    set_rfonts(r._r.get_or_add_rPr())
    if size: r.font.size = Pt(size)
    if bold is not None: r.font.bold = bold
    if italic is not None: r.font.italic = italic
    if small_caps is not None: r.font.small_caps = small_caps
    if superscript: r.font.superscript = True
    return r


def sect_pr(cols, continuous=True):
    s = copy.deepcopy(base_sect)
    for t in s.findall(qn("w:type")):
        s.remove(t)
    if continuous:
        t = OxmlElement("w:type"); t.set(qn("w:val"), "continuous"); s.insert(0, t)
    c = s.find(qn("w:cols"))
    if c is None:
        c = OxmlElement("w:cols"); s.append(c)
    c.set(qn("w:num"), str(cols)); c.set(qn("w:space"), "360")
    return s


def tiny_break_paragraph(sect):
    """An empty, 1-pt paragraph that ends a section."""
    p = OxmlElement("w:p"); ppr = OxmlElement("w:pPr")
    sp = OxmlElement("w:spacing"); sp.set(qn("w:before"), "0"); sp.set(qn("w:after"), "0")
    sp.set(qn("w:line"), "20"); sp.set(qn("w:lineRule"), "exact")
    ppr.append(sp); ppr.append(sect); p.append(ppr)
    return p


def borders(el_pr, tag, **sides):
    b = el_pr.find(qn(tag))
    if b is None:
        b = OxmlElement(tag); el_pr.append(b)
    for side, val in sides.items():
        e = b.find(qn("w:" + side))
        if e is None:
            e = OxmlElement("w:" + side); b.append(e)
        if val:
            e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(val)); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
        else:
            e.set(qn("w:val"), "nil")


def math_size(el, half_points):
    """Inline equations in tables and captions: same size as the text around them."""
    for mr in el.iter(M + "r"):
        rpr = mr.find(qn("w:rPr"))
        if rpr is None:
            rpr = OxmlElement("w:rPr")
            mrpr = mr.find(M + "rPr")
            (mrpr.addnext(rpr) if mrpr is not None else mr.insert(0, rpr))
        sz = rpr.find(qn("w:sz"))
        if sz is None:
            sz = OxmlElement("w:sz"); rpr.append(sz)
        sz.set(qn("w:val"), str(half_points))


# ----------------------------------------------------------------------------- page, fonts, styles
base_sect = body.find(qn("w:sectPr"))
for s in doc.sections:
    s.page_width, s.page_height = Inches(8.5), Inches(11)
    s.top_margin, s.bottom_margin = Inches(0.75), Inches(1.0)
    s.left_margin = s.right_margin = Inches(0.625)
    s.header_distance = s.footer_distance = Inches(0.5)

rpr_default = doc.styles.element.find(qn("w:docDefaults")).find(qn("w:rPrDefault")).find(qn("w:rPr"))
set_rfonts(rpr_default)
for st in doc.styles:
    try:
        if st.type == 1 or st.type == 2:      # paragraph or character styles
            set_rfonts(st.element.get_or_add_rPr(), "Courier New" if "Verbatim" in st.name else FONT)
            if st.type == 1:
                st.font.color.rgb = docx.shared.RGBColor(0, 0, 0)
    except Exception:
        pass

class _Styles:
    def __getitem__(self, name):
        for st in doc.styles:
            if st.name == name:
                return st
        raise KeyError(name)


S = _Styles()
style_font(S["Normal"], 10); para_format(S["Normal"].paragraph_format)
for name in ("Body Text", "First Paragraph"):
    style_font(S[name], 10)
    para_format(S[name].paragraph_format, WD_ALIGN_PARAGRAPH.JUSTIFY, first=10)
style_font(S["Compact"], 8); para_format(S["Compact"].paragraph_format, first=0)
style_font(S["Heading 1"], 10, bold=False, italic=False, small_caps=True)
para_format(S["Heading 1"].paragraph_format, WD_ALIGN_PARAGRAPH.CENTER, before=6, after=3, first=0, keep=True)
style_font(S["Heading 2"], 10, bold=False, italic=True)
para_format(S["Heading 2"].paragraph_format, WD_ALIGN_PARAGRAPH.LEFT, before=4, after=2, first=0, keep=True)
style_font(S["Image Caption"], 8, italic=False)
para_format(S["Image Caption"].paragraph_format, WD_ALIGN_PARAGRAPH.JUSTIFY, before=2, after=6, first=0)
style_font(S["Table Caption"], 8, italic=False, small_caps=True)
para_format(S["Table Caption"].paragraph_format, WD_ALIGN_PARAGRAPH.CENTER, before=6, after=2, first=0, keep=True)
for name in ("Captioned Figure", "Figure"):
    if name in [s.name for s in doc.styles]:
        para_format(S[name].paragraph_format, WD_ALIGN_PARAGRAPH.CENTER, before=6, after=0, first=0, keep=True)
if "Verbatim Char" in [s.name for s in doc.styles]:
    S["Verbatim Char"].font.size = Pt(9)

settings = doc.settings.element
hy = OxmlElement("w:autoHyphenation"); hy.set(qn("w:val"), "true"); settings.append(hy)

# ----------------------------------------------------------------------------- headings: I., A., ...
roman = ["I", "II", "III", "IV", "V", "VI"]
h1 = h2 = 0
for p in doc.paragraphs:
    if p.style.name == "Heading 1":
        h1 += 1; h2 = 0
        p.runs[0].text = f"{roman[h1 - 1]}. " + p.runs[0].text
    elif p.style.name == "Heading 2":
        h2 += 1
        p.runs[0].text = f"{chr(64 + h2)}. " + p.runs[0].text

# ----------------------------------------------------------------------------- figures and captions
fig = 0
for p in doc.paragraphs:
    if p.style.name == "Image Caption":
        fig += 1
        r = p.runs[0]
        r.text = f"Fig. {fig}. " + r.text
        math_size(p._p, 16)
widths = [2.9, 2.6]
for shape, w in zip(doc.inline_shapes, widths):
    ratio = shape.height / shape.width
    shape.width = Inches(w); shape.height = int(Inches(w) * ratio)

tab = 0
for p in doc.paragraphs:
    if p.style.name == "Table Caption":
        tab += 1
        first = p.runs[0]
        label = OxmlElement("w:r"); label_t = OxmlElement("w:t"); label_t.text = f"Table {roman[tab - 1]}"
        label.append(label_t); first._r.addprevious(label)
        br = OxmlElement("w:r"); br.append(OxmlElement("w:br")); first._r.addprevious(br)
        math_size(p._p, 16)


# column widths in twips: one column is 5040 (3.5 in), Table III spans both columns (10440 = 7.25 in)
WIDTHS = {0: [2100, 735, 735, 735, 735], 1: [1500, 3540], 2: [2900] + [1508] * 5, 3: [1500, 520, 520, 800, 520, 560, 600], 4: [1150, 3890]}


def set_widths(table, widths):
    tbl = table._tbl
    tpr = tbl.tblPr
    tw = tpr.find(qn("w:tblW"))
    if tw is None:
        tw = OxmlElement("w:tblW"); tpr.append(tw)
    tw.set(qn("w:type"), "dxa"); tw.set(qn("w:w"), str(sum(widths)))
    lay = tpr.find(qn("w:tblLayout"))
    if lay is None:
        lay = OxmlElement("w:tblLayout"); tpr.append(lay)
    lay.set(qn("w:type"), "fixed")
    grid = tbl.find(qn("w:tblGrid"))
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(w))
    for row in table.rows:
        for cell, w in zip(row.cells, widths):
            cell.width = docx.shared.Twips(w)

# ----------------------------------------------------------------------------- tables (pandoc's five)
row_rules = {0: lambda t: t == "Total", 2: lambda t: t in ("BERT", "Att + wce") or t.startswith("Att + wce, BiLSTM 384") and "×1" in t,
             3: lambda t: t.startswith("Macro-F1")}
tables = list(doc.tables)
for i, t in enumerate(tables):
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    tpr = t._tbl.tblPr
    for st in tpr.findall(qn("w:tblStyle")):
        tpr.remove(st)
    borders(tpr, "w:tblBorders", top=8, bottom=8, left=0, right=0, insideH=0, insideV=0)
    mar = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        e = OxmlElement("w:" + side); e.set(qn("w:w"), "40" if i == 3 else "70"); e.set(qn("w:type"), "dxa"); mar.append(e)
    tpr.append(mar)
    set_widths(t, WIDTHS[i])
    first_row_seen = set()
    for r_i, row in enumerate(t.rows):
        cells = row.cells
        text0 = "".join(n.text or "" for n in cells[0]._tc.iter(qn("w:t")))
        for c in cells:
            tcpr = c._tc.get_or_add_tcPr()
            if r_i == 0:
                borders(tcpr, "w:tcBorders", bottom=4)
            elif i in row_rules and row_rules[i](text0):
                borders(tcpr, "w:tcBorders", top=4)
            for cp in c.paragraphs:
                para_format(cp.paragraph_format, first=0)
                for r in list(cp.runs):
                    set_rfonts(r._r.get_or_add_rPr()); r.font.size = Pt(8)
                    if "±" in r.text and not r.text.startswith("±"):
                        head, tail = r.text.split("±", 1)
                        r.text = head
                        new = copy.deepcopy(r._r); r._r.addnext(new)
                        new.find(qn("w:t")).text = "±" + tail
                for r in cp.runs:
                    if r.text.startswith("±"):
                        r.font.size = Pt(6)
            math_size(c._tc, 16)

# Table III spans both columns: give it its own one-column section
t3 = tables[2]._tbl
cap3 = t3.getprevious()
cap3.addprevious(tiny_break_paragraph(sect_pr(2)))          # ends the two-column text before it
t3.addnext(tiny_break_paragraph(sect_pr(1)))                # ends the one-column section of the table

# ----------------------------------------------------------------------------- numbered equations
tex = SRC.joinpath("main.tex").read_text()
continues = [not re.match(r"\s*\n\s*\n", tex[m.end():]) for m in re.finditer(r"\\end\{equation\}", tex)]
eqs = [p for p in doc.paragraphs if p._p.findall(".//" + M + "oMathPara")]
assert len(eqs) == len(continues) == 10, (len(eqs), len(continues))
for n, (p, cont) in enumerate(zip(eqs, continues), start=1):
    nxt = p._p.getnext()
    t = doc.add_table(rows=1, cols=2)
    tbl = t._tbl
    p._p.addprevious(tbl)
    tpr = tbl.tblPr
    borders(tpr, "w:tblBorders", top=0, bottom=0, left=0, right=0, insideH=0, insideV=0)
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tpr.append(lay)
    mar = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        e = OxmlElement("w:" + side); e.set(qn("w:w"), "0"); e.set(qn("w:type"), "dxa"); mar.append(e)
    tpr.append(mar)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_widths(t, [COL_TW - 400, 400])
    for cell in t.rows[0].cells:
        cell.vertical_alignment = docx.enum.table.WD_ALIGN_VERTICAL.CENTER
    c0, c1 = t.rows[0].cells
    c0._tc.remove(c0.paragraphs[0]._p)
    c0._tc.append(p._p)
    para_format(p.paragraph_format, WD_ALIGN_PARAGRAPH.CENTER, before=1, after=1, first=0)
    np_ = c1.paragraphs[0]
    para_format(np_.paragraph_format, WD_ALIGN_PARAGRAPH.RIGHT, before=1, after=1, first=0)
    run(np_, f"({n})", 10)
    if cont and nxt is not None and nxt.tag == qn("w:p"):
        ppr = nxt.find(qn("w:pPr"))
        if ppr is None:
            ppr = OxmlElement("w:pPr"); nxt.insert(0, ppr)
        ind = OxmlElement("w:ind"); ind.set(qn("w:firstLine"), "0"); ppr.append(ind)

# ----------------------------------------------------------------------------- front matter
first_body = body[0]


def new_para_before(anchor):
    p = OxmlElement("w:p"); anchor.addprevious(p)
    return docx.text.paragraph.Paragraph(p, doc._body)


title = new_para_before(first_body)
para_format(title.paragraph_format, WD_ALIGN_PARAGRAPH.CENTER, after=8, first=0)
run(title, "Imbalanced Maritime Incident Classification Using BERT–BiLSTM with Attention and LLM-Based Data Augmentation", 24)

authors = [("Mohammad Mahdi Arab", False, "math.mohammad.arab@shahroodut.ac.ir"),
           ("Hoda Mashayekhi", True, "hmashayekhi@shahroodut.ac.ir"),
           ("Maryam Khodabakhsh", False, "m_khodabakhsh@shahroodut.ac.ir")]
at = doc.add_table(rows=1, cols=3)
first_body.addprevious(at._tbl)
at.alignment = WD_TABLE_ALIGNMENT.CENTER
set_widths(at, [3480, 3480, 3480])
borders(at._tbl.tblPr, "w:tblBorders", top=0, bottom=0, left=0, right=0, insideH=0, insideV=0)
for cell, (name, star, mail) in zip(at.rows[0].cells, authors):
    lines = [(name, 11, False), ("Faculty of Computer Engineering", 10, True), ("Shahrood University of Technology", 10, True),
             ("Shahrood, Iran", 10, False), (mail, 10, False)]
    for k, (text, size, it) in enumerate(lines):
        cp = cell.paragraphs[0] if k == 0 else cell.add_paragraph()
        para_format(cp.paragraph_format, WD_ALIGN_PARAGRAPH.CENTER, first=0)
        run(cp, text, size, italic=it)
        if k == 0 and star:
            run(cp, "*", size, superscript=True)

note = new_para_before(first_body)
para_format(note.paragraph_format, WD_ALIGN_PARAGRAPH.CENTER, before=4, after=8, first=0)
run(note, "*Corresponding author: Hoda Mashayekhi (e-mail: hmashayekhi@shahroodut.ac.ir).", 8, italic=True)
note._p.get_or_add_pPr().append(sect_pr(1, continuous=False))   # end of the one-column title section

abstract = tex[tex.index(r"\begin{abstract}") + len(r"\begin{abstract}"):tex.index(r"\end{abstract}")].strip()
keywords = tex[tex.index(r"\begin{IEEEkeywords}") + len(r"\begin{IEEEkeywords}"):tex.index(r"\end{IEEEkeywords}")].strip()
clean = lambda s: s.replace("--", "–").replace("\\%", "%").replace("~", " ")
ab = new_para_before(first_body)
para_format(ab.paragraph_format, WD_ALIGN_PARAGRAPH.JUSTIFY, after=4, first=10)
run(ab, "Abstract", 9, bold=True, italic=True); run(ab, "—", 9, bold=True, italic=True)
run(ab, clean(abstract), 9, bold=True)
kw = new_para_before(first_body)
para_format(kw.paragraph_format, WD_ALIGN_PARAGRAPH.JUSTIFY, after=6, first=10)
run(kw, "Keywords—", 9, bold=True, italic=True); run(kw, clean(keywords), 9, bold=True)

# the rest of the paper: two columns, continuing on the same page
final = body.find(qn("w:sectPr"))
body.replace(final, sect_pr(2))

# ----------------------------------------------------------------------------- references
REFS = [
    ("Z. Zhao, X. Liu, L. Feng, M. Grifoll, and H. Feng, “Causation analysis of marine traffic accidents using deep learning approaches: A case study from China’s coasts,” ", "Systems", ", vol. 13, no. 4, p. 284, 2025."),
    ("H. Black, R. Kurt, and O. Turan, “A domain-adapted NLP pipeline for human factors classification of maritime accidents,” ", "Ocean Engineering", ", vol. 357, p. 125625, 2026."),
    ("I. Munaev, “MAIB incident type dataset: Labeled marine accident reports for text classification,” Hugging Face dataset https://huggingface.co/datasets/baker-street/maib-incident-reports-5K, 2025.", "", ""),
    ("Y. Zhang, B. Kang, B. Hooi, S. Yan, and J. Feng, “Deep long-tailed learning: A survey,” ", "IEEE Transactions on Pattern Analysis and Machine Intelligence", ", vol. 45, no. 9, pp. 10795–10816, 2023."),
    ("X. Qiu, T. Sun, Y. Xu, Y. Shao, N. Dai, and X. Huang, “Pre-trained models for natural language processing: A survey,” ", "Science China Technological Sciences", ", vol. 63, no. 10, pp. 1872–1897, 2020."),
    ("H. Yu, X. Xu, Z. Guo, T. Wei, and L. Xu, “Semantic modeling of ship collision reports: Ontology design, knowledge extraction, and severity classification,” ", "Journal of Marine Science and Engineering", ", vol. 14, p. 448, 2026."),
    ("S. Wang, Y. Hou, P. Qiu, K. Wang, and B. Wang, “Ship fire and explosion accident evolution modeling based on ontology-enhanced text mining and dynamic Bayesian network,” ", "Applied Sciences", ", vol. 16, p. 4984, 2026."),
    ("W. Nguyen et al., “Llamarine: Open-source maritime industry-specific large language model,” arXiv:2503.00203, 2025.", "", ""),
    ("H. Park, J. Jung, M. Seo, H. Choi, D. Cho, S. Park, and D.-G. Choi, “AIS-LLM: A unified framework for maritime trajectory prediction, anomaly detection, and collision risk assessment with explainable forecasting,” arXiv:2508.07668, 2025.", "", ""),
    ("T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal loss for dense object detection,” ", "IEEE Transactions on Pattern Analysis and Machine Intelligence", ", vol. 42, no. 2, pp. 318–327, 2020."),
    ("N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: Synthetic minority over-sampling technique,” ", "Journal of Artificial Intelligence Research", ", vol. 16, pp. 321–357, 2002."),
    ("M. Bayer, M.-A. Kaufhold, and C. Reuter, “A survey on data augmentation for text classification,” ", "ACM Computing Surveys", ", vol. 55, no. 7, p. 146, 2022."),
    ("S. Hochreiter and J. Schmidhuber, “Long short-term memory,” ", "Neural Computation", ", vol. 9, no. 8, pp. 1735–1780, 1997."),
    ("M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” ", "IEEE Transactions on Signal Processing", ", vol. 45, no. 11, pp. 2673–2681, 1997."),
    ("Z. Niu, G. Zhong, and H. Yu, “A review on the attention mechanism of deep learning,” ", "Neurocomputing", ", vol. 452, pp. 48–62, 2021."),
    ("Qwen Team, “Qwen2.5 technical report,” arXiv:2412.15115, 2024.", "", ""),
    ("C. Cortes and V. Vapnik, “Support-vector networks,” ", "Machine Learning", ", vol. 20, no. 3, pp. 273–297, 1995."),
    ("G. Salton and C. Buckley, “Term-weighting approaches in automatic text retrieval,” ", "Information Processing & Management", ", vol. 24, no. 5, pp. 513–523, 1988."),
    ("T. G. Dietterich, “Approximate statistical tests for comparing supervised classification learning algorithms,” ", "Neural Computation", ", vol. 10, no. 7, pp. 1895–1923, 1998."),
    ("J. El Zini and M. Awad, “On the explainability of natural language processing deep models,” ", "ACM Computing Surveys", ", vol. 55, no. 5, p. 103, 2022."),
]
h = doc.add_paragraph(style=S["Heading 1"]); h.add_run("References")
for n, (a, journal, b) in enumerate(REFS, start=1):
    p = doc.add_paragraph(style=S["Normal"])
    pf = p.paragraph_format
    para_format(pf, WD_ALIGN_PARAGRAPH.JUSTIFY, after=0)
    pf.left_indent, pf.first_line_indent = Inches(0.3), Inches(-0.3)
    pf.tab_stops.add_tab_stop(Inches(0.3))
    run(p, f"[{n}]\t" + a, 8)
    if journal:
        run(p, journal, 8, italic=True); run(p, b, 8)

cp = doc.core_properties
cp.title = "Imbalanced Maritime Incident Classification Using BERT–BiLSTM with Attention and LLM-Based Data Augmentation"
cp.author = "Mohammad Mahdi Arab, Hoda Mashayekhi, Maryam Khodabakhsh"
doc.save(OUT)
print("saved", OUT, "| equations", len(eqs), "| tables", len(tables), "| figures", fig)
