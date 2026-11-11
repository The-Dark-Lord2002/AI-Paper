"""Turn main.tex into a plain LaTeX body that pandoc converts well: citations and references resolved to the
numbers of the PDF, table rows inlined, TikZ figure replaced by its rendered image."""
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])
tex = src.joinpath("main.tex").read_text()

body = tex[tex.index(r"\section{Introduction}"):tex.index(r"\bibliographystyle")]
body = re.sub(r"(?m)^\s*%.*\n", "", body)                     # comment lines

# citations: IEEEtran numbers them in order of first citation; the cite package sorts and compresses
order = []
for keys in re.findall(r"\\cite\{([^}]*)\}", tex[tex.index(r"\begin{document}"):]):
    for k in keys.split(","):
        if k.strip() not in order:
            order.append(k.strip())
number = {k: i + 1 for i, k in enumerate(order)}


def cite(m):
    ns = sorted(number[k.strip()] for k in m.group(1).split(","))
    groups, run = [], [ns[0]]
    for n in ns[1:]:
        if n == run[-1] + 1:
            run.append(n)
        else:
            groups.append(run); run = [n]
    groups.append(run)
    parts = [f"[{g[0]}]--[{g[-1]}]" if len(g) >= 3 else ", ".join(f"[{n}]" for n in g) for g in groups]
    return ", ".join(parts)


body = re.sub(r"\\cite\{([^}]*)\}", cite, body)

refs = {"sec:lit": "II", "sec:method": "III", "sec:eval": "IV", "sec:conc": "V", "sec:loss": "III-E",
        "sec:data-level": "III-F", "sec:discussion": "IV-F", "fig:framework": "1", "fig:cm": "2",
        "tab:data": "I", "tab:settings": "II", "tab:results": "III", "tab:perclass": "IV", "tab:attention": "V"}
body = re.sub(r"\\ref\{([^}]*)\}", lambda m: refs[m.group(1)], body)

body = re.sub(r"\\inputtable\{([^}]*)\}", lambda m: src.joinpath(m.group(1)).read_text(), body)
body = re.sub(r"\{\\tiny\$\\pm\$([^}]*)\}", r"±\1", body)       # small "±sd" in Table III, shrunk later
body = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", r"\\includegraphics{fig1.png}", body, flags=re.S)
body = body.replace(r"\includegraphics{figures/confusion_matrix.pdf}", r"\includegraphics{fig2.png}")
body = body.replace(r"\begin{table*}", r"\begin{table}").replace(r"\end{table*}", r"\end{table}")
body = re.sub(r"\\(footnotesize|centered)\b", "", body)
body = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", body)
body = re.sub(r"\\label\{[^}]*\}", "", body)

out.write_text("\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n" + body + "\n\\end{document}\n")
print("citation order:", order)
