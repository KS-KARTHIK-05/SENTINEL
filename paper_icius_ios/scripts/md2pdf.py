#!/usr/bin/env python3
"""Export response_to_reviewers.md as an A4 PDF (pdfLaTeX, Times, Type 1 fonts).

Handles the Markdown used in that file: #/##/### headings, **bold**, *italic*, `code`, > quotes,
nested '-' lists, pipe tables, '---' rules and bare URLs. The first block (title + bold header lines)
becomes the page header with submission ID, title and authors.

    python paper_icius_ios/scripts/md2pdf.py paper_icius_ios/response_to_reviewers.md
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

UNICODE = r"""
\DeclareUnicodeCharacter{2192}{\ensuremath{\rightarrow}}
\DeclareUnicodeCharacter{2082}{\textsubscript{2}}
\DeclareUnicodeCharacter{2080}{\textsubscript{0}}
\DeclareUnicodeCharacter{03BB}{\ensuremath{\lambda}}
\DeclareUnicodeCharacter{03B5}{\ensuremath{\varepsilon}}
\DeclareUnicodeCharacter{03B3}{\ensuremath{\gamma}}
\DeclareUnicodeCharacter{03B2}{\ensuremath{\beta}}
\DeclareUnicodeCharacter{2212}{\ensuremath{-}}
\DeclareUnicodeCharacter{00B1}{\ensuremath{\pm}}
\DeclareUnicodeCharacter{2265}{\ensuremath{\geq}}
\DeclareUnicodeCharacter{00D7}{\ensuremath{\times}}
\DeclareUnicodeCharacter{00B0}{\ensuremath{^\circ}}
"""


def esc(t: str) -> str:
    out, i = [], 0
    for m in re.finditer(r'https?://\S+?(?=[\s)]|$)', t):
        out.append(_esc(t[i:m.start()])); out.append(r'\url{' + m.group(0).rstrip('.,;') + '}')
        if m.group(0)[-1] in '.,;':
            out.append(m.group(0)[-1])
        i = m.end()
    out.append(_esc(t[i:]))
    return ''.join(out)


def _esc(t: str) -> str:
    t = t.replace('\\', r'\textbackslash{}')
    for a, b in [('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('$', r'\$'), ('_', r'\_'), ('{', r'\{'), ('}', r'\}'),
                 ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}')]:
        t = t.replace(a, b)
    t = t.replace(r'\textbackslash\{\}', r'\textbackslash{}')
    t = re.sub(r'`([^`]+)`', r'\\texttt{\1}', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', t)
    t = re.sub(r'(?<![\w*])\*([^*]+)\*(?![\w*])', r'\\emph{\1}', t)
    t = re.sub(r'"([^"]*)"', r"``\1''", t)
    return t


def convert(md: str) -> str:
    lines = md.split('\n')
    # header block: '# title' then bold metadata lines until first blank line after them
    head, i = [], 0
    while i < len(lines) and (lines[i].startswith('#') or lines[i].startswith('**') or not lines[i].strip()):
        if lines[i].strip():
            head.append(lines[i])
        i += 1
        if head and not lines[i - 1].strip() and len(head) > 1:
            break
    body, list_depth, in_table = [], 0, False

    def close_lists(to=0):
        nonlocal list_depth
        while list_depth > to:
            body.append(r'\end{itemize}'); list_depth -= 1

    for ln in lines[i:]:
        s = ln.rstrip()
        if s.lstrip().startswith('|'):
            cells = [c.strip() for c in s.strip().strip('|').split('|')]
            if all(re.fullmatch(r':?-+:?', c) for c in cells):
                continue
            if not in_table:
                if not ln.startswith(' '):
                    close_lists()
                body.append(r'\begin{center}\small\begin{tabular}{' + 'l' * len(cells) + r'}\toprule')
                body.append(' & '.join(r'\textbf{' + esc(c) + '}' for c in cells) + r' \\\midrule')
                in_table = True
            else:
                body.append(' & '.join(esc(c) for c in cells) + r' \\')
            continue
        if in_table:
            body.append(r'\bottomrule\end{tabular}\end{center}'); in_table = False
        m = re.match(r'^( *)- (.*)', s)
        if m:
            depth = len(m.group(1)) // 2 + 1
            while list_depth < depth:
                body.append(r'\begin{itemize}\itemsep1pt'); list_depth += 1
            close_lists(depth)
            body.append(r'\item ' + esc(m.group(2)))
            continue
        if s.startswith('  ') and list_depth and s.strip():
            body.append(esc(s.strip()))           # continuation of a list item
            continue
        if not s.strip():
            if list_depth:
                body.append('')                   # keep list open across a blank line
                continue
            body.append(''); continue
        close_lists()
        if s.startswith('## '):
            body.append(r'\section*{' + esc(s[3:]) + '}')
        elif s.startswith('### '):
            body.append(r'\subsection*{' + esc(s[4:]) + '}')
        elif s.startswith('# '):
            body.append(r'\section*{' + esc(s[2:]) + '}')
        elif s.strip() == '---':
            body.append(r'\medskip\hrule\medskip')
        elif s.startswith('> '):
            body.append(r'\begin{quote}\small ' + esc(s[2:]) + r'\end{quote}')
        else:
            body.append(esc(s))
    close_lists()
    if in_table:
        body.append(r'\bottomrule\end{tabular}\end{center}')
    title = esc(head[0].lstrip('# '))
    meta = r'\\'.join(esc(h) for h in head[1:])
    return (r"""\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{mathptmx}\renewcommand{\ttdefault}{pcr}
\usepackage[a4paper,margin=2.2cm]{geometry}\usepackage{booktabs}\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}\usepackage{parskip}
""" + UNICODE + r"""
\pagestyle{fancy}\fancyhf{}\renewcommand{\headrulewidth}{0.4pt}
\fancyhead[L]{\small ICIUS 2026 -- Submission UN006 -- Response to the reviewers}
\fancyhead[R]{\small\thepage}
\begin{document}
{\Large\bfseries """ + title + r"""\par}\medskip
{\small """ + meta + r"""\par}\medskip
""" + '\n'.join(body) + '\n\\end{document}\n')


if __name__ == '__main__':
    src = sys.argv[1]
    tex = os.path.splitext(src)[0] + '.tex'
    with open(src) as f:
        out = convert(f.read())
    with open(tex, 'w') as f:
        f.write(out)
    d = os.path.dirname(os.path.abspath(tex))
    for _ in range(2):
        subprocess.run(['pdflatex', '-interaction=nonstopmode', os.path.basename(tex)], cwd=d,
                       stdout=subprocess.DEVNULL, check=False)
    print('wrote', os.path.splitext(tex)[0] + '.pdf')
