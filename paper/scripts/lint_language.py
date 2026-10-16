#!/usr/bin/env python3
"""Flag informal or overclaiming language in the manuscript (reviewer comment R2.10).

    python scripts/lint_language.py sections/*.tex response_to_reviewers.tex
Exit code 1 if any banned phrase is found; 'guarantee' is allowed only within
three lines of a \\ref{prop:...}/\\ref{cor:...}/\\ref{lem:...} reference.
"""
import re
import sys

BANNED = [
    'bulletproof', 'brutal', 'ironclad', 'flawless', 'perfect', 'perfectly', 'blistering',
    'violently', 'hardcore', 'massive', 'massively', 'beautifully', 'pathetic', 'cratering',
    'zero data loss', 'lost zero data', 'total network survival', 'effortlessly',
    'incredibly', 'insanely', 'crushes', 'slams', 'rip out', 'ripping', 'bolt',
    'data-loss paradox', 'literal spring', 'thermal ghost', 'smartest', 'exactly 1.2 ms',
]
GUARANTEE = re.compile(r'\bguarantee[sd]?\b', re.I)
FORMAL = re.compile(r'\\ref\{(prop|cor|lem):')


def main(paths):
    bad = 0
    for path in paths:
        lines = open(path, encoding='utf-8').read().splitlines()
        for n, line in enumerate(lines, 1):
            text = line.split('%')[0].lower()
            for w in BANNED:
                if re.search(r'\b' + re.escape(w) + r'\b', text):
                    print(f'{path}:{n}: banned phrase "{w}": {line.strip()[:100]}')
                    bad += 1
            if GUARANTEE.search(text):
                ctx = '\n'.join(lines[max(0, n - 4):n + 3])
                if not FORMAL.search(ctx):
                    print(f'{path}:{n}: "guarantee" without a nearby formal result: '
                          f'{line.strip()[:100]}')
                    bad += 1
    print(f'{bad} issue(s)')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
