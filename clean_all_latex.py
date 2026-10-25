import re

with open(r'C:\Users\user\.gemini\antigravity\scratch\ml_assignment_2\paper\cvpr_paper.typ', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace LaTeX commands with Typst equivalents
text = re.sub(r'\\label\{([^}]+)\}', r'<\1>', text)
text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', text)
text = re.sub(r'\\bar\{([^}]+)\}', r'macron(\1)', text)

replacements = {
    '\\alpha': 'alpha',
    '\\approx': 'approx',
    'pprox': 'approx',
    '\\max': 'max',
    '\\ge': '>=',
    '\\bb': 'bb',
    '\\leftarrow': '<-',
    '\\sigma': 'sigma',
    '\\rho': 'rho'
}

for k, v in replacements.items():
    text = text.replace(k, v)

with open(r'C:\Users\user\.gemini\antigravity\scratch\ml_assignment_2\paper\cvpr_paper.typ', 'w', encoding='utf-8') as f:
    f.write(text)

print("Replaced all remaining LaTeX commands.")
