import re

with open(r'C:\Users\user\.gemini\antigravity\scratch\ml_assignment_2\paper\cvpr_paper.typ', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace citations with bracketed numbers
citations = {
    '#cite("turk1991face", "hastie2009elements")': "[9, 6]",
    '#cite("schroff2015facenet")': "[8]",
    '#cite("breiman1984classification")': "[4]",
    '#cite("breiman1996bagging")': "[1]",
    '#cite("freund1997decision", "zhu2009multi")': "[5, 10]",
    '#cite("freund1997decision")': "[5]",
    '#cite("zhu2009multi")': "[10]",
    '#cite("breiman2001random")': "[3]",
    '#cite("kohavi1996bias", "breiman1996bias")': "[7, 2]"
}
for k, v in citations.items():
    text = text.replace(k, v)

# Clean up math LaTeX relics
text = text.replace(r'cal(D)^(*b)', r'cal(D)^(b)')
text = text.replace(r'h^{*b}(x)', r'h^(b)(x)')
text = text.replace(r'\{', '{').replace(r'\}', '}')
text = text.replace(r'\cal(D)', 'cal(D)')
text = text.replace(r'\hat{y}_i^{(t)}', 'hat(y)_i^((t))')
text = text.replace(r'\hat{y}', 'hat(y)')
text = text.replace(r'\cdot', 'dot')
text = text.replace(r'\bb(I)', 'bb(I)')
text = text.replace(r'\in', 'in')
text = text.replace(r'\times', 'times')
text = text.replace(r'\subset.eq', 'subset.eq')
text = text.replace(r'\ln', 'ln')
text = text.replace(r'\exp', 'exp')
text = text.replace(r'X_{\text{test}}^{\text{masked}}', 'X_"test"^"masked"')
text = text.replace(r'X_{\text{train}}', 'X_"train"')
text = text.replace(r'X_{\text{val}}', 'X_"val"')
text = text.replace(r'X_{\text{test}}', 'X_"test"')
text = text.replace(r'Y_{\text{test}}^{\text{onehot}}', 'Y_"test"^"onehot"')

with open(r'C:\Users\user\.gemini\antigravity\scratch\ml_assignment_2\paper\cvpr_paper.typ', 'w', encoding='utf-8') as f:
    f.write(text)

print("Typst text cleaned of LaTeX syntax.")
