import re

with open(r'C:\Users\user\.gemini\antigravity\scratch\ml_assignment_2\paper\cvpr_paper.typ', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace \text{...} with "..."
text = re.sub(r'\\text\{([^}]+)\}', r'"\1"', text)

# Replace \arg\max with arg max
text = text.replace(r'\arg\max', 'arg max')
text = text.replace(r'\ll', '<<')
text = text.replace(r'\to', '->')
text = text.replace(r'\infty', 'infinity')
text = text.replace(r'\forall', 'forall')
text = text.replace(r'\neq', '!=')
text = text.replace(r'\quad', ' ')
text = text.replace(r'\dots', '...')
text = text.replace(r'\sum', 'sum')
text = text.replace(r'\left(', '(').replace(r'\right)', ')')

# Clean up any leftover Bias^2 or B^2
text = text.replace(r'Bias$^2$', r'Bias#super[2]')
text = text.replace(r'Bias^2', r'Bias#super[2]')

with open(r'C:\Users\user\.gemini\antigravity\scratch\ml_assignment_2\paper\cvpr_paper.typ', 'w', encoding='utf-8') as f:
    f.write(text)

print("Typst math syntax cleaned.")
