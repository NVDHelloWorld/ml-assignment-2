import json

with open(r'c:\Users\user\Downloads\ML_assignment_2_.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(r'C:\Users\user\.gemini\antigravity\scratch\ml_assignment_2\extracted_notebook.py', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        out.write(f"\n# {'='*60}\n# CELL {i} ({cell['cell_type']})\n# {'='*60}\n")
        out.write(''.join(cell['source']) + '\n')

print("Extraction complete. Cells extracted:", len(nb['cells']))
