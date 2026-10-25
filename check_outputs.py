import json

with open(r'c:\Users\user\Downloads\ML_assignment_2_.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    outputs = cell.get('outputs', [])
    if outputs:
        print(f"=== Cell {i} Outputs ({len(outputs)}) ===")
        for out in outputs:
            if out.get('output_type') == 'stream':
                print(''.join(out.get('text', [])))
            elif out.get('output_type') in ['display_data', 'execute_result']:
                data = out.get('data', {})
                if 'text/plain' in data:
                    print(''.join(data['text/plain']))
                if 'image/png' in data:
                    print(f"[Image PNG present, len: {len(data['image/png'])}]")
            elif out.get('output_type') == 'error':
                print(f"Error: {out.get('ename')}: {out.get('evalue')}")
