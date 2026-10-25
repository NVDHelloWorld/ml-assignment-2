with open(r'C:\Users\user\.gemini\antigravity\scratch\ml_assignment_2\paper\cvpr_paper.typ', encoding='utf-8') as f:
    text = f.read()

stack = []
matches = {'(': ')', '[': ']', '{': '}'}

for idx, ch in enumerate(text):
    line_no = text[:idx].count('\n') + 1
    if ch in '([{':
        stack.append((ch, idx, line_no))
    elif ch in ')]}':
        if not stack:
            print(f"Extra closing {ch} at line {line_no}")
        else:
            top, top_idx, top_line = stack.pop()
            if matches[top] != ch:
                print(f"Mismatched {top} from line {top_line} with {ch} at line {line_no}")

if stack:
    print(f"Unclosed delimiters ({len(stack)}):")
    for top, top_idx, top_line in stack:
        print(f"  Unclosed {top} from line {top_line}: {text.splitlines()[top_line-1]}")
else:
    print("All brackets matched!")
