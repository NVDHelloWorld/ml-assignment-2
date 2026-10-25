import typst

with open(r'C:\Users\user\.gemini\antigravity\scratch\ml_assignment_2\paper\cvpr_paper.typ', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(1, len(lines) + 1):
    sub = ''.join(lines[:i])
    with open('tmp_test.typ', 'w', encoding='utf-8') as out:
        out.write(sub)
    try:
        typst.compile('tmp_test.typ', output='tmp_test.pdf')
    except Exception as e:
        err = str(e)
        if 'unclosed delimiter' in err:
            # Check if this line introduced the unclosed delimiter
            pass
        else:
            # Different error or syntax
            pass

# Let's find first line where unclosed delimiter starts
last_ok = 0
for i in range(1, len(lines) + 1):
    sub = ''.join(lines[:i])
    # Add closing if needed or test
    with open('tmp_test.typ', 'w', encoding='utf-8') as out:
        out.write(sub)
    try:
        typst.compile('tmp_test.typ', output='tmp_test.pdf')
        last_ok = i
    except Exception:
        pass

print(f"Total lines: {len(lines)}, last strictly compiling line prefix: {last_ok}")
