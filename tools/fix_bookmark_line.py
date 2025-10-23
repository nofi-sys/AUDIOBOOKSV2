from pathlib import Path

p = Path('qc_app.py')
s = p.read_text(encoding='utf8')
lines = s.splitlines()
changed = False
for i, line in enumerate(lines):
    if 'Punto guardado (fila' in line and 'self._log' in line:
        # Replace whole line with a clean f-string
        lines[i] = '            self._log(f"Punto guardado (fila {idx + 1}, t={abs_time:.2f}s)")'
        changed = True
        break
if changed:
    p.write_text('\n'.join(lines) + ('\n' if s.endswith('\n') else ''), encoding='utf8')
else:
    print('No change')

