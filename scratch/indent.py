with open("app.py", "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

start_idx = 398
end_idx = 451

for i in range(start_idx, end_idx):
    if lines[i]:
        lines[i] = "    " + lines[i]

with open("app.py", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
