import re
with open("kernels/moe/naive_moe.cu", "r") as f:
    text = f.read()

print("Functions found:")
for m in re.finditer(r'void moe_forward_([a-zA-Z0-9_]+)\(', text):
    print(m.group(1))
