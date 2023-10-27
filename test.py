import re

# a = re.search(r"\(([A-Za-z]*)\)@@$", "a(b@@b(b)@@")
# print(a.group()[1:-3])
b = 0
a = b + 1
print(a)
a = [[]]
print(a == [[]])

print(eval("('IDTTFBT', 'HOMO', '-5.23 eV')"))

from ast import literal_eval
z = "(aa, 4, 5)"

print(tuple(i for i in z.strip('()').split(", ")))