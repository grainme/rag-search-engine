res = [1, 2, 3, 4]

res1 = []
for e in res:
    res1.append(e * 2)
print(res1)

res2 = list(e * 2 for e in res)
print(res2)

res3 = [e * 2 for e in res]
print(res3)
