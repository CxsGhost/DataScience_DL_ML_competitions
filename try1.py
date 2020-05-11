

def f(n):
    return lambda x, y = [n]: (y.append(x), sum(y))[1]
print(f(4)(2))
print(f(4)(1))



a = [2, 3]
print(a.index(2))
