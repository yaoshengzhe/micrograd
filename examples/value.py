import micrograd

if __name__ == '__main__':
    a = micrograd.core.Value(2.0, label='a')
    print(a)

    b = micrograd.core.Value(-3.0, label='b')
    print(b)

    c = a * b
    c.label = 'c'
    print(c)

    d = c + micrograd.core.Value(10, label='e')
    d.label = 'd'
    print(d)

    f = micrograd.core.Value(-2.0, label='f')
    L = d * f
    L.label = 'L'

    print(L.draw())
