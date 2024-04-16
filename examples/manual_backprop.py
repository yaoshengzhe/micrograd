import micrograd

if __name__ == '__main__':
    x1 = micrograd.core.Value(2.0, label='x1')
    x2 = micrograd.core.Value(0.0, label='x2')

    w1 = micrograd.core.Value(-3.0, label='w1')
    w2 = micrograd.core.Value(1.0, label='w2')

    b = micrograd.core.Value(6.7, label='b')

    x1w1 = x1*w1
    x1w1.label = 'x1*w1'
    x2w2 = x2*w2
    x2w2.label = 'x2*w2'

    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = 'x1*w1 + x2*w2'

    n = x1w1x2w2 + b
    n.label = 'n'
    o = n.tanh()
    
    print(o.draw())
