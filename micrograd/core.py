'''Micrograd core abstractions.'''

import graphviz

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), '*')

    @staticmethod
    def _trace(root):
        nodes, edges = set(), set()
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
        build(root)
        return nodes, edges

    def draw(self):
        dot = graphviz.Digraph(format='svg', graph_attr={'rankdir': 'LR'})

        nodes, edges = Value._trace(self)
        for n in nodes:
            uid = str(id(n))
            dot.node(name=uid, label="{ %s | data %.4f }" % (n.label, n.data, ), shape='record')

            if n._op:
                dot.node(name=uid+n._op, label=n._op)
                dot.edge(uid+n._op, uid)

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)

        return dot
