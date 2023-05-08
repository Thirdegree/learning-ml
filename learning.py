from __future__ import annotations
from collections.abc import Sequence
from typing import Callable, Literal, TypeVar
from itertools import chain
# from micrograd.engine import Value

from dataclasses import InitVar, dataclass
import math
import numpy as np
from graphviz import Digraph  # type: ignore

# https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=104s

T_Data = TypeVar('T_Data', bound=float)

OPS = Literal['+', '*', 'tanh', 'exp', '/', 'pow']


def noop_backward() -> None:
    pass


@dataclass
class Value:
    data: float
    label: str | None = None
    grad: float | None = None
    _backward: Callable[[], None] = noop_backward
    _op: OPS | None = None
    _children: InitVar[tuple[Value, ...] | None] = None

    def __post_init__(self, _children: tuple[Value, ...] | None) -> None:
        self._prev = _children or ()

    def __repr__(self) -> str:
        return f'Value({self.data=})'

    def __hash__(self) -> int:
        return hash(id(self))

    def __add__(self, other: Value | float) -> Value:
        _other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + _other.data, _op='+', _children=(self, _other))

        def _backward() -> None:
            if self.grad is None:
                self.grad = 0.
            if _other.grad is None:
                _other.grad = 0.
            assert out.grad is not None
            self.grad += out.grad
            _other.grad += out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Value | float) -> Value:
        return self + -other

    def __radd__(self, other: Value | float) -> Value:
        return self + other

    def __mul__(self, other: Value | float) -> Value:
        _other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * _other.data, _op='*', _children=(self, _other))

        def _backward() -> None:
            if self.grad is None:
                self.grad = 0.
            if _other.grad is None:
                _other.grad = 0.
            assert out.grad is not None
            self.grad += out.grad * _other.data
            _other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __rmul__(self, other: Value | float) -> Value:
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        return self * other**-1

    def __pow__(self, other: float) -> Value:
        out = Value(
            self.data**other,
            label=f'**{other}',
            _op='pow',
            _children=(self, ),
        )

        def _backward() -> None:
            if self.grad is None:
                self.grad = 0.
            self.grad += (other * self.data**(other - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), _children=(self, ), _op='exp')

        def _backward() -> None:
            if self.grad is None:
                self.grad = 0.
            assert out.grad is not None
            self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        e = (2 * self).exp()
        return (e - 1) / (e + 1)

    def backward(self) -> None:
        if self.grad is None:
            self.grad = 1.0
        topo = []
        visited = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in set(v._prev):
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for child in reversed(topo):
            child._backward()


@dataclass
class Neuron:
    nin: InitVar[int]

    def __post_init__(self, nin: int) -> None:
        self.w = [Value(np.random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(np.random.uniform(-1, 1))

    def __call__(self, x: Sequence[Value]) -> Value:
        return (sum((wi * xi for wi, xi in zip(self.w, x)), self.b)).tanh()

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


@dataclass
class Layer:
    nin: InitVar[int]
    nout: InitVar[int]

    def __post_init__(self, nin: int, nout: int) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: Sequence[Value]) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return list(chain.from_iterable(n.parameters() for n in self.neurons))


@dataclass
class MLP:
    nin: InitVar[int]
    nouts: InitVar[list[int]]

    def __post_init__(self, nin: int, nouts: list[int]) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x: Sequence[Value]) -> Sequence[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return list(chain.from_iterable(layer.parameters() for layer in self.layers))


def trace(root: Value) -> tuple[set[Value], set[tuple[Value, Value]]]:
    nodes, edges = set(), set()

    def build(v: Value) -> None:
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root: Value) -> Digraph:
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)
    known_ids = set()
    for n in nodes:
        uid = str(id(n))
        known_ids.add(uid)
        dot.node(name=uid,
                 label=f"{{ {n.label or ''} | data: {n.data:.4f} | grad: {n.grad or 0.:.4f}}}",
                 shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        assert n2._op is not None
        if str(id(n1)) not in known_ids:
            print(str(id(n1)), n1, n2)
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


def main() -> None:
    # inputs
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    # weights
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    # bias
    b = Value(6.8813835870195432, label='b')
    # x1*w1 + x2*w2 + b
    x1w1 = x1 * w1
    x1w1.label = 'x1w1'
    x2w2 = x2 * w2
    x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = 'x1w1 + x2w2'
    n = x1w1x2w2 + b
    n.label = 'n'
    # o = n.tanh()
    e = (2 * n).exp()
    o = (e - 1) / (e + 1)
    o.label = 'o'

    o.backward()
    draw_dot(o).view()


def main2() -> None:
    _xs = [[2., 3., -1.], [3., -1., 0.5], [0.4, 1., 1.], [1., 1., -1.]]
    xs = [[Value(v) for v in sx] for sx in _xs]
    ys = [1., -1., -1., 1.]

    mlp = MLP(3, [4, 4, 1])
    ypred = [mlp(x)[0] for x in xs]
    # print(ypred)

    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    assert isinstance(loss, Value)
    loss.backward()
    print(loss)
    for k in range(2000):
        ypred = [mlp(x)[0] for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        assert isinstance(loss, Value)

        for p in mlp.parameters():
            # zero-grad
            p.grad = None

        loss.backward()
        for p in mlp.parameters():
            assert p.grad is not None
            p.data += -0.1 * p.grad
        print(k, loss.data)
    __import__('pprint').pprint(loss)
    __import__('pprint').pprint(ys)
    __import__('pprint').pprint(ypred)
    draw_dot(loss).view()


if __name__ == "__main__":
    # main()
    main2()
