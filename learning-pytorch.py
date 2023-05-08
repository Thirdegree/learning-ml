import torch


def tensor(data: float) -> torch.Tensor:
    t = torch.Tensor([data]).double()
    t.requires_grad = True
    return t


def main() -> None:
    x1 = tensor(2.0)
    x2 = tensor(0.0)
    w1 = tensor(-3.0)
    w2 = tensor(1.0)
    b = tensor(6.8813835870195432)
    n = x1 * w1 + x2 * w2 + b
    o = torch.tanh(n)
    print(o.data)
    o.backward()
    print('----')
    print(f'{x2.grad=}')
    print(f'{w2.grad=}')
    print(f'{x1.grad=}')
    print(f'{w1.grad=}')


if __name__ == "__main__":
    main()
