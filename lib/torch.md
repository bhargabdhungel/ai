# Basics

## Tensors

```python
  torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False)
  torch.tensor([[1, 2], [3, 4]])
  torch.tensor([1, 2, 3], dtype=torch.float64)
  torch.tensor([1, 2, 3], dtype=torch.float64, requires_grad=True)
  torch.ones(2, 2)
  torch.zeros(2, 2)
  torch.eye(2)
  torch.rand(2, 2)
  torch.randn(2, 2)
  torch.arange(0, 10)
  torch.linspace(0, 10, 5)
  torch.logspace(0, 10, 5)
  torch.zeros_like(x)
  torch.normal(mean=0, std=1, size=(2, 2))
  torch.randint(low=0, high=10, size=(2, 2))
  torch.randperm(10)
  torch.full((2, 2), 7)

```

## Tensors attributes

```python
  x = torch.tensor([1, 2, 3], dtype=torch.float64, requires_grad=True)
  x.dtype
  x.device
  x.shape
  x.requires_grad
```

## Tensors operations

```python
  x = torch.tensor([1, 2, 3], dtype=torch.float64)
  y = torch.tensor([4, 5, 6], dtype=torch.float64)
  z = x + y
  z = x - y
  z = x * y
  z = x / y
  z = x @ y
  z = x ** 2
  z = x.sum()
  z = x.mean()
  z = x.std()
  z = x.max()
  z = x.min()
  z = x.argmax()
  z = x.argmin()
  z = x.abs()
  z = x.sqrt()
  z = x.exp()
  z = x.log()
  z = x.sigmoid()
  z = x.relu()
  z = x.tanh()
  z = x.sin()
  z = x.cos()
  z = x.tan()
  z = x.pow(2)
  z = x.clamp(min=0, max=1)
  z = x.detach() // detach the tensor from the computation graph
```

## Tensors operations in-place

```python
  x.add_(1)
  x.sub_(1)
  x.mul_(2)
  x.div_(2)
  x.pow_(2)
  x.clamp_(min=0, max=1)
```

## Tensors reshaping

```python
  x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64)
  z = x.view(2, 3)
  z = x.reshape(3, 2)
  z = x.t()
  z = x.permute(1, 0)
  z = x.squeeze()
```

## Tensors indexing

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64)

  z = x[0, 0]
  z = x[0, :]
  z = x[:, 0]
  z = x[0:2]
  z = x[0:2, 0]
  z = x[0:2, :]
  z = x[:, 0:2]
  z = x[x > 3]
  z = x[(x > 3) & (x < 6)]
```

## Tensors concatenation

```python
  x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
  y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float64)
  z = torch.cat((x, y), dim=0)
  z = torch.cat((x, y), dim=1)
  z = torch.stack((x, y), dim=0)
  z = torch.stack((x, y), dim=1)

```

## Tensors reduction

```python
  z = x.sum()
  z = x.sum(dim=0)
  z = x.sum(dim=1)
  z = x.mean()
  z = x.mean(dim=0)
  z = x.mean(dim=1)
  z = x.std()
  z = x.std(dim=0)
  z = x.std(dim=1)
  z = x.max()
  z = x.max(dim=0)
  z = x.max(dim=1)
  z = x.min()
  z = x.min(dim=0)
  z = x.min(dim=1)
  z = x.argmax()
  z = x.argmax(dim=0)
  z = x.argmax(dim=1)
  z = x.argmin()
  z = x.argmin(dim=0)
  z = x.argmin(dim=1)
```

## Tensors broadcasting

```python
  z = x + y
  z = x - y
  z = x * y
  z = x / y
```
