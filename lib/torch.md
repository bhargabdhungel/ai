# Basics

## Tensors

```python
  torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False)
  torch.tensor([[1, 2], [3, 4]])
  torch.tensor([1, 2, 3], dtype=torch.float64)
  torch.tensor([1, 2, 3], dtype=torch.float64, requires_grad=True)
```

## Tensors attributes

```python
  x.dtype
  x.device
  x.shape
  x.requires_grad
  x.grad
  x.grad_fn
```

## Tensors operations

```python
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
  z = x.detach()
  z = x.numpy()
  z = x.item()
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
  z = x.view(2, 3)
  z = x.view(6)
  z = x.view(-1)
  z = x.view(1, 1, 2, 3)
  z = x.reshape(2, 3)
  z = x.reshape(6)
  z = x.reshape(-1)
  z = x.reshape(1, 1, 2, 3)
  z = x.permute(1, 0)
  z = x.t()
  z = x.flatten()
  z = x.squeeze()
  z = x.unsqueeze(1)
```

## Tensors indexing

```python
  z = x[0]
  z = x[0, 0]
  z = x[0, :]
  z = x[:, 0]
  z = x[0:2]
  z = x[0:2, 0]
  z = x[0:2, :]
  z = x[:, 0:2]
  z = x[x > 3]
  z = x[x > 3].view(-1)
  z = x[x > 3].view(-1).detach().numpy()
```

## Tensors concatenation

```python
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
