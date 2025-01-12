implementing gradient from scratch  --ref ( andrej karpathy)

1. tensor module to replicate Tensor of pytorch in python.
    From karpathy implementation I implement Context (ctx), inplace for _prev/children

2. Instead of reversed_topology for the chain rule backward, i have used stack.
