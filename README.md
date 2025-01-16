implementing gradient from scratch  --ref ( andrej karpathy)

1. tensor module to replicate Tensor of pytorch in python.
    From karpathy implementation I implement Context (ctx), inplace for _prev/children

2. Using topological sorting by the andrej for creating the backward computation graph and reversed topology to execute the backward. ( contains set which doesnot allow the visiting the same node multiple times)

3. 
