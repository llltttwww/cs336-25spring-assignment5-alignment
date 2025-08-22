import torch
from cs336_alignment.post_training_utils import masked_mean, masked_normalize

# 定义输入
ratio = torch.tensor([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
], dtype=torch.float32, requires_grad=True)

advs = torch.tensor([
    [2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2],
], dtype=torch.float32)

masks = torch.tensor([
    # generation 1: 4 tokens
    [1, 1, 1, 1, 0, 0, 0],
    # generation 2: 7 tokens
    [1, 1, 1, 1, 1, 1, 1],
], dtype=torch.float32)

# 最大生成长度
max_gen_len = 7

# 两种归一化方式
masked_mean_result = masked_mean(ratio * advs, masks, dim=1)
masked_normalize_result = masked_normalize(
    ratio * advs, masks, dim=1, normalize_constant=max_gen_len
)

print("masked_mean:", masked_mean_result)
print("masked_normalize:", masked_normalize_result)

# 反向传播测试
masked_mean_result.mean().backward()
print("ratio.grad:", ratio.grad)
