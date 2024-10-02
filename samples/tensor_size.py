import torch

# CUDAメモリをリセット
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()

# メモリ使用量を取得（テンソル作成前）
start_mem = torch.cuda.memory_allocated()

# テンソルを作成し、CUDAに転送
tensor = torch.randn(224, 224, 3, 64, device='cuda')

# メモリ使用量を取得（テンソル作成後）
end_mem = torch.cuda.memory_allocated()

# テンソル作成に使用されたメモリ量を計算（GB単位）
tensor_memory_used = (end_mem - start_mem) / (1024 ** 3)
print(f"Memory used by the tensor: {tensor_memory_used:.6f} GB")