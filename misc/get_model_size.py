def get_model_size(model):
    param_size = 0
    buffer_size = 0
    
    # 各パラメータのサイズを計算
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
        
    # 各バッファのサイズを計算
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
        
    total_size = param_size + buffer_size
    return total_size


