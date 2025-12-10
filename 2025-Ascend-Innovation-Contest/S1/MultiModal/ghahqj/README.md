# MindNLP 模型优化详细说明 (Qwen2)

本文档详细记录了针对 Qwen2-VL 模型的关键性能优化点，并附带了相应的核心代码实现。



## 1. Qwen2-VL 模型优化

### 1.1 算子融合与逻辑优化 (Modeling & Ops)

**优化痛点:** 

1. **RoPE 计算冗余**：原实现使用基础算子（乘、加、slice、cat）手动组合实现旋转位置编码（RoPE），导致算子下发数量多，计算图中碎片化严重，且显存读写频繁，影响推理性能。
2. **算子兼容性与效率**：原代码中使用 `tensor.swapaxes` 和 Python 切片操作，在昇腾（Ascend）硬件的静态图编译和执行中效率不如 MindSpore 的 `ops` 原语高效；同时 Attention 中的 Softmax 存在不必要的 float32 类型转换，增加了显存带宽压力。

**改进方案:** 

1. **RoPE 算子融合**：使用 MindSpore 特有的 `ops.rotary_position_embedding` 融合算子替代原本的手动实现，大幅减少算子调用开销并提升计算密度。 
2. **算子替换**：将 `swapaxes` 替换为 `ops.transpose`，将切片操作替换为 `ops.split`，获得更好的图编译性能。

**源码实现** (`modeling_qwen2_vl.py`):

**Python**

```python
# 1. 引入 MindSpore Ops
import mindspore.ops as ops

# 2. 优化 RoPE 实现 (apply_multimodal_rotary_pos_emb)
# 替换前：手动计算 cos/sin 乘法与 rotate_half
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)

# 替换后：使用融合算子
q_embed = mindspore.ops.rotary_position_embedding(q, cos, sin, mode=0)
k_embed = mindspore.ops.rotary_position_embedding(k, cos, sin, mode=0)

# 3. 优化 Vision RoPE (apply_rotary_pos_emb_vision)
# output = (tensor * cos) + (rotate_half(tensor) * sin)
output = mindspore.ops.rotary_position_embedding(tensor, cos, sin, mode=0)

# 4. 优化 Transpose 与 Softmax (Attention 模块)
# 替换前：attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=mindspore.float32).to(q.dtype)
# 替换后：移除多余类型转换，使用 ops.transpose 替代 swapaxes
q = ops.transpose(q, 0, 1)
k = ops.transpose(k, 0, 1)
v = ops.transpose(v, 0, 1)
# ...
attn_weights = nn.functional.softmax(attn_weights, dim=-1) # 移除 float32 cast
attn_output = ops.matmul(attn_weights, v)
attn_output = ops.transpose(attn_output, 0, 1)
```



## 评测结果

| 评测指标 | 平均得分 |
|---------|---------|
| 峰值显存得分 | 116.6667 |
| Prefill时延得分 | 110.7972    |
| Decode时延得分 | 113.3078     |
| **总分** | **113.5905** |
