import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    query: (B, H, Q_len, D)
    key:   (B, H, K_len, D)
    value: (B, H, V_len, D_v)
    mask:  (B, 1, Q_len, K_len) or broadcastable
    """
    # Step 1: QK^T
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))  # (B, H, Q_len, K_len)

    # Step 2: Scale
    dk = query.size(-1)
    scale = math.sqrt(dk)
    scaled_attention_logits = matmul_qk / scale

    # Step 3: Mask (before softmax)
    if mask is not None:
        scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 0, float('-inf'))

    # Step 4: Softmax (safe)
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    attention_weights = attention_weights.masked_fill(torch.isnan(attention_weights), 0.0)

    # Step 5: Final weighted sum
    output = torch.matmul(attention_weights, value)  # (B, H, Q_len, D_v)

    return output, attention_weights
