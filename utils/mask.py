import torch

def create_padding_mask(seq, pad_token=0):
    """
    seq: Tensor of shape (batch_size, seq_len)
    returns: (batch_size, 1, 1, seq_len)
    """
    mask = (seq == pad_token).unsqueeze(1).unsqueeze(2).float()  # (B, 1, 1, seq_len)
    return mask
def create_look_ahead_mask(size, device):
    """
    Mask out future positions in a sequence.

    Args:
        size: int (seq_len)
        device: the device for the mask

    Returns:
        (1, 1, size, size)
    """
    mask = torch.triu(torch.ones((size, size), device=device), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)

def create_masks(src, tgt, pad_token=0):
    """
    Creates all necessary masks for the Transformer.

    Args:
        src: (batch_size, src_seq_len)
        tgt: (batch_size, tgt_seq_len)
        pad_token: the token used for padding

    Returns:
        src_mask: (batch_size, 1, 1, src_seq_len)
        tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        memory_mask: same as src_mask
    """
    device = src.device
    src_mask = create_padding_mask(src, pad_token).to(device)  # (B, 1, 1, src_len)

    tgt_padding_mask = create_padding_mask(tgt, pad_token).to(device)  # (B, 1, 1, tgt_len)
    look_ahead_mask = create_look_ahead_mask(tgt.size(1), device)       # (1, 1, tgt_len, tgt_len)

    # Broadcast padding mask to match look-ahead mask shape
    expanded_tgt_padding_mask = tgt_padding_mask.expand(-1, 1, tgt.size(1), tgt.size(1))  # (B, 1, tgt_len, tgt_len)

    # Combine masks: final shape (B, 1, tgt_len, tgt_len)
    tgt_mask = torch.maximum(expanded_tgt_padding_mask, look_ahead_mask)

    return src_mask, tgt_mask, src_mask  # memory_mask is same as src_mask
