
def six_bit(v):
    # TODO: one super-scale for all the values, so the largest lands on 63 (6 bits)
    d = ...
    # TODO: each value as a whole number of d's, 0…63
    q = ...
    return q.to(torch.uint8), d

sc, d = six_bit(scale)        # the 8 sub-block scales
mq, dmin = six_bit(-lo)       # the 8 mins, stored as positive numbers
scale_k, lo_k = d * sc, -dmin * mq
q_k = torch.round((w.reshape(-1, 32) - lo_k) / scale_k).clamp(0, 15)   # llama.cpp re-fits the codes to the rounded scales
w_k = (q_k * scale_k + lo_k).reshape(-1)
bits = (256 * 4 + 8 * 6 + 8 * 6 + 2 * 16) / 256
print(f"fp16 scale + min per 32   {4 + 2 * 16 / 32:.3f} bits/weight   mean err {(w - w_hat).abs().mean().item():.5f}")
print(f"Q4_K super-block          {bits:.3f} bits/weight   mean err {(w - w_k).abs().mean().item():.5f}")
