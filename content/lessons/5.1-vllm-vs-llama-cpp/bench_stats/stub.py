def summarize(requests):
    # TODO: one wait per request, from its own arrival to its first token (seconds)
    ttft = ...
    # TODO: one value per request: ms per token, for the tokens after the first
    tpot = ...
    # TODO: every token made, over the wall-clock time from the first arrival to the last finish
    tok_per_s = ...
    # TODO: the typical wait: half the requests wait less than this, half wait more
    ttft_p50 = ...
    return ttft_p50, statistics.mean(tpot), tok_per_s


ttft_p50, tpot_ms, tok_per_s = summarize(requests)
print(f"TTFT p50 {ttft_p50:.2f} s   TPOT {tpot_ms:.1f} ms   throughput {tok_per_s:.0f} tok/s")
