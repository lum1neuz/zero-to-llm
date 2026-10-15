import statistics

# One benchmark run: 7 requests against your server, times in seconds from the start of the run
requests = [
    {"arrive": 0.00, "first": 0.18, "done": 2.58, "tokens": 121},
    {"arrive": 0.05, "first": 0.24, "done": 1.92, "tokens": 81},
    {"arrive": 0.40, "first": 0.61, "done": 4.13, "tokens": 161},
    {"arrive": 1.10, "first": 1.29, "done": 2.97, "tokens": 81},
    {"arrive": 1.90, "first": 2.95, "done": 4.27, "tokens": 61},
    {"arrive": 2.00, "first": 2.21, "done": 4.85, "tokens": 121},
    {"arrive": 2.70, "first": 2.90, "done": 4.50, "tokens": 81},
]

def summarize(requests):
    # TODO: one wait per request, from its own arrival to its first token (seconds)
    ttft = [r["first"] for r in requests]
    # TODO: one value per request: ms per token, for the tokens after the first
    tpot = [(r["done"] - r["first"]) / (r["tokens"] - 1) * 1000 for r in requests]
    # TODO: every token made, over the wall-clock time from the first arrival to the last finish
    tok_per_s = sum(r["tokens"] for r in requests) / (max(r["done"] for r in requests) - min(r["arrive"] for r in requests))
    # TODO: the typical wait: half the requests wait less than this, half wait more
    ttft_p50 = statistics.median(ttft)
    return ttft_p50, statistics.mean(tpot), tok_per_s


ttft_p50, tpot_ms, tok_per_s = summarize(requests)
print(f"TTFT p50 {ttft_p50:.2f} s   TPOT {tpot_ms:.1f} ms   throughput {tok_per_s:.0f} tok/s")
