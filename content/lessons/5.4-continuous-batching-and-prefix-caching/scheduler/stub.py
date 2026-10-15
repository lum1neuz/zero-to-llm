def continuous_batching(requests, max_batch):
    waiting = sorted(requests, key=lambda r: r["arrive"])
    running = []    # [request, tokens made so far]
    done = {}
    for step in range(1000):
        if not waiting and not running:
            break
        # TODO: admit waiting requests that have arrived, oldest first, while a slot is free
        ...
        # TODO: every running request makes one token this step
        ...
        # TODO: requests with all their tokens record this step in done and leave, freeing their slots
        ...
    return done


def mean_latency(done, requests):
    return sum(done[r["id"]] - r["arrive"] + 1 for r in requests) / len(requests)


for name, schedule in (("static", static_batching), ("continuous", continuous_batching)):
    done = schedule(requests, MAX_BATCH)
    print(f"{name:<11} last token at step {max(done.values()):>2}   mean latency {mean_latency(done, requests):.1f} steps")
