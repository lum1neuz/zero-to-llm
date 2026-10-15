# Each request: when it arrives (in decode steps) and how many tokens it wants
requests = [
    {"id": "a", "arrive": 0, "out": 12},
    {"id": "b", "arrive": 0, "out": 3},
    {"id": "c", "arrive": 1, "out": 2},
    {"id": "d", "arrive": 2, "out": 4},
    {"id": "e", "arrive": 2, "out": 3},
    {"id": "f", "arrive": 5, "out": 2},
]
MAX_BATCH = 3

def static_batching(requests, max_batch):
    done, step = {}, 0
    queue = sorted(requests, key=lambda r: r["arrive"])
    while queue:
        batch, queue = queue[:max_batch], queue[max_batch:]
        step = max(step, max(r["arrive"] for r in batch))   # wait until the whole batch has arrived
        step += max(r["out"] for r in batch)                # run until its longest request is done
        for r in batch:
            done[r["id"]] = step - 1                        # everyone is returned when the batch ends
    return done

def continuous_batching(requests, max_batch):
    waiting = sorted(requests, key=lambda r: r["arrive"])
    running = []    # [request, tokens made so far]
    done = {}
    for step in range(1000):
        if not waiting and not running:
            break
        # TODO: admit waiting requests that have arrived, oldest first, while a slot is free
        if waiting and waiting[0]["arrive"] <= step and len(running) < max_batch:
            running.append([waiting.pop(0), 0])
        # TODO: every running request makes one token this step
        for slot in running:
            slot[1] += 1
        # TODO: requests with all their tokens record this step in done and leave, freeing their slots
        for r, made in running:
            if made == r["out"]:
                done[r["id"]] = step
        running = [slot for slot in running if slot[1] < slot[0]["out"]]
    return done


def mean_latency(done, requests):
    return sum(done[r["id"]] - r["arrive"] + 1 for r in requests) / len(requests)


for name, schedule in (("static", static_batching), ("continuous", continuous_batching)):
    done = schedule(requests, MAX_BATCH)
    print(f"{name:<11} last token at step {max(done.values()):>2}   mean latency {mean_latency(done, requests):.1f} steps")
