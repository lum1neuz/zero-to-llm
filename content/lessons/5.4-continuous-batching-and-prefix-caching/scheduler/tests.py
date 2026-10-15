from harness import test, fail, learner_fn, script_error, stdout

# A different stream from the script's, with 2 slots. q finishes at step 0, so its slot is free from step 1;
# r and s arrive together at step 3 and t and u at step 4, so slots run short.
STREAM = [
    {"id": "p", "arrive": 0, "out": 5},
    {"id": "q", "arrive": 0, "out": 1},
    {"id": "r", "arrive": 3, "out": 2},
    {"id": "s", "arrive": 3, "out": 6},
    {"id": "t", "arrive": 4, "out": 1},
    {"id": "u", "arrive": 4, "out": 3},
    {"id": "v", "arrive": 9, "out": 2},
]
SLOTS = 2
BY_ID = {r["id"]: r for r in STREAM}


def simulate(reqs, max_batch, order=("admit", "make", "retire"), drain=False, one=False):
    """The reference scheduler, plus variants: phases in another order, admitting only into an empty batch,
    or admitting one request per step."""
    waiting = sorted(reqs, key=lambda r: r["arrive"])
    running, done = [], {}
    for step in range(200):
        if not waiting and not running:
            break
        for phase in order:
            if phase == "admit":
                if drain and running:
                    continue
                while waiting and waiting[0]["arrive"] <= step and len(running) < max_batch:
                    running.append([waiting.pop(0), 0])
                    if one:
                        break
            elif phase == "make":
                for slot in running:
                    slot[1] += 1
            else:
                for r, made in running:
                    if made == r["out"]:
                        done[r["id"]] = step
                running = [slot for slot in running if slot[1] < slot[0]["out"]]
    return done


RIGHT = simulate(STREAM, SLOTS)


def run(L):
    fn = learner_fn(L, "continuous_batching")
    out = fn([dict(r) for r in STREAM], SLOTS)
    if out is None:
        fail("`continuous_batching` returns nothing. Keep `return done` at the end.")
    if not isinstance(out, dict):
        fail(f"`continuous_batching` should return a dict of request id → finishing step, but it returns {type(out).__name__}.")
    return out


@test("all_finish", "Every request finishes")
def _(L):
    done = run(L)
    if not done:
        fail("`continuous_batching` finished no requests. Fill in the three steps: admit, make a token, retire.")
    missing = [r["id"] for r in STREAM if r["id"] not in done]
    if not missing:
        return
    finished = ", ".join(sorted(done))
    fail(f"Requests {', '.join(missing)} never finish. {finished} made all their tokens, but if they still hold their slots "
         "nobody else can join. Drop finished requests from `running` in the same step.")


def spans(done):
    """Each request makes one token per step while it runs, so it ran from done − out + 1 to done."""
    return {i: (d - BY_ID[i]["out"] + 1, d) for i, d in done.items() if i in BY_ID}


@test("arrivals", "No request starts before it arrives")
def _(L):
    for i, (start, end) in spans(run(L)).items():
        r = BY_ID[i]
        if start < r["arrive"]:
            fail(f"Request {i} arrives at step {r['arrive']} and wants {r['out']} tokens, so it can't finish before step "
                 f"{r['arrive'] + r['out'] - 1}. Yours finishes it at step {end}. Admit a request only once `arrive <= step`.")


@test("max_batch", "Never more than max_batch requests run at once")
def _(L):
    sp = spans(run(L))
    for step in range(max((e for _, e in sp.values()), default=0) + 1):
        ids = sorted(i for i, (s, e) in sp.items() if s <= step <= e)
        if len(ids) > SLOTS:
            fail(f"At step {step}, {len(ids)} requests are running ({', '.join(ids)}), but max_batch is {SLOTS}. "
                 "Admit only while `len(running) < max_batch`.")


@test("schedule", "Finished requests leave at once, and waiting ones take their slots")
def _(L):
    done = run(L)
    if done == RIGHT:
        return
    if done == simulate(STREAM, SLOTS, drain=True):
        fail("New requests only join once the whole batch has drained. Request q finishes at step 0, but its slot sits empty "
             "until p is done too. That's static batching again. Every step, any free slot takes the next waiting request.")
    if done == simulate(STREAM, SLOTS, one=True):
        fail("Only one request joins per step, even when two slots are free: t and u both wait at step 4. "
             "Keep admitting while a slot is free and someone is waiting.")
    for order in (("retire", "admit", "make"), ("admit", "retire", "make")):
        if done == simulate(STREAM, SLOTS, order=order):
            fail(f"Request q makes its only token at step 0, but yours records it at step {done.get('q')}, one step late. "
                 "Within a step, make the tokens first, then retire whoever just finished, so `done` gets the step of the last token.")
    i = next(i for i in RIGHT if done.get(i) != RIGHT[i])
    fail(f"Request {i} should make its last token at step {RIGHT[i]}, but yours says {done.get(i)}. Each step: admit into free "
         "slots, make one token for everyone running, then retire whoever is done.")


@test("script_runs", "The script runs and compares both schedules")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "schedule" not in L:
        fail("The loop at the bottom that runs both schedules is missing. Put it back.")
    if not stdout().strip():
        fail("Nothing printed. Put the print line back.")
