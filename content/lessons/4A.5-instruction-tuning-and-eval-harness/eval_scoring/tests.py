from harness import test, fail, learner_fn, as_float, script_error


def norm(L, s):
    out = learner_fn(L, "normalize_answer")(s)
    if out is Ellipsis:
        fail("`normalize_answer` still returns `...`, the placeholder.")
    if not isinstance(out, str):
        fail(f"`normalize_answer` should return a string, but it returns {type(out).__name__}.")
    return out


@test("normalized", "normalize_answer ignores case, punctuation and extra spaces")
def _(L):
    if norm(L, "Хоёр") != norm(L, "хоёр"):
        fail(f"\"Хоёр\" and \"хоёр\" normalize to \"{norm(L, 'Хоёр')}\" and \"{norm(L, 'хоёр')}\". A capital letter shouldn't make a right answer wrong. Lowercase first.")
    if norm(L, "улаанбаатар.") != norm(L, "Улаанбаатар"):
        fail(f"\"улаанбаатар.\" normalizes to \"{norm(L, 'улаанбаатар.')}\", so the full stop makes it miss. Drop everything that isn't a letter, a digit or a space.")
    if norm(L, "1162") == norm(L, "1163"):
        fail(f"\"1162\" and \"1163\" both normalize to \"{norm(L, '1162')}\", so any year matches any other. Keep the digits.")
    if norm(L, "Их  Хурал") == "иххурал":
        fail("\"Их  Хурал\" normalizes to \"иххурал\": every space is gone, so words run together and different phrases can collide. "
             "Keep single spaces between words.")
    if norm(L, " 1162 ") != "1162" or norm(L, "Их  Хурал") != "их хурал":
        fail(f"\" 1162 \" normalizes to \"{norm(L, ' 1162 ')}\" and \"Их  Хурал\" to \"{norm(L, 'Их  Хурал')}\". "
             "Spaces at the ends and doubled spaces in the middle should go: split the words and join them with single spaces.")
    if norm(L, "Орхон гол") == norm(L, "Орхон"):
        fail("\"Орхон гол\" and \"Орхон\" normalize to the same thing. Exact match should stay strict about the words themselves.")


@test("picks_option", "pick_option picks the best log-probability per character")
def _(L):
    pick = learner_fn(L, "pick_option")
    opts = ["Хүйтэн оргил", "Отгонтэнгэр", "Богд", "Хан Хэнтий"]
    lps = [-7.2, -9.9, -4.8, -13.0]
    out = pick(opts, lps)
    if out is Ellipsis:
        fail("`pick_option` still returns `...`, the placeholder.")
    if isinstance(out, str):
        fail(f"`pick_option` returns the option's text, \"{out}\". The script compares it with the right answer's index, so return the index.")
    if not isinstance(out, int) or isinstance(out, bool):
        fail(f"`pick_option` should return an index (an int), but it returns {type(out).__name__}.")
    if out == 0:
        out2 = pick(["Төв", "Сүхбаатарын талбай"], [-4.5, -10.8])
        if out2 != 1:
            fail(f"For [\"Төв\", \"Сүхбаатарын талбай\"] with log-probabilities [-4.5, -10.8] you pick {out2}. Per character that's "
                 "−1.5 against −0.6, so option 1 wins. Does your code use its arguments?")
        return
    if out == 2:
        fail("You pick \"Богд\", the option with the highest summed log-probability. But every extra character costs log-probability, "
             "so raw sums favour short options. Богд has −1.2 per character, Хүйтэн оргил −0.6. Compare per character.")
    if out == 3:
        fail("You pick the *lowest* score. Log-probabilities are negative, and the best option is the one closest to zero.")
    fail(f"Per character the scores are −0.60, −0.90, −1.20 and −1.30, so option 0 wins. You pick {out}.")


@test("script_runs", "The harness scores 75% exact match and 100% multiple choice")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    em, mc = as_float(L.get("em")), as_float(L.get("mc"))
    if em is None or mc is None:
        fail("The lines that compute `em` and `mc` are missing. Put them back.")
    if abs(em - 0.75) > 1e-9 or abs(mc - 1.0) > 1e-9:
        fail(f"You get exact match {100 * em:.0f}% and multiple choice {100 * mc:.0f}%, but the sample should give 75% and 100%. "
             "Did the questions or the script change?")
