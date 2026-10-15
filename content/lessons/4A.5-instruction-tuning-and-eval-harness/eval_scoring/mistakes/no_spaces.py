# Short-answer questions: (question, gold answer, the model's answer)
QA = [
    ("Монгол Улсын нийслэл аль хот вэ?", "Улаанбаатар", "улаанбаатар."),
    ("Морин хуур хэдэн утастай вэ?", "хоёр", "Хоёр"),
    ("Монголын хамгийн урт гол аль вэ?", "Орхон", "Орхон гол"),
    ("Чингис хаан хэдэн онд төрсөн бэ?", "1162", " 1162 "),
]

# Multiple choice: (question, options, the model's summed log-probability of each option, index of the right one)
MC = [
    ("Монголын хамгийн өндөр уул аль вэ?", ["Хүйтэн оргил", "Отгонтэнгэр", "Богд", "Хан Хэнтий"], [-7.2, -9.9, -4.8, -13.0], 0),
    ("Монгол Улсын мөнгөн тэмдэгт юу вэ?", ["төгрөг", "рубль", "юань", "доллар"], [-3.0, -4.5, -4.0, -5.4], 0),
    ("Хоёр бөхт тэмээ голчлон хаана амьдардаг вэ?", ["Говьд", "Хөвсгөлд", "Хэнтийд", "Архангайд"], [-4.0, -10.4, -7.7, -9.9], 0),
    ("Улаанбаатарын төв талбайг юу гэдэг вэ?", ["Сүхбаатарын талбай", "Чингисийн талбай", "Төв", "Зайсан"], [-10.8, -12.8, -4.5, -7.2], 0),
]

def normalize_answer(s):
    # TODO: lowercase, keep only letters, digits and spaces, and collapse runs of whitespace into single spaces
    return "".join(c for c in s.lower() if c.isalnum())


def exact_match(pred, gold):
    return normalize_answer(pred) == normalize_answer(gold)


def pick_option(options, logprobs):
    # TODO: the index of the option with the highest log-probability per character
    return max(range(len(options)), key=lambda i: logprobs[i] / len(options[i]))


em = sum(exact_match(pred, gold) for _, gold, pred in QA) / len(QA)
mc = sum(pick_option(opts, lps) == right for _, opts, lps, right in MC) / len(MC)
print(f"exact match {100 * em:.0f}%   multiple choice {100 * mc:.0f}%   score {50 * (em + mc):.1f}%")
