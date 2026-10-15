def count_letter(word, letter):
    # TODO: start the counter at 0
    count = 0
    for ch in word:
        # TODO: add 1 when ch matches
        if ch == letter:
            count += 1
    return count

print(count_letter("banana", "a"))
