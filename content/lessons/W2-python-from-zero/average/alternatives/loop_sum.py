def average(numbers):
    total = 0
    for n in numbers:
        total = total + n
    return total / len(numbers)

print(average([2, 4, 9]))
