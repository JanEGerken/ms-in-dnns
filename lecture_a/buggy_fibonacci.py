def fibonacci(n):
    if n <= 0:
        raise ValueError
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 1)


if __name__ == "__main__":
    n = int(input("Enter the number of terms in the Fibonacci sequence: "))
    for i in range(1, n + 1):
        result = fibonacci(i)
        print(f"The Fibonacci sequence of {i} is: {result}")
