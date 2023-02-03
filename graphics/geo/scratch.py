
def main():
    cache = {}
    def treegen(n:int, a, b, c):
        if (n, a, b, c) in cache:
            return cache[(n, a, b, c)]
        if n == 0:
            return 0
        if n == 1:
            return a
        ret = b * treegen(n-1, a, b, c)
        for i in range(1, n-1):
            l = treegen(i, a, b, c)
            r = treegen(n-i-1, a, b, c)
            ret += c * l * r
        cache[(n, a, b, c)] = ret
        return ret
    print([treegen(k, 1, 1, 1) for k in range(26)])
    print([treegen(k, 1, 1, 1) for k in range(100, 101)])


def main2():
    cache = {}
    def treegen(n:int, a, b, c):
        if (n, a, b, c) in cache:
            return cache[(n, a, b, c)]
        if n == 0:
            return 0
        if n == 1:
            return a
        ret = b * treegen(n-1, a, b, c)
        for i in range(1, n//2):
            l = treegen(i, a, b, c)
            r = treegen(n-i-1, a, b, c)
            ret += c * l * r
        if n % 2 == 1:
            lr = treegen(n//2, a, b, c)
            ret += lr * (lr + 1) // 2
        cache[(n, a, b, c)] = ret
        return ret
    print([treegen(k, 1, 1, 1) for k in range(26)])
    print([treegen(k, 1, 1, 1) for k in range(100, 101)])

def main0():
    def treegen(n:int, a, b, c):
        return 3**n
    print([treegen(k, 1, 1, 1) for k in range(26)])
    print([treegen(k, 1, 1, 1) for k in range(100, 101)])

if __name__ == "__main__":
    main0()
    main()
    main2()
