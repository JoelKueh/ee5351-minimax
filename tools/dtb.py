#!/usr/bin/python

import sys

BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

num = int(sys.argv[1])

for i in range(0, 8):
    for j in range(0, 8):
        if num & 1:
            print(BOLD + GREEN + "1" + RESET, end=" ")
        else:
            print(BOLD + RED + "0" + RESET, end=" ")
        num >>= 1
    print()
