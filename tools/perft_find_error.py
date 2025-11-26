#!/usr/bin/python

# This script recursively calls cli_debug and stockfish to compare the results
# of a perft at a specified position.
#
# To run, stockfish must be in the path.

import os
import sys
import subprocess
import re

fen = sys.argv[1]
depth = int(sys.argv[2])
fish_path = "stockfish"
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(script_dir + "/..")
rchess_path = root_dir + "/build/debug"
move_regex = r"([a-zA-z][1-8][a-zA-z][1-8][nbrq]?): ([1-9]\d*|0)"


def perft_get_counts(driver, fen, depth, moves):
    counts = {}

    try:
        # Prepare the command
        driver_cmd = f"position fen {fen} moves"
        for move in moves:
            driver_cmd += f" {move}"
        driver_cmd += f"\ngo perft {depth}\n"
        driver_cmd += "quit\n"

        # Start the subprocess and wait for it to complete.
        engine = subprocess.Popen([driver], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True)
        output = engine.communicate(input=driver_cmd, timeout=120)[0]

        # Loop through the output and parse it.
        for line in output.splitlines():
            matches = re.search(move_regex, line)
            if not matches:
                continue
            counts[matches.group(1)] = matches.group(2)
    except Exception as e:
        print(f"An error occured: {e}")

    return counts


# Compile the release version of the code.
comp_command = f"cd {root_dir}/build; cmake --build ./"
compiler = subprocess.run(comp_command, shell=True)

moves = []
while True:
    # Reset the missing and extra arrays.
    missing = {}
    extra = {}
    wrong = {}
    keys = []

    # Read the counts for the different engines.
    fish_counts = perft_get_counts(fish_path, fen, depth, moves)
    rchess_counts = perft_get_counts(rchess_path, fen, depth, moves)

    # Find all values that are either missing or wrong.
    for key, value in fish_counts.items():
        if key not in rchess_counts:
            missing[key] = value
        elif value != rchess_counts[key]:
            wrong[key] = rchess_counts[key]

    for key, value in rchess_counts.items():
        if key not in fish_counts:
            extra[key] = value

    # Prep header data.
    headers = [f"Depth: {depth}", "Expected", "Extra", "Wrong", "Missing"]
    keys += missing.keys()
    keys += extra.keys()
    keys += wrong.keys()
    keys.sort()

    # Pick only the first move to explore.
    if len(keys) == 0:
        break

    # Display the results of the comparison.
    for header in headers:
        print(f"{header:10}", end="")
    print()
    for key in keys:
        if key not in missing:
            missing[key] = ""
        if key not in extra:
            extra[key] = ""
        if key not in wrong:
            wrong[key] = ""
        if key not in fish_counts:
            fish_counts[key] = ""
        print(f"{key:10}{fish_counts[key]:10}{extra[key]:10}", end="")
        print(f"{wrong[key]:10}{missing[key]:10}")
    print()

    # Break out on the last iteration.
    if depth == 1:
        break

    # Make the first move in the keys list.
    depth -= 1
    moves.append(keys[0])

if len(keys) == 0:
    print("All keys match")
    exit()

# Print the moves that don't match.
for key in keys:
    if fish_counts[key] != "":
        print("Missing: ", end="")
    else:
        print("Extra: ", end="")
    for move in moves:
        print(move + " ", end="")
    print(key)

