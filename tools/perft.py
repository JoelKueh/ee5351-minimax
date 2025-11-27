#!/usr/bin/python

import tqdm
import argparse
import os
import subprocess
import re
import pexpect

# Get the path to the debug executable to test.
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(script_dir + "/..")
khess_path = root_dir + "/build/debug"

move_regex = r"([a-zA-z][1-8][a-zA-z][1-8][nbrq]?): ([1-9]\d*|0)"
end_regex = r"Nodes searched: ([1-9]\d*|0)"

# Handle arguments
parser = argparse.ArgumentParser(
        description='Move count tester for khess engine')
parser.add_argument('-f', '--full', action='store_true',
                    help='Perform full perft test')
full_perft = parser.parse_args(['-f', '--full'])

# List of positions to test.
POSITIONS = [
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
    '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1',
    'r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1',
    'r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1',
    'rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8',
    'r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10'
]

# List of depths to search to when you want to do a shorter search.
SHORT_DEPTHS = [
    6,
    5,
    7,
    6,
    6,
    5,
    5
]

SHORT_RESULTS = [
    119_060_324,
    193_690_690,
    178_633_661,
    706_045_033,
    706_045_033,
    89_941_194,
    164_075_551
]

# List of depths that you use when the '-f' argument is passed.
FULL_DEPTHS = [
    8,
    6,
    8,
    6,
    6,
    5,
    7
]

FULL_RESULTS = [
    84_998_978_956,
    8_031_647_685,
    3_009_794_393,
    706_045_033,
    706_045_033,
    89_941_194,
    287_188_994_746
]

def perform_test(fen, depth, result):
    # Get the list of moves that we will be searching.
    driver_cmd = f"position fen {fen}"
    driver_cmd += "\ngo perft 1\n"
    driver_cmd += "quit\n"
    move_count = 0

    # Start the subprocess and wait for it to complete.
    engine = subprocess.Popen([khess_path], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True)
    output = engine.communicate(input=driver_cmd, timeout=120)[0]

    # Loop through the output and parse it.
    for line in output.splitlines():
        matches = re.search(move_regex, line)
        if not matches:
            continue
        move_count += 1

    # Try the perft test.
    engine = pexpect.spawn(khess_path, encoding='utf-8')
    engine.sendline(f'position fen {fen}')
    engine.sendline(f'go perft {depth}')

    # Start the progress bar.
    progress = tqdm.tqdm(range(move_count), ascii=True)

    # Loop through the output and parse it.
    node_sum = 0
    while True:
        i = engine.expect([move_regex, end_regex], timeout=None)
        if i == 0:
            progress.update(1)
        else:
            node_sum = int(engine.match.group(1))
            progress.close()
            break

    # Exit the engine.
    engine.sendline('quit')

    if node_sum != result:
        print()
        print(f"fen: {fen}")
        print(f"expected: {result}")
        print(f"actual: {node_sum}")
        print("ERROR: NODE SUM DOES NOT MATCH")
        print()

    #for fen, depth, result in zip(POSITIONS, SHORT_DEPTHS, SHORT_RESULTS):
    #    print()
    #    print(f"{fen} | {depth} | {result}")
    #    perform_test(fen, depth, result)

for fen, depth, result in zip(POSITIONS, SHORT_RESULTS, SHORT_RESULTS):
    print()
    print(f"{fen} | {depth} | {result}")
    perform_test(fen, depth, result)

