# N Puzzle A* IDA* Solver

This repository contains a Python implementation of a sliding puzzle solver using A* and IDA* algorithms.

## Files Included

1. **puzzle_solver.py**: The main Python script for solving sliding puzzles.
2. **test_input_x.txt**: Test cases for the puzzle solver.
   - `test_input_1.txt` to `test_input_5.txt`: Solvable puzzles.
   - `test_input_6.txt` to `test_input_10.txt`: Unsolvable puzzles.

## Usage

Run the script with a test input file as follows:

```bash
python puzzle_solver.py <input_file>
```

Example:

```bash
python puzzle_solver.py test_input_1.txt
```

## Input File Format

- The first line specifies the solving algorithm: `A*` or `IDA*`.
- The second line is the maximum heuristic cost (`M`).
- The third line specifies the size of the grid (`k`).
- The next `k` lines describe the initial puzzle state.
- The final `k` lines describe the goal puzzle state.

Example:

```
A*
50
3
1 2 3
4 _ 5
6 7 8
1 2 3
4 5 6
7 8 _
```

## Notes

- The program checks solvability before attempting to solve the puzzle.
- For unsolvable puzzles, the program outputs: `This puzzle is unsolvable.`
