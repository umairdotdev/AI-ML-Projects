# Hidden Markov Coin Project

This project implements a Hidden Markov Model (HMM) to analyze sequences of coin toss outcomes and determine the most likely sequence of coin choices. The program uses a combination of transition and emission probabilities to compute these results.

## Files in the Repository

- **`chain-1.txt` and `chain-2.txt`**: Probability transition matrices representing Alice's tendencies for choosing coins.
- **`emit-1.txt` and `emit-2.txt`**: Emission probabilities for the coins showing Heads (`H`) and Tails (`T`).
- **`obs-1.txt` and `obs-2.txt`**: Sequences of observed coin toss outcomes (`H` as `1` and `T` as `0`).
- **`HiddenMarkovCoin.py`**: The Python implementation of the HMM analysis tool.

## How to Use

### Prerequisites
- Python 3.x installed
- Required Python libraries: `numpy`, `pandas`

### Running the Program
1. Place the input files (`chain-*.txt`, `emit-*.txt`, `obs-*.txt`) in the same directory as `HiddenMarkovCoin.py`.
2. Run the script by executing:
   ```bash
   python HiddenMarkovCoin.py
   ```
3. Follow the on-screen prompts to specify:
   - The chain file (e.g., `chain-1.txt`).
   - The emission file (e.g., `emit-1.txt`).
   - The observation file (e.g., `obs-1.txt`).
   - The output file name for the results.

### Outputs
The program outputs:
1. **Most Likely Sequence**: A sequence of coin choices corresponding to the observed outcomes.
2. **Probability of Observation Sequence**: The computed probability of the given observation sequence.
3. **Most Likely States**: The most probable state at any given step in the observation sequence.

## Example
Given `chain-1.txt` and `emit-1.txt` as inputs with the observation file `obs-1.txt`, the program will compute the following:
- Most likely sequence of coins.
- Probabilities of each sequence and their states.

## Extensions
The program also includes extended functionality:
1. Handling more general output alphabets.
2. Calculating the probability of an observation sequence.
3. Computing the most likely state at any step in the sequence.

## Background
This project is based on the concept where Alice uses two biased coins (X and Y) with distinct probabilities for Heads/Tails. Bob observes the outcomes but not which coin is tossed. The Hidden Markov Model helps deduce the most likely sequence of coin choices Alice made, given the observed results.
