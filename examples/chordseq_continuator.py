"""Constrained chord generation; --limit 0 opts into the full 2.5M-token corpus."""

import argparse
import gzip
from itertools import islice
from pathlib import Path

from ctor.sequence_model import SequenceModel


def open_chord_sequences():
    data_path = Path(__file__).resolve().parents[1] / 'data' / 'chord_sequences.txt'
    if data_path.exists():
        return open(data_path, 'r', encoding='utf-8')
    return gzip.open(f'{data_path}.gz', 'rt', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=256, help='Training phrases; 0 uses all 31,032')
    parser.add_argument('--samples', type=int, default=3)
    args = parser.parse_args()
    if args.limit < 0 or args.samples < 1:
        parser.error('limit must be non-negative and samples must be positive')
    model = SequenceModel(kmax=3)
    with open_chord_sequences() as stream:
        lines = stream if args.limit == 0 else islice(stream, args.limit)
        model.learn_sequences([chord.strip() for chord in line.split(';')[1:-1]] for line in lines)
    for length in (4, 8):
        constraints = {0: 'C', length - 1: 'C'}
        if length == 8:
            constraints[length // 2] = 'F#7'
        for _ in range(args.samples):
            sequence = model.sample_sequence(length, constraints=constraints)
            print(' '.join(sequence) if sequence is not None else 'No solution for these chord constraints.')


if __name__ == '__main__':
    main()
