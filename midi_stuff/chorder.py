import mido
import numpy as np

from midi_stuff.mini_muse import Realized_Chord


class Chorder:
    def __init__(self, chords):
        self.chords = chords

    def __str__(self):
        return f"Chorder with {len(self.chords)} chords"

    def __repr__(self):
        return f"Chorder with {len(self.chords)} chords"


if __name__ == '__main__':
    chorder = Chorder(Realized_Chord.create_chords("../data/nice_chords.mid"))
    print(chorder)
