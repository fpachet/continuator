import mido
import numpy as np

from ctor.continuator import Note
from midi_stuff.mini_muse import Realized_Chord


class Chorder:
    def __init__(self):
        self.chords = []

if __name__ == '__main__':
    chords = Realized_Chord.create_chords("../data/nice_chords.mid")
    print(len(chords))

