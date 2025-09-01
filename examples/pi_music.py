from muses.base.temporals import TemporalCollection

import random

if __name__ == '__main__':

    decimals = "314159265358979323846264338327950288419716939937510"
    m = TemporalCollection()
    start = 0
    durations = [.25, .5, 1]
    for n in decimals:
        dur = random.choice(durations)
        m.add_note(60 + int(n), start, dur, 70)
        start += dur
    m.save_midi("pi_music.mid")
