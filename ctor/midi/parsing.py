"""Shared note pairing for delta-timed MIDI streams, without mutating messages."""

from collections import defaultdict, deque

from midi_stuff.mini_muse import Note


def parse_notes(messages, *, time_scale=1.0):
    """Return notes in onset order, pairing channel/pitch with FIFO note-offs.

    Times use the caller's units multiplied by time_scale. Unmatched note-offs
    and unfinished notes are ignored: an incomplete recording cannot supply a
    measured duration. Sustain controls do not extend the physical key duration.
    """
    pending = defaultdict(deque)
    completed = []
    absolute_time = 0
    onset_index = 0
    for message in messages:
        if message.time < 0:
            raise ValueError("MIDI delta times must be non-negative")
        absolute_time += message.time
        if message.type not in ("note_on", "note_off"):
            continue
        key = (message.channel, message.note)
        if message.type == "note_on" and message.velocity > 0:
            pending[key].append((onset_index, absolute_time,
                                 Note(message.note, message.velocity, 0, absolute_time * time_scale)))
            onset_index += 1
        elif pending[key]:
            index, onset, note = pending[key].popleft()
            note.duration = (absolute_time - onset) * time_scale
            completed.append((index, note))
    completed.sort(key=lambda entry: entry[0])
    return [note for _, note in completed]
