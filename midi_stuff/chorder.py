import mido
import numpy as np

from ctor.continuator import Note
from midi_stuff.mini_muse import Realized_Chord


class Chorder:
    def extract_notes(self, midi_file):

        mid = mido.MidiFile(midi_file)
        resolution = mid.ticks_per_beat
        notes = []
        pending_notes = np.empty(128, dtype=object)
        pending_start_times = np.zeros(128)
        current_time = 0
        for track in mid.tracks:
            for msg in track:
                current_time += 2 * mido.tick2second(msg.time, ticks_per_beat=resolution, tempo=500000)  # in beats
                if msg.type == "note_on" and msg.velocity > 0:
                    new_note = Note(msg.note, msg.velocity, 0)
                    notes.append(new_note)  # Store MIDI note number
                    pending_notes[msg.note] = new_note
                    pending_start_times[msg.note] = current_time
                    new_note.set_start_time(current_time)
                    new_note.set_duration(1)  # beat
                if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    if pending_notes[msg.note] is None:
                        print("found 0 velocity note, skipping it")
                        continue
                    pending_note = pending_notes[msg.note]
                    duration = current_time - pending_start_times[msg.note]
                    pending_note.set_duration(duration)
                    pending_notes[msg.note] = None
                    pending_start_times[msg.note] = 0
        # sets the note status w/r their neighbors
        return np.array(notes)

    def create_chords(self, midi_file):
        notes = self.extract_notes(midi_file)
        chords = []
        current_chord = Realized_Chord([])
        max_time_current_chord = 0
        for note in notes:
            if note.start_time > max_time_current_chord:
                if current_chord.get_nb_notes() > 0:
                    chords.append(current_chord)
                current_chord = Realized_Chord([note])
                max_time_current_chord = max(max_time_current_chord, note.get_end_time())
            else:
                current_chord.append(note)
                max_time_current_chord = max(max_time_current_chord, note.get_end_time())
        if current_chord.get_nb_notes() > 0:
            chords.append(current_chord)
        return chords


if __name__ == '__main__':
    chords = Chorder().create_chords("../data/nice_chords.mid")
    print(len(chords))

