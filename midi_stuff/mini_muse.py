import mido
import numpy as np


class Note:
    def __init__(self, pitch, velocity, duration, start_time=0):
        self.pitch = pitch
        self.velocity = velocity
        # the duration in the original sequence in beats, assuming 120bpm
        self.duration = duration
        # the start time in the original sequence in beats, assuming 120bpm
        self.start_time = start_time
        # time between start and the start of preceding note, always > 0
        self.preceding_start_delta = 0  # in beats, assuming 120bpm
        # time between start and the end of preceding note. Negative if overlaps with preceding
        self.preceding_end_delta = 0  # in beats, assuming 120bpm
        # time between start of next note and end. Negative if overlaps with next
        self.next_start_delta = 0  # in beats, assuming 120bpm
        # time between end of next note and end
        self.next_end_delta = 0  # in beats, assuming 120bpm

    def __str__(self):
        return f"{self.pitch} @ [{self.start_time}, {self.get_end_time()}]"

    def __repr__(self):
        return f"{self.pitch} @ [{self.start_time}, {self.get_end_time()}]"

    def set_duration(self, d):
        self.duration = d

    def set_start_time(self, t):
        self.start_time = t

    def overlaps_left(self):
        # if overlap is greater than half the duration
        return self.preceding_end_delta < 0

    def overlaps_right(self):
        # if overlap is greater than half the duration
        return self.next_start_delta < 0

    def transpose(self, t):
        note = self.copy()
        note.pitch = self.pitch + t
        return note

    def copy(self):
        new_note = Note(self.pitch, self.velocity, self.duration, start_time=self.start_time)
        new_note.preceding_start_delta = self.preceding_start_delta
        new_note.preceding_end_delta = self.preceding_end_delta
        new_note.next_start_delta = self.next_start_delta
        new_note.next_end_delta = self.next_end_delta
        return new_note

    def get_end_time(self):
        return self.start_time + self.duration

    def is_compatible_with(self, note):
        # returns true if self and note have same polyphonic status
        return self.overlaps_right() == note.overlaps_left()

    def get_status_right(self):
        if self.next_end_delta <= 0:
            return 'inside'
        if self.next_start_delta < 0:
            return 'overlaps'
        return 'after'

    def get_status_left(self):
        if self.preceding_end_delta >= 0:
            return 'before'
        if abs(self.preceding_end_delta) < self.duration:
            return 'overlaps'
        return 'contains'

    def is_similar_realization(self, note):
        if self.pitch != note.pitch:
            return False
        if self.velocity != note.velocity:
            return False
        if self.duration != note.duration:
            return False
        if self.preceding_end_delta != note.preceding_end_delta:
            return False
        if self.preceding_start_delta != note.preceding_start_delta:
            return False
        if self.next_start_delta != note.next_start_delta:
            return False
        if self.next_end_delta != note.next_end_delta:
            return False
        return True


class Realized_Chord:
    # is a list of Note
    def __init__(self, notes):
        self.notes = notes

    def append(self, note):
        self.notes.append(note)

    def get_nb_notes(self):
        return len(self.notes)

    @classmethod
    def extract_notes(cls, midi_file):
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
    @classmethod
    def create_chords(cls, midi_file):
        notes = cls.extract_notes(midi_file)
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
