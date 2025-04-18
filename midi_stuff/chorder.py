import random

import mido
import threading
import time

from midi_stuff.mini_muse import Realized_Chord


class Chorder:

    def __init__(self, chords, inport=None, outport=None):
        self.chords = chords
        self.inport = mido.open_input(inport or mido.get_input_names()[0])
        self.outport = mido.open_output(outport or mido.get_output_names()[4])
        print(mido.get_output_names()[4])

        # Phrase tracking
        self.phrase = []  # list of (msg, delta_time_from_previous)
        self.pending_notes = []  # active notes
        self.last_event_time = time.time()
        self.last_msg_time = None
        self.active_notes = []

    def __str__(self):
        return f"Chorder with {len(self.chords)} chords"

    def __repr__(self):
        return f"Chorder with {len(self.chords)} chords"

    def stop_playing(self):
        self.stop_playing = True

    def set_input_port(self, port_name):
        with self.lock:
            try:
                if self.inport:
                    self.inport.close()
                self.inport = mido.open_input(port_name)
                print(f"🔄 Input port changed to: {port_name}")
            except Exception as e:
                print(f"❌ Failed to change output port: {e}")

    def set_output_port(self, port_name):
        with self.lock:
            try:
                if self.outport:
                    self.outport.close()
                self.outport = mido.open_output(port_name)
                print(f"🔄 Output port changed to: {port_name}")
            except Exception as e:
                print(f"❌ Failed to change output port: {e}")

    @staticmethod
    def list_ports():
        print("Available MIDI Input Ports:")
        for name in mido.get_input_names():
            print(f"  [IN]  {name}")
        print("\nAvailable MIDI Output Ports:")
        for name in mido.get_output_names():
            print(f"  [OUT] {name}")

    def play_chord(self, chord):
        for note in chord.notes:
            self.pending_notes.append(note.pitch)
            self.outport.send(
                mido.Message(
                    "note_on",
                    note=note.pitch,
                    velocity=note.velocity,
                    time=0,
                ))

    def on_note_on(self, note, velocity):
        print(f"Note ON: {note} (velocity={velocity})")
        candidates = [ch for ch in self.chords if (ch.get_highest_pitch() % 12) == (note % 12)]
        the_chord = random.choice(candidates)
        self.send_notes_off_except(note)
        self.play_chord(the_chord)
        self.active_notes.append(note)

    def send_notes_off_except(self, p):
        for i in range(0, 128):
            if i != p:
                self.outport.send(mido.Message(
                    "note_off",
                    note=i,
                    velocity=0,
                ))

    def on_note_off(self, note, velocity):
        print(f"Note OFF: {note}")
        for pitch in self.pending_notes:
            self.outport.send(mido.Message(
                "note_off",
                note=pitch,
                velocity=0,
            ))
        self.pending_notes=[]

    def run(self):
        print(f"🎧 Listening to MIDI input: {self.inport}")
        # with mido.open_input(self.inport) as port:
        for msg in self.inport:
            if msg.type == 'note_on' and msg.velocity > 0:
                self.on_note_on(msg.note, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                self.on_note_off(msg.note, msg.velocity)


if __name__ == '__main__':
    chorder = Chorder(Realized_Chord.create_chords("../data/nice_chords.mid", transpose=True))
    chorder.run()
