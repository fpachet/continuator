import mido
import threading
import time


class MidiPhraseListener:
    def __init__(self, input_port_name=None, output_port_name=None, phrase_timeout=1.0, on_phrase_callback=None):
        # Ports
        self.inport = mido.open_input(input_port_name or mido.get_input_names()[0])
        self.outport = mido.open_output(output_port_name or mido.get_output_names()[0])

        # Phrase tracking
        self.phrase_timeout = phrase_timeout  # seconds of inactivity before phrase ends
        self.phrase = []  # list of (msg, delta_time_from_previous)
        self.pending_notes = set()  # active notes
        self.last_event_time = time.time()
        self.last_msg_time = None
        self.on_phrase_callback = on_phrase_callback

        # Threading
        self.lock = threading.Lock()
        self.running = False
        self.timer_thread = threading.Thread(target=self._check_phrase_end, daemon=True)
        self.stop_playing = False

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

    def start(self):
        self.running = True
        self.timer_thread.start()
        print("Listening for MIDI input...")
        try:
            for msg in self.inport:
                self._handle_message(msg)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.stop_playing = True
        self.running = False
        self.inport.close()
        self.outport.close()
        print("Stopped.")

    def _handle_message(self, msg):
        now = time.time()
        with self.lock:
            self.last_event_time = now
            # Compute delta from previous message
            if self.last_msg_time is None:
                delta = 0.0
            else:
                delta = now - self.last_msg_time
            self.last_msg_time = now
            # Track note state
            if msg.type == 'note_on' and msg.velocity > 0:
                self.stop_playing = True
                self.pending_notes.add((msg.channel, msg.note))
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                self.pending_notes.discard((msg.channel, msg.note))
            # Store message with delta
            self.phrase.append((msg, delta))

    def _check_phrase_end(self):
        while self.running:
            with self.lock:
                idle_time = time.time() - self.last_event_time
                if idle_time > self.phrase_timeout and not self.pending_notes and self.phrase:
                    phrase_copy = self.phrase[:]
                    self.phrase = []
                    self.last_msg_time = None
                    threading.Thread(target=self._on_phrase_complete, args=(phrase_copy,), daemon=True).start()
            time.sleep(0.05)

    def _on_phrase_complete(self, mido_sequence):
        print("\nPhrase complete. Playing back...\n")
        real_mido = []
        for msg, delta in mido_sequence:
            msg.time = delta
            real_mido.append(msg)
        # create proper sequence of messages
        # for msg, delta in phrase:
        #     time.sleep(msg.time)
        #     self.outport.send(msg)
        if self.on_phrase_callback:
            self.on_phrase_callback(real_mido)

    def play_phrase(self, mido_sequence):
        self.stop_playing = False
        pending_note_ons_played_sequence = []
        for msg in mido_sequence:
            if self.stop_playing:
                pending_notes_being_played = [pending[1] for pending in self.pending_notes]
                for i in range(128):
                    if i not in pending_notes_being_played:
                        self.outport.send(mido.Message(
                            "note_off",
                            note=i,
                            velocity=0,
                        ))
                return
            time.sleep(msg.time)
            if msg.type == "note-on":
                pending_note_ons_played_sequence.append(msg.note)
            if msg.type == "note-off":
                if msg.note in pending_note_ons_played_sequence:
                    pending_note_ons_played_sequence.remove(msg.note)
            self.outport.send(msg)


if __name__ == "__main__":
    MidiPhraseListener.list_ports()  # List available ports

    # Use default ports (or specify input_port_name / output_port_name)
    listener = MidiPhraseListener()

    listener.start()
