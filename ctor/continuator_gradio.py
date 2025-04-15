import gradio as gr
import mido
import threading

from ctor.phrase_listener import MidiPhraseListener
from ctor.continuator import Continuator2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
from PIL import Image

class Continuator_gradio:
    listener = None
    continuator = Continuator2()

    def list_midi_ports(self):
        input_ports = mido.get_input_names()
        output_ports = mido.get_output_names()
        return input_ports, output_ports

    def refresh_ports(self):
        input_ports, output_ports = self.list_midi_ports()
        return (
            gr.update(choices=input_ports, value=input_ports[0] if input_ports else None),
            gr.update(choices=output_ports, value=output_ports[0] if output_ports else None)
        )

    # --- MIDI LISTENER CONTROL ---
    def start_midi_listener(self, input_port, output_port):
        if self.listener is not None:
            return "⚠️ Listener already running."
        try:
            self.listener = MidiPhraseListener(
                input_port_name=input_port,
                output_port_name=output_port,
                on_phrase_callback=self.create_continuation # Hook
            )
            threading.Thread(target=self.listener.start, daemon=True).start()
            return f"✅ Listening on:\nIN: {input_port}\nOUT: {output_port}"
        except Exception as e:
            listener = None
            return f"❌ Error: {e}"

# callback function called when a phrase is detected by the phrase_listenr
    def create_continuation(self, mido_sequence):
        # self.write_messages_to_midi(mido_sequence, 'midi_sequence.mid')
        phrase = self.continuator.get_phrase_from_mido(mido_sequence)
        if self.continuator.get_learn_input():
            self.continuator.learn_phrase(phrase, False)
        constraints = {}
        # constraints[0] = self.continuator.get_vp_for_pitch(62)
        constraints[len(phrase)] = self.continuator.get_end_vp()
        generated_sequence = self.continuator.sample_sequence(length=len(phrase) + 1, constraints=constraints)
        if generated_sequence is None:
            print("no solution gradio")
            return
        sequence_to_render = generated_sequence[:-1]
        rendered_sequence = self.continuator.realize_vp_sequence(sequence_to_render)
        mido_sequence = self.continuator.create_mido_sequence(rendered_sequence)
        self.listener.play_phrase(mido_sequence)

    def write_messages_to_midi(self, messages, filename="output.mid", ticks_per_beat=480):
        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        for msg in messages:
            msg.time = mido.second2tick(msg.time, ticks_per_beat, 500000)
            track.append(msg)
        mid.save(filename)
        print(f"✅ MIDI file saved as {filename}")

    def stop_midi_listener(self):
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
            return "🛑 Listener stopped."
        return "ℹ️ No listener is running."

    def apply_input_port_change(self, new_port_name):
        if self.listener is not None:
            try:
                self.listener.set_input_port(new_port_name)
                return f"🔈 Input port changed to: {new_port_name}"
            except Exception as e:
                return f"❌ Failed to change input port: {e}"
        else:
            return "ℹ️ Listener is not running."

    def apply_output_port_change(self, new_port_name):
        if self.listener is not None:
            try:
                self.listener.set_output_port(new_port_name)
                return f"🔈 Output port changed to: {new_port_name}"
            except Exception as e:
                return f"❌ Failed to change output port: {e}"
        else:
            return "ℹ️ Listener is not running."

    # --- PHRASE MENU AND DISPLAY ---

    def update_phrase_dropdown(self):
        choices =self.continuator.get_phrase_titles()
        return gr.update(choices=choices, value=choices[-1] if choices else None, label= f"{len(choices)} phrases")

    def show_phrase(self, index_label):
        if not index_label:
            return "No phrase selected."
        index = int(index_label.split()[0]) - 1
        phrase = self.continuator.get_phrase(index)
        return "\n".join(str(msg) for msg in phrase)

    def show_phrase_as_piano_roll(self, index_label):
        if not index_label:
            return None
        index = int(index_label.split()[0]) - 1
        phrase = self.continuator.get_phrase(index)
        # Draw piano roll and return as base64 image
        image_b64 = self.draw_piano_roll(phrase)
        return image_b64

    def draw_piano_roll(self, notes, note_range=(21, 108), beat_width_px=100, fig_height_px=300, min_fig_width_px=400):
        """
        Returns a PIL image. Auto-scales width based on phrase duration.
        - beat_width_px: pixels per beat horizontally
        - fig_height_px: total height in pixels
        """
        if not notes:
            return None
        min_note, max_note = note_range
        note_span = max_note - min_note + 1
        duration = max(note.start_time + note.duration for note in notes)
        # Calculate width and height in inches (for 100 dpi)
        fig_width_px = max(duration * beat_width_px, min_fig_width_px)
        fig_width = fig_width_px / 100
        fig_height = fig_height_px / 100
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_xlim(0, max(duration, 1.0))  # avoid zero-width view
        ax.set_ylim(min_note - 1, max_note + 1)
        ax.set_xlabel("Time (beats)")
        ax.set_ylabel("MIDI Pitch")
        ax.grid(True)
        for note in notes:
            rect = patches.Rectangle(
                (note.start_time, note.pitch),
                note.duration,
                0.8,
                linewidth=1,
                edgecolor='black',
                facecolor='skyblue'
            )
            ax.add_patch(rect)
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def save_selected_phrase(self, index_label):
        if not index_label:
            return None
        index = int(index_label.split()[0]) - 1
        phrase = self.continuator.get_phrase(index)
        midi_messages = self.continuator.create_mido_sequence(phrase)
        filename = f"phrase_{index + 1}.mid"
        self.write_messages_to_midi(midi_messages, filename)
        return filename

    def set_learn_input(self, choice):
        self.continuator.set_learn_input(choice=="Learn input")

    def open_midi_files(self, files):
        midi_files = [f.name for f in files if f.name.lower().endswith('.mid') or f.name.lower().endswith('.midi')]
        # print ("\n".join(midi_files) if midi_files else "No MIDI files found.")
        self.continuator.learn_files(midi_files)

    def clear_memory(self):
        self.continuator.clear_memory()

    def clear_last_phrase(self):
        self.continuator.clear_last_phrase()
    # --- BUILD GRADIO UI ---

    def launch(self):
        input_ports, output_ports = self.list_midi_ports()
        with gr.Blocks() as demo:
            gr.Markdown("## 🎹 Continuator")
            with gr.Tabs():
                with gr.TabItem("Real time"):
                    with gr.Row():
                        refresh_button = gr.Button("🔄 Refresh MIDI Ports")
                        in_dropdown = gr.Dropdown(label="🎧 MIDI Input Port", choices=input_ports, value=input_ports[0] if input_ports else None)
                        out_dropdown = gr.Dropdown(label="🔈 MIDI Output Port", choices=output_ports, value=output_ports[0] if output_ports else None)
                    with gr.Row():
                        start_button = gr.Button("▶️ Start Listening")
                        stop_button = gr.Button("⏹️ Stop Listening")
                        status_box = gr.Textbox(label="Status", lines=2)
                    refresh_button.click(fn=self.refresh_ports, outputs=[in_dropdown, out_dropdown])
                    start_button.click(fn=self.start_midi_listener, inputs=[in_dropdown, out_dropdown], outputs=status_box)
                    stop_button.click(fn=self.stop_midi_listener, outputs=status_box)
                    in_dropdown.change(fn=self.apply_input_port_change, inputs=in_dropdown, outputs=status_box)
                    out_dropdown.change(fn=self.apply_output_port_change, inputs=out_dropdown, outputs=status_box)
                    gr.Markdown("---")
                    with gr.Row():
                        phrase_selector = gr.Dropdown(label="🎵 Captured Phrases", choices=[], interactive=True, container=True, scale= 2)
                        refresh_phrase_list = gr.Button("📋 Refresh List")
                        save_button = gr.Button("💾 Save Phrase as MIDI")
                        clear_memory_button = gr.Button("💾 Clear memory")
                        clear_last_phrase_button = gr.Button("💾 Forget last phrase")
                        download_file = gr.File(label="⬇️ Download MIDI File")
                    save_button.click(
                        fn=self.save_selected_phrase,
                        inputs=phrase_selector,
                        outputs=download_file
                    )
                    clear_memory_button.click(
                        fn=self.clear_memory,
                    )
                    clear_last_phrase_button.click(
                        fn=self.clear_last_phrase,
                    )
                    refresh_phrase_list.click(fn=self.update_phrase_dropdown, outputs=phrase_selector)
                    phrase_output = gr.Image(label="🎹 Piano Roll", type="pil")
                    phrase_selector.change(fn=self.show_phrase_as_piano_roll, inputs=phrase_selector, outputs=phrase_output)
                with gr.TabItem("Midi files"):
                    file_input = gr.File(file_types=[".mid", ".midi"], label="Select MIDI file(s)",
                                         file_count="multiple")
                    load_button = gr.Button("🔄 Load MIDI files")
                    load_button.click(fn=self.open_midi_files, inputs=file_input)

                with gr.TabItem("Settings"):
                    choice = gr.Radio(choices=["Learn input", "Don't learn input"], label="Learn mode")
                    choice.change(fn=self.set_learn_input, inputs=choice)

        demo.launch()

# --- LAUNCH ---
if __name__ == "__main__":
    Continuator_gradio().launch()
