"""
Copyright (c) 2025 Ynosound.
All rights reserved.

See LICENSE file in the project root for full license information.
"""

import json
import os
import subprocess
import sys

import gradio as gr
import mido
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from ctor.phrase_listener import MidiPhraseListener
from ctor.continuator import Continuator2
from io import BytesIO
from PIL import Image

class Continuator_gradio:

    def __init__(self):
        self.continuator = Continuator2()
        self.listener = None
        input_ports, output_ports = self.list_midi_ports()
        if input_ports and output_ports:
            self.start_midi_listener(input_ports[0], output_ports[0])
        else:
            print("no input or output port available")

    def list_midi_ports(self):
        # this does not work, mido ports are somehow called only once with gradio
        # input_ports = mido.get_input_names()
        # output_ports = mido.get_output_names()
        # return input_ports, output_ports
        script_path = os.path.join(os.path.dirname(__file__), "midi_ports_poll.py")
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        ports = json.loads(result.stdout)
        print("STDOUT:", result.stdout)
        # print("STDERR:", result.stderr)
        return ports["inputs"], ports["outputs"]

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
                on_phrase_callback=self.create_continuation  # Hook
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
            self.continuator.learn_phrase(phrase, self.continuator.transpose)
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
        choices = self.continuator.get_phrase_titles()
        return gr.update(choices=choices, value=choices[-1] if choices else None, label=f"{len(choices)} phrases")

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
        return self.generate_pianoroll_image(phrase)

    def generate_pianoroll_image(self, notes, beat_resolution=16, figsize=(10, 6)):
        """
        Generates a piano roll image from a list of Note objects.

        Returns:
            A NumPy array (H x W x 3) suitable for gr.Image.
        """
        if not notes:
            return
        # Determine the total number of time steps
        end_times = [note.start_time + note.duration for note in notes]
        total_beats = max(end_times)
        total_time_steps = int(np.ceil(total_beats * beat_resolution))
        pianoroll = np.zeros((128, total_time_steps), dtype=int)

        for note in notes:
            pitch = note.pitch
            start_idx = int(note.start_time * beat_resolution)
            end_idx = int((note.start_time + note.duration) * beat_resolution)
            pianoroll[pitch, start_idx:end_idx] = 1

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(pianoroll[::-1], aspect='auto', cmap='Blues', interpolation='nearest')
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Pitch (MIDI)")
        ax.set_title("Piano Roll")
        ax.set_yticks(np.linspace(0, 127, 13))
        ax.set_yticklabels([int(127 - i) for i in np.linspace(0, 127, 13)])
        ax.set_xticks([])
        ax.grid(False)

        # Convert to NumPy image for gr.Image
        canvas = FigureCanvas(fig)
        buf = BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
        image_np = np.array(image)
        plt.close(fig)
        return image_np

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
        self.continuator.set_learn_input(choice == "Learn input")

    def set_transpose(self, choice):
        self.continuator.set_transpose(choice == "Transpose")

    def set_forget(self, choice):
        self.continuator.set_forget(choice == "Forget")

    def set_keep_last(self, choice):
        print("keep last " + str(choice))
        self.continuator.set_keep_last(choice)

    def open_midi_files(self, files):
        midi_files = [f.name for f in files if f.name.lower().endswith('.mid') or f.name.lower().endswith('.midi')]
        # print ("\n".join(midi_files) if midi_files else "No MIDI files found.")
        self.continuator.learn_files(midi_files, transposition=self.continuator.transpose)

    def clear_memory(self):
        self.continuator.clear_memory()

    def clear_last_phrase(self):
        self.continuator.clear_last_phrase()

    def set_generate_length(self, choice):
        self.continuator.generate_length = choice

    def generate_from_memory(self):
        generated_sequence = self.continuator.sample_sequence(length=self.continuator.generate_length, constraints=None)
        if generated_sequence is None:
            print("no sequence generated")
            return []
        sequence_to_render = generated_sequence[:]
        rendered_sequence = self.continuator.realize_vp_sequence(sequence_to_render)
        mido_sequence = self.continuator.create_mido_sequence(rendered_sequence)
        self.listener.play_phrase(mido_sequence)
        return rendered_sequence

    def save_generated_as_midi_file(self, sequence):
        if sequence is None:
            return
        midi_messages = self.continuator.create_mido_sequence(sequence)
        filename = f"generated_phrase.mid"
        self.write_messages_to_midi(midi_messages, filename)
        return filename

    # --- BUILD GRADIO UI ---

    def launch(self):
        input_ports, output_ports = self.list_midi_ports()
        with gr.Blocks() as demo:
            gr.Markdown("## 🎹 Continuator")
            with gr.Tabs():
                with gr.TabItem("Real time"):
                    with gr.Row():
                        refresh_button = gr.Button("🔄 Refresh MIDI Ports")
                        in_dropdown = gr.Dropdown(label="🎧 MIDI Input Port", choices=input_ports,
                                                  value=input_ports[0] if isinstance(input_ports, list) and input_ports else None)
                        out_dropdown = gr.Dropdown(label="🔈 MIDI Output Port", choices=output_ports,
                                                   value=output_ports[0] if isinstance(output_ports, list) and output_ports else None)
                    with gr.Row():
                        start_button = gr.Button("▶️ Start Listening")
                        stop_button = gr.Button("⏹️ Stop Listening")
                        status_box = gr.Textbox(label="Status", lines=2)
                    refresh_button.click(fn=self.refresh_ports, outputs=[in_dropdown, out_dropdown])
                    start_button.click(fn=self.start_midi_listener, inputs=[in_dropdown, out_dropdown],
                                       outputs=status_box)
                    stop_button.click(fn=self.stop_midi_listener, outputs=status_box)
                    in_dropdown.change(fn=self.apply_input_port_change, inputs=in_dropdown, outputs=status_box)
                    out_dropdown.change(fn=self.apply_output_port_change, inputs=out_dropdown, outputs=status_box)
                    gr.Markdown("## 🧠 Memory")
                    with gr.Row():
                        phrase_selector = gr.Dropdown(label="🎵 Captured Phrases", choices=[], interactive=True,
                                                      container=True, scale=2)
                        refresh_phrase_list = gr.Button("📋 Refresh List")
                        save_button = gr.Button("💾 Save Phrase as MIDI")
                        clear_memory_button = gr.Button("🧽 Clear memory")
                        clear_last_phrase_button = gr.Button("↩️ Forget last phrase")
                        download_file = gr.File(label="⬇️ Download MIDI File")
                    save_button.click(
                        fn=self.save_selected_phrase,
                        inputs=phrase_selector,
                        outputs=download_file
                    )
                    clear_memory_button.click(
                        fn=self.clear_memory,
                        outputs=phrase_selector).then(
                        fn=self.update_phrase_dropdown,
                        outputs=phrase_selector
                    )
                    clear_last_phrase_button.click(
                        fn=self.clear_last_phrase,
                        outputs=phrase_selector).then(
                        fn=self.update_phrase_dropdown,
                        outputs=phrase_selector
                    )
                    refresh_phrase_list.click(fn=self.update_phrase_dropdown, outputs=phrase_selector)
                    phrase_output = gr.Image(label="🎹 Piano Roll", type="pil")
                    phrase_selector.change(fn=self.show_phrase_as_piano_roll, inputs=phrase_selector,
                                           outputs=phrase_output)
                with gr.TabItem("Midi files"):
                    file_input = gr.File(file_types=[".mid", ".midi"], label="Select MIDI file(s)",
                                         file_count="multiple")
                    load_button = gr.Button("🔄 Load MIDI files")
                    generate_button = gr.Button("🪄 Generate")
                    sequence_length_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1,
                                                 label="Sequence length")
                    generated_phrase_output = gr.Image(label="🎹 Piano Roll", type="pil")
                    midi_download_output = gr.File(label="⬇️ Download MIDI")

                    sequence_length_slider.change(fn=self.set_generate_length, inputs=[sequence_length_slider])
                    load_button.click(fn=self.open_midi_files, inputs=file_input)
                    generated_sequence_state = gr.State()
                generate_button.click(
                    fn=self.generate_from_memory,
                    outputs=generated_sequence_state
                ).then(
                    fn=self.generate_pianoroll_image,
                    inputs=generated_sequence_state,
                    outputs=generated_phrase_output
                ).then(
                    fn=self.save_generated_as_midi_file,
                    inputs=generated_sequence_state,
                    outputs=midi_download_output
                )
                with gr.TabItem("Parameters"):
                    learn_choice = gr.Radio(choices=["Learn input", "Don't learn input"], label="Learn mode",
                                            value="Learn input")
                    learn_choice.change(fn=self.set_learn_input, inputs=learn_choice)
                    transpose_choice = gr.Radio(choices=["Transpose", "Don't transpose"], label="Transpose",
                                                value="Don't transpose")
                    transpose_choice.change(fn=self.set_transpose, inputs=transpose_choice)
                    forget_choice = gr.Radio(choices=["Don't forget", "Forget"], label="Forget", value="Don't forget")
                    forget_choice.change(fn=self.set_forget, inputs=forget_choice)
                    keep_last_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1,
                                                 label="Keep only N last inputs")
                    keep_last_slider.change(fn=self.set_keep_last, inputs=[keep_last_slider])
        demo.launch()


# --- LAUNCH ---
if __name__ == "__main__":
    Continuator_gradio().launch()
