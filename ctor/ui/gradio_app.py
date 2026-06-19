"""
Copyright (c) 2025 Ynosound.
All rights reserved.

See LICENSE file in the project root for full license information.
"""

import json
import inspect
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import gradio as gr
import mido
import threading
import time
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
        self.generation_mode = "Fixed length"
        self.min_end_length = 4
        self.max_end_length = 64
        self.playback_thread = None
        self.playback_stop_event = threading.Event()
        self.input_ports, self.output_ports, self.initial_status = self.get_midi_ports_for_ui()

    def list_midi_ports(self):
        script_path = os.path.join(os.path.dirname(__file__), "midi_ports_poll.py")
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"midi_ports_poll.py failed with code {result.returncode}: {result.stderr}")

        if not result.stdout.strip():
            raise RuntimeError("midi_ports_poll.py returned empty output")

        ports = json.loads(result.stdout)
        return ports["inputs"], ports["outputs"]

    def get_midi_ports_for_ui(self):
        try:
            input_ports, output_ports = self.list_midi_ports()
        except Exception as e:
            return [], [], f"MIDI ports could not be listed: {e}"

        if input_ports and output_ports:
            return input_ports, output_ports, "Ready. Select MIDI ports, then start listening."
        if input_ports:
            return input_ports, [], "MIDI input found, but no MIDI output port is available."
        if output_ports:
            return [], output_ports, "MIDI output found, but no MIDI input port is available."
        return [], [], "No MIDI input or output ports found."

    def refresh_ports(self):
        input_ports, output_ports, status = self.get_midi_ports_for_ui()
        self.input_ports = input_ports
        self.output_ports = output_ports
        return (
            gr.update(choices=input_ports, value=input_ports[0] if input_ports else None),
            gr.update(choices=output_ports, value=output_ports[0] if output_ports else None),
            status,
            self.listener_summary(),
        )

    def listener_summary(self):
        return "Running" if self.listener is not None else "Stopped"

    def memory_summary(self):
        phrase_count = len(self.continuator.get_phrase_titles())
        if phrase_count == 0:
            return "No phrases learned yet."
        note_count = sum(len(self.continuator.get_phrase(index)) for index in range(phrase_count))
        return f"{phrase_count} phrase(s), {note_count} note(s) learned."

    def sync_ui_state(self, selected_phrase=None):
        return self.listener_summary(), self.memory_summary(), self.update_phrase_dropdown(selected_phrase)

    # --- MIDI LISTENER CONTROL ---
    def start_midi_listener(self, input_port, output_port):
        if self.listener is not None:
            return "Listener is already running."
        if not input_port or not output_port:
            return "Choose both a MIDI input and output port before starting."
        try:
            self.listener = MidiPhraseListener(
                input_port_name=input_port,
                output_port_name=output_port,
                on_phrase_callback=self.create_continuation  # Hook
            )
            threading.Thread(target=self.listener.start, daemon=True).start()
            return f"Listening on:\nIN: {input_port}\nOUT: {output_port}"
        except Exception as e:
            self.listener = None
            return f"Could not start listener: {e}"

    # callback function called when a phrase is detected by the phrase_listener
    def create_continuation(self, mido_sequence):
        # self.write_messages_to_midi(mido_sequence, 'midi_sequence.mid')
        phrase = self.continuator.get_phrase_from_mido(mido_sequence)
        if self.continuator.get_learn_input():
            self.continuator.learn_phrase(phrase, self.continuator.transpose)
        generated_sequence = self.generate_continuation_for_phrase(phrase)
        if generated_sequence is None:
            print("no solution gradio")
            return
        sequence_to_render = self.sequence_without_terminal_end(generated_sequence)
        rendered_sequence = self.continuator.realize_vp_sequence(sequence_to_render)
        mido_sequence = self.continuator.create_mido_sequence(rendered_sequence)
        if self.listener is not None:
            self.listener.play_phrase(mido_sequence)

    def generate_continuation_for_phrase(self, phrase):
        if self.generation_mode == "Until end":
            return self.continuator.continue_until_end(
                prefix=phrase,
                min_length=self.min_end_length,
                max_length=max(self.min_end_length, self.max_end_length),
            )
        constraints = {len(phrase): self.continuator.get_end_vp()}
        return self.continuator.continue_sequence(
            phrase,
            length=len(phrase) + 1,
            constraints=constraints,
        )

    def generate_from_memory_viewpoints(self):
        if self.generation_mode == "Until end":
            return self.continuator.continue_until_end(
                prefix=None,
                min_length=self.min_end_length,
                max_length=max(self.min_end_length, self.max_end_length),
            )
        return self.continuator.sample_sequence(
            length=self.continuator.generate_length,
            constraints=None,
        )

    def sequence_without_terminal_end(self, generated_sequence):
        if generated_sequence and generated_sequence[-1] == self.continuator.get_end_vp():
            return generated_sequence[:-1]
        return generated_sequence

    def write_messages_to_midi(self, messages, filename="output.mid", ticks_per_beat=480):
        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        for msg in messages:
            msg.time = mido.second2tick(msg.time, ticks_per_beat, 500000)
            track.append(msg)
        mid.save(filename)
        print(f"MIDI file saved as {filename}")

    def stop_midi_listener(self):
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
            return "Listener stopped."
        return "No listener is running."

    def stop_playback(self):
        self.playback_stop_event.set()
        if self.listener is not None:
            self.listener.stop_playing = True
        return "Playback stop requested."

    def apply_input_port_change(self, new_port_name):
        if self.listener is not None:
            try:
                self.listener.set_input_port(new_port_name)
                return f"Input port changed to: {new_port_name}"
            except Exception as e:
                return f"Failed to change input port: {e}"
        else:
            return "Listener is not running. The port will be used next time you start."

    def apply_output_port_change(self, new_port_name):
        if self.listener is not None:
            try:
                self.listener.set_output_port(new_port_name)
                return f"Output port changed to: {new_port_name}"
            except Exception as e:
                return f"Failed to change output port: {e}"
        else:
            return "Listener is not running. The port will be used next time you start."

    # --- PHRASE MENU AND DISPLAY ---

    def update_phrase_dropdown(self, selected_phrase=None):
        choices = self.continuator.get_phrase_titles()
        if not isinstance(choices, list):
            print("Warning: get_phrase_titles returned non-list:", choices)
            choices = []
        value = selected_phrase if selected_phrase in choices else choices[-1] if choices else None
        return gr.update(choices=choices, value=value, label=f"Captured phrases ({len(choices)})")

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

    def generate_pianoroll_image(self, notes, beat_resolution=16, figsize=(10, 4.5)):
        """
        Generates a piano roll image from a list of Note objects.

        Returns:
            A NumPy array (H x W x 3) suitable for gr.Image.
        """
        if not notes:
            return
        notes = list(notes)
        # Determine the total number of time steps
        end_times = [note.start_time + note.duration for note in notes]
        total_beats = max(end_times)
        total_time_steps = max(1, int(np.ceil(total_beats * beat_resolution)))
        min_pitch = max(0, min(note.pitch for note in notes) - 3)
        max_pitch = min(127, max(note.pitch for note in notes) + 3)
        pitch_count = max_pitch - min_pitch + 1
        pianoroll = np.zeros((pitch_count, total_time_steps), dtype=int)

        for note in notes:
            pitch = note.pitch
            start_idx = int(note.start_time * beat_resolution)
            end_idx = max(start_idx + 1, int((note.start_time + note.duration) * beat_resolution))
            pianoroll[pitch - min_pitch, start_idx:end_idx] = 1

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(pianoroll[::-1], aspect='auto', cmap='Blues', interpolation='nearest')
        ax.set_xlabel("Time")
        ax.set_ylabel("MIDI pitch")
        ax.set_title("Piano Roll")
        tick_count = min(8, pitch_count)
        tick_positions = np.linspace(0, pitch_count - 1, tick_count, dtype=int)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([max_pitch - pos for pos in tick_positions])
        ax.set_xticks([])
        ax.grid(False)
        fig.tight_layout()

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
        filename = os.path.join(tempfile.gettempdir(), f"continuator_phrase_{index + 1}.mid")
        self.continuator.save_midi(phrase, filename)
        return filename

    def save_selected_phrase_with_status(self, index_label):
        filename = self.save_selected_phrase(index_label)
        if filename is None:
            return None, "Select a captured phrase before saving."
        return filename, f"Saved selected phrase to {filename}."

    def select_phrase_outputs(self, index_label):
        if not index_label:
            return None, None, "Select a phrase to inspect it."
        return (
            self.show_phrase_as_piano_roll(index_label),
            self.save_selected_phrase(index_label),
            "Selected phrase is ready as a MIDI download.",
        )

    def sequence_to_timed_midi_messages(self, sequence):
        midi_file = self.continuator.create_mido_sequence(sequence)
        messages = []
        for msg in midi_file.tracks[0]:
            if msg.is_meta:
                continue
            msg = msg.copy(time=mido.tick2second(msg.time, midi_file.ticks_per_beat, 500000))
            messages.append(msg)
        return messages

    def _sleep_with_stop(self, seconds):
        deadline = time.time() + max(0, seconds)
        while time.time() < deadline:
            if self.playback_stop_event.is_set():
                return False
            time.sleep(min(0.02, deadline - time.time()))
        return not self.playback_stop_event.is_set()

    @staticmethod
    def _send_all_notes_off(outport):
        for note in range(128):
            outport.send(mido.Message("note_off", note=note, velocity=0))

    def _play_messages_on_output_port(self, messages, output_port):
        outport = None
        try:
            outport = mido.open_output(output_port)
            for msg in messages:
                if not self._sleep_with_stop(msg.time):
                    break
                outport.send(msg)
        finally:
            if outport is not None:
                self._send_all_notes_off(outport)
                outport.close()

    def play_selected_phrase(self, index_label, output_port):
        if not index_label:
            return "Select a captured phrase before playback."
        if not output_port and self.listener is None:
            return "Choose a MIDI output port before playback."

        self.stop_playback()
        self.playback_stop_event.clear()
        index = int(index_label.split()[0]) - 1
        phrase = self.continuator.get_phrase(index)
        messages = self.sequence_to_timed_midi_messages(phrase)

        if self.listener is not None:
            self.playback_thread = threading.Thread(target=self.listener.play_phrase, args=(messages,), daemon=True)
        else:
            self.playback_thread = threading.Thread(
                target=self._play_messages_on_output_port,
                args=(messages, output_port),
                daemon=True,
            )
        self.playback_thread.start()
        return f"Playing selected phrase ({len(phrase)} note(s))."

    def set_learn_input(self, choice):
        self.continuator.set_learn_input(choice == "Learn input")
        return f"Learn mode set to: {choice}."

    def set_decay_mode(self, choice):
        self.continuator.set_decay_mode(choice)
        return f"Decay mode set to: {choice}."

    def set_transpose(self, choice):
        self.continuator.set_transpose(choice == "Transpose")
        return f"Transpose mode set to: {choice}."

    def set_forget(self, choice):
        self.continuator.set_forget(choice == "Forget")
        return f"Forget policy set to: {choice}."

    def set_keep_last(self, choice):
        self.continuator.set_keep_last(int(choice))
        return f"When forgetting is enabled, keep the last {int(choice)} input(s)."

    def open_midi_files(self, files):
        if not files:
            return "Select at least one MIDI file to load.", self.memory_summary(), self.update_phrase_dropdown()
        midi_files = [f.name for f in files if f.name.lower().endswith('.mid') or f.name.lower().endswith('.midi')]
        if not midi_files:
            return "No .mid or .midi files were selected.", self.memory_summary(), self.update_phrase_dropdown()
        try:
            before = len(self.continuator.get_phrase_titles())
            self.continuator.learn_files(midi_files, transposition=self.continuator.transpose)
            learned = len(self.continuator.get_phrase_titles()) - before
        except Exception as e:
            return f"Could not load MIDI files: {e}", self.memory_summary(), self.update_phrase_dropdown()
        return f"Loaded {len(midi_files)} MIDI file(s), learned {learned} phrase(s).", self.memory_summary(), self.update_phrase_dropdown()

    def clear_memory(self):
        self.continuator.clear_memory()
        return self.update_phrase_dropdown()

    def clear_memory_with_confirmation(self, confirm):
        if not confirm:
            return "Check Confirm clear memory before clearing.", self.update_phrase_dropdown(), None, self.memory_summary()
        self.continuator.clear_memory()
        return "Memory cleared.", self.update_phrase_dropdown(), None, self.memory_summary()

    def clear_last_phrase(self):
        self.continuator.clear_last_phrase()
        return self.update_phrase_dropdown()

    def clear_last_phrase_with_status(self):
        before = len(self.continuator.get_phrase_titles())
        self.continuator.clear_last_phrase()
        after = len(self.continuator.get_phrase_titles())
        if before == after:
            status = "No phrase to forget."
        else:
            status = "Forgot the most recent phrase."
        return status, self.update_phrase_dropdown(), None, self.memory_summary()

    def set_generate_length(self, choice):
        self.continuator.generate_length = int(choice)
        return f"Fixed generation length set to {int(choice)}."

    def set_generation_mode(self, choice):
        self.generation_mode = choice
        return f"Generation mode set to: {choice}."

    def update_generation_mode_inputs(self, choice):
        self.set_generation_mode(choice)
        fixed = choice == "Fixed length"
        return (
            gr.update(visible=fixed),
            gr.update(visible=not fixed),
            gr.update(visible=not fixed),
            f"Generation mode set to: {choice}.",
        )

    def set_min_end_length(self, choice):
        self.min_end_length = int(choice)
        return f"Minimum end length set to {self.min_end_length}."

    def set_max_end_length(self, choice):
        self.max_end_length = int(choice)
        return f"Maximum end length set to {self.max_end_length}."

    def generate_from_memory(self):
        generated_sequence = self.generate_from_memory_viewpoints()
        if generated_sequence is None:
            print("no sequence generated")
            return []
        sequence_to_render = self.sequence_without_terminal_end(generated_sequence)
        rendered_sequence = self.continuator.realize_vp_sequence(sequence_to_render)
        mido_sequence = self.continuator.create_mido_sequence(rendered_sequence)
        if self.listener is not None:
            self.listener.play_phrase(mido_sequence)
        return rendered_sequence

    def generate_from_memory_with_status(self):
        try:
            sequence = self.generate_from_memory()
        except Exception as e:
            return None, f"Generation failed: {e}"
        if not sequence:
            return None, "No sequence could be generated. Load or play more material, or loosen the generation constraints."
        listener_note = " It was also sent to the MIDI output." if self.listener is not None else ""
        return sequence, f"Generated {len(sequence)} note(s).{listener_note}"

    def save_generated_as_midi_file(self, sequence):
        if sequence is None:
            return
        filename = os.path.join(tempfile.gettempdir(), "continuator_generated_phrase.mid")
        self.continuator.save_midi(sequence, filename)
        return filename

    # --- BUILD GRADIO UI ---

    def launch(self, **launch_kwargs):
        css = """
        #app-title { margin-bottom: 0.25rem; }
        .compact-row button { min-width: 9rem; }
        """
        with gr.Blocks(title="Continuator") as demo:
            gr.Markdown("## Continuator", elem_id="app-title")
            with gr.Row():
                status_box = gr.Textbox(label="Status", value=self.initial_status, lines=3, interactive=False, scale=3)
                listener_box = gr.Textbox(label="Listener", value=self.listener_summary(), interactive=False, scale=1)
                memory_box = gr.Textbox(label="Memory", value=self.memory_summary(), interactive=False, scale=2)
            with gr.Row(elem_classes=["compact-row"]):
                top_start_button = gr.Button("Start listening", variant="primary")
                top_stop_button = gr.Button("Stop listening")
            with gr.Tabs():
                with gr.TabItem("Perform"):
                    with gr.Row():
                        in_dropdown = gr.Dropdown(label="MIDI input port", choices=self.input_ports,
                                                  value=self.input_ports[0] if self.input_ports else None, scale=2)
                        out_dropdown = gr.Dropdown(label="MIDI output port", choices=self.output_ports,
                                                   value=self.output_ports[0] if self.output_ports else None, scale=2)
                        refresh_button = gr.Button("Refresh ports", scale=1)
                    with gr.Row(elem_classes=["compact-row"]):
                        start_button = gr.Button("Start listening", variant="primary")
                        stop_button = gr.Button("Stop listening")
                    refresh_button.click(fn=self.refresh_ports, outputs=[in_dropdown, out_dropdown, status_box, listener_box])
                    start_button.click(fn=self.start_midi_listener, inputs=[in_dropdown, out_dropdown],
                                       outputs=status_box).then(fn=self.listener_summary, outputs=listener_box)
                    stop_button.click(fn=self.stop_midi_listener, outputs=status_box).then(fn=self.listener_summary, outputs=listener_box)
                    in_dropdown.change(fn=self.apply_input_port_change, inputs=in_dropdown, outputs=status_box)
                    out_dropdown.change(fn=self.apply_output_port_change, inputs=out_dropdown, outputs=status_box)
                    gr.Markdown("### Live memory")
                    with gr.Row():
                        phrase_selector = gr.Dropdown(label="Captured phrases (0)", choices=[], interactive=True,
                                                      container=True, scale=2)
                        save_button = gr.Button("Save selected")
                        clear_last_phrase_button = gr.Button("Forget last")
                    download_file = gr.File(label="Selected phrase MIDI")
                    save_button.click(
                        fn=self.save_selected_phrase_with_status,
                        inputs=phrase_selector,
                        outputs=[download_file, status_box]
                    ).then(fn=self.memory_summary, outputs=memory_box)
                    clear_last_phrase_button.click(
                        fn=self.clear_last_phrase_with_status,
                        outputs=[status_box, phrase_selector, download_file, memory_box])
                    with gr.Group():
                        with gr.Row(elem_classes=["compact-row"]):
                            gr.Markdown("### Selected phrase piano roll")
                            play_selected_button = gr.Button("Play selected", variant="primary", scale=1)
                            stop_playback_button = gr.Button("Stop playback", scale=1)
                        phrase_output = gr.Image(label="Piano roll", type="pil")
                    phrase_selector.change(
                        fn=self.select_phrase_outputs,
                        inputs=phrase_selector,
                        outputs=[phrase_output, download_file, status_box],
                    ).then(fn=self.memory_summary, outputs=memory_box)
                    play_selected_button.click(
                        fn=self.play_selected_phrase,
                        inputs=[phrase_selector, out_dropdown],
                        outputs=status_box,
                    )
                    stop_playback_button.click(fn=self.stop_playback, outputs=status_box)
                with gr.TabItem("Train"):
                    file_input = gr.File(file_types=[".mid", ".midi"], label="MIDI file(s)",
                                         file_count="multiple")
                    load_button = gr.Button("Load MIDI files", variant="primary")
                    load_button.click(fn=self.open_midi_files, inputs=file_input, outputs=[status_box, memory_box, phrase_selector])
                    gr.Markdown("### Memory management")
                    with gr.Row():
                        confirm_clear_memory = gr.Checkbox(label="Confirm clear memory", value=False)
                        clear_memory_button = gr.Button("Clear memory")
                    clear_memory_button.click(
                        fn=self.clear_memory_with_confirmation,
                        inputs=confirm_clear_memory,
                        outputs=[status_box, phrase_selector, phrase_output, memory_box],
                    )
                with gr.TabItem("Generate"):
                    generation_mode_choice = gr.Radio(
                        choices=["Fixed length", "Until end"],
                        label="Generation mode",
                        value="Fixed length"
                    )
                    sequence_length_slider = gr.Slider(minimum=1, maximum=100, step=1, value=self.continuator.generate_length,
                                                 label="Fixed sequence length")
                    min_end_length_slider = gr.Slider(minimum=1, maximum=200, step=1, value=self.min_end_length,
                                                      label="Minimum length until end", visible=False)
                    max_end_length_slider = gr.Slider(minimum=1, maximum=500, step=1, value=self.max_end_length,
                                                      label="Maximum length until end", visible=False)
                    generate_button = gr.Button("Generate", variant="primary")
                    generated_phrase_output = gr.Image(label="Generated piano roll", type="pil")
                    midi_download_output = gr.File(label="Generated MIDI")
                    generated_sequence_state = gr.State()
                    generation_mode_choice.change(
                        fn=self.update_generation_mode_inputs,
                        inputs=generation_mode_choice,
                        outputs=[sequence_length_slider, min_end_length_slider, max_end_length_slider, status_box],
                    )
                    sequence_length_slider.change(fn=self.set_generate_length, inputs=[sequence_length_slider], outputs=status_box)
                    min_end_length_slider.change(fn=self.set_min_end_length, inputs=[min_end_length_slider], outputs=status_box)
                    max_end_length_slider.change(fn=self.set_max_end_length, inputs=[max_end_length_slider], outputs=status_box)
                generate_button.click(
                    fn=self.generate_from_memory_with_status,
                    outputs=[generated_sequence_state, status_box]
                ).then(
                    fn=self.generate_pianoroll_image,
                    inputs=generated_sequence_state,
                    outputs=generated_phrase_output
                ).then(
                    fn=self.save_generated_as_midi_file,
                    inputs=generated_sequence_state,
                    outputs=midi_download_output
                )
                with gr.TabItem("Advanced"):
                    learn_choice = gr.Radio(choices=["Learn input", "Don't learn input"], label="Learn mode",
                                            value="Learn input")
                    learn_choice.change(fn=self.set_learn_input, inputs=learn_choice, outputs=status_box)
                    transpose_choice = gr.Radio(choices=["Transpose", "Don't transpose"], label="Transpose",
                                                value="Don't transpose")
                    transpose_choice.change(fn=self.set_transpose, inputs=transpose_choice, outputs=status_box)
                    forget_choice = gr.Radio(choices=["Don't forget", "Forget"], label="Forget", value="Don't forget")
                    forget_choice.change(fn=self.set_forget, inputs=forget_choice, outputs=status_box)
                    keep_last_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1,
                                                 label="Keep only N last inputs")
                    keep_last_slider.change(fn=self.set_keep_last, inputs=[keep_last_slider], outputs=status_box)
                    decay_mode_choice = gr.Radio(choices=["full", "late", "middle", "early"], label="Decay Mode",
                                            value="full")
                    decay_mode_choice.change(fn=self.set_decay_mode, inputs=decay_mode_choice, outputs=status_box)
            top_start_button.click(fn=self.start_midi_listener, inputs=[in_dropdown, out_dropdown],
                                   outputs=status_box).then(fn=self.listener_summary, outputs=listener_box)
            top_stop_button.click(fn=self.stop_midi_listener, outputs=status_box).then(
                fn=self.listener_summary, outputs=listener_box)
            if hasattr(gr, "Timer"):
                timer = gr.Timer(value=2)
                timer.tick(fn=self.sync_ui_state, inputs=phrase_selector, outputs=[listener_box, memory_box, phrase_selector])

        launch_options = dict(launch_kwargs)
        launch_parameters = inspect.signature(demo.launch).parameters
        if "theme" in launch_parameters:
            launch_options.setdefault("theme", gr.themes.Soft())
        if "css" in launch_parameters:
            launch_options.setdefault("css", css)
        return demo.launch(**launch_options)


# --- LAUNCH ---
if __name__ == "__main__":
    Continuator_gradio().launch()
