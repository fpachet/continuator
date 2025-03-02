import pretty_midi
import numpy as np
import matplotlib.pyplot as plt


def plot_piano_roll(midi_path, fs=100):
    """
    Load a MIDI file with pretty_midi, generate a piano-roll, and plot it.

    Parameters:
    -----------
    midi_path : str
        Path to the MIDI file.
    fs : int
        Frames (samples) per second to use when constructing the piano roll.
        Higher values = finer temporal resolution but bigger arrays.
    """
    # Parse the MIDI file
    pm = pretty_midi.PrettyMIDI(midi_path)

    # Generate the piano roll (shape is [128 x T])
    # Each entry in the array is the velocity (0–127) of a note at that time step
    piano_roll = pm.get_piano_roll(fs=fs)

    # Plot using matplotlib
    plt.figure()
    plt.imshow(
        piano_roll,
        aspect='auto',
        origin='lower',
        interpolation='nearest'
    )
    plt.xlabel("Time (frames at fs = {})".format(fs))
    plt.ylabel("MIDI pitch")
    plt.title("PrettyMIDI Piano Roll")
    plt.show()


def notes_to_pretty_midi(notes, program=0):
    """
    Convert a list of note specifications into a pretty_midi.PrettyMIDI object.

    Parameters
    ----------
    notes : list of tuples
        Each tuple should be (pitch, start_time, end_time, velocity).
        - pitch: int (0–127)
        - start_time: float (time in seconds)
        - end_time: float (time in seconds)
        - velocity: int (0–127)
    program : int
        The MIDI program number (instrument). 0 is a piano by General MIDI standard.

    Returns
    -------
    pm : pretty_midi.PrettyMIDI
        A PrettyMIDI object containing all the notes in one instrument track.
    """
    # Create a PrettyMIDI object
    pm = pretty_midi.PrettyMIDI()

    # Create an Instrument instance for a specific program (sound)
    instrument = pretty_midi.Instrument(program=program)

    # For each note in the list, create a pretty_midi.Note object and add to the instrument
    for pitch, start, end, velocity in notes:
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=end
        )
        instrument.notes.append(note)

    # Add the instrument to the PrettyMIDI object
    pm.instruments.append(instrument)

    return pm

# Example usage:
plot_piano_roll('../../data/ctor2_output.mid', fs=100)
# We'll specify that it will have a tempo of 80bpm.
