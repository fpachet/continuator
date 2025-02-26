import mido
import sounddevice as sd
import numpy as np
import soundfile as sf
import ctypes
import time

# Load the Faust-generated shared library (modify path if needed)
FAUST_LIB = "synth.so"  # Change to your compiled Faust shared library
faust = ctypes.CDLL(FAUST_LIB)

# Define Faust functions
faust.init.argtypes = [ctypes.c_int]
faust.init.restype = None
faust.setParamValue.argtypes = [ctypes.c_char_p, ctypes.c_float]
faust.setParamValue.restype = None
faust.compute.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
faust.compute.restype = None

# Initialize Faust synth
fs = 44100
faust.init(fs)

# MIDI to Frequency Conversion
def midi_to_freq(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))

# Load and play MIDI file
