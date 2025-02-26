import dawdreamer as daw
from scipy.io import wavfile
SAMPLE_RATE = 44100
engine = daw.RenderEngine(SAMPLE_RATE, 512)  # 512 block size
faust_processor = engine.make_faust_processor("faust")
faust_processor.set_dsp_string(
    """
    declare name "MySine";
    freq = hslider("freq", 440, 0, 20000, 0);
    gain = hslider("vol[unit:dB]", 0, -120, 20, 0) : ba.db2linear;
    process = freq : os.osc : _*gain <: si.bus(2);
    """
    )
print(faust_processor.get_parameters_description())
engine.load_graph([
                   (faust_processor, [])
])
faust_processor.set_parameter("/MySine/freq", 440.)  # 440 Hz
faust_processor.set_parameter("/MySine/vol", -6.)  # -6 dB volume

engine.set_bpm(120.)
engine.render(4., beats=True)  # render 4 beats.
audio = engine.get_audio()  # shaped (2, N samples)
wavfile.write('sine_demo.wav', SAMPLE_RATE, audio.transpose())

# Change settings and re-render
faust_processor.set_parameter("/MySine/freq", 880.)  # 880 Hz
engine.render(4., beats=True)
# and so on...