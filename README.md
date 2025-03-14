# A Python implementation of a constrainable Continuator

A revival of th Continuator system, using a combination of variable-order Markov model and belief propagation to enforce positional constraints.

It is based on the following papers:
- Pachet, F. The Continuator: Musical Interaction with Style. Journal of New Music Research, 32(3):333-341, 2003
- Papadopoulos, A., Pachet, F., Roy, P. and Sakellariou, J. Exact Sampling for Regular and Markov Constraints with Belief Propagation. 21th Principles and Practice of Constraint Programming Conference (CP 2015), Cork (Ireland), 2015
- Pachet, F., Roy, P. and Barbieri, G. Finite-Length Markov Processes with Constraints. Proceedings of the 22nd International Joint Conference on Artificial Intelligence (IJCAI), pages 635-642, Barcelona, Spain, July 2011
- Roy, P. and Pachet, F. Enforcing Meter in Finite-Length Markov Sequences. 27th Conference on Artificial Intelligence (AAAI 2013), Bellevue, Washington (USA), June 2013

## Features

- Efficient implementation of variable-order markov model
- Combination with a viewpoint system that enables the generation of musically plausible material
- Combination with a belief propagation system to enforce positional constraints (that are retro propagated)
- many tricks here and there to maximize musical quality


### Dependencies

The project requires the following Python packages:
numpy~=2.2.3
mido~=1.2.10
scipy~=1.15.2
torch~=2.6.0

## Usage

```python
from core.max_entropy import MaxEntropyModel
from utils.midi_processor import MIDIProcessor

# Initialize the model
model = MaxEntropyModel()

# Process MIDI data
processor = MIDIProcessor()
training_data = processor.load_midi_files('path/to/midi/files')

# Train the model
model.train(training_data)

# Generate new melody
new_melody = model.generate(length=32)

# Save the generated melody
processor.save_midi(new_melody, 'output.midi')
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

