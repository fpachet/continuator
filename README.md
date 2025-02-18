# Maximum Entropy Melody Generator

A melody generation system implementing an efficient Maximum Entropy model using PyTorch and transformers. This project provides a novel approach to musical composition by leveraging statistical modeling and deep learning techniques for creating coherent and diverse melodies.

It is based on the paper:
Sakellariou, J., Tria, F., Loreto, V. et al. Maximum entropy models capture melodic styles. Sci Rep 7, 9172 (2017). https://doi.org/10.1038/s41598-017-08028-4

which is available at: https://www.nature.com/articles/s41598-017-08028-4


## Features

- Efficient implementation of Maximum Entropy models for melody generation
- Optimized context creation and partition function computation
- MIDI file processing and generation capabilities
- Integration with PyTorch and transformers for enhanced model performance
- Scalable architecture for handling complex musical patterns

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/max-entropy-melody-generator.git
cd max-entropy-melody-generator
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

The project requires the following Python packages:
- numpy ~= 1.24.2
- mido ~= 1.2.10
- scipy ~= 1.10.1
- torch ~= 2.6.0
- transformers ~= 4.48.3

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

## Project Structure

```
├── core/
│   ├── max_entropy.py        # Main implementation of MaxEntropy model
│   └── model_utils.py        # Utility functions for model operations
├── data/
│   └── midi/                 # Directory for MIDI training data
├── utils/
│   ├── midi_processor.py     # MIDI file processing utilities
│   └── data_utils.py         # Data preprocessing utilities
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Implementation Details

The implementation focuses on two key aspects:

1. **Efficient Context Creation**
- Optimized context window management
- Efficient feature extraction from musical sequences
- Smart caching of frequently used contexts

2. **Partition Function Computation**
- Improved algorithms for partition function calculation
- Parallelized computation using PyTorch
- Memory-efficient implementation for large-scale models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

