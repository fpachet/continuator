# A Python implementation of a constrainable Continuator

A reimplementation of the Continuator system, using a combination of variable-order Markov model and belief propagation to enforce positional constraints.
Note that this is the only system, to my knowledge, able to produce controllable sequences (with guarantees) with unary/positional constraints.
These are extremely powerful and can turn seemingly "random" walks into actual music with intention.

It is inspired from the following papers:
- Pachet, F. The Continuator: Musical Interaction with Style. Journal of New Music Research, 32(3):333-341, 2003
- Papadopoulos, A., Pachet, F., Roy, P. and Sakellariou, J. Exact Sampling for Regular and Markov Constraints with Belief Propagation. 21th Principles and Practice of Constraint Programming Conference (CP 2015), Cork (Ireland), 2015
- Pachet, F., Roy, P. and Barbieri, G. Finite-Length Markov Processes with Constraints. Proceedings of the 22nd International Joint Conference on Artificial Intelligence (IJCAI), pages 635-642, Barcelona, Spain, July 2011
- Roy, P. and Pachet, F. Enforcing Meter in Finite-Length Markov Sequences. 27th Conference on Artificial Intelligence (AAAI 2013), Bellevue, Washington (USA), June 2013

Three reasons why this kind of approach remains interesting, in spite of the existence of more powerful sequence prediction algorithms such as transformers:
- you can learn **online** and even real time (not the case with transformers)
- you need **fewer** data to produce interesting material. Try the example with Bach prelude in C for instance.
- you can **control** the generation, notably with positional constraints like in this project. More complex constraints like meter can also be implemented (with polynomial complexity)


## Features

- Efficient yet simple implementation of variable-order markov model
- Use of a viewpoint system that enables the handling of rhythmic structure without the cost of heavy tokenization
- Sampling combines the variable-order Markov model with exact finite-chain inference for positional constraints. The current implementation uses an iterative sparse forward-backward solver for the constrained chain, while keeping the older recursive BP code as a reference/test path.
- Many tricks here and there to maximize musical quality

## Authors
- [François Pachet](https://github.com/fpachet)

### Dependencies
The project requires Python 3.11 at least, as well as the following packages:
numpy~=2.2.3
mido~=1.2.10
gradio
matplotlib

## Installation

1. clone the repository: 
```bash
git clone https://github.com/fpachet/continuator.git
cd continuator
```
2. create virtual environment and install dependencies:
```bash
python3 -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows (PowerShell):
# You may need first to install Microsoft C++ Build Tools (a minimal Visual Studio subset that includes cl.exe, the C compiler).
# Go to the official page.

.\venv\Scripts\Activate.ps1

python3 -m pip install .
# If this causes a problem like "error: externally managed environment", then try:
# python3 -m pip install . --break-system-packages

```
3. launch and then click on the url displayed in the terminal:
```bash
   python3 -m ctor.continuator_gradio
```


## Usage in Python
See "examples" folder for ints, characters, words and chord sequence generation examples.

```python
import ctor.continuator

# Initialize the model
midi_file_path = "../../data/prelude_c.mid"
generator = ctor.continuator.Continuator2(midi_file_path, 4, transposition=False)

# set positional constraints as a dictionary index -> viewpoint
constraints = {}
# to start with a "start"
# constraints[0] = generator.get_start_vp()
# to force arbitrary value at arbitrary position, here a D3 as first note
constraints[0] = generator.get_vp_for_pitch(62)
# to end with an "end"
constraints[19] = generator.get_end_vp()

# generate the viewpoint sequence with some length
generated_sequence = generator.sample_sequence(length=20, constraints=constraints)

# remove extra start or end viewpoint if needed
sequence_to_render = generated_sequence[0:-1]

# realize the sequence with actual notes
rendered_sequence = generator.realize_vp_sequence(sequence_to_render)

# save the generated sequence
generator.save_midi(rendered_sequence, "../../data/constrained_prelude.mid", tempo=-1)
```

### Generation APIs

There are three related generation modes:

- `sample_sequence(length, constraints=...)` generates a fixed-length sequence.
- `continue_sequence(prefix, length, constraints=...)` generates a fixed-length continuation. The prefix is conditioning context only and is not included in the returned sequence.
- `continue_until_end(prefix=..., min_length=..., max_length=...)` generates a variable-length continuation that stops when the end viewpoint is first reached.

Constraints are always indexed over the returned generated sequence, not over the prefix plus the generated sequence. For example, in `continue_sequence(prefix=[1, 2], length=3, constraints={0: 3})`, position `0` refers to the first generated element after the prefix.

The current constrained sampler uses an iterative forward-backward pass on the first-order chain for feasibility and marginals, then combines that information with the variable-order continuation model during sampling.

The legacy dictionary format for constraints is still supported:

```python
constraints = {0: generator.get_vp_for_pitch(62), 19: generator.get_end_vp()}
sequence = generator.sample_sequence(length=20, constraints=constraints)
```

For new code, a small constraint builder is also available:

```python
from ctor.constraints import ConstraintProblem

constraints = ConstraintProblem(length=20)
constraints.at(0).equals(generator.get_vp_for_pitch(62))
constraints.at(19).equals(generator.get_end_vp())

sequence = generator.sample_sequence(length=20, constraints=constraints)
```

### Migration notes

For front-end integrations, prefer the high-level `Continuator2` methods in `ctor.continuator`:

- Existing calls to `sample_sequence(length=..., constraints=...)` still work.
- Use `continue_sequence(prefix, length=..., constraints=...)` for fixed-length real-time continuations after a played phrase.
- Use `continue_until_end(prefix=..., min_length=..., max_length=...)` when the continuation should decide its own length but end on the model's end viewpoint.

The generated continuation returned by `continue_sequence` and `continue_until_end` excludes the prefix. This is useful for MIDI playback because the UI should play only the newly generated material.

The Gradio interface now exposes both fixed-length generation and "until end" generation. Internally it calls the new `Continuator2` continuation methods rather than constructing the old BP graph directly.

The old recursive BP implementation remains in the project for comparison tests and reference, but it is no longer the default generation path.

## User interface
Currently continuator can be run as:
- python code on midi files (input and output)
- real time midi with rt-midi on a local machine, command line
- real time midi with rt-midi on a local machine, with a gradio interface from a browser

A web client-server version will be available soon, as so far attempts at using gradio and javascript failed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
