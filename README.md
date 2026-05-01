# A Python implementation of a constrainable Continuator

A reimplementation of the Continuator system, using a combination of variable-order Markov modeling and exact finite-chain inference to enforce positional constraints.
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
- Sampling combines the variable-order Markov model with exact finite-chain inference for positional constraints. The current implementation uses an iterative sparse forward-backward solver for the constrained chain.
- Many tricks here and there to maximize musical quality

## Architecture documentation

The current implementation is documented in:

- [`docs/current_architecture.md`](docs/current_architecture.md): overview of the classic engine, learning flow, constrained sampling, MIDI layer, and known design pressure points.
- [`docs/class_map.md`](docs/class_map.md): concise class-level reference for the current code.
- [`docs/public_api.md`](docs/public_api.md): compatibility surface used by external front ends such as `continuator_front`.
- [`docs/context_bp_design.md`](docs/context_bp_design.md): notes on the experimental context-BP core.

## Authors
- [François Pachet](https://github.com/fpachet)

### Dependencies
The project requires Python 3.11 at least.

The core package depends only on:
- `numpy`
- `mido`

Optional features are installed with extras:
- `continuator[gradio]` for the Gradio UI dependencies
- `continuator[realtime-midi]` for live MIDI port support through `python-rtmidi`
- `continuator[local-ui]` for both the Gradio UI and realtime MIDI support

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
3. To use the local Gradio/realtime MIDI interface, install the optional UI dependencies:
```bash
python3 -m pip install ".[local-ui]"
```
4. launch and then click on the url displayed in the terminal:
```bash
   python3 -m ctor.continuator_gradio
```
The compatibility command above is kept for existing users. New code may also
launch the UI implementation directly with:
```bash
   python3 -m ctor.ui.gradio_app
```
If the default Gradio ports are already in use, set an explicit port:
```bash
   GRADIO_SERVER_PORT=8060 python3 -m ctor.continuator_gradio
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

The experimental generic context-BP engine can be selected explicitly without
changing the MIDI-facing `Continuator2` API:

```python
from ctor.engines import make_sequence_engine

engine = make_sequence_engine("context_bp", kmax=4)
engine.learn_sequence(["C", "D", "E"])
sequence = engine.continue_until_end(prefix=["C"], min_length=2, max_length=8)
```

For MIDI experiments with the new core, use the separate experimental facade:

```python
from ctor.context_bp_continuator import ContextBPContinuator

generator = ContextBPContinuator(kmax=4)
```

### Migration notes

For front-end integrations, prefer the high-level `Continuator2` methods in `ctor.continuator`:

- Existing calls to `sample_sequence(length=..., constraints=...)` still work.
- Existing client code that passes `sample_sequence(..., start_vp=...)` is supported for compatibility with `continuator_front`; `start_vp` is treated as a hidden handoff viewpoint and is not included in the returned sequence.
- Use `continue_sequence(prefix, length=..., constraints=...)` for fixed-length real-time continuations after a played phrase.
- Use `continue_until_end(prefix=..., min_length=..., max_length=...)` when the continuation should decide its own length but end on the model's end viewpoint.

The generated continuation returned by `continue_sequence` and `continue_until_end` excludes the prefix. This is useful for MIDI playback because the UI should play only the newly generated material.

The Gradio interface exposes both fixed-length generation and "until end" generation. Internally it calls the high-level `Continuator2` continuation methods.

Identity viewpoint models, such as integers, characters, and words, no longer store realization address lists because generated viewpoints are already the realized objects.

### Importing from another project

This repository can be installed as a package named `continuator`. The importable modules keep their existing names, so client code can still use imports such as:

```python
from ctor.continuator import Continuator2
```

For a Hugging Face Space, add a pinned Git dependency to the backend `requirements.txt` instead of copying `ctor/`, `midi_stuff/`, or `utils/` into the front-end repository:

```text
continuator @ git+https://github.com/fpachet/continuator.git@v1.2.2
```

The base package intentionally does not install `gradio`, `matplotlib`, or `python-rtmidi`. Hosted front ends such as `continuator_front` should depend on those packages only if they use them directly.

During development, a commit SHA is also acceptable when you want to test a specific unreleased commit:

```text
continuator @ git+https://github.com/fpachet/continuator.git@<commit-sha>
```

After that, `continuator_front` can remove its vendored Continuator source and import the installed package directly.

## User interface
Currently continuator can be run as:
- python code on midi files (input and output)
- real time midi with rt-midi on a local machine, command line
- real time midi with rt-midi on a local machine, with a gradio interface from a browser

A web client-server version is available on a Huggingface space at:
https://huggingface.co/spaces/pachet/continuator_front


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
