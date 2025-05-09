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
- you can learn **online** and even real time (not the cas with transformers)
- you need **fewer** data to produce interesting material. Try the example with Bach prelude in C for instance.
- you can **control** the generation, notably with positional constraints like in this project. More complex constraints like meter can also be implemented (with polynomial complexity)


## Features

- Efficient yet simple implementation of variable-order markov model
- Use of a viewpoint system that enables the handling of rhythmic structure without the cost of heavy tokenization
- Sampling is a combination of Markov with a belief propagation system that enforces positional constraints (that are duly retro propagated)
- Many tricks here and there to maximize musical quality

## Authors
- [François Pachet](https://github.com/fpachet)

### Dependencies

The project requires the following Python packages:
numpy~=2.2.3
mido~=1.2.10
gradio
matplotlib

## Installation

1. clone the repository: 
```bash
git clone https://github.com/fpachet/continuator.git
```
2. install with pip install:
```bash
cd continuator
python3 -m venv venv && source venv/bin/activate && python3 -m pip install .
```
3. launch and then click on the url displayed in the command line:
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

