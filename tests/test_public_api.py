import inspect
import sys

from ctor.continuator import Continuator2
from ctor.sequence_model import SequenceModel
from ctor.vo_regular_bp import VORegularBPContinuator


def test_default_facade_and_generation_signatures():
    assert issubclass(Continuator2, VORegularBPContinuator)
    for method, names in (
        (Continuator2, ['midi_file', 'kmax', 'transposition', 'seed']),
        (Continuator2.sample_sequence, ['prefix', 'length', 'constraints', 'start_vp',
                                      'raise_on_fail']),
        (Continuator2.continue_until_end, ['prefix', 'min_length', 'max_length', 'end_vp']),
    ):
        assert set(names) <= set(inspect.signature(method).parameters)


def test_core_public_imports_and_single_backend():
    from ctor.belief_propag import NoSolutionErrorInBP
    from ctor.constraints import ConstraintProblem
    from ctor.midi import MidiContinuatorBase
    assert issubclass(Continuator2, MidiContinuatorBase)
    assert issubclass(NoSolutionErrorInBP, Exception)
    assert ConstraintProblem is not None
    assert SequenceModel is not None
    assert 'ctor.classic' not in sys.modules
    assert 'ctor.context_bp' not in sys.modules


def test_ui_package_does_not_eagerly_import_gradio():
    import ctor.ui
    assert ctor.ui.__all__ == ['Continuator_gradio']
    assert 'ctor.ui.gradio_app' not in sys.modules
    from ctor.ui.phrase_listener import MidiPhraseListener
    assert MidiPhraseListener is not None


def test_version_matches_distribution_metadata():
    import tomllib
    from pathlib import Path
    import ctor
    project = tomllib.loads((Path(__file__).parents[1] / 'pyproject.toml').read_text())
    assert ctor.__version__ == project['project']['version'] == '2.0.0'
