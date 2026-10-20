"""Build-artifact check: test an installed wheel outside the source checkout.

Usage: python scripts/verify_distribution.py dist/*.whl dist/*.tar.gz
Requires network access for pip to resolve the pinned runtime dependencies.
"""

import argparse
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import venv
import zipfile


def verify(wheel, sdist):
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert 'ctor/midi/parsing.py' in names
        retired = (
            'ctor/classic/', 'ctor/context_bp/', 'ctor/core/', 'utils/',
            'ctor/chain_solver.py', 'ctor/context_bp_continuator.py',
            'ctor/variable_order_markov.py', 'ctor/vo_regular_bp_continuator.py',
            'ctor/continuator_gradio.py', 'ctor/midi_ports_poll.py', 'ctor/phrase_listener.py',
        )
        assert not any(name.startswith(retired) for name in names)
    with tempfile.TemporaryDirectory(prefix='continuator-install-') as temp:
        root = Path(temp)
        stage = root / 'contract'
        stage.mkdir()
        # Copy test material only. No ctor/midi_stuff source can shadow the wheel.
        with tarfile.open(sdist) as archive:
            for member in archive.getmembers():
                parts = PurePosixPath(member.name).parts[1:]
                if not member.isfile() or not parts or '..' in parts:
                    continue
                if parts[0] not in ('tests', 'benchmarks', 'data', 'pyproject.toml'):
                    continue
                target = stage.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('wb') as dest:
                    shutil.copyfileobj(source, dest)
        assert (stage / 'data/prelude_c.mid').is_file(), 'MIDI fixture missing from source distribution'
        environment = root / 'venv'
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / 'bin/python'
        subprocess.run([str(python), '-I', '-m', 'pip', 'install', f'{wheel}[dev]'], check=True, cwd=stage)
        program = '''
import pathlib, sys
sys.path.insert(0, str(pathlib.Path.cwd()))
import ctor, pytest
assert pathlib.Path(ctor.__file__).is_relative_to(pathlib.Path(sys.prefix)), ctor.__file__
raise SystemExit(pytest.main(['-q', 'tests']))
'''
        subprocess.run([str(python), '-I', '-c', program], cwd=stage, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    parser.add_argument('sdist', type=Path)
    args = parser.parse_args()
    verify(args.wheel.resolve(), args.sdist.resolve())
