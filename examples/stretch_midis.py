from pathlib import Path

from muses.base.temporals import Piece

if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parents[1]
    piece = Piece.load_midi(repo_root / "data" / "prelude_nano_gpt.mid")
    piece.stretch(.4)
    piece.save_midi(repo_root / "data" / "prelude_nano_gpt_stretched.mid")
