from muses.base.temporals import Piece

piece = Piece.load_midi("../data/prelude_nano_gpt.mid")
piece.stretch(.4)
piece.save_midi("../data/prelude_nano_gpt_stretched.mid")