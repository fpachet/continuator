from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable


@dataclass(frozen=True, eq=False)
class BoundaryToken:
    name: str

    def __repr__(self) -> str:
        return f"<{self.name}>"


class Vocabulary:
    """Bidirectional mapping between hashable symbols and compact integer ids."""

    def __init__(self, *, start_symbol: Hashable | None = None, end_symbol: Hashable | None = None):
        self.start_symbol = BoundaryToken("START") if start_symbol is None else start_symbol
        self.end_symbol = BoundaryToken("END") if end_symbol is None else end_symbol
        self.symbol_to_id: dict[Hashable, int] = {}
        self.id_to_symbol: list[Hashable] = []
        self.start_id = self.add(self.start_symbol)
        self.end_id = self.add(self.end_symbol)

    def __len__(self) -> int:
        return len(self.id_to_symbol)

    def add(self, symbol: Hashable) -> int:
        if symbol in self.symbol_to_id:
            return self.symbol_to_id[symbol]
        symbol_id = len(self.id_to_symbol)
        self.symbol_to_id[symbol] = symbol_id
        self.id_to_symbol.append(symbol)
        return symbol_id

    def encode(self, symbol: Hashable) -> int:
        try:
            return self.symbol_to_id[symbol]
        except KeyError as e:
            raise KeyError(f"Unknown symbol: {symbol!r}") from e

    def encode_or_add(self, symbol: Hashable) -> int:
        return self.add(symbol)

    def decode(self, symbol_id: int) -> Any:
        return self.id_to_symbol[symbol_id]

    def has_symbol(self, symbol: Hashable) -> bool:
        return symbol in self.symbol_to_id
