import time
import re
from music21 import pitch
from sympy.strategies.branch.traverse import top_down


class ChordNamer:
    _midi_dict = {}

    # init method or constructor
    def __init__(self, file_path):
        self._parse_midi_mapping(file_path)

    def _parse_midi_mapping(self, file_path):
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or "=>" not in line:
                    continue  # Skip empty or malformed lines
                pitches_part, name_part = line.split("=>", 1)
                pitches_str = pitches_part.strip().strip("[]")
                try:
                    pitches = tuple(
                        int(p.strip()) for p in pitches_str.split(",") if p.strip()
                    )
                    names = name_part.strip().split(";")
                    names = [n.strip() for n in names]
                    self._midi_dict[pitches] = names
                except ValueError as e:
                    print(f"Skipping line due to error: {line}\n{e}")

    def name_for(self, list_of_pitches):
        # 1) translate to start at 48
        delta = list_of_pitches[0] - 48
        pitch_list = [n - delta for n in list_of_pitches]
        # 2) fold notes higher than 72
        pitch_list2 = []
        for n in pitch_list:
            new_n = n
            while new_n > 72:
                new_n -= 12
            if new_n not in pitch_list2:
                pitch_list2.append(new_n)
        chord_names = self._midi_dict[tuple(sorted(pitch_list2))]
        return [self.transpose_name(name, delta) for name in chord_names]

    # todo
    def notes_for_name(self, name: str):
        pass

    def transpose_name(self, chord_name, semitones: int) -> str:
        # Match root note at start, possibly with sharp/flat
        match_root = re.match(r"([A-G][b#]?)(.*)", chord_name)
        if not match_root:
            return chord_name  # Return as-is if parsing fails
        root, rest = match_root.groups()
        transposed_root = pitch.Pitch(root).transpose(semitones).name
        # Look for slash bass note (e.g., /C#)
        if "/" in rest:
            chord_part, bass = rest.rsplit("/", 1)
            transposed_bass = pitch.Pitch(bass).transpose(semitones).name
            return f"{transposed_root}{chord_part}/{transposed_bass}"
        else:
            return f"{transposed_root}{rest}"

    def show_synonyms(self):
        histo = {}
        for k in self._midi_dict:
            for v in self._midi_dict[k]:
                if v not in histo:
                    histo[v] = 1
                else:
                    histo[v] = histo[v] + 1
        for key, value in sorted(histo.items(), key=lambda item: item[1], reverse=True):
            print(f"{key}: {value}")


if __name__ == "__main__":
    t0 = time.perf_counter_ns()
    cn = ChordNamer("../../data/all_chords_C.txt")
    t1 = time.perf_counter_ns()
    print(f"time to parse: {(t1-t0)/1_000_000}")
    t0 = time.perf_counter_ns()
    print(cn.name_for([48, 52, 55]))
    print(cn.name_for([48, 53, 55]))
    print(cn.name_for([48, 54, 55, 59]))
    print(cn.name_for([47, 54, 55, 59]))
    print(cn.name_for([32, 36, 45, 46, 51, 82, 84, 86]))
    t1 = time.perf_counter_ns()
    print(f"time to compute: {(t1-t0)/1_000_000}")
    # cn.show_synonyms()
