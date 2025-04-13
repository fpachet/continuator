from datasets import load_dataset

ds = load_dataset("ohollo/lmd_chords")
chords_list = ds["train"]["chords"]
with open("../../data/chordsequences/chord_sequences.txt", "w") as file:
    for chord_seq in chords_list:
        # If chord_seq is a list, join the chords with a space
        chords = "; ".join(chord_seq['symbol'])
        file.write(chords + "\n")
