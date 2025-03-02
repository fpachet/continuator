import argparse
import os

import mido
import music21


def get_mirror_axis(tonic):
    return tonic + 3.5

def mirror_note_over_axis(note, axis):
    original_note_distance = axis - note
    return int(axis + original_note_distance)


def find_average_octave_of_tracks(mid):
    octaves = {}
    for i, track in enumerate(mid.tracks):
        track_avg_notes = []
        for message in track:
            if message.type == 'note_on':
                track_avg_notes.append(message.note)
        if len(track_avg_notes) > 0:
            track_avg_note = sum(track_avg_notes) / len(track_avg_notes)
            octaves[i] = track_avg_note
            try:
                track_name = track.name
            except AttributeError:
                track_name = i
            print(track_name, track_avg_note)
    return octaves


def mirror_all_notes(mid, mirror_axis, ignored_channels):
    for track in mid.tracks:
        for message in track:
            if message.type == 'note_on' or message.type == 'note_off':
                if message.channel not in ignored_channels:
                    mirrored_note = mirror_note_over_axis(message.note, mirror_axis)
                    message.note = mirrored_note
    return


def transpose_back_to_original_octaves(mid, original_octaves, new_octaves, ignored_channels):
    for i, track in enumerate(mid.tracks):
        if i in original_octaves:
            notes_distance = original_octaves[i] - new_octaves[i]
            octaves_to_transpose = round(notes_distance / 12)
            for message in track:
                if message.type == 'note_on' or message.type == 'note_off':
                    if message.channel not in ignored_channels:
                        transposed_note = message.note + (octaves_to_transpose * 12)
                        message.note = int(transposed_note)


def invert_tonality(mid, tonic, ignored_channels, adjust_octaves):
    mirror_axis = get_mirror_axis(tonic)

    if adjust_octaves:
        print("---")
        print("original average note values:")
        original_octaves = find_average_octave_of_tracks(mid)

    mirror_all_notes(mid, mirror_axis, ignored_channels)

    if adjust_octaves:
        print("---")
        print("new average note values:")
        new_octaves = find_average_octave_of_tracks(mid)

        transpose_back_to_original_octaves(mid, original_octaves, new_octaves, ignored_channels)

        print("---")
        print("adjusted average note values:")
        find_average_octave_of_tracks(mid)
    return


def main(input_file, tonic, ignored_channels, adjust_octaves):
    root, ext = os.path.splitext(input_file)
    mid = mido.MidiFile(input_file)

    invert_tonality(mid, tonic, ignored_channels, adjust_octaves)
    mid.save(root + "_negative" + ext)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Negative Harmonize a midi file.')
#
#     parser.add_argument('file', metavar='f',
#                         help='the input midi file (with no extension .mid)')
#     parser.add_argument('--tonic', type=int, default=60,
#                         help='the tonic')
#     parser.add_argument('--ignore', type=int, nargs="+", default=[],
#                         help='the midi channels to ignore (usually 9 for drums)')
#     parser.add_argument('--adjust-octaves', dest='adjust_octaves', action='store_true',
#                         help='transpose octaves to keep bass instruments low')
#     parser.set_defaults(adjust_octaves=False)
#
#     args = parser.parse_args()
#
#     main(input_file=args.file, tonic=args.tonic, ignored_channels=args.ignore, adjust_octaves=args.adjust_octaves)

# file_path = '../../data/keith/train/K7_MD.mid'
file_path = "../../data/ctor2_output.mid"
mid = mido.MidiFile(file_path)
my_score: music21.stream.Score = music21.converter.parse(file_path)
k = my_score.analyze('key')
print(k)
invert_tonality(mid, k.tonic.midi, [9], True)
mid.save("../../data/negative_ctor2.mid")
my_score: music21.stream.Score = music21.converter.parse("../../data/negative.mid")
k = my_score.analyze('key')
print(k)
