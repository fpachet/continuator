from music21 import key, pitch

class PitchProfile:
    def krumhansl_weights(self, weightType):
        if weightType == 'major':
            return [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39,
                    3.66, 2.29, 2.88]
        elif weightType == 'minor':
            return [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    def get_weights(self, pitch, type):
        return self.rotate_right(list(self.krumhansl_weights(type)), pitch.pitchClass)

    def rotate_right(self, arr, n):
        n = n % len(arr)  # handle cases where n > len(arr)
        return  arr[-n:] + arr[:-n]

if __name__ == '__main__':
    k = key.Key('G', 'minor')
    pitch.Pitch("C")
    pf = PitchProfile().get_weights(pitch.Pitch("Eb"), 'major')
    print(pf)
    print(pitch.Pitch('A#2').midi)
