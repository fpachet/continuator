class Note:
    def __init__(self, pitch, velocity, duration, start_time=0):
        self.pitch = pitch
        self.velocity = velocity
        # the duration in the original sequence in quarter-note beats
        self.duration = duration
        # the start time in the original sequence in quarter-note beats
        self.start_time = start_time
        # time between start of next note and end. Negative if overlaps with next
        self.next_start_delta = 0  # in quarter-note beats
        self.overlaps_left_flag = False

    def __str__(self):
        return f"{self.pitch} @ [{self.start_time}, {self.get_end_time()}]"

    def __repr__(self):
        return f"{self.pitch} @ [{self.start_time}, {self.get_end_time()}]"

    def set_duration(self, d):
        self.duration = d

    def set_start_time(self, t):
        self.start_time = t

    def overlaps_left(self):
        return self.overlaps_left_flag

    def overlaps_right(self):
        return self.next_start_delta < 0

    def transpose(self, t):
        note = self.copy()
        note.pitch = self.pitch + t
        return note

    def copy(self):
        new_note = Note(self.pitch, self.velocity, self.duration, start_time=self.start_time)
        new_note.next_start_delta = self.next_start_delta
        new_note.overlaps_left_flag = self.overlaps_left_flag
        return new_note

    def get_end_time(self):
        return self.start_time + self.duration
