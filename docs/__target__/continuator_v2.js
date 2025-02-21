// Transcrypt'ed from Python, 2025-02-20 17:34:34
var random = {};
var time = {};
import {AssertionError, AttributeError, BaseException, DeprecationWarning, Exception, IndexError, IterableError, KeyError, NotImplementedError, RuntimeWarning, StopIteration, UserWarning, ValueError, Warning, __JsIterator__, __PyIterator__, __Terminal__, __add__, __and__, __call__, __class__, __envir__, __eq__, __floordiv__, __ge__, __get__, __getcm__, __getitem__, __getslice__, __getsm__, __gt__, __i__, __iadd__, __iand__, __idiv__, __ijsmod__, __ilshift__, __imatmul__, __imod__, __imul__, __in__, __init__, __ior__, __ipow__, __irshift__, __isub__, __ixor__, __jsUsePyNext__, __jsmod__, __k__, __kwargtrans__, __le__, __lshift__, __lt__, __matmul__, __mergefields__, __mergekwargtrans__, __mod__, __mul__, __ne__, __neg__, __nest__, __or__, __pow__, __pragma__, __pyUseJsNext__, __rshift__, __setitem__, __setproperty__, __setslice__, __sort__, __specialattrib__, __sub__, __super__, __t__, __terminal__, __truediv__, __withblock__, __xor__, _sort, abs, all, any, assert, bin, bool, bytearray, bytes, callable, chr, delattr, dict, dir, divmod, enumerate, filter, float, getattr, hasattr, hex, input, int, isinstance, issubclass, len, list, map, max, min, object, oct, ord, pow, print, property, py_TypeError, py_iter, py_metatype, py_next, py_reversed, py_typeof, range, repr, round, set, setattr, sorted, str, sum, tuple, zip} from './org.transcrypt.__runtime__.js';
import * as __module_time__ from './time.js';
__nest__ (time, '', __module_time__);
import * as __module_random__ from './random.js';
__nest__ (random, '', __module_random__);
var __name__ = '__main__';
export var Note =  __class__ ('Note', [object], {
	__module__: __name__,
	get __init__ () {return __get__ (this, function (self, pitch, velocity, duration, start_time) {
		if (typeof start_time == 'undefined' || (start_time != null && start_time.hasOwnProperty ("__kwargtrans__"))) {;
			var start_time = 0;
		};
		self.pitch = pitch;
		self.velocity = velocity;
		self.duration = duration;
		self.start_time = start_time;
	});},
	get set_duration () {return __get__ (this, function (self, d) {
		self.duration = d;
	});},
	get set_start_time () {return __get__ (this, function (self, t) {
		self.start_time = t;
	});}
});
export var Continuator2 =  __class__ ('Continuator2', [object], {
	__module__: __name__,
	get __init__ () {return __get__ (this, function (self, midi_file, kmax, transposition) {
		if (typeof kmax == 'undefined' || (kmax != null && kmax.hasOwnProperty ("__kwargtrans__"))) {;
			var kmax = 5;
		};
		if (typeof transposition == 'undefined' || (transposition != null && transposition.hasOwnProperty ("__kwargtrans__"))) {;
			var transposition = false;
		};
		self.midi_file = midi_file;
		self.kmax = kmax;
		self.prob_to_keep_singletons = 1 / 3;
		self.notes_original = self.extract_notes ();
		var all_notes = [];
		if (transposition) {
			for (var t = -(6); t < 6; t++) {
				var all_notes = all_notes + self.transpose_notes (self.notes_original, t);
			}
			self.notes = all_notes;
		}
		else {
			self.notes = self.notes_original;
		}
		self.prefixes_to_continuations = [];
		self.build_vo_markov_model ();
	});},
	get transpose_notes () {return __get__ (this, function (self, notes, t) {
		return (function () {
			var __accu0__ = [];
			for (var n of notes) {
				__accu0__.append (Note (n.pitch + t, n.velocity, n.duration, __kwargtrans__ ({start_time: n.start_time})));
			}
			return __accu0__;
		}) ();
	});},
	get extract_notes () {return __get__ (this, function (self) {
		var mid = mido.MidiFile (self.midi_file);
		var notes = [];
		var pending_notes = np.empty (128, __kwargtrans__ ({dtype: object}));
		var pending_start_times = np.zeros (128);
		var current_time = 0;
		for (var track of mid.tracks) {
			for (var msg of track) {
				current_time += msg.time;
				if (msg.py_metatype == 'note_on' && msg.velocity > 0) {
					var new_note = Note (msg.note, msg.velocity, 0);
					notes.append (new_note);
					pending_notes [msg.note] = new_note;
					pending_start_times [msg.note] = current_time;
					new_note.set_start_time (current_time);
					new_note.set_duration (120);
				}
				if (msg.py_metatype == 'note_off') {
					var pending_note = pending_notes [msg.note];
					var duration = current_time - pending_start_times [msg.note];
					pending_note.set_duration (duration);
					pending_notes [msg.note] = null;
					pending_start_times [msg.note] = 0;
				}
			}
		}
		return notes;
	});},
	get build_vo_markov_model () {return __get__ (this, function (self) {
		self.prefixes_to_continuations = np.empty (self.kmax, __kwargtrans__ ({dtype: object}));
		for (var k = 0; k < self.kmax; k++) {
			var prefixes_to_cont_k = dict ({});
			for (var i = 0; i < len (self.notes) - k; i++) {
				if (i < k + 1) {
					continue;
				}
				var current_ctx = self.get_viewpoint_tuple (tuple (range ((i - k) - 1, i)));
				if (!__in__ (current_ctx, prefixes_to_cont_k)) {
					prefixes_to_cont_k [current_ctx] = [];
				}
				prefixes_to_cont_k [current_ctx].append (i);
			}
			self.prefixes_to_continuations [k] = prefixes_to_cont_k;
		}
	});},
	get get_viewpoint () {return __get__ (this, function (self, index) {
		var note = self.notes [index];
		return tuple ([note.pitch, note.duration / 100]);
	});},
	get get_viewpoint_tuple () {return __get__ (this, function (self, indices_tuple) {
		var vparray = (function () {
			var __accu0__ = [];
			for (var id of indices_tuple) {
				__accu0__.append (self.get_viewpoint (id));
			}
			return __accu0__;
		}) ();
		return tuple (vparray);
	});},
	get sample_sequence () {return __get__ (this, function (self, start_note, length) {
		if (typeof length == 'undefined' || (length != null && length.hasOwnProperty ("__kwargtrans__"))) {;
			var length = 50;
		};
		var current_seq = [start_note];
		for (var _ = 0; _ < length; _++) {
			var cont = self.get_continuation (current_seq);
			if (cont == -(1)) {
				print ('restarting from scratch');
				var cont = random.choice (range (len (self.notes)));
			}
			current_seq.append (cont);
		}
		return current_seq;
	});},
	get get_continuation () {return __get__ (this, function (self, current_seq) {
		var vp_to_skip = null;
		for (var k = self.kmax; k > 0; k--) {
			if (k > len (current_seq)) {
				continue;
			}
			var continuations_dict = self.prefixes_to_continuations [k - 1];
			var ctx = tuple (current_seq.__getslice__ (-(k), null, 1));
			var viewpoint_ctx = self.get_viewpoint_tuple (ctx);
			if (__in__ (viewpoint_ctx, continuations_dict)) {
				var all_conts = continuations_dict [viewpoint_ctx];
				var all_cont_vp = (function () {
					var __accu0__ = [];
					for (var i of all_conts) {
						__accu0__.append (self.get_viewpoint (i));
					}
					return set (__accu0__);
				}) ();
				if (len (all_cont_vp) == 1 && k > 0) {
					if (random.random () > 1 / (k + 1)) {
						var vp_to_skip = all_cont_vp.py_pop ();
						continue;
					}
					else {
						var vp_to_skip = null;
					}
				}
				if (vp_to_skip !== null) {
					var all_conts_tu_use = (function () {
						var __accu0__ = [];
						for (var c of all_conts) {
							if (self.get_viewpoint (c) != vp_to_skip) {
								__accu0__.append (c);
							}
						}
						return __accu0__;
					}) ();
				}
				else {
					var all_conts_tu_use = all_conts;
				}
				var next_continuation = random.choice (all_conts_tu_use);
				return next_continuation;
			}
		}
		return -(1);
		print ('no continuation found');
	});},
	get get_pitch_string () {return __get__ (this, function (self, sequence_of_notes) {
		return ''.join ((function () {
			var __accu0__ = [];
			for (var note of sequence_of_notes) {
				__accu0__.append (str (note.pitch) + ' ');
			}
			return __accu0__;
		}) ());
	});},
	get save_midi () {return __get__ (this, function (self, idx_sequence, output_file) {
		var mid = mido.MidiFile ();
		var track = mido.MidiTrack ();
		mid.tracks.append (track);
		var sequence = [];
		var start_time = 0;
		for (var i of idx_sequence) {
			var note = self.notes [i];
			var note_copy = Note (note.pitch, note.velocity, note.duration);
			if (i != 0) {
				var delta = note.start_time - self.notes [i - 1].start_time;
				start_time += delta;
			}
			note_copy.set_start_time (start_time);
			sequence.append (note_copy);
		}
		var mido_sequence = [];
		for (var note of sequence) {
			mido_sequence.append (mido.Message ('note_on', __kwargtrans__ ({note: note.pitch, velocity: note.velocity, time: note.start_time})));
			mido_sequence.append (mido.Message ('note_off', __kwargtrans__ ({note: note.pitch, velocity: 0, time: note.start_time + int (note.duration)})));
		}
		mido_sequence.py_sort (__kwargtrans__ ({key: (function __lambda__ (msg) {
			return msg.time;
		})}));
		var current_time = 0;
		for (var msg of mido_sequence) {
			var delta = msg.time - current_time;
			msg.time = delta;
			track.append (msg);
			current_time += delta;
		}
		mid.save (output_file);
	});},
	get get_longest_subsequence_with_train () {return __get__ (this, function (self, sequence_of_idx) {
		var train_string = generator.get_pitch_string (generator.notes);
		var sequence_of_notes = (function () {
			var __accu0__ = [];
			for (var id of sequence_of_idx) {
				__accu0__.append (generator.notes [id]);
			}
			return __accu0__;
		}) ();
		var sequence_string = generator.get_pitch_string (sequence_of_notes);
		var match = SequenceMatcher (null, train_string, sequence_string, __kwargtrans__ ({autojunk: false})).find_longest_match ();
		var nb_notes_common = train_string.__getslice__ (match.a, match.a + match.size, 1).count (' ');
		return nb_notes_common;
	});}
});
export var midi_file_path = '../data/prelude_c.mid';
export var t0 = time.perf_counter_ns ();
export var generator = Continuator2 (midi_file_path, 4, __kwargtrans__ ({transposition: true}));
export var t1 = time.perf_counter_ns ();
print ('total time: {}'.format ((t1 - t0) / 1000000));
export var start_note = 0;
export var generated_sequence = generator.sample_sequence (start_note, __kwargtrans__ ({length: 200}));
generator.save_midi (generated_sequence, '../data/ctor2_output.mid');
print ('Generated Sequence:', generated_sequence);
print ('{} notes in commun with train'.format (generator.get_longest_subsequence_with_train (generated_sequence)));

//# sourceMappingURL=continuator_v2.map