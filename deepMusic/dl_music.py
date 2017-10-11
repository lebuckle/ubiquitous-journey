import os
from music21 import *
import sys
# Add the ptdraft folder path to the sys.path list
# sys.path.append('../')
from test_grammar import *
from functions import *
import lstm


def get_grammar(s):
	# Extract the notes from the stream
	notes	= s.flat.notes
	noteStream = s.flat.notes.stream()
	no_notes = len(notes)
	test_chord = chord.fromForteClass('3-11')
	print("Extracting {0} notes" .format(no_notes))

	extracted_notes = stream.Voice()
	extracted_chords = stream.Voice()
	extracted_chords.insert(5, test_chord)
	test_chord = chord.fromForteClass('3-10')
	extracted_chords.insert(5, test_chord)
	test_chord = chord.fromForteClass('3-11')
	extracted_chords.insert(5, test_chord)
	test_chord = chord.fromForteClass('3-9')
	extracted_chords.insert(5, test_chord)
	print("No notes: {}" .format(no_notes))
	for i in range(no_notes):
		if(noteStream[i].isNote):
			# print("Note {0}: {1}" .format(i,noteStream[i]))
			extracted_notes.insert(i, noteStream[i])
		elif (noteStream[i].isChord):
			# print("Chord {0}: {1}" .format(i,noteStream[i]))
			extracted_chords.insert(i, noteStream[i])

	# print("\n\nextracted_notes: {}" .format(extracted_notes.show('text')))

	abstract_grammars = []
	parsed = parse_melody(extracted_notes, extracted_chords)
	abstract_grammars.append(parsed)

	return extracted_chords, abstract_grammars


n1 = note.Note('B')
if n1.isNote :
	print("Is a note")

# fp = common.getSourceFilePath() / 'original_metheny.mid'
fp = "Test_input.midi"
print("File: {}" .format(fp))
mf = midi.MidiFile()

mf.open(str(fp))
mf.read()

mf.close()

len(mf.tracks)

s = midi.translate.midiFileToStream(mf)

# print("Notes: {}" .format(len(s.flat.notesAndRests)))

# Extract the chords and grammars from the stream
chords, grammars = get_grammar(s)
# print("Grammars: {}" .format(grammars))

# Get corpus
corpus, values, val_indices, indices_val = get_corpus_data(grammars)

print('corpus length:{0} = {1}' .format(len(corpus), corpus))
print('corpus 0:{0} {1}' .format(corpus[0], type(corpus[0])))
print('total # of values:', len(values))

# Build model
max_len = 2
N_epochs = 1
max_tries = 1000
diversity = 0.5

model = lstm.build_model(corpus=corpus, val_indices=val_indices, 
												 max_len=max_len, N_epochs=N_epochs)

# set up audio stream
out_stream = stream.Stream()

# generation loop
curr_offset = 0.0
loopEnd = len(chords)
print("No chords: {}" .format(loopEnd))
for loopIndex in range(1, loopEnd):
	# get chords from file
	curr_chords = stream.Voice()
	for j in chords[loopIndex]:
		curr_chords.insert((j.offset % 4), j)	

	# generate grammar
	curr_grammar = generate_grammar(model=model, corpus=corpus, 
												abstract_grammars=grammars, 
																	values=values, val_indices=val_indices, 
																	indices_val=indices_val, 
																	max_len=max_len, max_tries=max_tries,
																	diversity=diversity)

	curr_grammar = curr_grammar.replace(' A',' C').replace(' X',' C')

	# Pruning #1: smoothing measure
	curr_grammar = prune_grammar(curr_grammar)

	# Get notes from grammar and chords
	curr_notes = unparse_grammar(curr_grammar, curr_chords)

	# Pruning #2: removing repeated and too close together notes
	curr_notes = prune_notes(curr_notes)

	# quality assurance: clean up notes
	curr_notes = clean_up_notes(curr_notes)

	# print # of notes in curr_notes
	print('After pruning: %s notes' % (len([i for i in curr_notes
	if isinstance(i, note.Note)])))

	# insert into the output stream
	for m in curr_notes:
		out_stream.insert(curr_offset + m.offset, m)
	for mc in curr_chords:
		out_stream.insert(curr_offset + mc.offset, mc)

	curr_offset += 4.0

bpm = 130
out_stream.insert(0.0, tempo.MetronomeMark(number=bpm))

# Play the final stream through output (see 'play' lambda function above)
# play = lambda x: midi.realtime.StreamPlayer(x).play()
# play(out_stream)

# save stream
mf = midi.translate.streamToMidiFile(out_stream)
print("Print stream")
print("Out stream: {}" .format(out_stream))
out_fn = "output.midi"
mf.open(out_fn, 'wb')
mf.write()
mf.close()

mf = midi.translate.streamToMidiFile(s)
mf.open("Testing_output.midi", 'wb')
mf.write()
mf.close()
# notes.show('text')

# m = stream.Voice()
# for i in notes:
# 	# print("i type: {} {}" .format(type(i), i))
# 	if notes[i].isNote :
# 		print("Note")
# 	m.insert(i.offset, i)
# print("\n\nm: {}" .format(m.show('text')))
