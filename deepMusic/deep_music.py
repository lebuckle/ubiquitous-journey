from __future__ import print_function
import sys

from music21 import *
import numpy as np

p1 = stream.Part()
k1 = key.KeySignature(0) # key of C
n1 = note.Note('B')
r1 = note.Rest()
c1 = chord.Chord(['A', 'B-'])
p1.append([k1, n1, r1, c1])
p1.show('text')


print("\n\tNotes")
noteStream = p1.notes.stream()
no_notes = len(noteStream)
print("No notes: {}" .format(no_notes))
for i in range(no_notes):
	if(noteStream[i].isNote):
		print("Note {0}: {1}" .format(i,noteStream[i]))
	elif (noteStream[i].isChord):
		print("Chord {0}: {1}" .format(i,noteStream[i]))

# noteStream.show('text')

# Add a measure
m1 = stream.Measure()
n2 = note.Note("D")
m1.insert(0, n2)
p1.append(m1)
p1.flat.notes.stream().show('text')

notes  = p1.flat.notes
print("Extracting notes")
notes.show('text')
print("notes type: {}" .format(type(notes)))
# for i in notes:
# 	print("m: {0}" .format(notes[i]))
# s = stream.Stream()
# n = note.Note('a')
# n.quarterLength = .5
# n.storedInstrument = instrument.Cowbell()
# s.append(n)

# n = note.Note('F-6')
# n.quarterLength = 1
# s.repeatAppend(n,2)

# # Second note
# # n1 = note.Note('a')
# # n1.quarterLength = .5

# # s.repeatAppend(n, 4)
# # s.repeatAppend(n1, 4)
# mf = midi.translate.streamToMidiFile(s)
# len(mf.tracks)

# len(mf.tracks[0].events)

# mf.open("Test output.midi", 'wb')
# mf.write()
# mf.close()

# 		Instruments
# p1 = converter.parse("tinynotation: 4/4 c4  d  e  f  g  a  b  c'  c1")
# p2 = converter.parse("tinynotation: 4/4 d  e  f  g  a  b  c'  c1")
# p1.getElementsByClass('Measure')[0].insert(0.0, instrument.Piccolo())
# p1.getElementsByClass('Measure')[0].insert(1.0, instrument.AltoSaxophone())
# # p1.getElementsByClass('Measure')[1].insert(1.0, instrument.Triangle())
# p1.getElementsByClass('Measure')[1].insert(1.0, instrument.Horn())
# p1.getElementsByClass('Measure')[1].insert(2.0, instrument.Piano())
# p2.getElementsByClass('Measure')[0].insert(0.0, instrument.Horn())
# s = stream.Score()
# # Insert first
# s.insert(0, p1)
# # Insert second
# s.insert(0, p2)

# noteStream = p1.notes.stream()
# noteStream.show('text')
# p1.notes.stream().show('text')
# mf = midi.translate.streamToMidiFile(s)
# mf.open("Test output.midi", 'wb')
# mf.write()
# mf.close()