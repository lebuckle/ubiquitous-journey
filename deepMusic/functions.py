import numpy as np
import random
from itertools import zip_longest
from music21 import *


''' Get corpus data from grammatical data '''
def get_corpus_data(abstract_grammars):
    print("\n\n\n\nThis Corpus")
    corpus = [x for sublist in abstract_grammars for x in sublist.split(' ')]
    values = set(corpus)
    val_indices = dict((v, i) for i, v in enumerate(values))
    indices_val = dict((i, v) for i, v in enumerate(values))

    # print("Corpus type: {}" .format(type(corpus)))
    return corpus, values, val_indices, indices_val

def generate_grammar(model, corpus, abstract_grammars, values, val_indices,
                       indices_val, max_len, max_tries, diversity):
    curr_grammar = ''
    # np.random.randint is exclusive to high
    start_index = np.random.randint(0, len(corpus) - max_len)
    sentence = corpus[start_index: start_index + max_len]    # seed
    running_length = 0.0
    while running_length <= 4.1:    # arbitrary, from avg in input file
        # transform sentence (previous sequence) to matrix
        x = np.zeros((1, max_len, len(values)))
        for t, val in enumerate(sentence):
            if (not val in val_indices): print(val)
            x[0, t, val_indices[val]] = 1.

        next_val = __predict(model, x, indices_val, diversity)

        # fix first note: must not have < > and not be a rest
        if (running_length < 0.00001):
            tries = 0
            while (next_val.split(',')[0] == 'R' or 
                len(next_val.split(',')) != 2):
                # give up after 1000 tries; random from input's first notes
                if tries >= max_tries:
                    print('Gave up on first note generation after', max_tries, 
                        'tries')
                    # np.random is exclusive to high
                    rand = np.random.randint(0, len(abstract_grammars))
                    next_val = abstract_grammars[rand].split(' ')[0]
                else:
                    next_val = __predict(model, x, indices_val, diversity)

                tries += 1

        # shift sentence over with new value
        sentence = sentence[1:] 
        sentence.append(next_val)

        # except for first case, add a ' ' separator
        if (running_length > 0.00001): curr_grammar += ' '
        curr_grammar += next_val

        length = float(next_val.split(',')[1])
        running_length += length

    return curr_grammar

''' Helper function to generate a predicted value from a given matrix '''
def __predict(model, x, indices_val, diversity):
    preds = model.predict(x, verbose=0)[0]
    next_index = __sample(preds, diversity)
    next_val = indices_val[next_index]

    return next_val

''' Helper function to sample an index from a probability array '''
def __sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

''' Smooth the measure, ensuring that everything is in standard note lengths 
    (e.g., 0.125, 0.250, 0.333 ... ). '''
def prune_grammar(curr_grammar):
    pruned_grammar = curr_grammar.split(' ')

    for ix, gram in enumerate(pruned_grammar):
        terms = gram.split(',')
        terms[1] = str(__roundUpDown(float(terms[1]), 0.250, 
            random.choice([-1, 1])))
        pruned_grammar[ix] = ','.join(terms)
    pruned_grammar = ' '.join(pruned_grammar)

    return pruned_grammar

''' Helper function to down num to the nearest multiple of mult. '''
def __roundDown(num, mult):
    return (float(num) - (float(num) % mult))

''' Helper function to round up num to nearest multiple of mult. '''
def __roundUp(num, mult):
    return __roundDown(num, mult) + mult

''' Helper function that, based on if upDown < 0 or upDown >= 0, rounds number 
    down or up respectively to nearest multiple of mult. '''
def __roundUpDown(num, mult, upDown):
    if upDown < 0:
        return __roundDown(num, mult)
    else:
        return __roundUp(num, mult)

''' Remove repeated notes, and notes that are too close together. '''
def prune_notes(curr_notes):
    for n1, n2 in __grouper(curr_notes, n=2):
        if n2 == None: # corner case: odd-length list
            continue
        if isinstance(n1, note.Note) and isinstance(n2, note.Note):
            if n1.nameWithOctave == n2.nameWithOctave:
                curr_notes.remove(n2)

    return curr_notes

''' Helper function, from recipes, to iterate over list in chunks of n 
    length. '''
def __grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
    # return izip_longest(*args, fillvalue=fillvalue)

''' Perform quality assurance on notes '''
def clean_up_notes(curr_notes):
    removeIxs = []
    for ix, m in enumerate(curr_notes):
        # QA1: ensure nothing is of 0 quarter note len, if so changes its len
        if (m.quarterLength == 0.0):
            m.quarterLength = 0.250
        # QA2: ensure no two melody notes have same offset, i.e. form a chord.
        # Sorted, so same offset would be consecutive notes.
        if (ix < (len(curr_notes) - 1)):
            if (m.offset == curr_notes[ix + 1].offset and
                isinstance(curr_notes[ix + 1], note.Note)):
                removeIxs.append((ix + 1))
    curr_notes = [i for ix, i in enumerate(curr_notes) if ix not in removeIxs]

    return curr_notes