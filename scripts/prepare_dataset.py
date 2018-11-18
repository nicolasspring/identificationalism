#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Tim Graf
# RAM-Peak: 7MB
# Runtime: 8min 30s on SurfacePro4

import gzip
import random

def split_corpus(infile, targetdir, n=1000):
    '''Split a corpus into devset, trainingset and testset and writes
    each set into a seperate file.
    Args:
        infile: an opened textfile with annotated text.
        targetdir: directory where the splitted sentences shall
                   be written to.
        n: desired size of test- and devset in sentences.
    '''
    sample(iter_sents(infile), n, targetdir)


def iter_sents(infile):
    '''Iterate over sentences in the corpus and yield them.
    Args: 
        infile: an opened textfile with annotated text.
    Returns: an iterator over the sentences.
    '''
    sent = []
    for line in infile:
        yield line
        #line = line.split()
        #if line[1] == '$.':
         #   sent.append(line[0])
         #   yield sent
         #   sent = []
        #else:
        #    sent.append(line[0])


def sample(iterator, k, targetdir):
    '''Sample sentences from an iterator into dev-, training- and
    testset and write them to outfiles.
    Args:
        iterator: an iterator over sentences
        k: desired test-, and devset size
        targetdir: desired directory of outputfiles
    '''
    with open(targetdir + 'new_train.csv', 'w', encoding='utf-8') as training, \
    open(targetdir + 'test1.csv', 'w', encoding='utf-8') as dev, \
    open (targetdir + 'test2.csv', 'w', encoding='utf-8') as test:
        testdev = []
        for t, sent in enumerate(iterator):
            if t < k*2:
                testdev.append(sent)
            else:
                m = random.randint(0,t)
                if m < k*2:
                    training.write(testdev[m])
                    testdev[m] = sent
                else:
                    training.write(sent)
        for sent in testdev[:k]:
            dev.write(sent)
        for sent in testdev[k:]:
            test.write(sent)


def main():
    with open('../csv/train.csv', mode='r', encoding='utf-8') as f:
        split_corpus(f, '../csv/testfiles/', 2500)
        #for i in iter_sents(f):
            #print(i)
            

if __name__ == '__main__':
    main()
