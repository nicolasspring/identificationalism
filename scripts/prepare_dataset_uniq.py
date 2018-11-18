#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import shuffle

def make_sets(f):
    targetdir = '../csv/testfiles/'
    with open(targetdir + 'new_train.csv', 'w', encoding='utf-8') as training, \
      open(targetdir + 'new_test.csv', 'w', encoding='utf-8') as test:
        sents = set()
        header = f.readline()
        training.write(header)
        test.write(header)
        for line in f:
            sents.add(line)
        shuffle(list(sents))
        len_test_set = 5000
        n = 0
        for sent in sents:
            if n < 5000:
                test.write(sent)
            else:
                training.write(sent)
            n += 1



def main():
    with open('../csv/train.csv', mode='r', encoding='utf-8') as f:
        make_sets(f)

if __name__ == '__main__':
    main()