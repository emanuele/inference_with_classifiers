#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

"""
Confusion matrices from ICANN2011 MEG competition.
See http://www.cis.hut.fi/icann2011/meg/megicann_proceedings.pdf
p.14
"""

import numpy as np

# Huttunen et al.
huttunen = np.array([[94,  29, 16, 10,   1],
                     [22, 100, 10, 18,   1],
                     [25,  16, 51, 10,   0],
                     [ 3,   4, 12, 85,  21],
                     [ 2,   2,  4 , 3, 114]])

# Santana et al.
santana = np.array([[67,  54, 14, 15,   0],
                    [25, 110,  5, 11,   0],
                    [19,  14, 57, 12,   0],
                    [ 1,   1,  5, 59,  59],
                    [ 1,   0,  0,  4, 120]])

# Jylänki et al.
jylanki = np.array([[67, 32, 43,  8,   0],
                    [36, 89, 18,  8,   0],
                    [30,  6, 61,  4,   1],
                    [ 6,  6, 11, 78,  24],
                    [ 1,  0,  1,  8, 115]])

# Tu & Sun
tu = np.array([[56, 55, 36,  3,   0],
               [30, 96, 21,  4,   0],
               [33, 22, 46,  1,   0],
               [ 4,  3,  3, 95,  20],
               [ 1,  0,  0, 11, 113]])
