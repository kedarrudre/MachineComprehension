#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
import os
import sys
from collections import defaultdict
import numpy as np
import re

# Common useful functions
def formatForPrint(str):
    s = re.sub(r"\\newline", "\\n", str)
    return s

def formatForProcessing(str):
    s = re.sub(r"\\newline", " ", str)
    s = re.sub(r"\'s", "", s)
    return s
