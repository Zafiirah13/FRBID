#!usr/bin/env python
"""
Authors : Zafiirah Hosenie
Email : zafiirah.hosenie@gmail.com or zafiirah.hosenie@postgrad.manchester.ac.uk
Affiliation : The University of Manchester, UK.
License : MIT
Status : Under Development
Description :
Python implementation for FRBID: Fast Radio Burst Intelligent Distinguisher.
This code is tested in Python 3 version 3.5.3  
"""

import os
import sys
import json


def makedirs(d):
    '''Make a directory if it does not exist'''
    if not os.path.exists(d):
        os.makedirs(d)
            
def ensure_dir(f):
    '''Ensure that the directory exist, if it does not exist, create one automatically'''
    
    d = os.path.dirname(f)
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except:
            pass
