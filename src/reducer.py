#!/usr/bin/env python
from itertools import groupby
from operator import itemgetter
import sys

def get_data(split_type = "\t"):
	"""get input for Reducer"""
	for line in sys.stdin:
		yield line.strip("\n").split(split_type)

def reducer():
	"""Reducer function for OOHack"""
	data = get_data(split_type = "\t")
	
	for current_word, record in groupby(data,itemgetter(0)):
		total = sum(int(record) for current_word, record in record)
		print "%s\t%d" % (current_word,total)
		
if __name__ == "__main__":
	reducer()