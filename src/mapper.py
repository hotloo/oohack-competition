#!/usr/bin/env python

import sys

def get_data(split_type = " "):
	"""data generator for mapper"""
	for line in sys.stdin:
		# line validity check
	
		yield line.strip("\n").split(split_type)

def mapper():
	"""Mapper function for OOHack"""
	
	data = get_data(split_type = " ")
	for record in data:
		# check lines validtity
		for word in record:
			print "%s\t%d" % (word,1)
			
if __name__ == "__main__":
	mapper()