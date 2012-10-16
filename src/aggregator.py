"""This is the files to pull all the necessary information about a company"""

import MySQLdb
import numpy as np

class Aggregator(object):
	"""docstring for Aggregator
	   
	   	Aggregator class for importing the data into a sensible numpy array
	"""
	def __init__(self, host = "locahost", user = "root", passwd = "1234", db_name = "OOHack"):
		super(Aggregator, self).__init__()
		self.db = MySQLdb.connect( host = host, user = user, passwd = passwd, db_name = db_name)
		
	def create_company(self):
		"""docstring for create_company
			
			create a company attributes
		"""
		train_data = np.array(zeros(*,*))
		test_data = np.array(zeros(*,*))
		
		cursor = self.db.cursor()
		self.create_company_id(train_data,test_data,cursor)
		self.create_company_external()
		self.create_company_milestone()
		self.create_company_office()
		self.create_company_relationship()
		self.create_company_videoembed()
		self.create_company_competitor()
		self.create_company_funding()
		self.create_company_ipo()
		self.split_company(train_data,test_data)
		self.export_data(train_data,test_data)
	
	def split_company(self):
		"""docstring for split_company"""
		pass
		
	def create_company_id(self):
		"""docstring for create_company_id"""
		pass
	
	def create_company_external(self):
		"""docstring for create_company_external"""
		pass
		
	def create_company_milestone(self):
		"""docstring for create_company_milestone"""
		pass
	
	def create_company_office(self):
		"""docstring for fname"""
		pass
		
	def create_company_relationship(self):
		"""docstring for create_company_relationship"""
		pass
		
	def create_company_videoembed(self):
		"""docstring for create_company_videoembed"""
		pass
	
	def create_company_competitor(self):
		"""docstring for create_company_competitor"""
		pass
		
	def create_company_funding(self):
		"""docstring for create_company_funding"""
		pass
		
	def create_company_ipo(self):
		"""docstring for create_company_ipo"""
		pass
		
	def export_data(self):
		"""docstring for export_data"""
		pass