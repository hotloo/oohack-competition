
"""This is the files to pull all the necessary information about a company"""

import MySQLdb
import numpy as np
import ipdb
import datetime
import time

class Aggregator(object):
	"""docstring for Aggregator
	   
	   	Aggregator class for importing the data into a sensible numpy array
	"""
	def __init__(self, host = "localhost", user = "root", passwd = "1234", db_name = "OOHack"):
		super(Aggregator, self).__init__()
		self.db = MySQLdb.connect( host = host, user = user, passwd = passwd, db = db_name)
		
	def create_company(self,filename):
		"""docstring for create_company
			
			create a company attributes
		"""
				
		cursor = self.db.cursor()
		self.create_company_id(cursor)
		self.create_company_external(cursor)
		self.create_company_milestone()
		self.create_company_office()
		self.create_company_relationship()
		self.create_company_videoembed()
		self.create_company_competitor()
		self.create_company_funding()
		self.create_company_ipo(cursor)
		self.split_company()
		self.export_data(filename)
	
	def split_company(self):
		"""docstring for split_company"""
		pass
		
	def create_company_id(self,cursor):
		"""docstring for create_company_id"""
		print "Company ID creation"
		query = "SELECT id, name, number_of_employees, founded_year, founded_month, founded_day, deadpooled_year, deadpooled_month, deadpooled_day FROM COMPANY"
		
		self.num_company = cursor.execute(query)
		
		self.data = np.array(np.zeros((self.num_company,1)))
		self.company_name = dict()
		
		for row in cursor.fetchall():
			self.data[row[0] - 1] = 0
			self.company_name[str(row[1])] = int(row[0]) - 1
			
			# Number of employee
			try:
				try:
					self.data[row[0] - 1, 1] = float(row[2])
				except TypeError:
					self.data[row[0] - 1, 1] = np.nan
			except IndexError:
				self.data = np.concatenate( ( self.data, np.zeros((self.num_company,1)) ), 1)
				try:
					self.data[row[0] - 1, 1] = float(row[2])
				except TypeError:
					self.data[row[0] - 1, 1] = np.nan
			# Launch Time
			try:
				try:
					self.data[row[0] - 1, 2] = time.mktime(datetime.datetime(row[3],row[4],row[5]).timetuple())
				except (TypeError,ValueError):
					self.data[row[0] - 1, 2] = np.nan
			except IndexError:
				self.data = np.concatenate( ( self.data, np.zeros((self.num_company,1)) ), 1)
				try:
					self.data[row[0] - 1, 2] = time.mktime(datetime.datetime(row[3],row[4],row[5]).timetuple())
				except (TypeError,ValueError):
					self.data[row[0] - 1, 2] = np.nan
			# Alive attributes
			try:
				if row[6] == None or row[7] == None or row[8] == None :
					self.data[row[0] - 1, 3] = 1.0
				else:
					self.data[row[0] - 1, 3] = 0.0
			except IndexError:
				self.data = np.concatenate( ( self.data, np.zeros((self.num_company,1)) ), 1)
				if row[6] == None or row[7] == None or row[8] == None :
					self.data[row[0] - 1, 3] = 1.0
				else:
					self.data[row[0] - 1, 3] = 0.0

		print "\t...Done"
		
	def create_company_external(self,cursor):
		"""docstring for create_company_external"""
		print "Company external creation"
		query = "SELECT company_id,count(id) FROM company_external_link GROUP BY company_id"
		
		cursor.execute(query)
		
		for row in cursor.fetchall():
			try:
				self.data[row[0] - 1,4] = row[1]
			except IndexError:
				self.data = np.concatenate( (self.data, np.zeros( (self.num_company, 1) ) ) ,1 )
				self.data[row[0] - 1,4] = row[1]
		print "\t...Done"
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
		
	def create_company_ipo(self,cursor):
		"""docstring for create_company_ipo"""
		print "IPO label creation"
		query = "SELECT company_id FROM ipo"
		
		cursor.execute(query)
		
		for row in cursor.fetchall():
			self.data[row[0] - 1] = 1
		
		print "\t...Done"
		
	def export_data(self,file):
		"""docstring for export_data"""
		train_file = file + "_train"
		test_file = file + "_test"
		
		positive_example = self.data[np.where(self.data[:,0] == 1)[0],:]
		negative_example = self.data[np.where(self.data[:,0] == 0)[0],:]
		
		rand_perm = np.random.permutation(negative_example.shape[0])
		
		np.save(train_file,np.concatenate((positive_example,negative_example[rand_perm[0:804],:])))
		np.save(test_file,negative_example[rand_perm[805:],:])
		
if __name__ == "__main__":
	company_profile = Aggregator()
	company_profile.create_company("../data/oohack")
