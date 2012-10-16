
"""This is the files to pull all the necessary information about a company"""

import MySQLdb
import numpy as np
import ipdb
import datetime
import time
import zlib

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
		self.create_company_milestone(cursor)
		self.create_company_office(cursor)
		self.create_company_relationship(cursor)
		self.create_company_videoembed(cursor)
		self.create_company_competitor(cursor)
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
					self.data[row[0] - 1, 1] = 0.0
			except IndexError:
				self.data = np.concatenate( ( self.data, np.zeros((self.num_company,1)) ), 1)
				try:
					self.data[row[0] - 1, 1] = float(row[2])
				except TypeError:
					self.data[row[0] - 1, 1] = 0.0
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
		query = "SELECT company_id,COUNT(id) FROM company_external_link GROUP BY company_id"
		
		cursor.execute(query)
		
		for row in cursor.fetchall():
			try:
				self.data[row[0] - 1,4] = row[1]
			except IndexError:
				self.data = np.concatenate( (self.data, np.zeros( (self.num_company, 1) ) ) ,1 )
				self.data[row[0] - 1,4] = row[1]
		print "\t...Done"
		
	def create_company_milestone(self,cursor):
		"""docstring for create_company_milestone"""
		print "Company milestone creation"
		query = "SELECT company_id,COUNT(id) FROM company_milestone GROUP BY company_id"
		cursor.execute(query)
		
		for row in cursor.fetchall():
			try:
				self.data[row[0] - 1,5] = row[1]
			except IndexError:
				self.data = np.concatenate( (self.data, np.zeros( (self.num_company, 1) ) ) ,1 )
				self.data[row[0] - 1,5] = row[1]
		
		print "\t...Done"
		
	
	def create_company_office(self,cursor):
		"""docstring for create_company_office"""
		print "Company office attribute creation"
		query = "SELECT company_id, country_code FROM company_office"
		cursor.execute(query)
		
		for row in cursor.fetchall():
			try:
				self.data[row[0] - 1,6] = zlib.adler32(str(row[1]))
			except IndexError:
				self.data = np.concatenate( (self.data, np.zeros( (self.num_company, 1) ) ) ,1 )
				self.data[row[0] - 1,6] = zlib.adler32(str(row[1]))
		
		print "\t...Done"
		
	def create_company_relationship(self,cursor):
		"""docstring for create_company_relationship"""
		print "Company relationship creatioin"
		query = "SELECT company_id,person_id FROM company_relationship"
		cursor.execute(query)
		
		company_relation_dict = dict()
		
		for row in cursor.fetchall():
			if str(row[0]) in company_relation_dict.keys():
				company_relation_dict[str(row[0])].append(row[1])
			else:
				company_relation_dict[str(row[0])] = [row[1]]
		
		query = "SELECT person_id,COUNT(company_id) FROM company_relationship GROUP BY person_id"
		cursor.execute(query)
		
		people_relation_dict = dict()
		for row in cursor.fetchall():
			people_relation_dict[str(row[0])] = float(row[1])
		
		for key in company_relation_dict.keys():
			for people in company_relation_dict[key]:
				try:
					self.data[float(key) - 1,7] = self.data[float(key) - 1,7] + float(people_relation_dict[str(people)])
				except IndexError:
					self.data = np.concatenate( (self.data, np.zeros( (self.num_company, 1) ) ) ,1 )
					self.data[float(key) - 1,7] = float(people_relation_dict[str(people)])
		
		print "\t...Done"
		
	def create_company_videoembed(self,cursor):
		"""docstring for create_company_videoembed"""
		print "Company milestone creation"
		query = "SELECT company_id,COUNT(id) FROM company_video_embed GROUP BY company_id"
		cursor.execute(query)
		
		for row in cursor.fetchall():
			try:
				self.data[row[0] - 1,8] = row[1]
			except IndexError:
				self.data = np.concatenate( (self.data, np.zeros( (self.num_company, 1) ) ) ,1 )
				self.data[row[0] - 1,8] = row[1]
		
		print "\t...Done"
	
	def create_company_competitor(self,cursor):
		"""docstring for create_company_competitor"""
		print "Company milestone creation"
		query = "SELECT company_id,COUNT(competitor_id) FROM competitor GROUP BY company_id"
		cursor.execute(query)
		
		for row in cursor.fetchall():
			try:
				self.data[row[0] - 1,9] = row[1]
			except IndexError:
				self.data = np.concatenate( (self.data, np.zeros( (self.num_company, 1) ) ) ,1 )
				self.data[row[0] - 1,9] = row[1]
		
		print "\t...Done"
		
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
