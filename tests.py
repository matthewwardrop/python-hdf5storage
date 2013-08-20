import unittest
from hdf5storage import Storage
import numpy as np

class TestUnitNewCreation(unittest.TestCase):
	
	def setUp(self):
		self.d = Storage('Test')	
	
	def test_set(self):
		self.d['cat'] = np.array(1)
		self.d['dog'] = 'test'
	
	def test_node_create(self):
		self.d.group('x',True)['cat'] = 0
	
	def test_repr(self):
		self.d.set_attrs(auto_nodes=True)
		self.d['blast'] = "tet"
		self.d.x['cat'] = 1
		self.d.y['dog'] = 2
		self.d.x.z['asd'] = 1
		print(self.d)
	
	def test_autonodes(self):
		d = Storage('Test',attrs={'auto_nodes':True})
		self.assertEquals( isinstance(d.x.y.u.asdasdasd,Storage), True)
	
	def test_attr(self):
		self.d.set_attrs(cat='cat')
		self.assertEquals(self.d.attrs['cat'],'cat')
	
	def test_dict(self):
		self.d['x'] = 1
		self.d['u'] = 2
		self.assertEquals(set(self.d),set(['x','u']))
		self.assertEquals(len(self.d),2)
	
	##### TEST DATA TYPES ##################################################
	def test_dict(self):
		self.d['test'] = {'dog': 3.2, 2.3: 1.5}
		self.d >> 'output.hdf5'
		d2 = self.d._load('output.hdf5')
		self.assertEqual(d2['test'],{'dog':3.2, 2.3: 1.5})
	
	def test_array(self):
		self.d['test'] = np.array([1,2,3])
		self.d >> 'output.hdf5'
		d2 = self.d._load('output.hdf5')
		self.assertEqual(d2['test'].tolist(),np.array([1,2,3]).tolist())
	
	def test_list(self):
		self.d['test'] = [1,2,3]
		self.d >> 'output.hdf5'
		d2 = self.d._load('output.hdf5')
		self.assertEqual(d2['test'],[1,2,3])
	
	##### Test Matlab ######################################################
	def test_repr(self):
		self.d.set_attrs(auto_nodes=True)
		self.d['blast'] = "tet"
		self.d.x['cat'] = 1
		self.d.y['dog'] = 2
		self.d.x.z['asd'] = 1
		
		self.d >> "test.mat"

if __name__ == '__main__':
    unittest.main()
