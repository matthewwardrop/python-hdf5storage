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
		self.d.node('x',True)['cat'] = 0
	
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
	
	def test_dict_like(self):
		self.d['x'] = 1
		self.d[2.5] = 2
		self.d[2.124] = 2
		self.d[80.0127665245] = 3
		self.assertEquals(self.d[80.0127665245],3)
		self.assertEquals(set(self.d),set(['x',2.5,2.124,80.0127665245]))
		self.assertEquals(len(self.d),4)
		self.d>>'test.hdf5'
		
		d = Storage._load('test.hdf5')
		self.assertEquals(set(d),set(['x',2.5,2.124,80.0127665245]))
		self.assertEquals(len(d),4)
		
		d.pop('x')
		self.assertEquals(set(d),set([2.5,2.124,80.0127665245]))
		self.assertEquals(len(d),3)

	##### TEST ATTRIBUTES ##################################################
	def test_attr(self):
		self.d.set_attrs(cat='cat')
		self.assertEquals(self.d.attrs['cat'],'cat')
		self.d.set_attrs(auto_nodes=True)
		self.d.x['dim'] = [np.array([1,2,3]),np.array([1,23,4])]
		self.d.node_attrs('x/dim/0',{'test':1,'test2':2})
		self.assertEquals(self.d.node_attrs('x/dim/0'),{'test':1})

		self.d >> 'output.hdf5'
		d2 = self.d._load('output.hdf5')
		self.assertEquals(d2.node_attrs('x/dim/0'),{'test':1})
	
	def test_nested_attr(self):
		self.d['test2'] = [np.array([1])]
		self.d.node_attrs('test2/0',attrs={'attr':1})
		
		self.d >> 'output2.hdf5'
		d2 = Storage._load('output2.hdf5')
		self.assertEquals(d2.node_attrs('test2/0'),{'attr':1})
	
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

	def test_warnings(self):
		self.d.set_attrs(auto_nodes=True)
		self.d.node('nodes',create=True)['test'] = 1

if __name__ == '__main__':
    unittest.main()
