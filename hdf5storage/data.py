import re
import tables

import warnings
warnings.filterwarnings('ignore',category=tables.NaturalNameWarning)

import numpy as np

from .interfaces import DataNode,DataGroup,DataLeaf,HDF5Node,HDF5Group,HDF5Leaf,HDF5LeafTable,HDF5LeafArray
from . import errors

from utility import decodeNumbers

#################### The Main DATA CLASS #######################################
#
# The DataNode object
class Storage(HDF5Group,DataGroup):
	'''
	Storage (name="",attrs={})
	
	The storage object that acts somewhat like a dictionary, and that can be
	output to disk as an HDF5 formatted data set (or as a series of matlab
	files). The Storage library is designed mostly for numeric data; though
	rudimentary support for strings exists also. Do not expect more complicated
	objects to work.

	In particular, the storage library is designed to work with any number 
	format with which numpy is familiar. The internal storage format used by
	pytables also uses numpy.
	
	Parameters
	----------
	name : The name to be used for the data set. Can be any string.
	attrs : Attributes for this data Storage node.
	
	Explanation
	-----------
	The Storage class inherits from HDF5Group and DataGroup. The HDF5Group class
	only defines internal methods and properties that are used to output the
	Data* objects to HDF5 format using pytables. These methods are prepended 
	with '_hdf5'. The DataGroup class defines the methods with which the user is
	most likely to be interested.

	The DataGroup class implements data structure methods which allows it to act
	much like the standard python dictionary object. i.e.
	>>> d = Storage()
	>>> d['test'] = 2.5

	pytables does not support keys being numeric; however HDF5Storage will allow
	you to set a numeric key. This means that numeric keys are transmitted as 
	strings to pytables. This is done in a lossless way; and when the HDF5 files
	are reloaded into HDF5Storage objects, they are converted back to numeric 
	keys. If you open the HDF5 files in external HDF5 viewers, a float key of
	`1.2e8` will be transformed to 'float(1.2e8)'; and so on. Thus:
	>>> d[2.5] = 3.6
	Works fine.

	HDF5Storage also supports the hierarchical nature of HDF5. This is accessed 
	by the node notation (or, where node names do not conflict with methods,
	by the attribute notation).
	>>> d.x # (This will return the DataGroup node at x)
	If x is not subnode of d which is also a DataGroup; this will result in an
	AttributeError being raised. Nodes can be automatically created by setting 
	an attribute with key 'auto_nodes' on the DataGroup objects.

	A summary of the methods and behaviour of Storage objects is presented below
	under "Examples".

	Examples
	--------

	Equivalent:
	d['test'] = 1
	d.node('test').value
	d.leaf('test').value

	d.x
	d.node('x')
	d.group('x')

	d.x.y.set_attrs(test=1,auto_nodes=True)
	d.node_attrs("x/y",attrs={'auto_nodes':True})

	d.x.y.node('test').set_attrs(auto=True)
	d.x.y.leaf('test').set_attrs(auto=True)
	d.node_attrs('x/y/test',attrs={'auto':True})
	'''
	
	def __init__(self,name="",attrs={}):
		self.set_name(name)
		self.__children = {}
		self.__attributes = {'type':'storage'}
		self.set_attrs(**attrs)
	
	######### USER FACING METHODS ##########################################
	
	# String representation
	def __repr__(self,depth=0):
		nl = "|"
		output = []
		if depth == 0:
			output = ["Storage:%s"%self.name]
		leaves = []
		nodes = []
		for node,value in self.__children.items():
			if isinstance(value,HDF5Group):
				nodes.append("+%s"%value.name)
				nodes.extend(map(lambda x: "%s%s"%(nl,x),value.__repr__(depth=depth+1)))
			else:
				leaves.append('-'+node)
				
		if len(nodes) > 0:
			output.extend(nodes)
		if len(leaves) > 0:
			output.extend(leaves)
		
		if depth > 0:
			return output
		return "\n|".join(output)
	
	def __dir__(self):
		return dir(type(self)) + list(self.groups)
	
	######################### DataGroup Methods ############################
	
	# More powerful methods
	def _node(self,node=None):
		if node in self.__children:
			return self.__children[node]
		raise errors.NoSuchNodeError("Storage object '%s' does not have node '%s'" % (self.name,node))
	
	@property
	def nodes(self):
		return list(self.__children.keys())
	
	def _group_generate(self,node):
		attrs = {}
		if 'auto_nodes' in self.attrs:
			attrs['auto_nodes'] = self.attrs['auto_nodes']
		return Storage(name=node,attrs=attrs)
	
	def _add_node(self,name,data=None,dtype=None,attrs={}):
		if isinstance(data,Storage): # Merge data type if name is none
			if name is None or name == '':
				self.__children.update(copy.deepcopy(data._children)) 
				return
		
		name = decodeNumbers(name)
		if isinstance(name,str):
			if name is None or not re.match("^[a-zA-Z][a-zA-Z\_]*$",name):
				raise errors.InvalidNodeNameError("Node names must be a string with length > 0 and that start with a letter. '%s' was provided."%name)
		elif isinstance(name,(int,float,long,complex)):
			pass
		else:
			raise errors.InvalidNodeNameError("'%s'"%name)

		self.__children[name] = getDataType(name=name,data=data,dtype=dtype,attrs=attrs)
		if isinstance(self.__children[name],DataGroup):
				if name in dir(type(self)):
					warnings.warn(errors.InaccessibleGroupNodeWarning("The name chosen for the group node '%s' will not be accessible as an attribute, because it clashes with the name of a method."%name))

	def _pop_node(self,node):
		return self.__children.pop(node)

	@property
	def attrs(self):
		return self.__attributes
	
	def set_attrs(self,**kwargs):
		for key, value in kwargs.items():
			self.__attributes[key] = value
	
	######################### HDF5 Methods #####################################
	#
	# Get the children of this root node.
	
	@property
	def _hdf5_group_children(self):
		return self.__children.values()
	
	@property
	def _hdf5_attrs(self):
		return self.__attributes
	
	@property
	def _hdf5_desc(self):
		return ""
	
	@classmethod
	def _hdf5_populate(cls,hdfNode):
		args = {}
		for attr in hdfNode._v_attrs._f_list():
			args[attr] = getattr(hdfNode._v_attrs,attr)
		return {'data':Storage._from_node(hdfNode,prefix=hdfNode._v_pathname),'args':args}
	
	######################### DATA ARCHIVAL METHODS ############################
	
	#
	# Save the data
	def __rshift__(self,location):
		if location.endswith('.mat'):
			name = location[:-4]
			md = {}
			for leaf in self.leaves:
				md[leaf] = self[leaf]
			import scipy.io as spio
			spio.savemat("%s.%s.mat"%(name,self._hdf5_name),md)
			for group in self.groups:
				self.node(group).__rshift__("%s.%s.mat"%(name,self._hdf5_name))
		else:
			h5file = tables.openFile(location, mode = "w", title = self._hdf5_name)
			self._hdf5_write(h5file,h5file.root);
			h5file.close()
	
	#
	# Restore the data
	@classmethod
	def _load(cls,location):
		h5file = tables.openFile(location, mode='r')
		data = cls._from_node(h5file.getNode('/'))
		h5file.close()
		return data
	
	@classmethod
	def _from_node(cls,node,prefix=""):
		retData = cls()
		retData.__examine_nodes(node,prefix=prefix)
		return retData
	
	def __examine_nodes(self,node,prefix=""):
		for leaf in node._v_leaves.values():
			populateDataType(self,leaf,prefix=prefix)
		for subnode in node._v_children.values():
			if subnode is not None:
				if not populateDataType(self,subnode,prefix=prefix):
					self.__examineNodes(subnode,prefix=prefix)

from datatypes import getDataType, populateDataType
