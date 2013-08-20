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
	output to disk as an HDF5 formatted data set.
	
	Parameters
	----------
	name : The name to be used for the data set. Can be any string.
	attrs : Attributes for this data Storage node.
	
	Explanation
	-----------
	
	Examples
	--------
	'''
	
	def __init__(self,name="",attrs={}):
		self.__name = name
		self.__children = {}
		self.__attributes = {'type':'data'}
		self.set_attrs(**attrs)
	
	@property
	def __auto_nodes(self):
		return self.attrs.get('auto_nodes',False)
	
	######### USER FACING METHODS ##########################################
	
	# Short hand methods
	def __getattr__(self,name):
		return self.group(name)
	
	# String representation
	def __repr__(self,depth=0):
		nl = "|"
		output = []
		if depth == 0:
			output = ["Storage:%s"%self.__name]
		leaves = []
		nodes = []
		for node,value in self.__children.items():
			if isinstance(value,HDF5Group):
				nodes.append("+%s"%value.__name)
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
		return dir(type(self)) + list(self.__children.keys())
	
	######### DICTIONARY IMITATION #########################################
	def __iter__(self):
		for x in self.leaves:
			yield x
	
	def __getitem__(self,key):
		value = self.__get_leaf(key)
		if isinstance(value,DataLeaf):
			return value.value
		return value
	
	def __setitem__(self,key,value):
		self.__add_leaf(name=key,data=value,dtype=None)
	
	def __len__(self):
		return len(self.leaves)
	
	######### PRIVATE METHODS ##############################################
	
	def __get_leaf(self,name):
		if name in self.__children and isinstance(self.__children[name],DataLeaf):
			return self.__children[name]
		raise errors.NoSuchLeafError("'%s'"%name)
	
	def __add_leaf(self,name,data=None,dtype=None,attrs={}):
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
			
		
		# TODO: Consider whether objects should be able to handle adding their own data
		#if self.__children.has_key(name) and getattr(self.__children.get(name),'_addData',None) is not None:
		#	if (self.__children.get(name)._addData(data=data,dtype=dtype)):
		#		return
		self.__children[name] = getDataType(name=name,data=data,dtype=dtype,attrs=attrs)
	
	######################### DataGroup Methods ############################
	
	# More powerful methods
	def group(self,node="",create=None,attrs={}):
		
		# If node is not the root level node for this data object, recurse down tree
		if isinstance(node,str):
			nodes = node.split('/')
		elif isinstance(node,(list,tuple)):
			nodes = list(node)
		else:
			raise errors.InvalidNodeError("'%s' is not a valid node identifier." % node)
		
		# Remove empty node strings
		while len(nodes) > 0 and nodes[0] == "":
			nodes.pop(0)
		
		# Resolve data node
		if len(nodes) == 0:
			return self
		else:
			if nodes[0] in self.__children:
				if isinstance(self.__children[nodes[0]],DataGroup):
					return self.__children[nodes[0]].group(nodes[1:],create=create)
				else:
					raise errors.NoSuchGroupError("While a child of name '%s' does exist; it is a leaf node. Use `Storage['%s']` to extract the value of this child node." % (name,name))
			else:
				if create if create is not None else self.__auto_nodes:
					if 'auto_nodes' in self.attrs and 'auto_nodes' not in attrs:
						attrs['auto_nodes'] = self.attrs['auto_nodes']
					self.__children[nodes[0]] = Storage(name=nodes[0],attrs=attrs)
					return self.__children[nodes[0]].group(nodes[1:],create=create)
		
		raise errors.NoSuchGroupError("'%s'" % node)
	
	@property
	def groups(self):
		return list(child for child in self.__children if isinstance(self.__children[child],DataGroup) )
	
	def leaf(self,node="/",data=None,dtype=None,attrs={},make_parents=True):
		if isinstance(node,str):
			nodes = node.split('/')
		parent_node = nodes[:-1]
		node = nodes[-1:][0]
		
		group = self.group(parent_node,create=make_parents)
		
		if group is self:
		
			if data is not None:
				return self.__add_leaf(node,data=data,dtype=dtype,attrs=attrs)
			else:
				return self.__get_leaf(node)
		
		return group.leaf(node,attrs=attrs,dtype=dtype,data=data)
	
	@property
	def leaves(self):
		return list(child for child in self.__children if isinstance(self.__children[child],DataLeaf) )
	
	def node_attrs(self,node="/",attrs=None):
		parent_node = node.split('/')[:-1]
		name = node.split('/')[-1:]
		
		group = self.group(parent_node,create=False)
		
		if group is self:
			if name in self.__children:
				nodeObj = self.__children[name]
				if isinstance(nodeObj,DataNode):
					if attrs is None:
						return nodeObj.attrs
					else:
						return nodeObj.set_attrs(**attrs)
				raise InvalidNodeError("'%s' is not a DataNode object." % name)
			raise InvalidNodeError("'%s'" % name)
		return group.node_attrs(name,attrs=attrs)
	
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
	def _hdf5_name_internal(self):
		return self.__name
	
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
				self.group(group).__rshift__("%s.%s.mat"%(name,self._hdf5_name))
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
