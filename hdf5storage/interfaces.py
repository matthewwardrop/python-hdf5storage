from abc import ABCMeta, abstractmethod, abstractproperty
###################### FOUNDATIONAL HDF5 CLASSES ###############################

class DataNode(object):
	__metaclass__ = ABCMeta
	
	@abstractproperty
	def attrs(self):
		pass
	
	def set_attrs(self,**kwargs):
		pass

class DataGroup(DataNode):
	
	@abstractmethod
	def group(self,node="",create=None,attrs={}):
		pass
	
	@abstractproperty
	def groups(self):
		pass
	
	@abstractmethod
	def leaf(self,node="/",data=None,dtype=None,attrs={},make_parents=True):
		pass
	
	@abstractproperty
	def leaves(self):
		pass
	
	@abstractmethod
	def node_attrs(self,node="/",attrs=None):
		pass

class DataLeaf(DataNode):
	__metaclass__ = ABCMeta
	
	@abstractproperty
	def value(self):
		pass
	
	@abstractmethod
	def set_value(self,value):
		pass
	
	def append(self,value):
		raise NotImplementedError

class HDF5Node(object):
	__metaclass__ = ABCMeta
	
	#
	# Returns the name to be used for this object's node
	@abstractproperty
	def _hdf5_name(self):
		raise NotImplementedError
	
	#
	# Returns the description of this object's node
	@abstractproperty
	def _hdf5_desc(self):
		return ""
	
	#
	# Returns the set of user attributes required to rebuild Python tree
	@abstractproperty
	def _hdf5_attrs(self):
		return {}
	
	@abstractmethod
	def _hdf5_populate(cls,hdfNode):
		pass
	
	@abstractmethod
	def _hdf5_write(cls,h5file,node):
		raise NotImplementedError

class HDF5Group(HDF5Node):
	__metaclass__ = ABCMeta
	
	#
	# Returns a series of HDF5Node inheritance objects
	@abstractproperty
	def _hdf5_group_children(self):
		return []
	
	#
	# Write this node to an HDF5 file
	def _hdf5_write(self,h5file,node):
		
		# Set node attributes
		for attribute,value in self._hdf5_attrs.items():
			node._f_setAttr(attribute,value)
		
		# Create subgroups
		for child in self._hdf5_group_children:
			if isinstance(child,HDF5Group):
				subgroup = h5file.createGroup(node,child._hdf5_name,child._hdf5_desc)
				child._hdf5_write(h5file, subgroup)
			else:
				child._hdf5_write(h5file, node)

class HDF5Leaf(HDF5Node):
	__metaclass__ = ABCMeta
	
	def _hdf5_leaf_write_attrs(self,leafObject):
		for attribute,value in self._hdf5_attrs.items():
			leafObject._f_setAttr(attribute,value)

class HDF5LeafTable(HDF5Leaf):
	__metaclass__ = ABCMeta
	
	@abstractproperty
	def _hdf5_leaf_table_structure(self):
		return NotImplementedError
	
	@abstractproperty
	def _hdf5_leaf_table_entries(self):
		return []
	
	def _hdf5_write(self,h5file,group):
		table = h5file.createTable(group, self._hdf5_name, self._hdf5_leaf_table_structure, self._hdf5_desc)
		#Actually write table
		entryDetails = table.row
		for entry in self._hdf5_leaf_table_entries:
			for key in entry:
				entryDetails[key] = entry[key]
			entryDetails.append()
		table.flush()
		self._hdf5_leaf_write_attrs(table)

class HDF5LeafArray(HDF5Leaf):
	__metaclass__ = ABCMeta
	
	@abstractproperty
	def _hdf5_leaf_array(self):
		return []
	
	def _hdf5_write(self,h5file,group):
		self._hdf5_leaf_write_attrs(h5file.createArray(group,self._hdf5_name, self._hdf5_leaf_array, self._hdf5_desc))
