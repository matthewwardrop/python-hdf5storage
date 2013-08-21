import types
from abc import ABCMeta, abstractmethod, abstractproperty
from utility import encodeNumbers
import errors
###################### FOUNDATIONAL HDF5 CLASSES ###############################

class DataNode(object):
	__metaclass__ = ABCMeta
	
	@property
	def name(self):
		return self.__name
	
	def set_name(self,name):
		self.__name = name
	
	def node(self,node="",create=None,generator=None):
		# If node is not the root level node for this data object, recurse down tree
		if isinstance(node,str):
			nodes = node.split('/')
		elif isinstance(node,(list,tuple)):
			nodes = list(node)
		elif isinstance(node,(int,long,complex,float)):
			nodes = [node]
		else:
			raise errors.InvalidNodeError("'%s' is not a valid node identifier." % node)
		
		create = create if create is not None else self.attrs.get('auto_nodes',False)

		# Remove empty node strings
		while len(nodes) > 0 and nodes[0] == "":
			nodes.pop(0)

		# Resolve data node
		if len(nodes) == 0:
			return self
		else:
			try:
				nodeObj = self._node(nodes[0])
			except errors.NoSuchNodeError as e:
				if create:
					if isinstance(generator,types.FunctionType):
						nodeObj = generator(nodes[0])
					else:
						nodeObj = self._group_generate(nodes[0])
					self._add_node(nodes[0],data=nodeObj)
				else:
					raise e
			
			if not isinstance(nodeObj,DataNode):
				raise errors.InvalidNodeError("Cannot find a valid node with name: '%s'" % nodes[0])
			
			return nodeObj.node(nodes[1:],generator=generator)
	
	def node_attrs(self,node="/",attrs=None):
		node = self.node(node)
		if attrs:
			node.set_attrs(**attrs)
		return node.attrs
	
	def add_node(self,name,parent="/",data=None,dtype=None,attrs={}):
		parent = self.node(parent)
		parent._add_node(name,data=data,dtype=dtype,attrs=attrs)

	# TODO : Make sure node is non-empty
	def pop_node(self,node="/"):
		if isinstance(node,str):
			node = node.split("/")
		parent = self.node(node=node[:-1])
		return parent._del_node(node[-1:][0])

	############# Make the artificial distinction between groups and leaves 

	@property
	def groups(self):
		return list(child for child in self.nodes if isinstance(self._node(child),DataGroup) )

	@property
	def leaves(self):
		return list(child for child in self.nodes if isinstance(self._node(child),DataLeaf) )

	'''def group(self,node="",create=None,attrs={}):
		
		generator = None
		if create if create is not None else self.attrs.get('auto_nodes',False):
			generator = self._group_generate
		
		node = self.node(node,generator)
		
		if isinstance(node,DataGroup):
			return node
		else:
			raise errors.NoSuchGroupError("While a child of name '%s' does exist; it is a leaf node. Use `Storage['%s']` to extract the value of this child node." % (node.name,node.name))
	
	def leaf(self,node="/",data=None,dtype=None,attrs={},make_parents=True):
		if isinstance(node,str):
			nodes = node.split('/')
		
		node = nodes[-1:][0]
		
		generator = None
		if make_parents == True:
				generator = self._group_generator(attrs)

		parent = self.node(nodes[:-1],generator=generator)
		
		if parent is self:
		
			if data is not None:
				return self.add_node(node,data=data,dtype=dtype,attrs=attrs)
			else:
				if isinstance(node,DataLeaf):
					return node
				else:
					raise errors.NoSuchLeafError("While a child of name '%s' does exist; it is a group node. Use `DataNode.%s` or `DataNode.node('%s')` to extract this child node." % (node.name,node.name,node.name))
		
		return parent.leaf(node,attrs=attrs,dtype=dtype,data=data)'''

	######## Methods to be overridden ##########################################

	def _group_generate(self,node):
		raise NotImplementedError("No way to generate new groups for '%s' objects." % self.__class__.__name__)

	############# Dictionary imitation #####################################
	def __iter__(self):
		for x in self.leaves:
			yield x
	
	def __getitem__(self,key):
		if key in self.leaves:
			return self.node(key).value
		raise errors.NoSuchLeafError("'%s'"%key)
	
	def __setitem__(self,key,value):
		self.add_node(name=key,data=value,dtype=None)
	
	def __len__(self):
		return len(self.leaves)
	
	def pop(self,key):
		if key in self.leaves:
			return self._pop_node(key)
		raise errors.NoSuchLeafError("'%s'"%key)
	
	def keys(self):
		return self.leaves

	def items(self):
		return list( (x,self[x]) for x in self.leaves )

	############# Expose groups as attributes ##############################
	# Short hand methods
	def __getattr__(self,name):
		if name in self.groups or not name.startswith('_') and self.attrs.get('auto_nodes',False):
			return self.node(name)
		raise AttributeError

	############# Methods to be overriden ##################################
	
	def _node(self,node):
		raise NotImplementedError("No way to traverse beyond '%s' objects." % self.__class__.__name__)
	
	@property
	def nodes(self):
		raise NotImplementedError("No way to list the children '%s' objects." % self.__class__.__name__)

	def _add_node(self,name,data=None,dtype=None,attrs={}):
		raise NotImplementedError("No way to add nodes to '%s' objects." % self.__class__.__name__)

	def _pop_node(self,name):
		raise NotImplementedError("No way to pop the nodes of '%s' objects." % self.__class__.__name__)

	@abstractproperty
	def attrs(self):
		pass
	
	@abstractproperty
	def set_attrs(self,**kwargs):
		pass

class DataGroup(DataNode):
	
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
	@property
	def _hdf5_name(self):
		return encodeNumbers(self._hdf5_name_internal)
	
	@property
	def _hdf5_name_internal(self):
		return self.name
	
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
