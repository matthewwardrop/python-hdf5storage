import os, math, copy
import tables
import numpy as np

from .interfaces import DataNode,DataGroup,DataLeaf,HDF5Node,HDF5Group,HDF5Leaf,HDF5LeafTable,HDF5LeafArray
from .data import Storage

#################### DATA TYPE CLASSES #########################################

#
# Get the appropriate data type for the input data
def getDataType(name,data=None,dtype=None,attrs={}):
	if isinstance(data,DataNode):
		data = copy.copy(data)
		data.set_name = name
		return data
	
	if dtype == 'array' or dtype is None and isinstance(data,np.ndarray):
		return DataArray(name,data,attrs=attrs)
	if dtype == 'storage' or dtype is None and isinstance(data,Storage):
		children = copy.deepcopy(data._children)
		dataObj = Storage(name=name,attrs=attrs)
		dataObj._children = children
		return dataObj
	if dtype == 'dict' or dtype is None and isinstance(data,dict):
		return DataDict(name,data,attrs=attrs) 
	if dtype == 'list' or dtype is None and isinstance(data,list):
		dataList = DataList(name,data,attrs=attrs)
		return dataList
	
	# LOOKING BAD! Let's try converting to np.array, and try again.
	try:
		data = np.array(data)
		return getDataType(name,data,dtype,attrs=attrs)
	except:
		pass
		
	#if dtype == 'var_array' or dtype is None and type(data) in [np.ndarray,list]:
	#	return DataVariableArray(name,data,**args)
	#if dtype == 'string' or type(data) in [str]:
	#	return DataString(name,data)
	raise ValueError, "Unknown data type for type %s (%s)" % (str(type(data)),str(data))

#
# Return true if populateDataType has handled all children of a node
def populateDataType(dataObj,hdfNode,extractOnly = False,prefix=''):
	try:
		type = hdfNode._f_getAttr('type')
	except AttributeError, e:
		if extractOnly:
			return None
		return False
	
	path = hdfNode._v_pathname[len(prefix):]
	name = os.path.basename(path)
	node = os.path.dirname(path) if os.path.dirname(path) != '' else "/"
	
	if not hdfNode._v_pathname.startswith(prefix):
		raise ValueError, "Invalid node (outside prefix)."
	
	if type == 'storage':
		obj = Storage
		dtype = 'storage'
	elif type == 'data_dict':
		obj = DataDict
		dtype = 'dict'
	elif type == 'data_array':
		obj = DataArray
		dtype = 'array'
	elif type == 'data_variablearray':
		obj = DataVariableArray
		dtype = 'var_array'
	elif type == "data_list":
		obj = DataList
		dtype = "list"
	else:
		raise ValueError, "Unknown data type %s"%type
	
	if issubclass(obj,HDF5Node):
		extracted = obj._hdf5_populate(hdfNode)
	else:
		raise Exception("Unknown type")
	
	if extractOnly:
		return extracted
	
	dataObj.add_node(name=name,parent=node,data=extracted['data'],dtype=dtype,attrs=extracted['args'])
	return True

class DataDict(HDF5LeafTable,DataLeaf):
	
	KEY_LENGTH = 30
	
	def __init__(self,name,data,attrs={}):
		self.set_name(name)
		self.__dict = data
		self.__props = attrs
	
	############## HDF5 Methods ############################################
	
	@property
	def _hdf5_desc(self):
		return ""
	
	@property
	def _hdf5_attrs(self):
		attrs = {'type':'data_dict'}
		attrs.update(self.__props)
		return attrs
	
	@property
	def _hdf5_leaf_table_structure(self):
		class DictTable(tables.IsDescription):
			key_string = tables.StringCol(self.KEY_LENGTH)
			key_float = tables.Float64Col()
			value_float = tables.Float64Col()
		return DictTable
	
	@property
	def _hdf5_leaf_table_entries(self):
		entries = []
		for key,value in self.__dict.items():
			if isinstance(key,str):
				entries.append( {'key_string':key, 'key_float':None, 'value_float':value} )
			else:
				entries.append( {'key_string':None, 'key_float':key, 'value_float':value} )
		return entries
	
	@classmethod
	def _hdf5_populate(cls,hdfNode):
		d = {}
		for row in hdfNode.iterrows():
			if math.isnan(row['key_float']):
				d[row['key_string']] = row['value_float']
			else:
				d[row['key_float']] = row['value_float']
		
		args = {}
		for attr in hdfNode._v_attrs._f_list():
			args[attr] = getattr(hdfNode._v_attrs,attr)
		
		return {'data':d,'args':args}
	
	############## DataLeaf Methods #######################################
	
	@property
	def value(self):
		return self.__dict
	
	@property
	def set_value(self,value):
		self.__dict = dict
	
	@property
	def attrs(self):
		return self.__props
	
	@property
	def set_attrs(self,**kwargs):
		self.__props.update(kwargs)
	
class DataList(HDF5Group,DataLeaf):
	
	def __init__(self,name,data,attrs={}):
		self.set_name(name)
		self.set_value(data)
		self.__props = attrs
	
	######### HDF5 Methods #################################################
	
	def _node(self,node):
		if node.isdigit():
			return self.__list[int(node)]
		raise errors.NoSuchNodeError("'%s'"%node)
	
	@property
	def _hdf5_desc(self):
		return ""
	
	@property
	def _hdf5_group_children(self):
		return self.__list
	
	@property
	def _hdf5_attrs(self):
		attrs = {'type':'data_list'}
		for prop in self.__props:
			attrs[prop] = self.__props[prop]
		return attrs
	
	@classmethod
	def _hdf5_populate(cls,hdfNode):
		
		found = {}
		
		for key,value in hdfNode._v_children.items():
			extracted = populateDataType(dataObj=None,hdfNode=value,extractOnly=True)
			#if isinstance(object,DataLeaf):
			#	extracted['data'].set_attrs(**extracted['args'])
			found[key] = extracted['data']
			
		
		final = []
		for i in xrange(len(found)):
			final.append(found["index_"+str(i)])
			
		args = {}
		for attr in hdfNode._v_attrs._f_list():
			args[attr] = getattr(hdfNode._v_attrs,attr)
		
		return {'data':final,'args':args}
	
	######## DataLeaf #####################################################
	@property
	def value(self):
		return map(lambda x: x.value if isinstance(x,DataLeaf) else x, self.__list)
	
	def set_value(self,value):
		self.__list = []
		for item in value:
			self.append(item)
	
	def append(self,value,dtype=None,**args):
		self.__list.append(getDataType("index_"+str(len(self.__list)),value,dtype,**args))
	
	@property
	def attrs(self):
		return self.__props
	
	def set_attrs(self,**kwargs):
		self.__props.update(kwargs)

class DataArray(HDF5LeafArray,DataLeaf):
	
	def __init__(self,name,data=None,attrs={}):
		self.set_name(name)
		self.__data = np.array(data)
		self.__props = attrs
	
	######### HDF5 Methods #################################################
	
	@property
	def _hdf5_desc(self):
		return ''
	
	@property
	def _hdf5_leaf_array(self):
		return self.__data
	
	@property
	def _hdf5_attrs(self):
		attrs = {'type':'data_array'}
		for prop in self.__props:
			attrs[prop] = self.__props[prop]
		return attrs
	
	@classmethod
	def _hdf5_populate(cls,hdfNode):
		args = {}
		for attr in hdfNode._v_attrs._f_list():
			args[attr] = getattr(hdfNode._v_attrs,attr)
		return {'data':hdfNode.read(),'args':args}
	
	######### Data Value Methods ###########################################
	
	@property
	def value(self):
		return self.__data
	
	def set_value(self,value):
		self.__data = np.array(value)
	
	@property
	def attrs(self):
		return self.__props
	
	def set_attrs(self,**kwargs):
		self.__props.update(kwargs)
