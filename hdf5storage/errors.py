
class HDF5StorageError(Exception):
	pass

class NodeExistsError(HDF5StorageError):
	pass

class InvalidNodeError(HDF5StorageError):
	pass

class NoSuchNodeError(HDF5StorageError):
	pass

class NoSuchGroupError(HDF5StorageError):
	pass

class NoSuchLeafError(HDF5StorageError):
	pass

class InvalidNodeNameError(HDF5StorageError):
	pass

class InaccessibleGroupNodeWarning(UserWarning):
	pass