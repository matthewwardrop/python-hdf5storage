import re

def encodeNumbers(name):
	if isinstance(name,str):
		return name
	elif isinstance(name,(int,long)):
		return "long(%d)" % name
	elif isinstance(name,float):
		return "float(%.17e)"%name
	elif isinstance(name,complex):
		return "complex(({0.real:.17e}{0.imag:+.17e}j))"%name
	raise ValueError("'%s' of type %s is not recognised."%(name,type(name)))

def decodeNumbers(name):
	if isinstance(name,(int,long,float,complex)):
		return name
	m = re.match("^([a-zA-Z]+)\(([\w\(\)\+\-]*\))$",name)
	if m is None:
		return name
	numtype,numstr = m.groups()
	if numtype is 'long':
		return long(numstr)
	elif numtype is 'int':
		return int(numstr)
	elif numtype is 'float':
		return float(numstr)
	elif numtype is 'complex':
		return complex(numstr)
	raise ValueError("'%s' of type %s is not recognised."%(name,type(name)))
	
	
