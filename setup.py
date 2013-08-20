#!/usr/bin/env python2

from distutils.core import setup

setup(name='HDF5Storage',
      version='0.1',
      description='A storage container for data (with attributes) that can be persisted as HDF5 files using pytables.',
      author='Matthew Wardrop',
      author_email='mister.wardrop@gmail.com',
      url='http://www.matthewwardrop.info/',
      #package_dir={'parameters':'.'},
      packages=['hdf5storage'],
     )
