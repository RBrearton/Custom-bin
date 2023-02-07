Some simple but extremely quick algorithms for use with
https://github.com/RBrearton/RSMapper.

# Installation

Note that this has a dependency on numpy. This uses both the python and numpy C api; you'll need CPython C headers to be installed on your system. In general, your compiler will need to know where your CPython headers are, and where your numpy C headers are. This is a very well trodden, operating system specific task; google is your friend!

Build by typing "make"

Build as a python module with the usual:
`python setup.py develop`

The library https://github.com/RBrearton/RSMapper depends on this library. If
you want to build this module to get RSMapper working, then the
`python setup.py develop`
command should work just fine for you.
