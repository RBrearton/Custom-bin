all: src/custom_bin/custom_bin.c setup.py
	python3 setup.py build_ext --inplace

test: all
	python3 test_mod.py