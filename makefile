all: src/mapper_c_utils/mapper_c_utils.c setup.py
	python3 setup.py build_ext --inplace

test: all
	python3 test_mod.py