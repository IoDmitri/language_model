from data_utils import *

def test_reading_files():
	file_generator = process_file_data("./data/ptb.test.txt", flatten=True)
	assert(set(file_generator) == set(file_generator))
	print "tests passed"

if __name__ == "__main__":
	test_reading_files()