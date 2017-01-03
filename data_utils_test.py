from data_utils import *

def test_reading_files():
	file_generator = process_file_data("./data/ptb.test.txt", flatten=True)
	s1 = set(file_generator)
	s2 = set(file_generator)
	assert(s1 == s2)
	print "tests passed"

if __name__ == "__main__":
	test_reading_files()