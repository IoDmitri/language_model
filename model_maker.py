import sys, getopt

from language_model import Language_model
from vocab import Vocab

def gen_reddit_model(f_name, save_dir, batch_size, max_steps, layers):
	model = Language_model(batch_size=batch_size, max_steps=max_steps, save_dir=save_dir, num_layers=layers, max_epochs=10)
	model.train_on_file(f_name)

def main(argv):
	f_name = None
	save_dir = None
	batch_size = 175
	max_steps = 25
	layers = 2

	try:
		opts, args = getopt.getopt(argv,"f:s:b:m:l:")
	except getopt.GetoptError:
		print 'ex occured'
		sys.exit(2) 

	for opt, arg in opts:
		if opt == "-f":
			_print_arg(arg, "f_name")
			f_name = arg
		elif opt == "-s":
			_print_arg(arg, "save_dir")
			save_dir = arg
		elif opt == "-b":
			_print_arg(arg, "batch_size")
			batch_size = int(arg)
		elif opt == "-m":
			_print_arg(arg, "max_steps")
			max_steps = int(arg)
		elif opt == "-l":
			_print_arg(arg, "num_layers")
			layers = int(arg)

	if not f_name:
		print "a file name must be provided, use the -f option"
		sys.exit(3)

	print "f_Name {}".format(f_name)
	print "save_dir {}".format(save_dir)
	print "batch_size {}".format(batch_size)
	print "max_steps {}".format(max_steps)
	print "layers {}".format(layers)

	gen_reddit_model(f_name, save_dir, batch_size, max_steps, layers) 

def _print_arg(name, arg):
	print "using {0} for {1}".format(name, arg)


if __name__ == "__main__":
	main(sys.argv[1:])
