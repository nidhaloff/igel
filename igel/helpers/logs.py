import logging.config
from os import path
conf = path.join(path.dirname(path.abspath(__file__)), 'logging.ini')
print("conf file: ", conf)


def create_logger(name):
	logging.config.fileConfig(conf, disable_existing_loggers=False)
	logger = logging.getLogger(name)
	return logger



