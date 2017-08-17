import os
import glog

class Config(object):
	def __init__(self, output_path, config_file):
		self.output_path = output_path
		self.config_file = config_file
		self.items = {}
		self.load_config()
		if 'gpu' in self.items.keys() and self.items['gpu'] in range(4):
			os.environ['THEANO_FLAGS'] = 'device=gpu%d,allow_gc=False'%self.items['gpu']

	def load_config(self):
		with open(self.config_file) as f:
			lines = f.readlines()
			if '----' in lines:
				lines = lines[:lines.index('----')]
			self.items_load = dict([(l.split('=')[0], self.parse_variable(l.split('=')[1].strip(), self.output_path)) for l in lines if '=' in l and not l.startswith('#')])

			if 'begin_loops' not in self.items.keys():
				self.items['begin_loops'] = 0

			if self.items != self.items_load:
				self.items = self.items_load
				glog.info('Loaded config file from: %s'%self.config_file+'\n'+''.join(lines))

	def parse_variable(self, var, output_path):
		try:
			return eval(var)
		except:
			return str(var).replace('DIR', output_path)
