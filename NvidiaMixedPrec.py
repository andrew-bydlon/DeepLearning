from tensorflow.compat.v1 import ConfigProto, InteractiveSession
import os

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = InteractiveSession(config=config)