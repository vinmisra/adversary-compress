import sys, os, pdb
import train_AE
import CAN
from pylearn2.config import yaml_parse

#command line input is name of experiment
NAME = sys.argv[1]

#path constants
#DATA_DIR = "/Users/vmisra/data" #local
DATA_DIR = "/users/ubuntu/data" #AWS
MODELS_DIR = os.path.join(DATA_DIR,"CAN_experiments/models",NAME)
SAVE_MODEL_PATH = os.path.join(MODELS_DIR,NAME+".pkl")
SAVE_YAML_PATH = os.path.join(MODELS_DIR,NAME+".yaml")
SAVE_LOG_PATH = os.path.join(MODELS_DIR,NAME+".log")

if not os.path.exists(MODELS_DIR):
	os.makedirs(MODELS_DIR)

#I/O logging
class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath,'w')
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
    def flush(self):
        pass
sys.stdout = Logger(SAVE_LOG_PATH)

#construct and dump YAML
YAML_path = "exp/"+NAME+".yaml"
YAML = open(YAML_path,'r').read() % {"DATA_DIR":DATA_DIR,"SAVE_PATH":SAVE_MODEL_PATH}
open(SAVE_YAML_PATH,'w').write(YAML)


#run experiment
yaml_parse.load(YAML).main_loop()
