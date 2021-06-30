# NVIDIA

# The purpose of this module is to provide a space to allow global variables to work properly inside of the unittest framework

# An example case is loading the TF weights file.
## It is ~4GB and read-only, so why not just load it once into a global variable

tf_weights = None
tf_tensors = None

pyt_model = None
pyt_checkpoint = None
