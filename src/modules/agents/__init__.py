REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .ff_agent import FFAgent
REGISTRY['ff_agent'] = FFAgent

from .rnn_sd_agent import RNN_SD_Agent
REGISTRY["rnn_sd"] = RNN_SD_Agent
