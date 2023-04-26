from .q_learner import QLearner
from .qtran_learner import QLearner as QTranLearner
from .q_learner_w import QLearner as WeightedQLearner
from .qnam_learner import QNAM_Learner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["w_q_learner"] = WeightedQLearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["qnam_learner"] = QNAM_Learner