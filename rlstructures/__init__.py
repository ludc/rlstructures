__version__="0.2"
__deprecated_message__=False
import sys
from rlstructures.core import masked_tensor,masked_dicttensor,DictTensor,TemporalDictTensor,Trajectories
import rlalgos.logger
import rlalgos.tools
sys.modules["rlstructures.logger"]=rlalgos.logger
sys.modules["rlstructures.tools"]=rlalgos.tools
from rlstructures.e_batcher.agent import E_Agent,replay_agent_stateless,replay_agent
from .agent import Agent
from rlstructures.env import VecEnv
from rlstructures.e_batcher import E_Batcher
import rlstructures.core
sys.modules["rlstructures.dicttensor"]=rlstructures.core
import rlstructures.deprecated.logging
sys.modules["rlstructures.logging"]=rlstructures.deprecated.logging
