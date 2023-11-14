import logging
from utilss.args_utils import get_args, update_args
from MultimodalBert.trainer.multimodal_trainer import MultimodalTrainer
import warnings
warnings.filterwarnings('ignore')

# logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt = '%m/%d/%Y %H:%M:%S',
#                     level = logging.INFO)
# logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    args = get_args()
    args = update_args(args)
    my_trainer = MultimodalTrainer(args, logger)
    my_trainer.do_train()
    # my_trainer.do_test()
    my_trainer.do_test_with_avg_model()
