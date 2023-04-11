import torch.utils.data as data

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def __len__(self):
        return 0
    
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     return parser
