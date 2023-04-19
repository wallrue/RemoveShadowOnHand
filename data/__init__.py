import importlib
import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.base_dataset import BaseDataset

def find_dataset_using_name(dataset_name):
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        print("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))
        exit(0)

    return dataset

def create_dataset(opt):
    dataset = find_dataset_using_name(opt.dataset_mode)
    instance = dataset()
    instance.initialize(opt)
    print("dataset [%s] was created" % (instance.name()))
    return instance

def train_val_split(dataset, val_split=1.0):
    valDataset = torch.utils.data.Subset(dataset, range(0, int(val_split * len(dataset))))
    trainDataset = torch.utils.data.Subset(dataset, range(int(val_split*len(dataset)), len(dataset)))
    return trainDataset, valDataset

# def train_val_split(dataset, val_split=0.25):
#     train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
#     datasets = {}
#     datasets['train'] = Subset(dataset, train_idx)
#     datasets['val'] = Subset(dataset, val_idx)
#     return datasets

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def __init__(self, opt):
        BaseDataLoader.__init__(self, opt)
        self.working_subset = 'main' #the kind of dataset to use ['train', 'valid']
        self.mainDataset, self.validDataset = train_val_split(create_dataset(opt), opt.validDataset_split)
        self.mainDataloader = torch.utils.data.DataLoader(
            self.mainDataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))
        
        if len(self.validDataset) != 0:
            self.validDataloader = torch.utils.data.DataLoader(
                self.validDataset,
                batch_size=opt.batch_size,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.num_threads))        
        
    def load_data(self):
        return self

    def __len__(self):
        return min(len(getattr(self, self.working_subset + 'Dataset')), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(getattr(self, self.working_subset + 'Dataloader')):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

            
# def CreateDataLoader(opt):
#     data_loader = CustomDatasetDataLoader()
#     data_loader.initialize(opt)
#     return data_loader

# def get_option_setter(dataset_name):
#     dataset_class = find_dataset_using_name(dataset_name)
#     return dataset_class.modify_commandline_options