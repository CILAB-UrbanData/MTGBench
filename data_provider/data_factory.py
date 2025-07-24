from data_provider.data_loader import MDTPRawloader
from torch.utils.data import DataLoader, Subset
import random

data_dict = {
    'MDTP': MDTPRawloader
}

def data_provider(args, flag):
    Data = data_dict[args.data]

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size

    if args.task_name ==  'TrafficLSTM':
        if args.data == 'MDTP':
            drop_last = True
            shuffle_flag = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
            normalization=args.normalization,
            S=args.S
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
