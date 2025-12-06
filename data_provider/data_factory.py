from data_provider.data_loader import MDTPRawloader, Trajnet_Dataset, \
    TRACKDataset, MDTPSingleLoader, Dataset_forTrGNN
from torch.utils.data import DataLoader, Subset
import random

data_dict = {
    'MDTP': MDTPRawloader,
    'MDTPsingle': MDTPSingleLoader,
    'Trajnet': Trajnet_Dataset,
    'TrGNN': Dataset_forTrGNN,
    'TRACK': TRACKDataset,
}

def data_provider(args, flag='train'):
    Data = data_dict[args.model]

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size

    if args.task_name == 'TrafficPrediction':
        if args.model == 'Trajnet':
            drop_last = False
            shuffle_flag = False
            data_set = Data(
                args=args,
                flag=flag,            
            )

        elif args.model == 'TrGNN' :
            drop_last = False
            shuffle_flag = True
            preprocess_path = args.preprocess_path 
            
            data_set = Data(
                args = args,
                flag = flag,
                preprocess_path = preprocess_path,
            )
        
        elif args.model == 'MDTP' or args.model == 'MDTPsingle':
            drop_last = True
            shuffle_flag = False
            data_set = Data(
                args = args,
                root_path=args.root_path,
                flag=flag,
                S=args.S
            )
        
        elif args.model == 'TRACK':
            drop_last = True 
            shuffle_flag = True 
            data_set = Data(
                flag = flag,
                args = args
            )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=0,
            drop_last=drop_last,
            collate_fn=data_set.collate_fn if hasattr(data_set, 'collate_fn') else None)
        return data_set, data_loader
    
    elif args.task_name == 'TRACK_pretrain':
        drop_last = True 
        shuffle_flag = True 
        data_set = Data(flag = flag, 
                        args = args)

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=0,
            drop_last=drop_last,
            collate_fn=data_set.collate_fn if hasattr(data_set, 'collate_fn') else None)
        return data_set, data_loader