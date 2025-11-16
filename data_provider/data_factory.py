from data_provider.data_loader import MDTPRawloader, SF20_forTrajnet_Dataset, \
    SF20_forTrGNN_Dataset, OtherForMDTP, DiDi_forTrGNN_Dataset, DiDi_forTrajnet_Dataset, TRACKDataset, MDTPSingleLoader, SF_forTrGNN_Dataset
from torch.utils.data import DataLoader, Subset
import random

data_dict = {
    'MDTP': MDTPRawloader,
    'MDTPsingle': MDTPSingleLoader,
    'OtherForMDTP': OtherForMDTP,
    'Trajnet': SF20_forTrajnet_Dataset,
    'TrGNN': SF_forTrGNN_Dataset,
    'DiDiTrGNN': DiDi_forTrGNN_Dataset,
    'DiDiTrajnet': DiDi_forTrajnet_Dataset,
    'TRACK': TRACKDataset,
}

def data_provider(args, flag='train'):
    Data = data_dict[args.data]

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size

    if args.task_name == 'TrafficPrediction':
        if args.data == 'Trajnet' or args.data == 'DiDiTrajnet':
            drop_last = False
            shuffle_flag = False
            data_set = Data(
                args=args,
                flag=flag,            
                root_path=args.root_path,
            )

        elif args.data == 'TrGNN' or args.data == 'DiDiTrGNN' :
            drop_last = False
            shuffle_flag = True
            data_set = Data(
                args = args,
                flag = flag
            )
        
        elif args.data == 'MDTP' or args.data == 'OtherForMDTP'or args.data == 'MDTPsingle':
            drop_last = True
            shuffle_flag = False
            data_set = Data(
                args = args,
                root_path=args.root_path,
                flag=flag,
                normalization=args.normalization,
                S=args.S
            )
        
        elif args.data == 'TRACK':
            drop_last = True 
            shuffle_flag = True 
            data_set = Data(
                data_root = args.root_path,
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
        data_set = Data(data_root = args.root_path, 
                        flag = flag, 
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