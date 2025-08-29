from data_provider.data_loader import MDTPRawloader, SF20_forTrajnet_Dataset, SF20_forTrGNN_Dataset, GaiyaForMDTP, DGM_Dataset
from utils.TRACK_wrapper_trllm_cont import TSLibDatasetWrapper
from torch.utils.data import DataLoader, Subset
import random

data_dict = {
    'MDTP': MDTPRawloader,
    'GaiyaForMDTP': GaiyaForMDTP,
    'Trajnet': SF20_forTrajnet_Dataset,
    'TrGNN': SF20_forTrGNN_Dataset,
    'TRACK_trllm_cont': TSLibDatasetWrapper,
    'DGM' : DGM_Dataset
}

def data_provider(args, flag='train'):
    Data = data_dict[args.data]

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size

    if args.task_name ==  'TrafficLSTM':
        if args.data == 'MDTP' or args.data == 'GaiyaForMDTP':
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
            drop_last=drop_last,
            collate_fn=data_set.collate_fn if args.data == 'MDTP' or args.data == 'GaiyaForMDTP' else None)
        return data_set, data_loader

    elif args.task_name == 'TrafficPrediction':
        if args.data == 'Trajnet':
            drop_last = False
            shuffle_flag = False
            data_set = Data(
                args=args,
                flag=flag,            
                root_path=args.root_path,
            )

        elif args.data == 'TrGNN':
            drop_last = False
            shuffle_flag = True
            data_set = Data(
                args = args,
                flag = flag,
                start_date = args.start_date,
                end_date = args.end_date,
                root_path = args.root_path
            )
        
        elif args.data == 'MDTP' or args.data == 'GaiyaForMDTP':
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
            num_workers=0,
            drop_last=drop_last,
            collate_fn=data_set.collate_fn if hasattr(data_set, 'collate_fn') else None)
        return data_set, data_loader

    elif args.task_name == 'TRACK_trllm_cont':
        if args.data == 'TRACK_Gaiya':
            data_set = Data(args)
            train_loader, eval_loader, test_loader = data_set.get_data()
            data_feature = data_set.get_data_feature()
        return data_set, train_loader, eval_loader, test_loader, data_feature
    
    elif args.task_name == 'ODflow_generation':
        if args.data == 'DGM':
            
            dataset = DGM_Dataset(
                list_IDs=args.list_IDs,
                tileid2oa2features2vals=args.tileid2oa2features2vals,
                o2d2flow=args.o2d2flow,
                oa2features=args.oa2features,
                oa2pop=args.oa2pop,
                oa2centroid=args.oa2centroid,
                dim_dests=args.dim_dests,
                frac_true_dest=args.frac_true_dest,
                model=args.model
            )
            # 根据flag决定是否打乱数据
            shuffle = True if flag == 'train' else False
            # 使用DGM_Dataset自定义的collate_fn
            data_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=shuffle,
                collate_fn=dataset.my_collate
            )
            return dataset, data_loader
