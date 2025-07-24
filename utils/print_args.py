def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    print(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    print(f'  {"Data Path:":<20}{args.data_path:<20}{"Checkpoints:":<20}{args.checkpoints:<20}')
    print()

    if args.model == 'MDTP':
        print("\033[1m" + "Model Parameters" + "\033[0m")
        print(f'  {"In Feats:":<20}{args.in_feats:<20}{"Gcn Hidden:":<20}{args.gcn_hidden:<20}')
        print(f'  {"Lstm Hidden:":<20}{args.lstm_hidden:<20}{"Fusion:":<20}{args.fusion:<20}')
        print(f'  {"N Nodes:":<20}{args.N_nodes:<20}{"Num Layers:":<20}{args.num_layers:<20}')
        print(f'  {"GradClip:":<20}{args.grad_clip:<20}{"Dropout:":<20}{args.dropout:<20}')
        print(f'  {"S:":<20}{args.S:<20}')
        print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    print(f'  {"Des:":<20}{args.des:<20}{"Loss:":<20}{args.loss:<20}')
    print(f'  {"Lradj:":<20}{args.lradj:<20}{"Use Amp:":<20}{args.use_amp:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    print()

