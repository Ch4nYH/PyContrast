from os.path import join


def get_paths(dataset, root='/ccvl/net/ccvl15/shuhao/domain_adaptation/datasets'):
    if dataset == 'synapse':
        root_dir = join(root, 'Synapse_pancreas/train_processed_noaug')
    elif dataset == 'synapse_aug':
        root_dir = join(root, 'Synapse_pancreas/train_processed_aug')
    elif dataset == 'msd_spleen':
        root_dir = join(root, 'MSD/Task09_Spleen/train_processed_aug')
    elif dataset == 'msd_liver':
        root_dir = join(root, 'MSD/Task03_Liver/train_processed_aug')
    elif dataset == 'msd_pancreas':
        root_dir = join(root, 'MSD/Task07_Pancreas/train_processed_aug')
    elif dataset == 'nih_pancreas':
        root_dir = join(root, 'nih_pancreas/train_processed_aug')
    else:
        print('ERROR: dataset {} not defined'.format(dataset))
        exit(-1)

    print('processing dataset {}'.format(dataset))
    if dataset == 'synapse':
        train_list = join(root_dir, '../train.lst')
        test_root = root_dir.replace('train_processed_noaug', 'test_processed')
    else:
        train_list = join(root_dir, '../train_processed_aug.lst')
        test_root = root_dir.replace('train_processed_aug', 'test_processed')

    test_list = join(root_dir, '../test.lst')

    return root_dir, train_list, test_root, test_list


def get_paths_multiorgan(dataset, root='/ccvl/net/ccvl15/shuhao/domain_adaptation/datasets'):
    if dataset == 'synapse':
        root_dir = join(root, 'Synapse/train_processed_noaug')
    else:
        print('ERROR: dataset {} not defined'.format(dataset))
        exit(-1)

    print('processing dataset {}'.format(dataset))
    train_list = join(root_dir, '../train.lst')
    test_root = root_dir.replace('train', 'test')
    test_list = join(root_dir, '../test.lst')

    return root_dir, train_list, test_root, test_list


def get_test_paths(dataset, root='/ccvl/net/ccvl15/shuhao/domain_adaptation/datasets'):
    if dataset == 'synapse':
        root_dir = join(root, 'Synapse_pancreas/test')
    elif dataset == 'msd_spleen':
        root_dir = join(root, 'MSD/Task09_Spleen/test')
    elif dataset == 'msd_liver':
        root_dir = join(root, 'MSD/Task03_Liver/test')
    elif dataset == 'msd_pancreas':
        root_dir = join(root, 'MSD/Task07_Pancreas/test')
    elif dataset == 'nih_pancreas':
        root_dir = join(root, 'nih_pancreas/test')
    else:
        print('ERROR: dataset {} not defined'.format(dataset))
        exit(-1)

    list_path = join(root_dir, '../test.lst')

    return root_dir, list_path


def get_test_paths_multiorgan(dataset, root='/ccvl/net/ccvl15/shuhao/domain_adaptation/datasets'):
    if dataset == 'synapse':
        root_dir = join(root, 'Synapse/test')
    elif dataset == 'msd_spleen':
        root_dir = join(root, 'MSD/Task09_Spleen/test')
    elif dataset == 'msd_liver':
        root_dir = join(root, 'MSD/Task03_Liver/test')
    elif dataset == 'msd_pancreas':
        root_dir = join(root, 'MSD/Task07_Pancreas/test')
    elif dataset == 'nih_pancreas':
        root_dir = join(root, 'nih_pancreas/test')
    else:
        print('ERROR: dataset {} not defined'.format(dataset))
        exit(-1)

    list_path = join(root_dir, '../test.lst')

    return root_dir, list_path
