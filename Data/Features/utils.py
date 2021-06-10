import scipy.io as scio
from os.path import join


def read_list_from_file(file_path):
    """Reads list of classes from a text file
    """
    with open(file_path, 'r') as f:
        output_list = f.readlines()
    output_list = [x.strip() for x in output_list]
    return output_list


def load_from_xlsa17(data_path, ds_name):
    # load mat files
    ds_path = join(data_path, ds_name)
    mat_feat = scio.loadmat(join(ds_path, 'res101.mat'))
    mat_att_split = scio.loadmat(join(ds_path, 'att_splits.mat'))

    image_files = [join(*x[0][0].split('/')[-2:]) for x in mat_feat['image_files']]
    labels = mat_feat['labels']-1
    allclasses_names = [[x[0][0]] for x in mat_att_split['allclasses_names']]
    att = mat_att_split['att'].T
    test_seen_loc = mat_att_split['test_seen_loc'] - 1
    test_unseen_loc = mat_att_split['test_unseen_loc'] - 1
    train_loc = mat_att_split['train_loc'] - 1
    trainval_loc = mat_att_split['trainval_loc'] - 1
    val_loc = mat_att_split['val_loc'] - 1
    return {'image_files': image_files, 'labels': labels, 'allclasses_names': allclasses_names,
            'att': att, 'test_seen_loc': test_seen_loc, 'test_unseen_loc': test_unseen_loc,
            'train_loc': train_loc, 'trainval_loc': trainval_loc, 'val_loc': val_loc}
