"""
A model for training unary potentials used for a CRF.

Yujia Li, 08/2013

* Updates *
- Added mean vector avg to preprocessing. Now for preprocessing the mean is 
  subtracted then std is used to normalize the variance.

"""

import numpy as np
import gnumpy as gnp
import gnn.nn as nn
import gnn.config as cfg

def prep_unary_train_data(feature_list, seg_list, K, n_per_class, std=None, avg=None):
    """
    Prepare training data for unary potentials.

    feature_list: a list of N feature matrices for N images, each feature
        matrix will be of size (H*W)*D
    seg_list: N*(H*W) segmentation matrix
    K: number of classes
    n_per_class: can be either a single integer or an iterable object of length
        K, where K is the number of classes.

    std: optional standard deviation vector, D dimensional
    avg: optional mean vector, D dimensional

    return: x, t, std
        x is the data
        t is the labels
        std is the standard deviation used to normalize x
        avg is the mean used to normalize x
    """

    if type(n_per_class) != list:
        per_class = [n_per_class] * K
    else:
        per_class = n_per_class

    assert(len(per_class) == K)

    n_cases = len(feature_list)
    n_dim = feature_list[0].shape[1]

    # count the number of positive and negative examples first
    total_for_class = [0] * K
    for i in range(n_cases):
        s = seg_list[i]
        for k in range(K):
            class_count = (s == k).sum()
            total_for_class[k] += min(class_count, per_class[k])
            if class_count < per_class[k]:
                print 'Image %d, only have %d pixels for class %d (target %d)' % (
                        i, class_count, k, per_class[k])

    n_total = sum(total_for_class)
    x = np.zeros((n_total, n_dim), dtype=np.single)
    t = np.zeros(n_total, dtype=np.int)

    i_start = 0
    for i in range(n_cases):
        f = feature_list[i]
        s = seg_list[i]
        idx = np.arange(s.size)

        for k in range(K):
            n_chosen = min((s == k).sum(), per_class[k])
            c_idx = idx[s == k]
            r_idx = np.random.permutation(c_idx.size)
            c_idx = c_idx[r_idx[:n_chosen]]

            x[i_start:i_start + n_chosen] = f[c_idx]
            t[i_start:i_start + n_chosen] = k

            i_start += n_chosen

    if std == None:
        std = x.std(axis=0)
    if avg == None:
        avg = x.mean(axis=0)
    return (x - avg) / std, t, std, avg

def make_train_val_idx(n_cases, train_proportion):
    """
    Generate indices for training and validation data, therefore splitting the
    whole data set.

    Return: train_idx, val_idx
    """
    n_train = int(n_cases * train_proportion)
    r_idx = np.random.permutation(n_cases)
    return r_idx[:n_train], r_idx[n_train:]

def get_unary(feature_list, unary_model, std=None, avg=0):
    """
    Compute unary potentials for a set of images using given unary model.

    feature_list: a list of N feature matrices, N is the number of images
    unary_model: a learned unary model, has a function named forward,
        unary_model.forward(x) will compute the unary potentials for data 
        matri x. For a M*D matrix x, the output of this function will be a M*K
        matrix, where K is the number of classes, and the element at (m,k) will
        be the negative potential for pixel m taking label k.
        - If it is a neural net, it is just the original forward function and
        the activation matrix before feeding into the output will be returned.
        - No need to take the log since this is already in log scale.
        - This is assumed to accept only gnumpy garrays.
    std: if set, this will be used to normalize the scale of each dimension in
        the feature vector.
    avg: if set, this will be used to subtract mean of each dimension.

    Return: a list of N unary matrices, each has size M*K, where M is the number
        of pixels and K is the number of classes.
    """
    unary = []
    for i in range(len(feature_list)):
        if std != None:
            features = (feature_list[i] - avg) / std
        else:
            features = (feature_list[i] - avg)
        unary.append(-unary_model.forward(gnp.garray(features)).asarray())

    return unary

def generate_default_unary(feature_list, seg_list, K, n_per_class):
    """
    Learn unary potentials on given data using default settings.

    feature_list: a list of N feature matrices, N is the number of images
    seg_list: N*D segmentation matrix, D is the number of pixels in each image
    K: number of classes
    n_per_class: number of pixels to sample from each class in each image

    Return: (unary_model, std)
        unary_model: a neural net, can be used to make unary potentials
        std: standard deviation for each dimension of the feature vectors, used
            to normalize data before feeding into the neural net.
    """
    x, t, std, avg = prep_unary_train_data(feature_list, seg_list, K, n_per_class)
    train_idx, val_idx = make_train_val_idx(t.size, 0.5)
    train_data = nn.Data()
    train_data.X = x[train_idx]
    train_data.T = t[train_idx]
    train_data.K = K

    val_data = nn.Data()
    val_data.X = x[val_idx]
    val_data.T = t[val_idx]
    val_data.K = K

    print '%d training cases, %d validation cases' % (train_idx.size, val_idx.size)

    net = nn.NN()
    net.load_train_data(train_data)
    net.load_val_data(val_data)
    
    net_cfg = cfg.Config('/u/yujiali/Desktop/Research/PythonToolbox/imgtools/unary_potential.cfg')
    net.init_net_without_loading_data(net_cfg)
    net.train()

    net.load_net('%s/%s' % (net_cfg.output_dir, 'best_net.pdata'))

    return net, std

def generate_default_unary_validation(train_flist, train_segs, val_flist, 
        val_segs, K, n_per_class_train, n_per_class_val, cfg_file=None, cfg_obj=None):
    """
    Learn unary potentials on given data using default settings.  The
    difference between this method and generate_default_unary is this method
    allows to use a validation set.

    train_flist, train_segs: training data
    val_flis, val_segs: validation data
    K: number of classes
    n_per_class_train: number of pixels to sample from each class in each image.
    n_per_class_val: same as n_per_class_train, except it is for validation set.
    cfg_file: configuration file used to train the model. Not used if cfg_obj
        is set. If none of the two are set, default settings will be used.
    cfg_obj: parsed configuration file object, used to train the model. 

    Return: (unary_model, std, best_acc)
        unary_model: a neural net, can be used to make unary potentials
        std: standard deviation for each dimension of the feature vectors, used
            to normalize data before feeding into the neural net.
        avg: mean for each dimension of the feature vectors, used to normalize
            data before feeding into the neural net
        best_acc: best validation accuracy
    """
    x, t, std, avg = prep_unary_train_data(train_flist, train_segs, K, n_per_class_train)
    xval, tval, _, _ = prep_unary_train_data(val_flist, val_segs, K, n_per_class_val, std, avg)
    train_data = nn.Data()
    train_data.X = x
    train_data.T = t
    train_data.K = K

    val_data = nn.Data()
    val_data.X = xval
    val_data.T = tval
    val_data.K = K

    print '%d training cases, %d validation cases' % (t.size, tval.size)

    net = nn.NN()
    net.load_train_data(train_data)
    net.load_val_data(val_data)
    
    if cfg_obj != None:
        net_cfg = cfg_obj
    elif cfg_file != None:
        net_cfg = cfg.Config(cfg_file)
    else:
        net_cfg = cfg.Config('/u/yujiali/Desktop/Research/PythonToolbox/imgtools/unary_potential.cfg')
    net.init_net_without_loading_data(net_cfg)
    best_acc = net.train()

    net.load_net('%s/%s' % (net_cfg.output_dir, 'best_net.pdata'))

    return net, std, avg, best_acc

def combine_unary_potentials(unary_list, weight_list):
    """Linearly combine a list of different unary potentials using the given 
    weights.

    unary_list: a list of unary potentials
    weight_list: a sequence of real value weights, same length as unary_list

    Return: the combined unary potential.
    """
    u = []
    n_unary_types = len(unary_list)
    n_data_points = len(unary_list[0])

    for i in range(n_data_points):
        new_u = 0
        for j in range(n_unary_types):
            new_u = new_u + unary_list[j][i] * weight_list[j]
        u.append(new_u)

    return u

