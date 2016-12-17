import cv2
import numpy as np
import scipy.io as scio
import skimage.segmentation
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler, normalize
from scipy.cluster.vq import *

import os

dirs = ['lf', 'model', 'spf']
for d in dirs:
    if not os.path.isdir(d):
        os.mkdir(d)

labels = 55

stat = np.loadtxt('stat.txt', delimiter=',')
index = np.where(stat>100)[0][1:]

image_dir = '../database/photos'
label_dir = '../database/annotations/pixel-level'

def bag_of_words(name, index, image_dir, label_dir):
    '''
    calculate feature descriptor of the given image 
    '''
    image_path = '%s/%s.jpg' % (image_dir, name)
    label_path = '%s/%s.mat' % (label_dir, name)

    # read image and convert to grayscale
    gray = cv2.imread(image_path, 0)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # load pixel-level annotation and get unique label number
    label = scio.loadmat(label_path)['groundtruth']
    number = np.unique(label)
    
    # use sift to extract features of the grayscaled image
    sift = cv2.xfeatures2d.SIFT_create()

    lf_dsc = np.array([])
    #for i in np.intersect1d(number, index):
    for i in number[1:]:
        mask = np.array(label, np.uint8)
        mask[np.where(mask!=i)] = 0
        mask[np.where(mask==i)] = 1

        kp, dsc = sift.detectAndCompute(gray, mask = mask)
        if dsc is not None:
            copy = np.insert(dsc, 0, i, axis=1)
            if len(lf_dsc) == 0:
                lf_dsc = np.array(copy)
            else:
                lf_dsc = np.vstack((lf_dsc, copy))

    save_path = 'lf/%s.txt' % name
    np.savetxt(save_path, lf_dsc, delimiter=',', fmt='%1.0f')

def cluster(data_list, k):
    BOW = cv2.BOWKMeansTrainer(k)
    for data in data_list:
        BOW.add(data)

    dic = BOW.cluster()
    save_path = 'codebook_%d.txt' % k
    np.savetxt(save_path, dic, delimiter=',', fmt='%1.4f')
    return dic

def generate_codebook(n, index, image_dir, label_dir):
    '''
    get masked local feature of each image
    '''
    train = np.arange(1,1005)[2:3]

    for i in train[n::4]:
        name = '0'*(4-len(str(i))) + str(i)
        bag_of_words(name, index, image_dir, label_dir)

def get_superpixel_descriptor(image_path, label_path):
    '''
    For a given image, 
    extract superpixel regions first, 
    then extract the feature descriptor of each superpixel, 
    finally, label the class as the most frequent one
    '''
    # read image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # load pixel-level annotation
    label = scio.loadmat(label_path)['groundtruth']

    # slic superpixel and get most frequent label in each superpixel
    segments = skimage.segmentation.slic(image, n_segments = 500, sigma = 5)
    result = get_superpixel_label(label, segments)

    # for each foreground region, use sift to extract local feature
    lf_dsc = np.array([])
    sift = cv2.xfeatures2d.SIFT_create()
    for seg_num, super_label in result.items():
        mask = np.array(segments, np.uint8)
        mask[np.where(mask!=seg_num)] = 0
        mask[np.where(mask==seg_num)] = 1

        kp, dsc = sift.detectAndCompute(gray, mask = mask)
        if dsc is not None:
            for l in super_label:
                copy = np.insert(dsc, 0, l, axis=1)
                if len(lf_dsc) == 0:
                    lf_dsc = np.array(copy)
                else:
                    lf_dsc = np.vstack((lf_dsc, copy))

    return lf_dsc

def get_superpixel_label(pixel_label, segments):
    count = {}
    for i in xrange(len(segments)):
        for j in xrange(len(segments[0])):
            region = segments[i][j]
            label = pixel_label[i][j]
            if region not in count:
                count[region] = {label:1}
            elif label not in count[region]:
                count[region][label] = 1
            else:
                count[region][label] += 1

    result = {}
    for region_id, dic in count.items():
        most = max(dic.values())
        num = [label for label, cnt in dic.items() 
                if cnt == most and label != 0]
        if len(num) > 0:
            result[region_id] = num

    return result

def check_labels():
    '''
    check the label number in pixel-level annotation

    The result is in all 1004 samples, there are 55 different labels, 
    not 59. This is the raw data issue.

    So select first 70% and replace three with #738, #787 and #772 (these 
    have label 8 and 15). Now we have all the 55 labels
    '''
    a = []
    train = get_training_data()
    for i in train:
        name = '0'*(4-len(str(i))) + str(i)
        label_path = '../../database/annotations/pixel-level/%s.mat' % name
        label = scio.loadmat(label_path)['groundtruth']
        number = label.flatten()
        a = np.union1d(a, np.unique(number))

    print a
    np.savetxt('foreground_label.txt', a[1:], delimiter=',', fmt='%1.0f')

def parallel():
    import pp

    ppservers = ()
    job_server = pp.Server(4, ppservers=ppservers)
    inputs = range(4)
    jobs = [(i, job_server.submit(generate_codebook, (i, index, image_dir, label_dir), 
            (bag_of_words,), 
            ("cv2", "numpy as np", "scipy.io as scio"))) 
            for i in inputs]
    for i, job in jobs:
        job()
    job_server.print_stats()

def ovr_svm_train(k):
    from os import listdir
    from os.path import isfile, join, getsize

    path = 'spf/'
    files = [join(path,f) for f in listdir(path) 
            if isfile(join(path, f)) and getsize(join(path,f)) > 0]
    feature = np.loadtxt(files[0], delimiter=',')
    for f in files[1:]:
        feature = np.vstack((feature, np.loadtxt(f, delimiter=',')))

    label = feature[:,1]
    feature = feature[:,2:]

    # scale original data
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #feature = scaler.fit_transform(feature)
    #assert(np.isfinite(feature).all())

    # train the model
    print "training ..."
    ovr = OneVsRestClassifier(
            estimator=SVC(kernel='rbf', probability=True), 
            n_jobs=-1
            )
    fit = ovr.fit(feature, label)

    from sklearn.externals import joblib
    joblib.dump(ovr, 'model/model.pkl')
    print 'done'

def ovr_svm_test(k):
    from os import listdir
    from os.path import isfile, join, getsize

    path = 'spf/'
    files = [join(path,f) for f in listdir(path) 
            if isfile(join(path, f)) and getsize(join(path,f)) > 0]
    feature = np.loadtxt(files[0], delimiter=',')
    for f in files[1:]:
        feature = np.vstack((feature, np.loadtxt(f, delimiter=',')))

    label = feature[:,1]
    feature = feature[:,2:]

    from sklearn.externals import joblib
    ovr = joblib.load('model/model.pkl')

    # scale original data
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #feature = scaler.fit_transform(feature)

    print 'predicting ...'
    acc = 0
    y = ovr.predict(feature)
    for i in xrange(len(feature)):
        if y[i] == label[i]:
            acc+=1
    print 1.0*acc/len(feature)

def bow_feature_train(voc, num_sample):
    '''
    generate SIFT BOW features to be trained by svm
    '''
    from os.path import getsize

    '''
    # calculate number of samples on region level
    # each picture has several unique labels
    num_sample = 0
    for i in xrange(1, 1005):
        name = '0'*(4-len(str(i))) + str(i)
        lf_path = '../../database/bow/lf/%s.txt' % name

        if getsize(lf_path) > 0:
            data = np.loadtxt(lf_path, delimiter=',')
            label = data[:,0]
            num_sample += len(np.unique(label))
    print num_sample
    '''
    k = len(voc)

    # Calculate the histogram of features
    im_features = np.zeros((num_sample, k), 'float32')
    im_label = []
    num = 0
    for i in range(1, 1005)[2:4]:
        name = '0'*(4-len(str(i))) + str(i)
        lf_path = 'lf/%s.txt' % name
        if getsize(lf_path) > 0:
            data = np.loadtxt(lf_path, delimiter=',')
            label = data[:,0]
            dsc = data[:,1:]

            for l in np.unique(label):
                words, dist = vq(dsc[np.where(label==l)], voc)
                for w in words:
                    im_features[num][w] += 1
                im_label.append(l)
                num += 1
    
    im_label = np.array(im_label)

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*num_sample+1) / (1.0*nbr_occurences + 1)), 'float32')

    # Perform L2 normalization
    im_features = im_features*idf
    im_features = normalize(im_features, norm='l2')
    save_path = 'bow_feature_%d.txt' % k
    np.savetxt(save_path, im_features, delimiter=',', fmt='%1.4f')
    save_path = 'bow_label_%d.txt' % k
    np.savetxt(save_path, im_label, delimiter=',', fmt='%1.0f')

    from sklearn.externals import joblib
    save_path = "bof_%d.pkl" % k
    joblib.dump((im_features, idf, k, voc), save_path, compress=3)

def bow_feature(image, voc, segments):
    '''
    compute bow feature for a given image with a mask

    1. use sift to extract the features of the masked region

    2. compute the sifted feature with vocabulary, and get the bow feature

    @param
        image is a grayscale image matrix
        voc is the generated bag of words clustering vocabulary
        segments is the sliced segmentation
    '''
    from scipy.cluster.vq import *
    from sklearn.preprocessing import normalize
    sift = cv2.xfeatures2d.SIFT_create()
    k = len(voc)
    number = np.unique(segments)
    im_features = np.zeros((len(number), k), 'float32')

    num = 0
    seg_num = []
    for n in number:
        mask = np.array(segments, np.uint8)
        mask[np.where(mask!=n)] = 0
        mask[np.where(mask==n)] = 1
        kp, dsc = sift.detectAndCompute(image, mask = mask)

        if dsc == None:
            num += 1
            seg_num.append(-1)
        else:
            words, dist = vq(dsc, voc)
            for w in words:
                im_features[num][w] += 1
            num += 1
            seg_num.append(n)

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0+len(number)) / (1.0*nbr_occurences + 1)), 'float32')

    # Perform L2 normalization
    im_features = im_features*idf
    im_features = normalize(im_features, norm='l2')

    return (im_features, np.array(seg_num))

def bow_predict(image, voc, mask):
    feature = bow_feature(image, voc, mask)
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature = scaler.fit_transform(feature)

    from sklearn.externals import joblib
    ovr = joblib.load('model/model.pkl')
    p = ovr.predict_proba(feature)
    y = ovr.predict(feature)
    return (y, p)

def bow():
    '''
    read bag of word features of superpixel in all the images, and 
    cluster into k classes
    '''
    from os import listdir
    from os.path import isfile, join, getsize

    path = 'lf/'
    files = [join(path,f) for f in listdir(path) 
            if isfile(join(path, f)) and getsize(join(path,f)) > 0]
    files = np.sort(files)

    data_list = []
    num_sample = 0
    for f in files:
        data = np.loadtxt(f, delimiter=',')
        label = data[:,0]
        num_sample += len(np.unique(label))
        data = np.array(data[:,1:], np.float32)
        data_list.append(data)

    print num_sample

    #K = [100, 300, 500, 1000]
    K = [1000]
    for k in K:
        print k
        print 'clustering ... '
        voc = cluster(data_list, k)
        print 'calculating bow features of each region in each pic'
        bow_feature_train(voc, num_sample)

def test_bow_predict_slic():
    name = '0003'
    # load labels in mat file
    label_path = '%s/%s.mat' % (label_dir, name)
    label = scio.loadmat(label_path)['groundtruth']

    # load codebook
    codebook_path = 'codebook_%d.txt' % 1000
    voc = np.loadtxt(codebook_path, delimiter=',')

    # load image
    image_path = '%s/%s.jpg' % (image_dir, name)
    image = cv2.imread(image_path)
    image2 = cv2.imread(image_path, 0)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    from sklearn.externals import joblib
    ovr = joblib.load('model/model.pkl')

    # segmentation
    segments = skimage.segmentation.slic(image, n_segments = 500, sigma = 5)
    result = get_superpixel_label(label, segments)

    #fg_label = np.loadtxt('fg_label.txt', delimiter=',')
    (feature, seg_num) = bow_feature(image2, voc, segments)
    #print len(feature), len(feature[0]), len(seg_num)
    acc = 0
    total = 0
    fg_label = np.loadtxt('label.txt', delimiter=',')
    for i in xrange(len(seg_num)):
        n = seg_num[i]
        if n in result:
            # dsc not None and label not 0
            p = ovr.predict_proba(feature[i])
            y = ovr.predict(feature[i])
            max_prob = np.max(p[0])
            max_index = np.argmax(p[0])
            max_label = fg_label[max_index]
            print max_prob, max_index, max_label, y[0], result[n][0]
            #print p[0][np.where(fg_label==y[0])[0]], p[0][np.where(fg_label==result[n][0])[0]]
            if y[0] == result[n][0]:
                acc += 1
            total += 1

    print acc, '/', total
    print 'done'

def selected_files():
    name = np.loadtxt('rects.txt')
    return np.array(name, np.uint32)

def batch_slic_bow_feature():
    names = selected_files()[1:2]
    import pp

    ppservers = ()
    job_server = pp.Server(4, ppservers=ppservers)
    inputs = range(4)
    jobs = [(i, job_server.submit(slic_bow_feature, (i, image_dir, label_dir), 
            (get_superpixel_label, bow_feature), 
            ("cv2", "numpy as np", "scipy.io as scio", "skimage.segmentation", "scipy.cluster.vq", "sklearn.preprocessing"))) 
            for i in names]
    for i, job in jobs:
        job()
    job_server.print_stats()

def slic_bow_feature(n, image_dir, label_dir):
    name = '0'*(4-len(str(n))) + str(n)
    # load labels in mat file
    label_path = '%s/%s.mat' % (label_dir, name)
    label = scio.loadmat(label_path)['groundtruth']
    number = np.unique(label)[1:]

    # load image
    image_path = '%s/%s.jpg' % (image_dir, name)
    image = cv2.imread(image_path)
    image2 = cv2.imread(image_path, 0)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # segmentation
    segments = skimage.segmentation.slic(image, n_segments = 500, sigma = 5)
    result = get_superpixel_label(label, segments)

    # load codebook
    codebook_path = 'codebook_%d.txt' % 1000
    voc = np.loadtxt(codebook_path, delimiter=',')

    fg_label = np.loadtxt('fg_label.txt', delimiter=',')
    (feature, seg_num) = bow_feature(image2, voc, segments)
    #print len(feature), len(feature[0]), len(seg_num)
    spf = np.array([])
    for i in xrange(len(seg_num)):
        n = seg_num[i]
        if n in result:
            # dsc not None and label not 0
            copy = np.array(feature[i])
            copy = np.insert(copy, 0, result[n][0], axis=0)
            copy = np.insert(copy, 0, n, axis=0)
            if len(spf) == 0:
                spf = np.array(copy)
            else:
                spf = np.vstack((spf, copy))
    save_path = 'spf/%s.txt' % name
    np.savetxt(save_path, spf, delimiter=',', fmt='%1.4f')

def test_bow_predict_mat():
    name = '0001'
    # load labels in mat file
    label_path = '%s/%s.mat' % (label_dir, name)
    label = scio.loadmat(label_path)['groundtruth']
    number = np.unique(label)[1:]

    # load image
    image_path = '%s/%s.jpg' % (image_dir, name)
    image = cv2.imread(image_path)
    image2 = cv2.imread(image_path, 0)

    codebook_path = 'codebook_%d.txt' % 1000
    voc = np.loadtxt(codebook_path, delimiter=',')

    for l in number:
        mask = np.array(label, np.uint8)
        mask[np.where(mask!=l)] = 0
        mask[np.where(mask==l)] = 1

        y = bow_predict(image2, voc, mask)[0]
        print l, y

if __name__ == '__main__':
    parallel()
    bow()
    k = 1000
    batch_slic_bow_feature()
    ovr_svm_train(k)
    test_bow_predict_slic()
