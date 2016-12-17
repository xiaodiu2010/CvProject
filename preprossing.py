import numpy as np
import scipy.io as scio
import skimage.io as sio
from matplotlib import pyplot as plt
from skimage import segmentation,color
import os
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pickle
import argparse


def clothset():
    """
    :return: 800 * 550 * 59 array
     every 800 * 550 is a likelihood of each garment appearing in
    a particular relative location of the human pose.
    For convenience, using the size 800 * 550 rather than sliding box
    When used in mrf, it need to resize back to original size.
    This will be modified later.
    """
    result = np.zeros((800, 550, 59))
    label_path = './database/annotations/pixel-level/'
    for i, f in enumerate(os.listdir(label_path)[1:1005]):
        f = label_path + f
        labels = scio.loadmat(f)['groundtruth']
        labels = labels[:800, :]
        uni_label = np.unique(labels)
        for j in uni_label:
            index = np.where(labels == j)
            result[:,:,j][index] += 1
    result[:,:,0] = 0
    result = result.astype(np.int32)
    for i in range(1,59):
        maxval = float(np.max(np.max(result[:,:,i])))
        if maxval == 0:
            result[:,:,i] = 0
        else:
            result[:,:,i] = result[:,:,i]/maxval
    for i in range(59):
        name = 'clothset' + str(i) + '.csv'
        np.savetxt('./database/clothset/'+name,result[:,:,i],fmt='%d',delimiter=',')




def super_unary_data():
    ## useful pics
    result_path = './database/photos/'
    position = scio.loadmat('./database/pose_result.mat')['result']
    position = position.astype(np.int32)
    pics = os.listdir(result_path)
    pic_path = './database/photos/'
    label_path = './database/annotations/pixel-level/'
    total = []
    for j, f in enumerate(pics):
        pic = pic_path + f
        num = int(re.split('\.',f)[0])
        position_num = position[position[:,0] == num,:][0]
        centerx, centery = compute_center(position_num)
        label = label_path + f.replace('jpg', 'mat')
        img = sio.imread(pic)
        img = img[:800,:]
        img_lab = color.rgb2lab(img)
        labels = scio.loadmat(label)['groundtruth']
        labels = labels[:800,:]
        label_set = np.unique(labels)
        img[np.where(labels ==0)] = 0
        img_lab[np.where(labels ==0)] = 0


        segments = segmentation.slic(img, n_segments=1000, sigma=5)
        for i in np.unique(segments):
            temp = []
            index = np.where(segments == i)
            temp_label = labels[index]
            label = classOpixel(temp_label)
            if label == 0:
                continue
            temp.append(label)
            temp_region = img[index]
            temp_region_lab = img_lab[index]
            feature = np.hstack((hist0img(temp_region) / float(temp_region.shape[0]),
                                 hist0img(temp_region_lab) / temp_region_lab.shape[0]))
            print feature.shape
            temp = temp + list(feature)
            x = np.mean(index[0], dtype=float)
            y = np.mean(index[1], dtype=float)
            x_coor = [(i-x)/800 for i in centerx]
            y_coor = [(i-y)/550 for i in centery]
            temp = temp + x_coor
            temp = temp + y_coor
            total.append(temp)

    result_u = np.vstack(total)
    np.savetxt('./database/super_unaries.csv',result_u,delimiter=',',fmt='%5f')


def training_unary(label_use, method = 'LR'):
    result_u = np.loadtxt('./database/super_unaries.csv',delimiter=',')
    index_u = np.array([True if (i in label_use) else False for i in result_u[:,0]],dtype=bool)
    result_u = result_u[index_u,:]
    numodata = result_u.shape[0]
    indices = np.random.permutation(numodata)
    training_idx, test_idx = indices[:int(0.8*numodata)],indices[int(0.8*numodata):]
    training,test = result_u[training_idx,:],result_u[test_idx,:]
    x_all,y_all = result_u[:,1:],result_u[:,0]
    X,y = training[:,1:],training[:,0]
    X_test,y_test = test[:,1:],test[:,0]
    clf = RandomForestClassifier(n_estimators=50,criterion = 'entropy')
    clf.fit(x_all, y_all)
    y_pred = clf.predict(x_all)
    confse_matrix = confusion_matrix(y_all,y_pred)
    np.savetxt('./database/confuse_matrix.csv',confse_matrix,delimiter=',',fmt='%d')
    filename = './database/model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    return clf

def get_useful_labels(threshold = 50):
    ## useful_labels
    label_path = './database/annotations/pixel-level/'
    label_num = np.zeros(59, dtype=int)
    for f in os.listdir(label_path):
        path = label_path + f
        try:
            labels = scio.loadmat(path)['groundtruth']
            label_set = np.unique(labels)
            label_num[label_set] = label_num[label_set] + 1
        except MemoryError as e:
            print "not mat"

    label_name = scio.loadmat('./database/label_list.mat')['label_list'][0]
    label = range(59)
    label_set = np.vstack((label_num, label_name))
    label_set = np.vstack((label,label_set))
    label_use = label_set[0][label_num > threshold]
    np.savetxt('./database/label_use.csv',label_use,delimiter=',',fmt='%u')
    label_name = label_set[2][label_num > threshold]
    label_name = map(lambda x: x.astype(str,15),label_name)
    list(label_name)
    np.savetxt('./database/label_name.csv',label_name,delimiter=',',fmt='%s')
    return label_use


def compute_center(position_num):
    centerx = []
    centery = []
    for i in range(1,27):
        box = position_num[4*i-3:4*i+1]
        centerx.append((box[1]+box[3])/2)
        centery.append((box[0]+box[2])/2)
    return centerx,centery

def classOpixel(temp_label):
    label, counts = np.unique(temp_label, return_counts=True)
    return label[np.argmax(counts)]

def hist0img(region_arr):
    item = []
    item = np.array(item)
    if len(region_arr.shape) == 2:
        for i in np.arange(3):
            hist = np.histogram(region_arr[:, i], bins=10)[0]
            item = np.append(item, hist)
    else:
        hist = np.histogram(region_arr, bins=10,normed=True)[0]
        item = np.append(item, hist)
    return item

def draw_pics(img,name = 'draw_super_pixel',path = './database/other/'):
    fig = plt.figure()
    _ = plt.imshow(img)
    path = path+ name + 'png'
    fig.savefig(path)
    plt.close()


def main(pretrain=True):
    # initial model
    """
    Run these model if you want build project from scratch.
    Remember: you need the whole dataset!
    """

    if pretrain:
        label_use = np.loadtxt('./database/label_use.csv',delimiter=',',dtype=int)
        clf = training_unary(label_use=label_use)
    else:
        clothset()
        super_unary_data()
        label_use = get_useful_labels(threshold=50)
        clf = training_unary(label_use=label_use)



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pretrain",  help="Wheather use pretrain model",  action="store_true")
    args = ap.parse_args()
    if args.pretrain:
        print 'use pretrain model'
    main(args.pretrain)
