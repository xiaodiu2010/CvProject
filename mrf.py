"""
1. BOW + SIFT
2. pose + color space + gabor
    input foreground img
    1.for every superpixels in foreground.
        label: the class appears most times in the region
        position: information: 52 relative postion value to every center
              position of pose boxes.
        color space: 60 bins, RGB and LAB color information
        gabor: later
    2.trainig a logistic model / random forest model
3. clothset
    Input every mat file
    For every class,compute a likelihood of each garment appearing in
    a particular relative location of the human pose.
    For convinience, using the size 600 * 800 rather than sliding box
    When used in mrf, it need to be resized back to original size.
    This will be modified later.
4. graph_cut test
    Test on grid graph_cut method.
5. edge_weight, this part serves for superpixels grouping
    input: foreground img
    1. slic img get 1000 superpixels
    2. compute Hweight and Vweight
        for every point, Hw(i,j) = 2 if j and j+1 is same region
        if not, Hw(i,j) = exp(-avg(x(j)-x(j+1))/10)
        same way for Vh(i,j)
    Min-cut or Max-flow's cost function is sum of weights of edges been cutted.
    So weights between two regions is inverse proportion to pixel differences.
"""

import numpy as np
import scipy.io as scio
import skimage.io as sio
from matplotlib import pyplot as plt
from skimage import segmentation, color
import pygco1.pygco as pygc
from preprossing import training_unary, get_useful_labels,compute_center, hist0img, classOpixel
import matplotlib.patches as mpatches
import re
import pickle
import os


def compute_edge_weights(img, segments):
    """
    :param img: foreground img
    :return: Hw,Vw
    """
    dim = img.shape
    Hw = np.zeros((dim[0], dim[1]-1), dtype=np.int32)
    Vw = np.zeros((dim[0]-1, dim[1]), dtype=np.int32)
    for i in range(dim[0]):
        for j in range(dim[1]-1):
            if segments[i, j] == segments[i, j+1]:
                Hw[i, j] = 2
            else:
                x1 = img[i, j]
                x2 = img[i, j+1]
                Hw[i, j] = exp_function(x1, x2)
    for j in range(dim[1]):
        for i in range(dim[0]-1):
            if segments[i, j] == segments[i+1, j]:
                Vw[i, j] = 2
            else:
                x1 = img[i, j]
                x2 = img[i+1, j]
                Vw[i, j] = exp_function(x1, x2)
    return Hw, Vw


def exp_function(x1, x2, theta=10):
    """
    :param x1: pixel value vectors
    :param x2: pixel value vectors
    :param theta: parameter
    :return: weight value of x1 and x2
    """
    diff_x = np.mean(x1-x2)/theta
    return np.exp(-2* diff_x)


def draw_super_pixel(image, name='draw_super_pixel'):
    # seg_image = segmentation.mark_boundaries(image, segments)
    fig = plt.figure()
    _ = plt.imshow(image)
    path = './database/other/'+name + 'png'
    fig.savefig(path)
    return fig


def cut_from_grid(unaries, Vw, Hw, wp, ww):
    dim = unaries.shape[0:2]
    n_disps = unaries.shape[2]
    pair = wp * np.eye(n_disps, dtype=np.int32)
    potts_cut = pygc.cut_grid_graph(unaries, pair, ww*Vw, ww*Hw)
    result = potts_cut.reshape(dim)
    return result


def mrf_process(f,clf,position,label_use,label_name,wp=-5,ww=1,withclothset = False,wset = 1):

    img = sio.imread('./database/photos/'+ f)
    mat = f.replace('jpg', 'mat')
    label_path = './database/annotations/pixel-level/'+mat
    labels = scio.loadmat(label_path)['groundtruth']
    img = img[:800,:]
    labels = labels[:800,:]
    img_labels = np.unique(labels)
    img[labels == 0] = 0
    num = int(re.split('\.', f)[0])
    position_num = position[position[:, 0] == num, :][0]


    segments = segmentation.slic(img, n_segments=1000, sigma=5)
    Hw, Vw = compute_edge_weights(img, segments)
    print Hw.shape, Vw.shape

    label_use_img = np.array([True if i in img_labels[1:] else False for i in label_use[1:]],dtype = bool)
    label_use_img = np.where(label_use_img == True)[0]

    label_name =  np.hstack(('null', label_name[label_use_img + 1]))

    classnum = img_labels.shape[0]

    differences = get_unaries(segments,img_labels,labels,classnum,clf,img,position_num,
                              label_use_img,withclothset = withclothset,wset = wset)

    unaries = differences.copy("C").astype(np.int32)
    print unaries.shape

    result = cut_from_grid(unaries, Vw, Hw, wp=wp, ww=ww)
    new_result = result.copy()
    for i in np.unique(new_result):
        if i ==0:
            continue
        else:
            new_result[np.where(new_result==i)] = label_use[label_use_img[i-1]+1]
            print label_use[label_use_img[i-1]] +1
    print np.unique(result)
    print np.unique(new_result)
    print np.sum(np.array(new_result == labels))
    rate = np.sum(np.array(new_result == labels)) / (800 * 550.0)

    draw_result(result,name =str(num),label_name= label_name,classnum = classnum)
    return  result, rate



def draw_result(result,name,label_name,classnum,path = './database/mrfresult/'):
    fig = plt.figure()
    im = plt.imshow(result)
    # get the colors of the values, according to the
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in range(classnum)]
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=label_name[i])) for i in range(len(label_name))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1),loc =2, borderaxespad=0.)
    path = path + name + '.png'
    fig.savefig(path)
    return fig


def get_unaries(segments,img_labels,labels,classnum,clf,img,position_num,label_use_img,withclothset = False,wset=1):
    # labels: mat file
    classnum = label_use_img.shape[0] + 1
    differences = np.zeros((800,550,classnum))
    max_disp = classnum
    centerx, centery = compute_center(position_num)
    img_lab = color.rgb2lab(img)
    total =[]
    for i in np.unique(segments):
        temp = []
        index = np.where(segments == i)
        temp_region = img[index]
        temp_label = labels[index]
        label = classOpixel(temp_label)
        if label == 0:
            continue
        temp.append(i)
        temp_region_lab = img_lab[index]
        feature = np.hstack((hist0img(temp_region) / float(temp_region.shape[0]),
                             hist0img(temp_region_lab) / temp_region_lab.shape[0]))
        #print feature.shape
        temp = temp + list(feature)
        x = np.mean(index[0], dtype=float)
        y = np.mean(index[1], dtype=float)
        x_coor = [(i - x) / 800 for i in centerx]
        y_coor = [(i - y) / 550 for i in centery]
        temp = temp + x_coor
        temp = temp + y_coor
        total.append(temp)

    result_u = np.vstack(total)
    x_test = result_u[:,1:]
    probs = clf.predict_proba(x_test)[:,label_use_img]
    probs = 100 - probs*100
    print probs.shape
    differences[:, :, 0][labels != 0] = 100
    for i in range(1,classnum):
        differences[:,:,i][labels ==0] = 100
    for i,num in enumerate(result_u[:,0]):
        region = np.where(segments == num)
        temp = differences[region]
        temp[:,1:] = probs[i,:]
        differences[region] = temp

    if withclothset:
        for i ,num in enumerate(img_labels):
            if num ==0:
                continue
            print num
            path = './database/clothset/' + 'clothset' + str(num) + '.csv'
            clothset = np.loadtxt(path, delimiter=',', dtype=int)
            clothset = 100- clothset*100./np.max(clothset)
            differences[:,:,i] = differences[:,:,i] + wset* clothset
    return differences

def main():
    ## load model
    filename = './database/model.sav'
    label_use = np.loadtxt('./database/label_use.csv',delimiter=',',dtype=int)
    clf = pickle.load(open(filename, 'rb'))
    position = scio.loadmat('./database/pose_result.mat')['result']
    position = position.astype(np.int32)
    label_name = np.loadtxt('./database/label_name.csv',dtype=str)
    # initial model
    pic_path = './database/photos/'

    """
    wprange = -1*np.linspace(2,20,10)
    wwrange = np.linspace(1,10,10)
    ratemat = np.zeros((wprange.shape[0],wwrange.shape[0]))
    for i,wp in enumerate(wprange):
        for j,ww in enumerate(wwrange):
            result,rate = mrf_process(f, clf, position, label_use, label_name, wp=wp, ww=ww, withclothset=False, wset=1)
            ratemat[i,j] = rate
    """
    ratemat = []
    for f in os.listdir(pic_path)[1:2]:
        result,rate = mrf_process(f,clf,position,label_use,label_name,wp=-6,ww=10,withclothset = False,wset = 1)
        np.savetxt('./database/mrfresult/' + re.split('\.',f)[0] + '.csv',result,delimiter=',',fmt='%d')
        ratemat.append(rate)




if __name__ == '__main__':
    main()
