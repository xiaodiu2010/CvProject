"""
Some image processing utilities.

Yujia Li, 08/2013
"""
from scipy import misc
import os

def save_imlist(imlist, output_dir, name_format="%d.jpg"):
    """
    Save a list of images to a specified directory.

    You can specify the format of the file name, then image indices will be
    filled in to create file names.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(imlist)):
        file_name = '%s/%s' % (output_dir, name_format % i)
        misc.imsave(file_name, imlist[i])

        if (i+1) % 100 == 0:
            print 'Saved %d images...' % (i+1)
        
def resize_imlist(imlist, new_size):
    """
    Resize all images in the list to a new size.
    """
    new_imlist = []
    n_imgs = len(imlist)

    for i in range(n_imgs):
        new_imlist.append(misc.imresize(imlist[i], new_size, interp='bilinear'))

    return new_imlist

