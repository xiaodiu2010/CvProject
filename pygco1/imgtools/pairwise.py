"""
Implementation of pairwise potentials to be used in CRFs.

Yujia Li, 08/2013
"""

import struct
import numpy as np

def read_pb_file(file_name):
    """
    Read a probability of boundary matrix from a file.

    pb file format:
    offset      size    type        description
    -------------------------------------------
    0           4       uint32      number of dimensions of the first matrix (=D)
    4           4       uint32      size of the first dimension 
    8           4       uint32      size of the second dimension
    ...
    D*4         4       uint32      size of the Dth dimension
    (D+1)*4     4       float32     first element in the first matrix
    (D+2)*4     4       float32     second element in the first matrix
    ...

    If more than one matrices are in one file, then the other matrices will come 
    right after the first one. Elements in the matrices are stored in an order that
    the first dimension is the most significant dimension (changes most slowly),
    and the last dimension is the least significant dimension (changes fastest).

    It's also in column major ordering, as in matlab.

    Return a matrix the same size as the image used to compute pb.
    """
    f = open(file_name, 'rb')
    n_dim = struct.unpack('I', f.read(4))[0]

    d = [0] * n_dim
    n_pix = 1
    for i in range(n_dim):
        d[i] = struct.unpack('I', f.read(4))[0]
        n_pix *= d[i]

    B = np.empty(n_pix, np.float32)
    for i in range(n_pix):
        B[i] = struct.unpack('f', f.read(4))[0]

    B = B.reshape(d[::-1]).T

    # try to read the next pb array
    t = f.read(4)
    if len(t) == 4:     # more arrays available, read all of them
        A = [B]

        while len(t) > 0:
            n_dim = struct.unpack('I', t)[0]
            n_pix = 1
            d = [0] * n_dim
            for i in range(n_dim):
                d[i] = struct.unpack('I', f.read(4))[0]
                n_pix *= d[i]

            B = np.empty(n_pix, np.float32)
            for i in range(n_pix):
                B[i] = struct.unpack('f', f.read(4))[0]

            A.append(B.reshape(d[::-1]).T)
            t = f.read(4)
    else:
        A = B

    f.close()

    return A

def load_pb_from_dir(file_dir, fname_format, start_idx, end_idx):
    """
    Read all pb files from a specified directory into a list.

    file_dir: directory of the pb files
    fname_format: format of the pb file names, should allow one integer index
    start_idx, end_idx: the index should go from start_idx to end_idx - 1

    Return: pb, a list of pb matrices.
    """
    pb = []
    for i in range(start_idx, end_idx):
        fname = '%s/%s' % (file_dir, fname_format % i)
        pb.append(read_pb_file(fname))

    return pb

def get_uniform_smoothness_pw_single_image(imsz):
    """
    Generate uniform smoothness pairwise potential for a single image of size 
    imsz. Pixel indices are assumed to be in row-major order.

    In a uniform smoothness pairwse potential, for any pair of neighboring
    pixels i and j, p(i,j) = 1

    imsz: a tuple of two integers H,W for height and width of the image

    return: edges, edge_weights
        edges is a E*2 matrix, E is the number of edges in the grid graph.
            For 4-connected graphs, E=(H-1)*W + H*(W-1). Each row is a pair of
            pixel indices for an edge
        edge_weights is a E-dimensional vector of 1's.
    """
    H, W = imsz
    E = (H - 1) * W + H * (W - 1)

    edges = np.empty((E, 2), dtype=np.int)
    edge_weights = np.ones(E, dtype=np.single)
    idx = 0

    # horizontal edges
    for row in range(H):
        edges[idx:idx+W-1,0] = np.arange(W-1) + row * W
        edges[idx:idx+W-1,1] = np.arange(W-1) + row * W + 1
        idx += W-1

    # vertical edges
    for col in range(W):
        edges[idx:idx+H-1,0] = np.arange(0, (H-1)*W, W) + col
        edges[idx:idx+H-1,1] = np.arange(W, H*W, W) + col
        idx += H-1

    return [edges, edge_weights]

def get_uniform_smoothness_pw(imlist):
    """
    Generate uniform smoothness pairwise potentials for a list of images.

    In a uniform smoothness pairwse potential, for any pair of neighboring
    pixels i and j, p(i,j) = 1

    imlist: a list of images
    return: pw, a list of pairwise potentials each of size E*3, where E is the
        number of edges for the corresponding image.
    """
    pw = []

    for i in range(len(imlist)):
        pw.append(get_uniform_smoothness_pw_single_image(imlist[i].shape[:2]))

    return pw

def get_boykov_jolly_pw_single_image(img, sigma=None):
    """
    Compute Boykov-Jolly style of pairwise potentials for a single image.
    In this type of pairwise potentials, for a pair of neighboring pixels i
    and j, the pairwise potential is defined as

        p(i,j) = exp(-(Ip - Iq)^2 / (2*sigma^2)) / dist(p,q)

    where Ip and Iq are simply intensity values, and can be easily substituted
    by any feature vector. In fact, (Ip - Iq)^2 can be replaced by any
    similarity metric between the two pixels p and q (more generally speaking,
    the whole term can be replaced by a similarity metric). For a 4-connected 
    graph, dist(p,q) for a pair of neighboring pixels is always 1, and thus can
    be ignored.

    Take a look at Boykov and Jolly's ICCV'01 paper for more details.

    img: a color or gray image of size H*W, it should be a numpy array of size
        (H,W) for gray images or (H,W,3) for color images
    sigma: the standard deviation parameter, if not set, the default value
        E[(Ip - Iq)^2]) will be used.

    return: edges, edge_weights
        edges is a E*2 matrix, E is the number of edges in the grid graph.
            For 4-connected graphs, E=(H-1)*W + H*(W-1). Each row is a pair of
            pixel indices for an edge
        edge_weights is a E-dimensional vector of 1's.
    """
    H, W = img.shape[:2]
    E =  H * (W - 1) + (H - 1) * W

    edges = np.empty((E, 2), dtype=np.int)
    edge_weights = np.empty(E, dtype=np.single)
    idx = 0


    # normalize
    im = img.astype(np.single) / 255

    is_color = (len(img.shape) == 3 and img.shape[2] == 3)
    if not is_color:
        assert (img.shape == 2)

    # horizontal edges
    for row in range(H):
        edges[idx:idx+W-1,0] = np.arange(W-1) + row * W
        edges[idx:idx+W-1,1] = np.arange(W-1) + row * W + 1
        idx += W-1

    v_h = (im[:,0:W-1] - im[:,1:W])**2
    if is_color:
        v_h = v_h.sum(axis=2)

    # vertical edges
    for col in range(W):
        edges[idx:idx+H-1,0] = np.arange(0, (H-1)*W, W) + col
        edges[idx:idx+H-1,1] = np.arange(W, H*W, W) + col
        idx += H-1

    v_v = (im[0:H-1,:] - im[1:H,:])**2
    if is_color:
        v_v = v_v.sum(axis=2)

    if sigma == None:
        prec = 1.0 / (2 * (v_h.sum() + v_v.sum()) / E)
    else:
        prec = 1.0 / (2 * sigma * sigma)

    edge_weights[:H*(W-1)] = np.exp(-v_h.flatten() * prec)
    edge_weights[H*(W-1):] = np.exp(-v_v.T.flatten() * prec)

    return [edges, edge_weights]

def get_boykov_jolly_pw(imlist, sigma=None):
    """
    Compute Boykov-Jolly style pairwise potentials for a list of images.

    See function get_boykov_jolly_pw_single_image for more details.

    imlist: a list of images
    sigma: the standard deviation parameter, if not set, a defualt sigma will
        be computed and used for each image.

    return: pw, a list of pairwise potetnials, each element is a tuple, the
        first element is the E*2 edge matrix, the second element is the
        E-dimensional edge weight vector.
    """
    pw = []

    for i in range(len(imlist)):
        pw.append(get_boykov_jolly_pw_single_image(imlist[i], sigma))

    return pw

def create_visualization_for_pw_single_image(edges, edge_weights, imsz):
    """
    Create visualization for pairwise potentials of a single image, for grid
    graph only.

    The visualization is to transfer pairwise potentials into an image the
    same size as the original image. The process is simple, for each pixel, 
    look at the two edges coming from the north and west then take the maximum
    of the negative edge weights. Then for each pixel we get a number related 
    to the likelihood of an edge at that point, this is used for visualization.

    edges: E*2 edge matrix, E is the number of edges
    edge_weights: E-dimensional vector for edge weights, (edges, edge_weights)
        are generated by one of the get_pw functions
    imsz: size of the image, (H,W)

    return: a visualization image of size H*W
    """
    H, W = imsz

    pw = np.ones(imsz, dtype=np.single) * edge_weights.max()
    pw[:,1:W] = edge_weights[:H*(W-1)].reshape(H,W-1)
    pw[1:H,:] = np.fmin(edge_weights[H*(W-1):].reshape(W,H-1).T, pw[1:H,:])

    return -pw

def create_visualization_for_pw(pwlist, imsz):
    """
    Create visualization images for a list of pairwise potentials for a list
    of images. All the images should have the same size.

    pwlist: a list of pairwise potentials, each element is a tuple of (edges,
        edge_weights)
    imsz: size of images, a tuple (H,W)

    return: imlist, a list of visualized images.
    """
    imlist = []
    for i in range(len(pwlist)):
        imlist.append(create_visualization_for_pw_single_image(
            pwlist[i][0], pwlist[i][1], imsz))

    return imlist

def combine_pw(pwlist, weights):
    """
    Linearly combine a list of pairwise potentials, using coefficients in
    weights.

    pwlist: a list of K types of pairwise potentials
    weights: a K-element weight vector, one weight for each type of pairwise
        potential.
    """
    K = len(weights)
    n_cases = len(pwlist[0])

    pw = []
    for i in range(n_cases):
        edge_weights = pwlist[0][i][1] * 0
        for k in range(K):
            edge_weights += pwlist[k][i][1] * weights[k]
        pw.append([pwlist[0][i][0], edge_weights])

    return pw

