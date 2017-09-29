from __future__ import print_function
from past.builtins import xrange

import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter


def extract_features(imgs, feature_fns, verbose=False):
  """
  Given pixel data for images and several feature functions that can operate on
  single images, apply all feature functions to all images, concatenating the
  feature vectors for each image and storing the features for all images in
  a single matrix.

  Inputs:
  - imgs: N x H X W X C array of pixel data for N images.
  - feature_fns: List of k feature functions. The ith feature function should
    take as input an H x W x D array and return a (one-dimensional) array of
    length F_i.
  - verbose: Boolean; if true, print progress.

  Returns:
  An array of shape (N, F_1 + ... + F_k) where each column is the concatenation
  of all features for a single image.
  """
  num_images = imgs.shape[0]
  if num_images == 0:
    return np.array([])

  # Use the first image to determine feature dimensions
  feature_dims = []
  first_image_features = []
  for feature_fn in feature_fns:
    feats = feature_fn(imgs[0].squeeze())
    assert len(feats.shape) == 1, 'Feature functions must be one-dimensional'
    feature_dims.append(feats.size)
    first_image_features.append(feats)

  # Now that we know the dimensions of the features, we can allocate a single
  # big array to store all features as columns.
  total_feature_dim = sum(feature_dims)
  imgs_features = np.zeros((num_images, total_feature_dim))
  imgs_features[0] = np.hstack(first_image_features).T

  # Extract features for the rest of the images.
  for i in xrange(1, num_images):
    idx = 0
    for feature_fn, feature_dim in zip(feature_fns, feature_dims):
      next_idx = idx + feature_dim
      imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
      idx = next_idx
    if verbose and i % 1000 == 0:
      print('Done extracting features for %d / %d images' % (i, num_images))

  return imgs_features


def rgb2gray(rgb):
  """Convert RGB image to grayscale

    Parameters:
      rgb : RGB image

    Returns:
      gray : grayscale image
  
  """
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def hog_feature(im):
  """Compute Histogram of Gradient (HOG) feature for an image
  
       Modified from skimage.feature.hog
       http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog
     
     Reference:
       Histograms of Oriented Gradients for Human Detection
       Navneet Dalal and Bill Triggs, CVPR 2005
     
    Parameters:
      im : an input grayscale or rgb image
      
    Returns:
      feat: Histogram of Gradient (HOG) feature
    
  """
  
  # convert rgb to grayscale if needed
  if im.ndim == 3:
    image = rgb2gray(im)
  else:
    image = np.at_least_2d(im)

  sx, sy = image.shape # image size
  orientations = 9 # number of gradient bins
  cx, cy = (8, 8) # pixels per cell

  gx = np.zeros(image.shape)
  gy = np.zeros(image.shape)
  gx[:, :-1] = np.diff(image, n=1, axis=1) # compute gradient on x-direction
  gy[:-1, :] = np.diff(image, n=1, axis=0) # compute gradient on y-direction
  grad_mag = np.sqrt(gx ** 2 + gy ** 2) # gradient magnitude
  grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90 # gradient orientation

  n_cellsx = int(np.floor(sx / cx))  # number of cells in x
  n_cellsy = int(np.floor(sy / cy))  # number of cells in y
  # compute orientations integral images
  orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
  for i in range(orientations):
    # create new integral image for this orientation
    # isolate orientations in this range
    temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                        grad_ori, 0)
    temp_ori = np.where(grad_ori >= 180 / orientations * i,
                        temp_ori, 0)
    # select magnitudes for those orientations
    cond2 = temp_ori > 0
    temp_mag = np.where(cond2, grad_mag, 0)
    orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[cx/2::cx, cy/2::cy].T
  
  return orientation_histogram.ravel()


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
  """
  Compute color histogram for an image using hue.

  Inputs:
  - im: H x W x C array of pixel data for an RGB image.
  - nbin: Number of histogram bins. (default: 10)
  - xmin: Minimum pixel value (default: 0)
  - xmax: Maximum pixel value (default: 255)
  - normalized: Whether to normalize the histogram (default: True)

  Returns:
    1D vector of length nbin giving the color histogram over the hue of the
    input image.
  """
  ndim = im.ndim
  bins = np.linspace(xmin, xmax, nbin+1)
  hsv = matplotlib.colors.rgb_to_hsv(im/xmax) * xmax
  imhist, bin_edges = np.histogram(hsv[:,:,0], bins=bins, density=normalized)
  imhist = imhist * np.diff(bin_edges)

  # return histogram
  return imhist


def daisy(im, step=4, radius=15, rings=3, histograms=8, orientations=8,
          normalization='l1', sigmas=None, ring_radii=None):
    '''
    Revised from skimage.feature.daisy:
    
    Extract DAISY feature descriptors densely for the given image.

    DAISY is a feature descriptor similar to SIFT formulated in a way that
    allows for fast dense extraction. Typically, this is practical for
    bag-of-features image representations.

    The implementation follows Tola et al. [1]_ but deviate on the following
    points:

      * Histogram bin contribution are smoothed with a circular Gaussian
        window over the tonal range (the angular range).
      * The sigma values of the spatial Gaussian smoothing in this code do not
        match the sigma values in the original code by Tola et al. [2]_. In
        their code, spatial smoothing is applied to both the input image and
        the center histogram. However, this smoothing is not documented in [1]_
        and, therefore, it is omitted.

    Parameters
    ----------
    im : (M, N) array
        Input image (greyscale).
    step : int, optional
        Distance between descriptor sampling points.
    radius : int, optional
        Radius (in pixels) of the outermost ring.
    rings : int, optional
        Number of rings.
    histograms  : int, optional
        Number of histograms sampled per ring.
    orientations : int, optional
        Number of orientations (bins) per histogram.
    normalization : [ 'l1' | 'l2' | 'daisy' | 'off' ], optional
        How to normalize the descriptors

          * 'l1': L1-normalization of each descriptor.
          * 'l2': L2-normalization of each descriptor.
          * 'daisy': L2-normalization of individual histograms.
          * 'off': Disable normalization.

    sigmas : 1D array of float, optional
        Standard deviation of spatial Gaussian smoothing for the center
        histogram and for each ring of histograms. The array of sigmas should
        be sorted from the center and out. I.e. the first sigma value defines
        the spatial smoothing of the center histogram and the last sigma value
        defines the spatial smoothing of the outermost ring. Specifying sigmas
        overrides the following parameter.

            ``rings = len(sigmas) - 1``

    ring_radii : 1D array of int, optional
        Radius (in pixels) for each ring. Specifying ring_radii overrides the
        following two parameters.

            ``rings = len(ring_radii)``
            ``radius = ring_radii[-1]``

        If both sigmas and ring_radii are given, they must satisfy the
        following predicate since no radius is needed for the center
        histogram.

            ``len(ring_radii) == len(sigmas) + 1``

    visualize : bool, optional
        Generate a visualization of the DAISY descriptors

    Returns
    -------
    descs : array
        Grid of DAISY descriptors for the given image as an array
        dimensionality  (P, Q, R) where

            ``P = ceil((M - radius*2) / step)``
            ``Q = ceil((N - radius*2) / step)``
            ``R = (rings * histograms + 1) * orientations``

    descs_img : (M, N, 3) array (only if visualize==True)
        Visualization of the DAISY descriptors.

    References
    ----------
    .. [1] Tola et al. "Daisy: An efficient dense descriptor applied to wide-
           baseline stereo." Pattern Analysis and Machine Intelligence, IEEE
           Transactions on 32.5 (2010): 815-830.
    .. [2] http://cvlab.epfl.ch/software/daisy
    '''
    import numpy as np
    from scipy import sqrt, pi, arctan2, cos, sin, exp
    from scipy.ndimage import gaussian_filter
    
    
    im = img_as_float(im)
    
    # Take mean of RGB channels
    im = np.mean(im, axis=2)


    # Validate parameters.
    if sigmas is not None and ring_radii is not None \
            and len(sigmas) - 1 != len(ring_radii):
        raise ValueError('`len(sigmas)-1 != len(ring_radii)`')
    if ring_radii is not None:
        rings = len(ring_radii)
        radius = ring_radii[-1]
    if sigmas is not None:
        rings = len(sigmas) - 1
    if sigmas is None:
        sigmas = [radius * (i + 1) / float(2 * rings) for i in range(rings)]
    if ring_radii is None:
        ring_radii = [radius * (i + 1) / float(rings) for i in range(rings)]
    if normalization not in ['l1', 'l2', 'daisy', 'off']:
        raise ValueError('Invalid normalization method.')

    # Compute image derivatives.
    dx = np.zeros(im.shape)
    dy = np.zeros(im.shape)
    dx[:, :-1] = np.diff(im, n=1, axis=1)
    dy[:-1, :] = np.diff(im, n=1, axis=0)

    # Compute gradient orientation and magnitude and their contribution
    # to the histograms.
    grad_mag = sqrt(dx ** 2 + dy ** 2)
    grad_ori = arctan2(dy, dx)
    orientation_kappa = orientations / pi
    orientation_angles = [2 * o * pi / orientations - pi
                          for o in range(orientations)]
    hist = np.empty((orientations,) + im.shape, dtype=float)
    for i, o in enumerate(orientation_angles):
        # Weigh bin contribution by the circular normal distribution
        hist[i, :, :] = exp(orientation_kappa * cos(grad_ori - o))
        # Weigh bin contribution by the gradient magnitude
        hist[i, :, :] = np.multiply(hist[i, :, :], grad_mag)

    # Smooth orientation histograms for the center and all rings.
    sigmas = [sigmas[0]] + sigmas
    hist_smooth = np.empty((rings + 1,) + hist.shape, dtype=float)
    for i in range(rings + 1):
        for j in range(orientations):
            hist_smooth[i, j, :, :] = gaussian_filter(hist[j, :, :],
                                                      sigma=sigmas[i])

    # Assemble descriptor grid.
    theta = [2 * pi * j / histograms for j in range(histograms)]
    desc_dims = (rings * histograms + 1) * orientations
    descs = np.empty((desc_dims, im.shape[0] - 2 * radius,
                      im.shape[1] - 2 * radius))
    descs[:orientations, :, :] = hist_smooth[0, :, radius:-radius, radius:-radius]
    idx = orientations
    for i in range(rings):
        for j in range(histograms):
            y_min = radius + int(round(ring_radii[i] * sin(theta[j])))
            y_max = descs.shape[1] + y_min
            x_min = radius + int(round(ring_radii[i] * cos(theta[j])))
            x_max = descs.shape[2] + x_min
            descs[idx:idx + orientations, :, :] = hist_smooth[i + 1, :,
                                                              y_min:y_max,
                                                              x_min:x_max]
            idx += orientations
    descs = descs[:, ::step, ::step]
    descs = descs.swapaxes(0, 1).swapaxes(1, 2)

    # Normalize descriptors.
    descs /= np.sum(descs, axis=2)[:, :, np.newaxis]

    # Flatten the descriptors as features
    return descs.flatten()


def convert(image, dtype, force_copy=False, uniform=False):
    """
    From skimage package:
    
    Convert an image to the requested data-type.
    Warnings are issued in case of precision loss, or when negative values
    are clipped during conversion to unsigned integer types (sign loss).
    Floating point values are expected to be normalized and will be clipped
    to the range [0.0, 1.0] or [-1.0, 1.0] when converting to unsigned or
    signed integers respectively.
    Numbers are not shifted to the negative side when converting from
    unsigned to signed integer types. Negative values will be clipped when
    converting to unsigned integers.
    Parameters
    ----------
    image : ndarray
        Input image.
    dtype : dtype
        Target data-type.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    uniform : bool, optional
        Uniformly quantize the floating point range to the integer range.
        By default (uniform=False) floating point values are scaled and
        rounded to the nearest integers, which minimizes back and forth
        conversion errors.
    References
    ----------
    .. [1] DirectX data conversion rules.
           http://msdn.microsoft.com/en-us/library/windows/desktop/dd607323%28v=vs.85%29.aspx
    .. [2] Data Conversions. In "OpenGL ES 2.0 Specification v2.0.25",
           pp 7-8. Khronos Group, 2010.
    .. [3] Proper treatment of pixels as integers. A.W. Paeth.
           In "Graphics Gems I", pp 249-256. Morgan Kaufmann, 1990.
    .. [4] Dirty Pixels. J. Blinn. In "Jim Blinn's corner: Dirty Pixels",
           pp 47-57. Morgan Kaufmann, 1998.
    """
    import numpy as np

    image = np.asarray(image)
    dtypeobj_in = image.dtype
    dtypeobj_out = np.dtype(dtype)
    dtype_in = dtypeobj_in.type
    dtype_out = dtypeobj_out.type
    kind_in = dtypeobj_in.kind
    kind_out = dtypeobj_out.kind
    itemsize_in = dtypeobj_in.itemsize
    itemsize_out = dtypeobj_out.itemsize

    if dtype_in == dtype_out:
        if force_copy:
            image = image.copy()
        return image

    if not (dtype_in in _supported_types and dtype_out in _supported_types):
        raise ValueError("Can not convert from {} to {}."
                         .format(dtypeobj_in, dtypeobj_out))

    def sign_loss():
        warn("Possible sign loss when converting negative image of type "
             "{} to positive image of type {}."
             .format(dtypeobj_in, dtypeobj_out))

    def prec_loss():
        warn("Possible precision loss when converting from {} to {}"
             .format(dtypeobj_in, dtypeobj_out))

    def _dtype_itemsize(itemsize, *dtypes):
        # Return first of `dtypes` with itemsize greater than `itemsize`
        return next(dt for dt in dtypes if np.dtype(dt).itemsize >= itemsize)

    def _dtype_bits(kind, bits, itemsize=1):
        # Return dtype of `kind` that can store a `bits` wide unsigned int
        def compare(x, y, kind='u'):
            if kind == 'u':
                return x <= y
            else:
                return x < y

        s = next(i for i in (itemsize, ) + (2, 4, 8) if compare(bits, i * 8,
                                                                kind=kind))
        return np.dtype(kind + str(s))

    def _scale(a, n, m, copy=True):
        """Scale an array of unsigned/positive integers from `n` to `m` bits.
        Numbers can be represented exactly only if `m` is a multiple of `n`.
        Parameters
        ----------
        a : ndarray
            Input image array.
        n : int
            Number of bits currently used to encode the values in `a`.
        m : int
            Desired number of bits to encode the values in `out`.
        copy : bool, optional
            If True, allocates and returns new array. Otherwise, modifies
            `a` in place.
        Returns
        -------
        out : array
            Output image array. Has the same kind as `a`.
        """
        kind = a.dtype.kind
        if n > m and a.max() < 2 ** m:
            mnew = int(np.ceil(m / 2) * 2)
            if mnew > m:
                dtype = "int{}".format(mnew)
            else:
                dtype = "uint{}".format(mnew)
            n = int(np.ceil(n / 2) * 2)
            warn("Downcasting {} to {} without scaling because max "
                 "value {} fits in {}".format(a.dtype, dtype, a.max(), dtype))
            return a.astype(_dtype_bits(kind, m))
        elif n == m:
            return a.copy() if copy else a
        elif n > m:
            # downscale with precision loss
            prec_loss()
            if copy:
                b = np.empty(a.shape, _dtype_bits(kind, m))
                np.floor_divide(a, 2**(n - m), out=b, dtype=a.dtype,
                                casting='unsafe')
                return b
            else:
                a //= 2**(n - m)
                return a
        elif m % n == 0:
            # exact upscale to a multiple of `n` bits
            if copy:
                b = np.empty(a.shape, _dtype_bits(kind, m))
                np.multiply(a, (2**m - 1) // (2**n - 1), out=b, dtype=b.dtype)
                return b
            else:
                a = a.astype(_dtype_bits(kind, m, a.dtype.itemsize), copy=False)
                a *= (2**m - 1) // (2**n - 1)
                return a
        else:
            # upscale to a multiple of `n` bits,
            # then downscale with precision loss
            prec_loss()
            o = (m // n + 1) * n
            if copy:
                b = np.empty(a.shape, _dtype_bits(kind, o))
                np.multiply(a, (2**o - 1) // (2**n - 1), out=b, dtype=b.dtype)
                b //= 2**(o - m)
                return b
            else:
                a = a.astype(_dtype_bits(kind, o, a.dtype.itemsize), copy=False)
                a *= (2**o - 1) // (2**n - 1)
                a //= 2**(o - m)
                return a

    if kind_in in 'ui':
        imin_in = np.iinfo(dtype_in).min
        imax_in = np.iinfo(dtype_in).max
    if kind_out in 'ui':
        imin_out = np.iinfo(dtype_out).min
        imax_out = np.iinfo(dtype_out).max

    # any -> binary
    if kind_out == 'b':
        if kind_in in "fi":
            sign_loss()
        prec_loss()
        return image > dtype_in(dtype_range[dtype_in][1] / 2)

    # binary -> any
    if kind_in == 'b':
        result = image.astype(dtype_out)
        if kind_out != 'f':
            result *= dtype_out(dtype_range[dtype_out][1])
        return result

    # float -> any
    if kind_in == 'f':
        if np.min(image) < -1.0 or np.max(image) > 1.0:
            raise ValueError("Images of type float must be between -1 and 1.")
        if kind_out == 'f':
            # float -> float
            if itemsize_in > itemsize_out:
                prec_loss()
            return image.astype(dtype_out)

        # floating point -> integer
        prec_loss()
        # use float type that can represent output integer type
        image = image.astype(_dtype_itemsize(itemsize_out, dtype_in,
                                             np.float32, np.float64))
        if not uniform:
            if kind_out == 'u':
                image *= imax_out
            else:
                image *= imax_out - imin_out
                image -= 1.0
                image /= 2.0
            np.rint(image, out=image)
            np.clip(image, imin_out, imax_out, out=image)
        elif kind_out == 'u':
            image *= imax_out + 1
            np.clip(image, 0, imax_out, out=image)
        else:
            image *= (imax_out - imin_out + 1.0) / 2.0
            np.floor(image, out=image)
            np.clip(image, imin_out, imax_out, out=image)
        return image.astype(dtype_out)

    # signed/unsigned int -> float
    if kind_out == 'f':
        if itemsize_in >= itemsize_out:
            prec_loss()
        # use float type that can exactly represent input integers
        image = image.astype(_dtype_itemsize(itemsize_in, dtype_out,
                                             np.float32, np.float64))
        if kind_in == 'u':
            image /= imax_in
            # DirectX uses this conversion also for signed ints
            # if imin_in:
            #     np.maximum(image, -1.0, out=image)
        else:
            image *= 2.0
            image += 1.0
            image /= imax_in - imin_in
        return np.asarray(image, dtype_out)

    # unsigned int -> signed/unsigned int
    if kind_in == 'u':
        if kind_out == 'i':
            # unsigned int -> signed int
            image = _scale(image, 8 * itemsize_in, 8 * itemsize_out - 1)
            return image.view(dtype_out)
        else:
            # unsigned int -> unsigned int
            return _scale(image, 8 * itemsize_in, 8 * itemsize_out)

    # signed int -> unsigned int
    if kind_out == 'u':
        sign_loss()
        image = _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out)
        result = np.empty(image.shape, dtype_out)
        np.maximum(image, 0, out=result, dtype=image.dtype, casting='unsafe')
        return result

    # signed int -> signed int
    if itemsize_in > itemsize_out:
        return _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out - 1)

    image = image.astype(_dtype_bits('i', itemsize_out * 8))
    image -= imin_in
    image = _scale(image, 8 * itemsize_in, 8 * itemsize_out, copy=False)
    image += imin_out
    return image.astype(dtype_out)


def img_as_float(image, force_copy=False):
    """Convert an image to double-precision (64-bit) floating point format.
    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    Returns
    -------
    out : ndarray of float64
        Output image.
    Notes
    -----
    The range of a floating point image is [0.0, 1.0] or [-1.0, 1.0] when
    converting from unsigned or signed datatypes, respectively.
    If the input image has a float type, intensity values are not modified
    and can be outside the ranges [0.0, 1.0] or [-1.0, 1.0].
    """
    return convert(image, np.float64, force_copy)


pass
