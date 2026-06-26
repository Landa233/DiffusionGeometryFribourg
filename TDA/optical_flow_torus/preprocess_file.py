"""Python port of preprocessFile.m and its helper functions.

Pipeline for turning a Middlebury/Sintel ``.flo`` optical-flow file into a set
of preprocessed 3x3 flow patches, following the method of Lee, Pedersen and
Mumford ("The Nonlinear Statistics of High-Contrast Patches in Natural
Images").

Each patch is stored as a row vector of length ``2 * patch_size**2``. The first
half of the vector is the horizontal (u) flow component and the second half the
vertical (v) component. Both components are flattened in column-major order to
match the original MATLAB ``reshape`` behaviour.
"""

import numpy as np


def read_flow_file(filename):
    """Read a ``.flo`` flow file into an ``(height, width, 2)`` array.

    Port of readFlowFile.m (originally by Deqing Sun, after Daniel Scharstein).
    """
    TAG_FLOAT = 202021.25

    if not filename:
        raise ValueError("read_flow_file: empty filename")
    if not filename.endswith(".flo"):
        raise ValueError(
            "read_flow_file: filename %s should have extension '.flo'" % filename
        )

    with open(filename, "rb") as fid:
        tag = np.fromfile(fid, np.float32, count=1)
        width = np.fromfile(fid, np.int32, count=1)
        height = np.fromfile(fid, np.int32, count=1)

        if tag.size == 0 or tag[0] != TAG_FLOAT:
            raise ValueError(
                "read_flow_file(%s): wrong tag (possibly big-endian machine?)"
                % filename
            )
        width = int(width[0])
        height = int(height[0])
        if width < 1 or width > 99999:
            raise ValueError("read_flow_file(%s): illegal width %d" % (filename, width))
        if height < 1 or height > 99999:
            raise ValueError(
                "read_flow_file(%s): illegal height %d" % (filename, height)
            )

        n_bands = 2
        tmp = np.fromfile(fid, np.float32, count=width * height * n_bands)

    # Data is stored row-major as (height, width, bands) interleaved.
    tmp = tmp.reshape((height, width, n_bands))
    img = np.empty((height, width, n_bands), dtype=np.float64)
    img[:, :, 0] = tmp[:, :, 0]  # horizontal component
    img[:, :, 1] = tmp[:, :, 1]  # vertical component
    return img


def sample_from_data_set(filepath, numpatches, patchsize, rng=None):
    """Sample ``numpatches`` random ``patchsize x patchsize`` patches.

    Port of sampleFromDataSet.m. Returns an array of shape
    ``(numpatches, 2 * patchsize**2)`` with patches stored in rows.
    """
    if rng is None:
        rng = np.random.default_rng()

    data = read_flow_file(filepath)
    m, n, _ = data.shape

    if numpatches < 1:
        raise ValueError("Number of patches needs to be positive.")
    if patchsize < 1:
        raise ValueError("Patch size must be positive.")
    if patchsize > m or patchsize > n:
        raise ValueError("Patch size must be smaller than data matrix")

    # Upper-left coordinates of each patch (note: duplicates are possible).
    patchrows = rng.integers(0, m - patchsize + 1, size=numpatches)
    patchcols = rng.integers(0, n - patchsize + 1, size=numpatches)

    patches = np.zeros((numpatches, 2 * patchsize ** 2))
    for i in range(numpatches):
        r, c = patchrows[i], patchcols[i]
        p = data[r:r + patchsize, c:c + patchsize, :]
        # Column-major flatten to match MATLAB reshape: u-component first
        # (read down columns), then v-component.
        patches[i, :] = p.reshape(-1, order="F")
    return patches


def get_d_norm_matrix(patch_size):
    """Build the D-norm matrix used to compute patch contrast.

    Port of getDNormMatrix.m. See Lee, Pedersen & Mumford, p. 88.
    """
    size = patch_size ** 2
    D = np.zeros((size, size))
    for i in range(size):  # 0-based row index
        count = 0
        if i >= patch_size:            # has a neighbour above
            D[i, i - patch_size] = -1
            count += 1
        if i % patch_size != 0:        # has a neighbour to the left
            D[i, i - 1] = -1
            count += 1
        if i % patch_size != patch_size - 1:  # has a neighbour to the right
            D[i, i + 1] = -1
            count += 1
        if i < size - patch_size:      # has a neighbour below
            D[i, i + patch_size] = -1
            count += 1
        D[i, i] = count
    return D


def compute_contrast_norm(patches):
    """Append the contrast D-norm of each patch as an extra last column.

    Port of computeContrastNorm.m. Input rows are flow vectors of length
    ``2 * n**2``; output has one extra column holding the contrast norm.
    """
    mrows, ncols = patches.shape
    n = np.sqrt(ncols / 2)
    if n != np.floor(n):
        raise ValueError(
            "Improperly sized patch data matrix. Number of columns should be "
            "2*n^2 for some integer n."
        )
    n = int(n)
    D = get_d_norm_matrix(n)

    mtx = np.hstack([patches, np.zeros((mrows, 1))])
    for i in range(mrows):
        u = patches[i, :n ** 2]
        v = patches[i, n ** 2:ncols]
        d = np.sqrt(u @ D @ u + v @ D @ v)
        mtx[i, ncols] = d
    return mtx


def get_high_contrast_patches(patches):
    """Keep only the top-fraction of patches by contrast norm (last column).

    Port of getHighContrastPatches.m.
    """
    FRACTION = 1.0 / 5.0
    mrows, ncols = patches.shape
    numberhighcontrast = int(np.ceil(mrows * FRACTION))  # at least one

    # Sort rows by the contrast norm (last column), ascending.
    order = np.argsort(patches[:, ncols - 1], kind="stable")
    patches = patches[order]
    return patches[mrows - numberhighcontrast:mrows, :]


def normalize_patches(patches):
    """Mean-center each flow component and scale to contrast norm 1.

    Port of normalizePatches.m. The last column is assumed to hold the
    contrast norm.
    """
    patches = patches.copy()
    mrows, ncols = patches.shape
    vecsize = (ncols - 1) / 2
    if vecsize != np.floor(vecsize):
        raise ValueError("Patch data matrix should have an odd number of columns.")
    vecsize = int(vecsize)

    # Mean-center the horizontal and vertical components separately.
    uavg = patches[:, :vecsize].mean(axis=1, keepdims=True)
    vavg = patches[:, vecsize:ncols - 1].mean(axis=1, keepdims=True)
    patches[:, :vecsize] -= uavg
    patches[:, vecsize:ncols - 1] -= vavg

    # Warn about near-zero norms before dividing.
    zerotest = np.where(patches[:, ncols - 1] < 1e-10)[0]
    if zerotest.size > 0:
        import warnings
        warnings.warn(
            "One of these vectors has norm very close to zero! "
            "Affected rows: %s" % zerotest.tolist()
        )

    # Normalize each patch by its contrast norm.
    patches[:, :ncols - 1] /= patches[:, ncols - 1:ncols]
    return patches


def knn_density_threshold(data, k, p):
    """Return the ``p`` percent densest points by k-th nearest neighbour distance.

    Port of kNNDensityThreshold.m. ``data`` rows are points (no norm column).
    Referred to as X(k, p) in the paper.
    """
    from scipy.spatial import cKDTree

    # The k nearest neighbours of a point include the point itself, so query
    # for k+1 and use the (k+1)-th distance (index k).
    tree = cKDTree(data)
    distances, _ = tree.query(data, k=k + 1)
    knn_distance = distances[:, k]

    m, n = data.shape
    numdatapoints = m * p / 100.0
    intnumdatapoints = int(np.floor(numdatapoints))
    if numdatapoints != intnumdatapoints:
        import warnings
        warnings.warn(
            "%g percent of %d points is not an integer. "
            "Taking the %d densest points" % (p, m, intnumdatapoints)
        )

    # Densest points = smallest distance to their k-th nearest neighbour.
    order = np.argsort(knn_distance, kind="stable")
    densestpointsindex = order[:intnumdatapoints]
    return data[densestpointsindex, :]


def preprocess_file(filepath, k, p, rng=None):
    """Preprocess a ``.flo`` file into a matrix of flow patches.

    Port of preprocessFile.m.

    Parameters
    ----------
    filepath : str
        Path of the ``.flo`` file to process.
    k : int
        k for the k-th nearest neighbour search.
    p : float
        Percentage of densest points to keep for the core subset.
    rng : numpy.random.Generator, optional
        Random generator used for patch sampling.

    Returns
    -------
    numpy.ndarray
        Matrix of preprocessed patches, one patch per row.
    """
    NUM_PATCHES = 385  # ~400,000 points across the 1041 Sintel frames.
    PATCH_SIZE = 3     # Always 3x3 patches.

    # Step 1: sample from the image.
    patches = sample_from_data_set(filepath, NUM_PATCHES, PATCH_SIZE, rng=rng)
    # Step 2: compute contrast norm.
    patches = compute_contrast_norm(patches)
    # Step 3: keep only high-contrast patches.
    hc_patches = get_high_contrast_patches(patches)
    # Step 4: random downsample (skipped, as in the original).
    # Step 5: normalize.
    hc_patches = normalize_patches(hc_patches)
    # Step 6: drop the contrast-norm column, then filter by density.
    n = hc_patches.shape[1]
    hc_patches = hc_patches[:, :n - 1]
    return knn_density_threshold(hc_patches, k, p)
