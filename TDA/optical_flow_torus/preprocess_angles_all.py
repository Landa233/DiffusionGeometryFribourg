"""Python port of preprocessFolderWithAnglesAll.m.

Walks a folder of subfolders of ``.flo`` optical-flow files, samples 3x3 flow
patches from every file, computes the contrast norm of each patch and the
dominant-flow angle (via PCA), and returns one matrix with every patch as a
row. Each row is ``[18 flow values, contrast norm, angle]`` (length 20).

The patch sampling / contrast-norm helpers are reused from
``preprocess_file.py`` so the two ports stay in sync.
"""

import os

import numpy as np

from preprocess_file import compute_contrast_norm, sample_from_data_set


def get_angles_with_pca(patch):
    """Return the 2D dominant flow direction of a 3x3 flow patch.

    Port of getAnglesWithPCA.m. ``patch`` is an 18-element vector: the first 9
    entries are the horizontal (u) components, the next 9 the vertical (v)
    components. The result is the first principal component of the 9x2 matrix
    whose rows are ``(u_i, v_i)``.
    """
    X = np.column_stack([patch[:9], patch[9:18]])
    # MATLAB's pca centers the columns and uses the SVD; the first principal
    # direction is the leading right-singular vector of the centered data.
    Xc = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return vt[0]


def compute_angles(patches):
    """Compute the flow angle (mod pi) of each patch.

    Port of computeAngles.m. Uses the first 18 columns of each row and returns
    a column vector of angles in ``[0, pi)``.
    """
    mrows, ncols = patches.shape
    if ncols < 18:
        raise ValueError("Data matrix has fewer than the expected 18 columns")

    xy = np.zeros((mrows, 2))
    for i in range(mrows):
        xy[i, :] = get_angles_with_pca(patches[i, :18])

    theta = np.arctan2(xy[:, 1], xy[:, 0])
    theta = np.mod(theta, np.pi)
    return theta.reshape(-1, 1)


def preprocess_folder_with_angles_all(folderpath, k=None, p=None, rng=None):
    """Preprocess every ``.flo`` file under ``folderpath``'s subfolders.

    Port of preprocessFolderWithAnglesAll.m.

    Parameters
    ----------
    folderpath : str
        Path to a folder whose subfolders contain ``.flo`` files.
    k, p :
        Unused; kept to match the MATLAB signature (the original k-NN density
        thresholding is commented out in the source).
    rng : numpy.random.Generator, optional
        Random generator used for patch sampling.

    Returns
    -------
    numpy.ndarray
        Matrix of preprocessed patches, one patch per row. Each row holds the
        18 flow values, the contrast norm, and the flow angle.
    """
    NUM_PATCHES = 385
    PATCH_SIZE = 3  # We always use 3x3 image patches.

    # MATLAB's dir() lists subfolders (including the folder itself via '.');
    # iterate the immediate subdirectories of folderpath.
    subfolders = sorted(
        entry.path for entry in os.scandir(folderpath) if entry.is_dir()
    )

    allpatches_list = []
    for tempdir in subfolders:
        # Get all .flo files in this subdirectory and process them.
        filelist = sorted(
            f for f in os.listdir(tempdir) if f.endswith(".flo")
        )
        numfiles = len(filelist)
        folderpatches = np.zeros((NUM_PATCHES * numfiles, 2 * PATCH_SIZE ** 2 + 1))
        for i, fname in enumerate(filelist):
            filepath = os.path.join(tempdir, fname)
            patches = sample_from_data_set(filepath, NUM_PATCHES, PATCH_SIZE, rng=rng)
            patches = compute_contrast_norm(patches)
            folderpatches[i * NUM_PATCHES:(i + 1) * NUM_PATCHES, :] = patches
        if numfiles > 0:
            allpatches_list.append(folderpatches)

    if not allpatches_list:
        return np.empty((0, 2 * PATCH_SIZE ** 2 + 2))

    allpatches = np.vstack(allpatches_list)

    # Compute angles over the entire collection and append as the last column.
    angles = compute_angles(allpatches)
    processedpatches = np.hstack([allpatches, angles])
    return processedpatches


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python preprocess_angles_all.py <folderpath>")
        sys.exit(1)

    result = preprocess_folder_with_angles_all(sys.argv[1])
    print()
    print("Processed patches shape:", result.shape)
