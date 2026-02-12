import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import diags, csr_matrix, coo_matrix, eye, bmat
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, svd
from opt_einsum import contract

from diffusion_geometry.operators import LinearOperator
from diffusion_geometry.tensors import Function
import plotly.express as px
from tqdm import tqdm
import os

import imageio

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def solve_differential_operator(
    operator: LinearOperator, initial_condition: Function, t_values
):

    # Diagonalise the differential operator in order to exponentiate it
    vals, vecs = operator.spectrum()
    vecs = vecs.coeffs

    # Compute the function values at each time step (T, n)
    initial_condition_eigenbasis = np.linalg.solve(
        vecs, initial_condition.coeffs
    )  # (n,)
    ft_eigenbasis = np.exp(t_values[:, None] * vals) * initial_condition_eigenbasis
    ft = contract("ij,tj->ti", vecs, ft_eigenbasis)

    return initial_condition.space.wrap(ft)


def gif_from_functions(
    ft,
    data,
    fps=30,
    range_color=None,
    color_continuous_scale="Viridis",
    filename="evolution",
):
    ft_pointwise = ft.to_pointwise_basis().real

    if range_color is None:
        range_color = [np.min(ft_pointwise.T), np.max(ft_pointwise.T)]

    frames = []

    print("Creating GIF with {ft.shape[0]} frames")
    for f in tqdm(ft_pointwise):
        fig = px.scatter(
            x=data[:, 0],
            y=data[:, 1],
            color=f,
            color_continuous_scale=color_continuous_scale,
            range_color=range_color,
        )
        fig.update_layout(
            coloraxis_showscale=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                showline=False,
                zeroline=False,
                title="",
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                showline=False,
                zeroline=False,
                title="",
            ),
            title="",
            margin=dict(l=0, r=0, t=0, b=0),
        )

        # Save the current frame as an image
        fig.write_image("temp.png")
        frames.append(imageio.imread("temp.png"))

    imageio.mimsave(filename + ".gif", frames, fps=fps, loop=0)
