import torch


def normalized_mean_error(pred_points, true_points):
    """
    Compute the Normalized Mean Error of the predicted points.

    The NME is the mean Euclidean distance between the predicted and the ground
    truth points, divided by the Euclidean distance between the outer points of
    the eyes. See also https://ibug.doc.ic.ac.uk/resources/300-W/.
    """

    if true_points.ndim == 2:
        assert true_points.shape[0] == 68
    elif true_points.ndim == 3:
        assert true_points.shape[1] == 68
    else:
        raise ValueError(f'Unexpected shape {true_points.shape}')

    inter_ocular_dist = torch.linalg.vector_norm(
        true_points[..., 36, :] - true_points[..., 45, :],
        dim=-1
    )
    eucl_dists = torch.linalg.vector_norm(
        pred_points - true_points,
        dim=-1
    )
    mean_eucl_dists = eucl_dists.mean(dim=-1)

    return mean_eucl_dists / inter_ocular_dist
