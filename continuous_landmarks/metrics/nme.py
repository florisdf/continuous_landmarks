def normalized_mean_error(pred_points, true_points):
    """
    Compute the Normalized Mean Error of the predicted points.

    The NME is the mean Euclidean distance between the predicted and the ground
    truth points, divided by the Euclidean distance between the outer points of
    the eyes. See also https://ibug.doc.ic.ac.uk/resources/300-W/.
    """

    num_of_images, num_of_points, num_coords = true_points.shape
    assert num_of_points == 68

    inter_ocular_dist = (
        true_points[..., 36, :] - true_points[..., 45, :]
    ).pow(2).sum(dim=-1).sqrt()
    dists_sum = (
        pred_points - true_points
    ).pow(2).sum(dim=-1).sqrt().sum(dim=-1)

    return dists_sum / (num_of_points * inter_ocular_dist)
