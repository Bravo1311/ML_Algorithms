def my_linear_regression(phi: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """ Computes the weights of a linear regression that fits the given data.
    
    Args:
        phi: Input feature matrix of shape (N, D) containing N samples of dimension D.
        ys: Target outputs of shape (N, 1) containing N 1-dimensional samples.
        
    Returns:
        A numpy array containing the regressed weights of shape (D, 1), containing one weight for each input dimension.
    """
    phi_aug = np.column_stack([np.ones((phi.shape[0], 1)), phi])
    phi_transpose = np.transpose(phi_aug)
    phi_dp = np.dot(phi_transpose, phi_aug)
    ys_dp = np.dot(phi_transpose, ys)
    # weights = np.dot(np.linalg.inv(phi_dp), ys_dp)
    weights = np.linalg.solve(phi_dp, ys_dp)
    return weights[1:]