def my_kmeans(xs: np.ndarray, init_centers: np.ndarray, n_iter: int):
    """ Runs the K-Means algorithm from a given initialization
    
    Args:
        xs: A 2D numpy array of shape (N, D) containing N samples of dimension D
        init_centers: A 2D numpy array of shape (K, D) containing the K initial cluster centers of dimension D.
        n_iter: The number of iterations for the K-Means algorithm.
    
    Returns:
        A (K, D) numpy array containing the final cluster centers.
    """
    
    init_centers = np.array(init_centers)
    if(n_iter==0):
        return init_centers
    else:
        clusters = {}
        for i in range(init_centers.shape[0]):
            clusters[tuple(init_centers[i])] = np.array([[]])
        for i in range(xs.shape[0]):
            min = distance(xs[i], init_centers[0])
            index = 0
            for j in range(init_centers.shape[0]):
                temp_dist = distance(xs[i], init_centers[j])
                if temp_dist < min:
                    min = temp_dist
                    index = j
            existing_arr = clusters[tuple(init_centers[index])]
            updated_arr = [[]]
            if existing_arr.size == 0:
                updated_arr = xs[i].reshape(1, -1)
            else:
                updated_arr = np.vstack((existing_arr, xs[i].reshape(1, -1)))
            
            clusters[tuple(init_centers[index])] = updated_arr

        result = np.array([])
        for key in clusters:
            new_center = np.sum(clusters[key], axis = 0)/clusters[key].shape[0]
            if result.size == 0:
                result = new_center.reshape(1, -1)
            else:
                result = np.vstack((result, new_center.reshape(1, -1)))
        # print('result---> ',result)
        result = my_kmeans(xs, result, n_iter-1)
        return np.array(result)

def distance(arr1:np.ndarray, arr2:np.ndarray):
    sum = 0
    for i in range(arr1.size):
        sum+=(arr1[i]-arr2[i])**2
    return np.sqrt(sum)