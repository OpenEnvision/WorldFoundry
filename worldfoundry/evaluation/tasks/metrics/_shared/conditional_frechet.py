import tensorflow as tf


def symmetric_matrix_square_root(mat, eps=1e-10):
    """Compute square root of a symmetric matrix.
    Note that this is different from an elementwise square root. We want to
    compute M' where M' = sqrt(mat) such that M' * M' = mat.
    Also note that this method **only** works for symmetric matrices.
    Args:
      mat: Matrix to take the square root of.
      eps: Small epsilon such that any element less than eps will not be square
        rooted to guard against numerical instability.
    Returns:
      Matrix square root of mat.
    """
    # Unlike numpy, tensorflow's return order is (s, u, v)
    s, u, v = tf.linalg.svd(mat)
    # sqrt is unstable around 0, just use 0 in such case
    si = tf.compat.v1.where(tf.less(s, eps), s, tf.sqrt(s))
    # Note that the v returned by Tensorflow is v = V
    # (when referencing the equation A = U S V^T)
    # This is unlike Numpy which returns v = V^T
    return tf.matmul(tf.matmul(u, tf.linalg.tensor_diag(si)), v, transpose_b=True)


def trace_sqrt_product(sigma, sigma_v):
    """Find the trace of the positive sqrt of product of covariance matrices.
    '_symmetric_matrix_square_root' only works for symmetric matrices, so we
    cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
    ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).
    Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
    We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
    Note the following properties:
    (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
      => eigenvalues(A A B B) = eigenvalues (A B B A)
    (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
      => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
    (iii) forall M: trace(M) = sum(eigenvalues(M))
      => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                    = sum(sqrt(eigenvalues(A B B A)))
                                    = sum(eigenvalues(sqrt(A B B A)))
                                    = trace(sqrt(A B B A))
                                    = trace(sqrt(A sigma_v A))
    A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
    use the _symmetric_matrix_square_root function to find the roots of these
    matrices.
    Args:
      sigma: a square, symmetric, real, positive semi-definite covariance matrix
      sigma_v: same as sigma
    Returns:
      The trace of the positive square root of sigma*sigma_v
    """

    # Note sqrt_sigma is called "A" in the proof above
    sqrt_sigma = symmetric_matrix_square_root(sigma)

    # This is sqrt(A sigma_v A) above.
    sqrt_a_sigmav_a = tf.matmul(sqrt_sigma, tf.matmul(sigma_v, sqrt_sigma))
    return tf.linalg.trace(symmetric_matrix_square_root(sqrt_a_sigmav_a))


def sample_covariance(a, b, invert=False):
    '''
    Sample covariance estimating
    a = [N,m]
    b = [N,m]
    '''
    assert (a.shape[0] == b.shape[0])
    assert (a.shape[1] == b.shape[1])
    N = a.shape[0]
    C = tf.matmul(tf.transpose(a), b) / N
    return tf.linalg.pinv(C) if invert else C


def no_embedding(x):
    return x


@tf.function
def cfid(y_true, y_predict, x_true, embeding=no_embedding, estimator=sample_covariance):
    '''
    CFID metric implementation, according to the formula described in the paper;
    https://arxiv.org/abs/2103.11521
    The formula:
    Given (x,y)~N(m_xy,C) and (x,y_h)~N(m_xy_h,C_h)
    Assume their joint Gaussian distribution:
    C = [[C_xx,   C_xy]
         [C_yx,   C_yy]]
    C_h = [[C_xx,   C_xy_h]
         [C_y_hx, C_y_hy_h]]
    m_xy = mean(x,y)
    m_xy_h = mean(x,y_h)
    Denote:
    C_y|x   = C_yy - C_yx @ (C_xx^-1) @ C_xy
    C_y_h|x = C_y_hy_h - C_y_hx @ (C_xx^-1) @ C_xy_h
    m_y     = mean(y)
    m_y_h   = mean(y_h)
    CFID((x,y), (x,y_h)) = ||m_y - m_y_h||^2 + Tr((C_yx-C_y_hx) @ (C_xx^-1) @ (C_xy-C_x_y_h)) + \
                                             + Tr(C_y|x + C_y_h|x) -2*Tr((C_y|x @ (C_y_h|x^(1/2)) @ C_y|x)^(1/2))
    The arguments:
    y_true    = [N,k1]
    y_predict = [N,k2]
    x_true    = [N,k3]
    embedding - Functon that transform [N,ki] -> [N,m], 'no_embedding' might be consider to used, if you working with same dimensions activations.
    estimator - Covariance estimator. Default is sample covariance estimator.
                The estimator might be switched to other estimators. Remmember that other estimator must support 'invert' argument
    '''

    y_predict = embeding(y_predict)
    y_true = embeding(y_true)
    x_true = embeding(x_true)

    assert ((y_predict.shape[0] == y_true.shape[0]) and (y_predict.shape[0] == x_true.shape[0]))
    assert ((y_predict.shape[1] == y_true.shape[1]) and (y_predict.shape[1] == x_true.shape[1]))

    # mean estimations
    m_y_true = tf.reduce_mean(y_true, axis=0)
    m_y_predict = tf.reduce_mean(y_predict, axis=0)
    m_x_true = tf.reduce_mean(x_true, axis=0)

    # covariance computations
    с_y_predict_x_true = estimator(y_predict - m_y_predict, x_true - m_x_true)
    с_y_true_x_true = estimator(y_true - m_y_true, x_true - m_x_true)

    с_x_true_y_true = estimator(x_true - m_x_true, y_true - m_y_true)
    c_x_true_y_predict = estimator(x_true - m_x_true, y_predict - m_y_predict)

    с_y_predict_y_predict = estimator(y_predict - m_y_predict, y_predict - m_y_predict)
    с_y_true_y_true = estimator(y_true - m_y_true, y_true - m_y_true)
    inv_с_x_true_x_true = estimator(x_true - m_x_true, x_true - m_x_true, invert=True)

    # conditional covariance estimations
    c_y_true_given_x_true = с_y_true_y_true - tf.matmul(с_y_true_x_true,
                                                        tf.matmul(inv_с_x_true_x_true, с_x_true_y_true))
    c_y_predict_given_x_true = с_y_predict_y_predict - tf.matmul(с_y_predict_x_true,
                                                                 tf.matmul(inv_с_x_true_x_true, c_x_true_y_predict))
    c_y_true_x_true_minus_c_y_predict_x_true = с_y_true_x_true - с_y_predict_x_true
    c_x_true_y_true_minus_c_x_true_y_predict = с_x_true_y_true - c_x_true_y_predict

    # Distance between Gaussians
    m_dist = tf.einsum('...k,...k->...', m_y_true - m_y_predict, m_y_true - m_y_predict)
    c_dist1 = tf.linalg.trace(tf.matmul(tf.matmul(c_y_true_x_true_minus_c_y_predict_x_true, inv_с_x_true_x_true),
                                        c_x_true_y_true_minus_c_x_true_y_predict))
    c_dist2 = tf.linalg.trace(c_y_true_given_x_true + c_y_predict_given_x_true) - 2 * trace_sqrt_product(
        c_y_predict_given_x_true, c_y_true_given_x_true)

    return m_dist + c_dist1 + c_dist2
