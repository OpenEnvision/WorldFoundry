import tensorflow as tf

from worldfoundry.evaluation.tasks.metrics._shared.conditional_frechet import (
    cfid,
    no_embedding,
    sample_covariance,
    symmetric_matrix_square_root,
    trace_sqrt_product,
)


@tf.function
def ssd(y_true, y_predict, x_true, estimator=sample_covariance, calculate_TrSV=False):
    '''
    y_true, y_predict, x_true should be normalized before calculating SSD
    for SSD:
        y_true: real image r
        y_predict： fake image f
        x_true： text es
        ssd = 1 - cos(m_f,m_s)  + ||d(C_ff|s) - d(C_rr|s)||^2
    for SSD-T:
        y_true: real caption s
        y_predict： fake caption sf
        x_true： real image r
        ssd = 1 - cos(m_fs,m_r)  + ||d(C_fsfs|s) - d(C_rr|s)||^2
    '''

    assert ((y_predict.shape[0] == y_true.shape[0]) and (y_predict.shape[0] == x_true.shape[0]))
    assert ((y_predict.shape[1] == y_true.shape[1]) and (y_predict.shape[1] == x_true.shape[1]))

    # mean estimations
    m_y_true = tf.reduce_mean(y_true, axis=0)
    m_y_predict = tf.reduce_mean(y_predict, axis=0)
    m_x_true = tf.reduce_mean(x_true, axis=0)

    # to connect SC better and we have proved that cos(m_f,m_s) \propto E(e_f, e_s)
    # we use E[(e_f, es)] for our calculation
    SS = 1 - tf.reduce_mean(tf.reduce_sum(tf.math.multiply(y_predict, x_true), axis=1))

    # covariance computations
    c_y_predict_x_true = estimator(y_predict - m_y_predict, x_true - m_x_true)
    c_y_true_x_true = estimator(y_true - m_y_true, x_true - m_x_true)
    c_x_true_y_true = estimator(x_true - m_x_true, y_true - m_y_true)
    c_x_true_y_predict = estimator(x_true - m_x_true, y_predict - m_y_predict)
    c_y_predict_y_predict = estimator(y_predict - m_y_predict, y_predict - m_y_predict)
    c_y_true_y_true = estimator(y_true - m_y_true, y_true - m_y_true)
    inv_c_x_true_x_true = estimator(x_true - m_x_true, x_true - m_x_true, invert=True)

    # conditional covariance estimations
    c_y_true_given_x_true = c_y_true_y_true - tf.matmul(c_y_true_x_true,
                                                        tf.matmul(inv_c_x_true_x_true, c_x_true_y_true))
    c_y_predict_given_x_true = c_y_predict_y_predict - tf.matmul(c_y_predict_x_true,
                                                                 tf.matmul(inv_c_x_true_x_true, c_x_true_y_predict))

    dSV = tf.math.sqrt(tf.math.square(tf.linalg.diag_part(c_y_predict_given_x_true - c_y_true_given_x_true)))
    dSV = tf.reduce_sum(dSV)

    if calculate_TrSV:
        mask = tf.cast(tf.linalg.diag(tf.ones([512])), dtype=tf.float64)
        с_y_predict_x_true = estimator(y_predict - m_y_predict, x_true - m_x_true)

        с_y_true_x_true = estimator(y_true - m_y_true, x_true - m_x_true)
        с_x_true_y_true = estimator(x_true - m_x_true, y_true - m_y_true)
        c_y_true_x_true_minus_c_y_predict_x_true = с_y_true_x_true - с_y_predict_x_true
        c_x_true_y_true_minus_c_x_true_y_predict = с_x_true_y_true - c_x_true_y_predict
        inv_с_x_true_x_true = estimator(x_true - m_x_true, x_true - m_x_true, invert=True)

        m_dist = tf.einsum('...k,...k->...', m_y_true - m_y_predict, m_y_true - m_y_predict)
        c_dist1 = tf.linalg.trace(tf.matmul(tf.matmul(c_y_true_x_true_minus_c_y_predict_x_true, inv_с_x_true_x_true),
                                            c_x_true_y_true_minus_c_x_true_y_predict))
        TrSV = tf.linalg.trace(c_y_true_given_x_true * mask + c_y_predict_given_x_true * mask) - 2 * trace_sqrt_product(
            c_y_predict_given_x_true * mask, c_y_true_given_x_true * mask)
        return SS + dSV, SS, dSV, (m_dist, c_dist1, TrSV)
    return SS + dSV, SS, dSV, None


__all__ = [
    "cfid",
    "no_embedding",
    "sample_covariance",
    "ssd",
    "symmetric_matrix_square_root",
    "trace_sqrt_product",
]

