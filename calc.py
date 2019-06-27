import os

# Uncomment this to run on the CPU only
os.environ['CUDA_VISIBLE_DEVICES']=''

import tensorflow as tf
import numpy as np
import h5py

def thompson_model(N, dim):
    tf.reset_default_graph()

    # Start with random coordinates from a normal dist
    # r0 = np.random.normal(size=[N,512], dtype=np.float32)
    # coord = tf.Variable(r0, name='coordinates')
    coord = tf.get_variable(name='coordinates', shape=[N, dim],
                            initializer=tf.random_normal_initializer())

    # Normalize the coordinates onto the unit sphere
    coord = coord/tf.reshape(tf.norm(coord,axis=1),(-1,1))

    def squared_diff(A):
        r = tf.reduce_sum(A*A, 1)
        r = tf.reshape(r, [-1, 1])
        return r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)

    RR = squared_diff(coord)

    # We don't need to compute the gradient over the symmetric distance
    # matrix, only the upper right half
    mask = np.triu(np.ones((N, N), dtype=np.bool_), 1)

    R = tf.sqrt(tf.boolean_mask(RR, mask))

    # Electostatic potential up to a constant, 1/r
    U = tf.reduce_sum(1/R)

    return U, coord

def minimize_thompson(N, dim, reported_U=None, limit=10**10):

    U, coord = thompson_model(N, dim)

    # Choose a high energy to start with
    previous_u = N**2

    learning_rate = 0.1
    LR = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.AdamOptimizer(learning_rate=LR).minimize(U)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)

        print (" iteration / N / U_current / delta / learning rate")

        for n in range(limit):
            for _ in range(100):
                sess.run(opt, feed_dict={LR:learning_rate})

            u = sess.run(U)
            delta_u = np.abs(previous_u - u)
            previous_u = u

            msg = "  {} {} {:0.14f} {:0.14f} {:0.10f}"

            if reported_U is None:
                print (msg.format(n, N, u, delta_u, learning_rate))
            else:
                print (msg.format(n, N, u-reported_U, delta_u, learning_rate))

            if np.isclose(delta_u,0,atol=1e-9):
                break

            # Even ADAM gets stuck, slowly decay the learning rate
            learning_rate *= 0.96

        u,c = sess.run([U,coord])
        saver.save(sess, r'saver/model.ckpt')

        return u, c

f_h5 = 'generate.h5'
with h5py.File(f_h5,'w') as h5:
    N = 100  # number of classes or points
    dim = 3 # dimension of the data
    h5.create_group(str(N))
    g = h5[str(N)]
    u, c = minimize_thompson(N, dim, limit=1000)
    g.attrs.create('energy', u)
    g['coordinates'] = c
