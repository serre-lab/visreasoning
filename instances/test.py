import tensorflow as tf
import numpy as np
import ipdb


if __name__ == '__main__':
    session = tf.InteractiveSession()

    array = np.ones((5,5))
    vtensor = tf.Variable(2*array)
    ctensor = tf.constant(3*array)

    vtensorpart = vtensor[0:3,0:3]
    ctensorpart = ctensor[0:3,0:3]

    session.run(tf.initialize_all_variables())
    fetched = session.run([vtensorpart,ctensorpart])

    print(fetched[0])
    print(fetched[1])
    ipdb.set_trace()