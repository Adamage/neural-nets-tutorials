import tensorflow as tf

tf.reset_default_graph()
input_depth = 1

with tf.Session() as session:
    embeddings = tf.Variable(tf.truncated_normal([20, input_depth], dtype=tf.float32))
    vec = tf.Variable(tf.truncated_normal([1, input_depth], dtype=tf.float32))
    session.run(tf.global_variables_initializer())

    diff = tf.abs(embeddings - vec)
    mean = tf.reduce_mean(diff, 1)
    ind = tf.argmin(mean)

    session.run(tf.gather(embeddings, ind))