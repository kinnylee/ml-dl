import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

sess = tf.Session()
binding = {a: 1.3, b: 2.5}
add = tf.add(a, b)
c = sess.run(add, feed_dict=binding)
print(c)