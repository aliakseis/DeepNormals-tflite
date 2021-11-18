from model import *


model = GenerateNet()
model.load('Net/DeepNormals/DeepNormals')

with tf.Session() as second_sess:
    # Clone the current model
    model2 = model
    # Delete the training ops
    del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
    # Save the checkpoint
    model2.save('checkpoint_'+".ckpt")
    # Write a text protobuf to have a human-readable form of the model
    tf.train.write_graph(second_sess.graph_def, '.', 'checkpoint_'+".pbtxt", as_text = True)
