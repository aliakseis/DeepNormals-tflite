import tensorflow as tf


# Convert the model.
converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model("./models/1/") 
tflite_model = converter.convert()

  # Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

