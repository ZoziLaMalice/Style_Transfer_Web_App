import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import PIL


def normalize(img):
  max_dim = 512

  img = tf.io.read_file(img)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float64) # Nous fait passer sur une échelle de 0 à 1
  
  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  
  return img


kand_img = normalize('./style/kandinsky.jpg')


def get_model(name):
  vgg = tf.keras.applications.VGG19(include_top=False)
  vgg.trainable = False

  model = tf.keras.Model([vgg.input], [vgg.get_layer(x).output for x in name])
  
  return model


def gram_matrix(input_tensor):
  
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  
  return result/(num_locations)


def clip(img):
  return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)


class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layer):
    super(StyleContentModel, self).__init__()
    
    self.vgg = get_model(style_layers + content_layer)
    
    self.style_layers = style_layers
    self.content_layers = content_layer
    
    self.num_style_layers = len(style_layers)
    
    self.vgg.trainable = False


  def __call__(self, inputs):
    
    inputs = tf.keras.applications.vgg19.preprocess_input(inputs*255)
    inputs = tf.image.resize(inputs, (224, 224))
    preds = self.vgg(inputs)
    
    style_outputs = preds[:self.num_style_layers]
    content_outputs = preds[self.num_style_layers:]
    
    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
    
    style_dict = {}
    content_dict = {}
    
    for (key, item) in zip(self.style_layers, style_outputs):
      style_dict[key] = item
    
    for (key, item) in zip(self.content_layers, content_outputs):
      content_dict[key] = item
    
    return {'content': content_dict, 'style': style_dict}


def style_content_loss(outputs, style_targets, content_targets, style_layers, content_layer, style_weight=1e-2, content_weight=1e-4):
  style_outputs = outputs['style']
  content_outputs = outputs['content']

  style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                          for name in style_outputs.keys()])
  style_loss *= style_weight / len(style_layers)

  content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                            for name in content_outputs.keys()])
  content_loss *= content_weight / len(content_layer)

  return content_loss + style_loss

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)
