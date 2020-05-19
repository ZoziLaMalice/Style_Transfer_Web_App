## Style Transfer Web App

# Project Description

The objective of this project was to become familiar with the Flask framework to develop Web App in Python.
So I decided to apply a Neural Style Transfer model.

A style transfer algorithm will merge two images, but in a special way.

One image is defined as a style image and the other image will be defined as a content image.
Then, the algorithm will smartly select the pixels of the two images in order to extract the style of one and the content of the other.

The user has two possibilities: 
* Custom model, based on VGG19, which is longer because he has to do a lot of iterations in order to find his weights
* TensorFlow Hub model (magenta arbitrary image stylization), which is much faster.

I used Bootsrap Templates for the CSS/js part, Flask for the web framework, TensorFlow 2.0 for deep learning models.
