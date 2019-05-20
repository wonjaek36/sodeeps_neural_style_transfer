import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import json
import logging
# import imageio
from skimage import io

# %matplotlib inline

class Style_Transfer():

    def __init__(self, config):
        self.config = config
        logging.basicConfig(stream = sys.stdout, level=logging.INFO)

    def load_pretrained_model(self):

        resources_path = self.config['DEFAULT']['RESOURCE_PATH']

        print (resources_path)
        print (os.path.join(resources_path, 'imagenet-vgg-verydeep-19.mat'))
        model = load_vgg_model(os.path.join(resources_path, 'imagenet-vgg-verydeep-19.mat'))
        print (model)
    
        return model

    def test_content_image(self):
        resources_path = self.config['DEFAULT']['RESOURCE_PATH']
        image_path = os.path.join(resources_path, 'images', 'louvre.jpg')
        content_image = imageio.imread(image_path)
        print (content_image.shape)
        imshow(content_image)
        plt.show()

    def test_style_image(self):
        resources_path = self.config['DEFAULT']['RESOURCE_PATH']
        image_path = os.path.join(resources_path, 'images', 'monet_800600.jpg')
        style_image = io.imread(image_path)
        imshow(style_image)
        plt.show()

    def run(self):
        resources_path = self.config['DEFAULT']['RESOURCE_PATH']
        images_path = self.config['IMAGES']['PATH']
        content_image_filename = self.config['IMAGES']['CONTENT']
        style_image_filename = self.config['IMAGES']['STYLE']
        output_path = self.config['OUTPUT']['PATH']
        generated_filename = self.config['OUTPUT']['OUTPUT']
        progress_filename = self.config['OUTPUT']['PROGRESS']
        style_layers_name = self.config['MODEL']['STYLE_LAYERS_NAME']
        style_layers_weight = self.config['MODEL']['STYLE_LAYERS_WEIGHT']
        num_iterations = self.config['MODEL']['ITERATION']
        STYLE_LAYERS = []

        for layer_name, layer_weight in zip(style_layers_name, style_layers_weight):
            STYLE_LAYERS.append( (layer_name, layer_weight ))

        # self.test_content_image()
        # self.test_style_image()

        tf.reset_default_graph() # Reset the graph
        sess = tf.InteractiveSession() # Start interactive session
        self.sess = sess # Assign sess as instance variable

        content_image = io.imread(os.path.join(images_path, content_image_filename))
        content_image = reshape_and_normalize_image(content_image)
        style_image = io.imread(os.path.join(images_path, style_image_filename))
        style_image = reshape_and_normalize_image(style_image)
        generated_image = generate_noise_image(content_image)
        # imshow(generated_image[0])

        model = self.load_pretrained_model()

        # Assign the content image to be the input of the VGG model
        sess.run(model['input'].assign(style_image))

        # Select the output tensor of layer conv4_2
        out = model['conv4_2']

        # Set a_C to be the hidden layer activation from the layer we have selected
        a_C = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute the content cost
        J_content = self.compute_content_cost(a_C, a_G)


        # Assign the input of the model to be the "style" image 
        sess.run(model['input'].assign(style_image))

        # Compute the style cost
        J_style = self.compute_style_cost(model, STYLE_LAYERS)
        J = self.total_cost(J_content, J_style)

        optimizer = tf.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(J)

        # Initialize global variables (you need to run the session on the initializer)
        sess.run(tf.global_variables_initializer())
        # Run the noisy input image (initial generated image) through the model. Use assign().
        sess.run(model['input'].assign(generated_image))

        for i in range(num_iterations):
    
            # Run the session on the train_step to minimize the total cost
            ### START CODE HERE ### (1 line)
            sess.run(train_step)
            ### END CODE HERE ###
            
            # Compute the generated image by running the session on the current model['input']
            ### START CODE HERE ### (1 line)
            generated_image = sess.run(model['input'])
            ### END CODE HERE ###

            # Print every 20 iteration.
            if i%20 == 0:
                Jt, Jc, Js = sess.run([J, J_content, J_style])
                print("Iteration " + str(i) + " :")
                print("total cost = " + str(Jt))
                print("content cost = " + str(Jc))
                print("style cost = " + str(Js))
                
                # save current generated image in the "/output" directory
                
                filename = str(i) + progress_filename

                save_image(os.path.join(output_path, filename), generated_image)
        
        # save last generated image
        save_image(os.path.join(output_path, generated_filename), generated_image)
        
        return generated_image


    def compute_content_cost(self, a_C, a_G):
        """
        Computes the content cost
        
        Arguments:
        a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
        
        Returns: 
        J_content -- scalar that you compute using equation 1 above.
        """
        
        ### START CODE HERE ###
        # Retrieve dimensions from a_G (≈1 line)
        m, n_H, n_W, n_C = a_G.shape.as_list()
        
        # Reshape a_C and a_G (≈2 lines)
        a_C_unrolled = tf.reshape(a_C, [m, n_H*n_W, n_C])
        a_G_unrolled = tf.reshape(a_G, [m, n_H*n_W, n_C])
        
        # compute the cost with tensorflow (≈1 line)
        diff_c_g = a_C_unrolled - a_G_unrolled
        diff_sum = tf.reduce_sum(tf.multiply(diff_c_g, diff_c_g))
        J_content = 1.0 * diff_sum / (4 * n_H * n_W * n_C)
        ### END CODE HERE ###
        
        return J_content


    def gram_matrix(self, A):
        """
        Argument:
        A -- matrix of shape (n_C, n_H*n_W)
        
        Returns:
        GA -- Gram matrix of A, of shape (n_C, n_C)
        """
        
        ### START CODE HERE ### (≈1 line)
        GA = tf.tensordot(A, tf.transpose(A), axes = 1)
        ### END CODE HERE ###
        
        return GA


    def compute_layer_style_cost(self, a_S, a_G):
        """
        Arguments:
        a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
        
        Returns: 
        J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
        """
    
        # Retrieve dimensions from a_G (≈1 line)
        m, n_H, n_W, n_C = a_G.shape.as_list()
        
        # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
        a_S = tf.reshape(a_S, [n_H*n_W, n_C])
        a_S = tf.transpose(a_S)
        a_G = tf.reshape(a_G, [n_H*n_W, n_C])
        a_G = tf.transpose(a_G)

        # Computing gram_matrices for both images S and G (≈2 lines)
        GS = self.gram_matrix(a_S)
        GG = self.gram_matrix(a_G)

        # Computing the loss (≈1 line)
        diff_sum = tf.reduce_sum((GS-GG) ** 2)
        J_style_layer = 1.0 * diff_sum / (4 * n_C ** 2 * (n_H*n_W)**2)
        
        return J_style_layer


    def compute_style_cost(self, model, STYLE_LAYERS):
        """
        Computes the overall style cost from several chosen layers
        
        Arguments:
        model -- our tensorflow model
        STYLE_LAYERS -- A python list containing:
                            - the names of the layers we would like to extract style from
                            - a coefficient for each of them
        
        Returns: 
        J_style -- tensor representing a scalar value, style cost defined above by equation (2)
        """
        
        # initialize the overall style cost
        J_style = 0
        sess = self.sess
        for layer_name, coeff in STYLE_LAYERS:

            # Select the output tensor of the currently selected layer
            out = model[layer_name]

            # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
            a_S = sess.run(out)

            # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
            # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
            # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
            a_G = out
            
            # Compute style_cost for the current layer
            J_style_layer = self.compute_layer_style_cost(a_S, a_G)

            # Add coeff * J_style_layer of this layer to overall style cost
            J_style += coeff * J_style_layer

        return J_style


    def total_cost(self, J_content, J_style, alpha = 10, beta = 40):
        """
        Computes the total cost function
        
        Arguments:
        J_content -- content cost coded above
        J_style -- style cost coded above
        alpha -- hyperparameter weighting the importance of the content cost
        beta -- hyperparameter weighting the importance of the style cost
        
        Returns:
        J -- total cost as defined by the formula above.
        """
        
        J = alpha * J_content + beta * J_style
        return J


if __name__ == "__main__":

    config = None
    with open(os.path.join('src', 'config.json'), 'r') as f:
        config = json.load(f)

    if config is not None:
        st = Style_Transfer(config)
        st.run()
        # np.random.seed(1)
