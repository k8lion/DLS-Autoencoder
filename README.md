# DLS-Autoencoder

Kaitlin Maile, Geetali Tyagi

Project constraints: implement autoencoder for 256x256 RGB images that minimizes L1Loss and has a latent vector no bigger than 8192 bytes.

This model utilizes 3 convolutional layers with 2-pixel stride to accomplish downsampling while increasing channels. 
The output of the final convolutional layer in the encode stage outputs a 16x16x16 vector, which is converted to the 
half datatype to fit with the 8192 byte requirement. The decode network mirrows the encode network by first converting 
back to the float datatype and then using transposed convultional layers to upsample the image dimensions while 
decreasing channels. The network was trained on the MSCOCO dataset.
