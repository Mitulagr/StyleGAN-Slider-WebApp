import streamlit as st
import numpy as np
import PIL.Image
from model import getGAN, clearCache
import torch
import time

Path = ""

GAN = getGAN()

# @st.cache_data(allow_output_mutation=True)
def generate_image(style):
    #img = GAN([style])
    #img = img[0].to('cpu').squeeze().permute(1,2,0) * 0.5 + 0.5
    #img = img.clip(0,1)
    return (torch.rand(1024,1024,3).numpy() * 255).astype(np.uint8) 
    #return (img.detach().numpy() * 255).astype(np.uint8)

# Create 32 sliders to control the latent vector


# Generate and display the image
# image = generate_image(latent_vector)
# image = PIL.Image.fromarray(image, 'RGB')
# st.image(image, caption='Generated Image')

# if st.button('Generate new image'):
clearCache()
now = time.time()
latent_vector = torch.zeros(1, 512)
for i in range(32):
    value = st.slider(f'Value {i}', -1.0, 1.0, 0.0, 0.001)
    latent_vector[0][i] = value
image = generate_image(latent_vector)
image = PIL.Image.fromarray(image, 'RGB')
st.image(image, caption='Generated Image')
print(time.time()-now)

