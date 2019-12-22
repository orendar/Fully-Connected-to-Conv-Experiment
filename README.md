# Fully-Connected-to-Conv-Experiment
Experimenting with converting a pretrained Linear layer to a Conv layer using the VGG architecture and the Oxford-IIIT-Pets dataset.

This repo contains the full code and plots from my blog post "Should You Convert Pretrained Fully-Connected Layers to Convolutional Layers?", which will be linked here shortly.

VGG_experiment_1stage notebook contains the single-stage experiment where both models are unfrozen and all layers are trained right from the start, as detailed in the blog post.

VGG_experiment_2stage contains the two-stage experiment where both models initially have their pretrained layers frozen and only their newly added layers trained using a high learning rate, and then the entire model is unfrozen and trained using a lower learning rate, again as detailed in the blog post.

The notebooks can be run top-to-bottom and dependencies can all be easily installed with pip/conda. Notice that setup.py is needed for the experiment notebooks - if you change anything in the setup.ipynb notebook, then you should download it as a .py file and put it in the same directory as the experiments so that it can be imported.
