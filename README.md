# source-position-reconstruction

This project is an attempt to develop a simulation, data from which are used for training the Neural Network model and predictions are done in order to identify Astonomical poin sources. The imitaion considerations are focusing on low resolution images obtained from telescopes such as Fermi-LAT, which is gamma ray telescope. The idea of the project is to reduce "reality gap" between the theoretical aspecst(used in simulations) and Galactic Center Excess in gamma-ray domain.

The modeling of simulation considers the gaussian profile for shource, for instrumental noise posisson noise and for diffused emission it considers perlin noise. Perlin noise is used with different octaves, which provides effect of emission form different segments thorough the depth of the view. And outputs resultant image of the region of interest. We also used the analytical relation between the flux from the source and number of source in the image.
