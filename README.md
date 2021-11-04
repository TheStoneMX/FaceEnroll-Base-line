# FaceEnroll-Base-line
Unlike Eigenfaces and Fisherfaces, where in most modern face verification systems, training and enrollment are two different steps. Training is performed on millions of images. On the other hand, enrollment is performed using a small set of images.

In case of OpenFace ( and also Dlib as discussed in the next chapter ), enrolling a person is simply passing a few images of the person through the network to obtain 128 dimensional feature descriptors corresponding to each image. In other words, we convert each image to a feature in a high-dimensional space.

In this high dimensional space, features belonging to the same person will be close to each other and far away for different persons.

