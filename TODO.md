Model Side
==========

Try a smaller model on a single GPU.

Data Side
=========

Data augmentation. Very important in the other implementations. You
shold also do this.

Larger patch size (128). Stride = 1.

Make a plot of the label distribution for each image.

Do a proper training-test split, i.e. by sample instead of by
patch. This way you ensure that you don't have adjacent patches in the
training and test sets.
