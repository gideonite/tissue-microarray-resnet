Must bring in multiple GPUs. Reread the original ResNet README and
they use eight! (batch size 32 on each but the images are bigger. So
the fact that I can only fit 64 is not crazy.)

Write an `accounting` functions just like Yu. Good idea. And yes, it
really is just that simple and you just save it every n iterations.

Figure out the scheduling.

Model Side
==========

Test the model on CIFAR-10 to debug the code.

Look into different SGD algorithms. Investigate what the Torch people
say about the ones that choose LR automagically.

Try a smaller model.

Data Side
=========

Collapse the 4 labels to 2 labels.

Balance the classes? How unbalanced are they currently? (Do this
again)

Data augmentation. Very important in the other implementations. You
shold also do this.
