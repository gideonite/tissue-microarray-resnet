Must bring in multiple GPUs. Reread the original ResNet README and
they use eight! (batch size 32 on each but the images are bigger. So
the fact that I can only fit 64 is not crazy.)

Write an `accounting` functions just like Yu. Good idea. And yes, it
really is just that simple and you just save it every n iterations.

Figure out the scheduling.
