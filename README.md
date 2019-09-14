# WMCNN-pytorch-Netmodel
accoding to the artical  [Aerial Image Super Resolution via Wavelet Multiscale Convolutional Neural Networks] If you use this code, please cite the paper.I  stripped out the network modle of the article using the pytorch
The code is based on the article [Aerial Image Super Resolution via Wavelet Multiscale Convolutional Neural Networks], wmcnn, because the code that comes with the article uses python and matlab mixed programming, and a pytorch version, because the code uses a lot of data sets to read and pre- The process of processing, and the environment is set up abnormally, the code cannot be run, so the relevant network structure is stripped and can be used directly. Without the training process, you can use it to quickly embed into your network structure without worrying about the input size.
attention
 
It should be noted that in the network, the paper is slightly different. At the end of the network structure, the individual feels that it drops directly from the 160feature map to 12. It may return to the loss of denoising performance, so a buffer convolution is added. Slowly down the feature map to 12



代码基于文章的[Aerial Image Super Resolution via Wavelet Multiscale Convolutional Neural Networks]，wmcnn,由于文章附带的代码使用了python和matlab混合编程，以及一个pytorch版本，由于代码中使用了大量的数据集读取和预处理的过程，而且环境搭建异常，无法运行代码，故剥离了相关的网络结构，可直接使用。缺失了训练过程，你可以使用它很快的嵌入到你网络结构中，而不需要担心输入尺寸的 。


 需要注意的是网络中和论文稍有不同的是，在网络的结构的末，个人感觉从160feature map 直接降到12，可能回到是会导致去噪性能损失，所以加了一个缓冲卷积，慢慢降feature map到12 
