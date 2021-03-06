# SceneRecognize
[AI Challenger](https://challenger.ai/competition/scene/subject): 对80个场景进行分类

# 模型实现
- 使用TensorFlow 的slim模块实现网络的基础结构。
- 构建深度残差网络。
# 模型特点
模型设计主要参考了VGG16和Inception-ResNet-V2的结果，并有所改进。<br>
- 在制作TFRecord时，将输入数据尺寸统一为299x299x3
- 将激活函数变为Leaky relu 增加模型非线性能力。
- 加入BN层，可选择去掉dropout，增加模型稳定性，减少参数量
- 使用Global Pooling层替换传统的全连接dense层。
- 使用核为1x1的Conv2D替换全连接层。
- 调整网络深度，以适应当前数据集

# 设计原则
- 对于相同的尺寸的输出特征图谱，每层必须含有相同数量的过滤器
- 如果特征图谱的尺寸减半，则过滤器的数量必须翻倍，以保持每层的时间复杂度
- 直接通过卷积层（stride=2）进行下采样，网络末端以全局的均值池化层结束

# 关键代码
``` Python
net = slim.batch_norm(net, is_training=True)
```
``` Python
# Global Average Pooling
    kernel_size = net.get_shape()[1:3]
    if kernel_size.is_fully_defined():
        net = slim.avg_pool2d(net, kernel_size, padding='VALID')
    else:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True)

    # dropout
    if dropout_keep_prob != 1:
        net = slim.dropout(net, dropout_keep_prob, is_training=True)

    # Use conv2d instead of fully_connected layers.
    net = slim.conv2d(net, 80, [1, 1])
```

# 参考文献
- [图像识别领域的深度残差学习](https://www.jianshu.com/p/f71ba99157c7)
