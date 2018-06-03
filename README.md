# SceneRecognize
[AI Challenger](https://challenger.ai/competition/scene/subject): 对80个场景进行分类

# 模型实现
模型设计主要参考了VGG16和Inception-ResNet-V2的结果，并有所改进。<br>
- 在制作TFRecord时，将输入数据尺寸统一为299x299x3
- 将激活函数变为Leaky relu 增加模型非线性能力。
- 加入BN层，可选择去掉dropout，增加模型稳定性，减少参数量
- 使用Global Pooling层替换传统的全连接层。
- 调整网络深度，以适应当前数据集
