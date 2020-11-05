## 本周工作总结：

#### 1) latex方面：

在vscode上配置好了latex以及相关extension，实践了git blame等相关命令。暂时未找到合适的会议为其制作latex模板。

#### 2) 5th method代码阅读

大体理解了第五名的整体思路以及其使用的数据集的具体细节：其采用的数据集均衡处理为不完全采样：确保每一个epoch中采到的真视频和假视频都没有重复，其次，以真视频的总数量对假视频进行随机采样。
其使用的数据集为对视频使用mtcnn提脸之后再次转为mp4视频。
其代码结构如下：

![code_structure](https://github.com/Andrew-Pei/deepfakes_detection/blob/master/pictures/code_structure.png)

接下来主要仿照dataset_videos.py和1_train_model_i3d.py对自己的代码进行修改。

