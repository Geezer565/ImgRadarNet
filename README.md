# ImgRadarNet｜图像与雷达融合网络

## 项目资料地图

- [26smartcar](https://github.com/Geezer565/26smartcar)：车辆基础、硬件接入和部署。
- [oncar](https://github.com/Geezer565/oncar)：最终车端运行、任务流程和上位机协同。
- [data_collect](https://github.com/Geezer565/data_collect)：采集、标注、训练和过程记录。
- [ImgRadarNet](https://github.com/Geezer565/ImgRadarNet)：图像与雷达融合网络的精简实现。
- [smartcar-dataset-v22](https://github.com/Geezer565/smartcar-dataset-v22)：最终 V22 数据资料。
- [smartcar-model-v22](https://github.com/Geezer565/smartcar-model-v22)：最终 V22 模型与车端转换结果。

这是智能车项目中“如何把一张图像和一圈雷达信息一起用于预测”的精简实现。它把模型、数据读取、训练和推理保留在很小的范围内，适合单独理解图像—雷达融合的基本思路。

## 这个仓库适合谁

- 想快速了解图像和雷达如何一起输入模型的人。
- 想从简单代码开始复现训练与推理流程的人。
- 想对照完整训练工程、理解核心网络结构的人。

## 文件说明

- `model.py`：融合网络结构。
- `dataset.py`：图像和雷达训练数据读取。
- `train.py`：训练入口。
- `infer.py`：单次推理示例。
- `test.py`：测试与验证。
- `utils.py`：通用辅助方法。

## 与完整工程的关系

这个仓库是模型思路的精简入口，不承担实车部署和完整实验记录：

- 完整采集、标注、训练、验证和转换流程在 [`data_collect`](https://github.com/Geezer565/data_collect)。
- 最终 V22 数据资料在 [`smartcar-dataset-v22`](https://github.com/Geezer565/smartcar-dataset-v22)。
- 最终 V22 模型和车端转换结果在 [`smartcar-model-v22`](https://github.com/Geezer565/smartcar-model-v22)。
- 实体车端运行请看 [`oncar`](https://github.com/Geezer565/oncar) 与 [`26smartcar`](https://github.com/Geezer565/26smartcar)。

## 使用提示

示例代码需要与实际数据格式保持一致。若要在自己的项目中使用，请先确认图像尺寸、雷达长度、标签含义和模型输出是否匹配。

## 归档说明

这是一份最终精简版，只保留最核心的代码，便于阅读和后续引用。大型数据、训练结果和车辆环境资料已拆分到对应仓库。
