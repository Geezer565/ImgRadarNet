# ImgRadarNet｜图像与雷达融合网络的精简实现

> 一个独立、精简的图像—雷达融合网络示例。它保留模型、数据读取、训练和推理的核心代码，适合快速理解项目中的融合思路。

## 项目资料地图

| 资料 | 作用 |
| --- | --- |
| [26smartcar](https://github.com/Geezer565/26smartcar) | 车辆基础、硬件接入与部署 |
| [oncar](https://github.com/Geezer565/oncar) | 最终车端运行、任务流程和上位机协同 |
| [data_collect](https://github.com/Geezer565/data_collect) | 采集、标注、训练和过程记录 |
| [ImgRadarNet](https://github.com/Geezer565/ImgRadarNet) | 图像与雷达融合网络的精简实现（本仓库） |
| [smartcar-dataset-v22](https://github.com/Geezer565/smartcar-dataset-v22) | 最终 V22 图像、标注与雷达数据 |
| [smartcar-model-v22](https://github.com/Geezer565/smartcar-model-v22) | 最终 V22 模型与车端转换结果 |

## 1. 适合用来做什么

- 快速理解图像和雷达如何一起进入一个预测模型。
- 作为阅读完整训练工程前的最小入口。
- 用自己的同格式数据验证训练、推理和结果检查流程。

它不承担车辆部署、完整数据处理或全部实验记录；这些内容请看 `data_collect`、`oncar` 和 `26smartcar`。

## 2. 文件说明

| 文件 | 作用 |
| --- | --- |
| `model.py` | 图像与雷达融合网络结构 |
| `dataset.py` | 图像和雷达训练数据读取 |
| `train.py` | 训练入口 |
| `infer.py` | 推理示例 |
| `test.py` | 测试与验证 |
| `utils.py` | 通用辅助方法 |

## 3. 使用前需要准备

要运行这份示例，需要先准备：

1. 与脚本读取方式匹配的图像、雷达和标签资料。
2. 能运行深度学习训练的 Python 环境。
3. 与数据格式一致的图像尺寸、雷达长度和标签定义。

如果目标是复现项目最后阶段的完整训练结果，应转到 `data_collect`，并使用独立保存的 V22 数据和模型资料。

## 4. 与完整工程的关系

```text
ImgRadarNet：理解核心网络
       ↓
data_collect：完整采集、训练与验证方法
       ↓
smartcar-dataset-v22 / smartcar-model-v22：最终数据与模型
       ↓
oncar / 26smartcar：实体车端运行
```

## 5. 公开与使用说明

本仓库目前不含原始数据、训练结果、现场图片或服务密钥，是最适合优先公开的一份资料。

### 使用许可

本仓库的原创代码采用 [MIT 许可](LICENSE)。在保留许可说明的前提下，任何人都可以学习、修改和再使用代码。

原始数据、训练模型和第三方内容不在此许可范围内，仍应遵循其各自的使用说明。

## 6. 归档说明

这是最终项目中的“最小可读版本”。它用于展示融合网络的结构和思路，不代表可以直接控制真实车辆。
