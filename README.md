# Seal-OCR-For-BLAR
“国土资源国土资源智能文档解析系统”项目

## 背景/需求
- OCR输入PDF文档正文内容。
- OCR所盖公章上面的文字，与公章下面的落款单位对比，如果一致则通过，不一致则提示用章错误，转为人工处理。
## 环境需求
-   Windows or Linux
-   Python >= 3.5
-   Pillow >= 6.1.0
-   torch >= 1.2.0
-   torchvision >= 0.4.1
-   opencv-python >= 4.2.0
-   scikit-image >= 0.17.2
-   scipy >= 1.3.1
-   cuda >= 10.0
-   Full conda list can be found in environment.txt and craft/requirements.txt

## 代码目录结构
```
├── CRAFT(弯曲文本检测模型)
│   ├── basenet
│   ├── data(印章原始数据)
│   ├── fail_log(用于保存后处理过程中出现的异常)
│   ├── polar-img(仿射矫正后的正常语序印章文本)
│   ├── result(CRAFT模型原始检测结果)
│   ├── imgproc.py(包含自定义后处理，实现根据result文件夹中的原始输出，经处理生成polar-img中的仿射矫正后的正常语序印章文本)
├── TPS(模糊文本识别模型)
├── creat_dataset(用于生成合成数据集，训练TPS模型)
```
## 项目技术方案
- 预期输入

![raw_input](https://github.com/GaoKangYu/Seal-OCR-For-BLAR/blob/main/readme_fig/raw_input.png)
- 识别区块划分

![task_decomposition](https://github.com/GaoKangYu/Seal-OCR-For-BLAR/blob/main/readme_fig/task_decomposition.png)
- 整体工作流程如下：

1、整体通过颜色滤波，区分开红色与黑色，红色、黑色分别输出保存，可以获得水平文本识别区A和弯曲文本识别区共存的图片P1、水平文本识别区B的图片P2。
![color_separation](https://github.com/GaoKangYu/Seal-OCR-For-BLAR/blob/main/readme_fig/color_separation.png)

2、以P1为输入，目标检测确定并抠取弯曲文本识别区（印章区域），以白色填充该区域，分别得到水平文本识别区A的图片P3和弯曲文本识别区域的图片P4。
3、以水平文本识别方案（CTPN-DenseNet-CTC，CTPN检测、DenseNet+CTC实现不定长识别），对水平文本识别区A和B对应图片P2、P3进行OCR，获取其文本并保存于txt文档。
![horizontal_text_recognition](https://github.com/GaoKangYu/Seal-OCR-For-BLAR/blob/main/readme_fig/horizontal_text_recognition.png)

4、以P4为输入，采用弯曲文本检测方案（CRAFT+后处理）完成输入印章、输出语序正确的水平文本图片过程，具体过程为：首先通过CRAFT模型获取单字的位置信息，从而确定印章区域的质心和半径，采用笛卡尔坐标转极坐标的方式将弯曲文字转为正常语序的水平文字图片P5。
![curved_text_processing_flow](https://github.com/GaoKangYu/Seal-OCR-For-BLAR/blob/main/readme_fig/curved_text_processing_flow.png)

5、以P5为输入，经过模糊文字识别方案（TPS），输出保存于txt中。至此，完成整个PDF的OCR工作。

- 附（用于训练TPS模型合成数据集的效果对比）
![synthetic_dataset](https://github.com/GaoKangYu/Seal-OCR-For-BLAR/blob/main/readme_fig/synthetic_dataset.png)
