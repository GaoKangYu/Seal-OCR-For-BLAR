# SealOCR

#### 介绍
此为印章识别项目数据图片极坐标转换后的识别测试代码，输入为极坐标变换后的LTR图片，存入./east/test/image/，输出为识别得到的文字。
整体数据流如下：
- 运行predict.py，读取./east/test/image/下图片，使用anchor覆盖文本区域，过程图存于./east/test/tmp/，获得检出文本区并使用矩形框标出的图片存于./east/test/predict/，裁剪矩形框区域，获得剔除背景的纯文本区域存于./crnn/test/recs/，读取./crnn/test/recs/下文件，输入CRNN网络，得到最后的预测结果。

#### 代码目录结构

- ./
- --predict.py 端到端预测，输入./east/test/image/，通过east、crnn得到OCR文字输出
- 
- east
- --data 存放处理数据集的部分代码
- --model 存放east权重文件，download.txt中存放的是权重的百度云地址
- --net 存放网络结构、训练代码、字典文件
- --test 存放图片，image下的图片为east输入图片
- --predict_east.py east预测代码
- 
- crnn
- --data 存放处理数据集的部分代码
- --model 存放CRNN权重文件，download.txt中存放的是7.12用于纠正测试权重的百度云地址
- --net 存放网络结构、训练代码、字典文件
- --test 存放文本区剪切后的east模型输出，CRNN输入图片
- --predict_crnn.py crnn预测代码

#### 待更新
- 无