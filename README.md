# DnCNN_Paddle
SimDR: Is 2D Heatmap Even Necessary for Human Pose Estimation?

[官方源码](https://github.com/leeyegy/SimDR)

[复现地址](https://github.com/ultranity/Paddle-SimDR)

## 1. 简介
2D 热图由于其高性能表示多年来一直主导人体姿势估计。然而，基于热图的方法有几个缺点：

- 1) 在现实场景中经常遇到的低分辨率图像中，性能急剧下降。

- 2）为了提高定位精度，可能需要多个上采样层来恢复从低到高的特征图分辨率，这在计算上是昂贵的。

- 3) 通常需要额外的坐标细化来减少缩小的热图的量化误差。

鉴于上面揭示的缺点，SimDR 是一种简单而有效的方案，将 **heatmap-free** 方法提升到 **heatmap-based** 方法的竞争性能水平，在低输入分辨率的情况下大大优于后者,摆脱了额外的后处理，并通过坐标表示设计减少了量化误差。此外，SimDR 允许直接删除某些方法中耗时的上采样模块，这可能会激发对人体姿态估计轻量级模型的新研究

## 2. 复现精度
验收指标：
COCO val HRNet-W48 384x288 keypint AP 76.9

Paddle 复现：
COCO val HRNet-W32 256x192 keypint AP 76.7
COCO val HRNet-W48 384x288 keypint AP 78.5

[模型训练日志及权重](https://pan.baidu.com/s/1N84NEcnRGyjkwerWanz3Gw#isr6)
## 3. 数据集
目前KeyPoint模型支持[COCO](https://cocodataset.org/#keypoints-2017)数据集和[MPII](http://human-pose.mpi-inf.mpg.de/#overview)数据集，数据集的准备方式请参考[关键点数据准备](docs/tutorials/PrepareKeypointDataSet_cn.md)。

[对应 AI Studio 数据集](https://aistudio.baidu.com/aistudio/datasetdetail/142126)
## 4. 环境依赖

PaddleDet = 2.4.0

## 5. 快速开始
复现基于 PaddleDetction 框架，基本使用方式PaddleDetction相同， 参见 [对应文档](configs/keypoint/README.md)

#### 单卡训练

```shell
python -u tools/train.py -c configs/keypoint/simdr/hrnet_w48_384x288.yml --eval --use_vdl=True --vdl_log_dir="./output"
```

#### 多卡训练

```shell
#COCO DataSet
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m paddle.distributed.launch tools/train.py -c configs/keypoint/simdr/hrnet_w48_384x288.yml  --eval --use_vdl=True --vdl_log_dir="./output"
```

### 模型评估

```shell
python -u tools/eval.py -c configs/keypoint/simdr/hrnet_w48_384x288.yml -o weights=./output/hrnet_w48_384x288/best_model.pdparams
```

### 模型部署
参见 PaddleDetection [对应文档](configs/keypoint/README.md)


#### 模型预测

​    注意：top-down模型只支持单人截图预测。

```shell
CUDA_VISIBLE_DEVICES=0 python3 tools/infer.py -c configs/keypoint/simdr/hrnet_w48_384x288.yml -o weights=./output/hrnet_w48_384x288/model_final.pdparams --infer_dir=../images/ --draw_threshold=0.5 --save_txt=True
```

### 推理过程：

模型动转静导出：

```shell
python -u tools/export_model.py -c configs/keypoint/simdr/hrnet_w48_384x288.yml -o weights=./output/hrnet_w48_384x288/model_final.pdparams filename=simdr_hrnet_w48_384x288 --output_dir=./output/
```

模型推理：

```
python ./deploy/python/keypoint_infer.py --device=gpu --model_dir=./output/simdr_hrnet_w48_384x288 --batch_size=2 --image_dir=./dataset/coco/test2017/
```

## 6. TIPC

首先安装AutoLog（规范化日志输出工具）

```shell
pip install  https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
```

运行命令：

```shell
%cd /home/aistudio/PaddleDetection/
#you
#!ln -s ./test_tipc/output/norm_train_gpus_0_autocast_null/ output_inference
# 准备数据
!bash test_tipc/prepare.sh ./test_tipc/configs/keypoint/simdr_hrnet_w48_384x288_train_infer_python.txt 'lite_train_lite_infer'

# 运行测试
!bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/keypoint/simdr_hrnet_w48_384x288_train_infer_python.txt 'lite_train_lite_infer'
```

运行结果见 [test-tipc.log](./test-tipc.log)
## 7. LICENSE

本项目的发布受[Apache 2.0 license](https://github.com/PaddlePaddle/models/blob/release/2.2/community/repo_template/LICENSE)许可认证。