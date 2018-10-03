# ncnn-face

## 基于MobileNet-SSD的人脸检测器ncnn版

![](https://i.imgur.com/QDGkI9r.jpg)


## 使用方法

下载转换好的模型，地址[releases](https://github.com/imistyrain/ncnn_face/releases)

编译并安装ncnn库，步骤参见[how to build ncnn library](https://github.com/Tencent/ncnn/wiki/how-to-build)

### Windows

双击打开Windows文件夹下的ncnn_face.sln编译即可

### Linux & arm

	mkdir build 
	cd build
	cmake ..
	make -j8

训练参见[https://github.com/chuanqi305/MobileNet-SSD](MobileNet-SSD)

模型转换参见[convert to ncnn model](https://github.com/Tencent/ncnn/wiki/how-to-use-ncnn-with-alexnet)

参考:

* [ncnn-mtcnn](https://github.com/Longqi-S/ncnn-mtcnn/)

* [MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)