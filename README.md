# keras-yolo3-voc
A Keras implementation of YOLOv3 (Tensorflow backend)  for  datas in Pascal VOC format

一、基础环境搭建（ubuntu 18.04 安装GPU +CUDA+cuDNN）：
1、安装Ubuntu 18.04
为了满足后续的模型训练需要，在安装操作系统时，将交换区设置得大一些，例如8G或16G，以防模型训练时读取大量样本导致内存溢出

2、安装显卡驱动

安装Ubuntu后默认安装的是开源版本的显卡驱动，为了后续能够在使用tensorflow-gpu时能更好地发挥GPU的性能，推荐安装NVIDIA官方版本的驱动
在Ubuntu里面，打开“软件和更新”，点击里面的“附加驱动”标签页，选择使用NVIDIA driver，然后点击“应用更改”进行官方驱动的安装，安装后重启电脑即可
重启电脑后，只要在电脑的设备信息里面看到“图形”是显示了是自己显卡型号，则说明NVIDIA官方显卡驱动安装成功了，之后就能用nvidia-smi命令了

3.安装CUDA
进入 https://developer.nvidia.com/cuda-downloads  ，依次选择 CUDA 类型然后下载对应的CUDA即可

4.安装Anaconda，从Anaconda官网（https://www.continuum.io/downloads）  上下载安装包，选择Linux系统，安装基于Python 3.6版本
对下载的文件授予可执行权限，然后进行安装：

bash Anaconda3-5.2.0-Linux-x86_64.sh  

当询问是否把Anaconda的bin添加到用户的环境变量中，选择yes

5.使用conda create命令创建虚拟环境到指定路径，并指定Python版本，同时可以将需要一起安装的包也一起指定：

conda create –n KerasYolo3Demo python=3.6 numpy scipy matplotlib jupyter

其中-n指定虚拟环境的名称（上面的是KerasYolo3Demo文件夹里）
默认安装的路径位于anaconda安装目录下的envs文件夹里面，也可以使用—prefix参数来重新指定虚拟环境路径
如果要查看有哪些虚拟环境，则执行以下命令：

conda info -envis

如果在创建conda虚拟环境时没有指定python的版本，则默认是使用anaconda安装目录下bin中的python版本。为了实现虚拟环境的隔离，必须指定python版本

6.激活虚拟环境
创建好conda虚拟环境后，在使用之前必须先进行激活。下面激活刚创建的KerasYolo3Demo虚拟环境，命令如下：

conda source activate KerasYolo3Demo

如果要注销退出当前的虚拟环境，则执行命令：

conda source deactivate tensorflow


7.安装tensorflow-gpu

conda source activate tensorflow
conda install tensorflow-gpu

conda将会检测tensorflow-gpu的最新版本以及相关的依赖包，包括调用NVIDIA显卡所需要的Cuda、Cudnn等依赖环境，都会自动按顺序进行安装

keras版本的yolo3还依赖于PIL工具包，如果之前没安装的，也要在anaconda中安装

# 安装keras-gpu版本
conda install keras-gpu
# 安装 PIL
conda install pillow

8.安装OpenCV（无法直接安装的，需要指定安装源）

conda install --channel https://conda.anaconda.org/menpo opencv3

9.安装PyCharm（Python开发IDE环境，社区版免费使用）
在Ubuntu里面安装PyCharm非常简单，在Ubuntu软件商城里面搜索“pycharm”，然后选择社区版“PyCharm CE”进行安装即可

为了能够在PyCharm中使用我们自己创建的conda虚拟环境，需要进行下配置。在Pycharm 的Files>>settings>>Project Interpreter>>Add local 
里面添加刚才创建的conda虚拟环境的目录下所在的Python 3.6程序，应用之后就可以使用我们自己使用的虚拟环境了


二、配置YOLO
1.下载YOLOv3源代码，后续可在此基础上进行改写：
https://github.com/wait1ess/keras-yolo3-voc

2.打开PyCharm，新建项目，将keras-yolo3的源代码导入到PyCharm中

3.YOLO官网上提供了YOLOv3模型训练好的权重文件，把它下载保存到电脑上，下载地址为
https://pjreddie.com/media/files/yolov3.weights

三、使用预训练权重进行目标检测
1.转换权重文件，将前面下载的yolo权重文件yolov3.weights转换成适合Keras的模型文件，转换代码如下：

source activate tensorflow
python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5

2.修改yolo.py里面的相关路径配置，主要是model_path,classes_path和gpu_num

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo_weights.h5',
        # "model_path":"model_data/trained_weights_final.h5",
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path":   'model_data/coco_classes.txt',
        # "classes_path": 'model_data/voc_classes.txt',
        "score" : 0.7,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 0,
    }

3、创建yolo实例，预测图片或视频
图片如下：

if __name__ == '__main__':
    yolo=YOLO()
    path = 'D:/VOCtrainval_06-Nov-2007/yoloV3conf/keras-yolo3-master/test001.jpg'
    try:
        image = Image.open(path)
    except:
        print('Open Error! Try again!')
    else:
        r_image = yolo.detect_image(image)
        r_image.show()

    yolo.close_session()

视频如下：

if __name__ == '__main__':
    video_path="D:/VOCtrainval_06-Nov-2007/yoloV3conf/keras-yolo3-master/test002.mp4"
    output_path="D:/VOCtrainval_06-Nov-2007/yoloV3conf/keras-yolo3-master/result002.mp4"
    detect_video(YOLO(), video_path, output_path)



四、训练自己的目标检测模型（应用于VOC格式数据）
1.以环境文件夹为根目录创建以下数据集
.
└── VOCdevkit     #根目录
    └── VOC2017   #不同年份的数据集，这里只下载了2007的，还有2007等其它年份的
        ├── Annotations        #存放xml文件，与JPEGImages中的图片一一对应，解释图片的内容等等
        ├── ImageSets          #该目录下存放的都是txt文件，txt文件中每一行包含一个图片的名称，末尾会加上±1表示正负样本
        │   ├── Action
        │   ├── Layout
        │   ├── Main
        │   └── Segmentation
        ├── JPEGImages         #存放源图片
        ├── SegmentationClass  #存放的是图片，语义分割相关
        └── SegmentationObject #存放的是图片，实例分割相关
        
2、安装标注工具labelImg，标注数据

http://host.robots.ox.ac.uk/pascal/VOC/

使用方法详见：https://blog.csdn.net/weixin_41065383/article/details/90637205


3、使用make_main_txt.py划分训练数据集，验证集以及测试集，保存于Main中

4、YOLO采用的标注数据文件，每一行由文件所在路径、标注框的位置（左上角、右下角）、类别ID组成，格式为：image_file_path x_min,y_min,x_max,y_max,class_id
例子如下：
path/00001.jpg xmin,ymin,xmax,ymax,class 
path/00002.jpg xmin,ymin,xmax,ymax,class xmin,ymin,xmax,ymax,class...

这种文件格式跟前面制作好的VOC_2007标注文件（Main中的txt）的格式不一样，Keras-yolo3里面提供了voc格式转yolo格式的转换脚本 voc_annotation.py

在转换格式之前，先打开voc_annotation.py文件，修改里面的classes的值为自己训练数据的label，然后执行即可，新的数据集标注文件保存于model_data目录

source activate tensorflow
python voc_annotation.py

5、创建类别文件voc_classes.txt

6、修改train.py里面的相关路径配置，主要有：annotation_path、classes_path、weights_path

    annotation_path = 'model_data/2007_train.txt'
    log_dir = 'model_data/logs/000/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'

7.训练，训练后的模型，默认保存路径为logs/000/trained_weights_final.h5
