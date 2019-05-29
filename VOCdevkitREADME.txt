1)JPEGImages文件夹
文件夹里包含了训练图片和测试图片，混放在一起

2)Annatations文件夹
文件夹存放的是xml格式的标签文件，每个xml文件都对应于JPEGImages文件夹的一张图片

3)ImageSets文件夹
Action存放的是人的动作，我们暂时不用

Layout存放的人体部位的数据。我们暂时不用

Main存放的是图像物体识别的数据，分为20类
Main里面应有test.txt , train.txt, val.txt ,trainval.txt（四个文件利用make_main_txt.py生成）