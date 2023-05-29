# -*- coding:utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2019/7/20 20:01

@author: Jock
list.txt:0/0_1.jpg 0 图片路径 表情标签

dog: 0
elephant: 1
giraffe: 2
guitar: 3
horse： 4
house： 5
person： 6
"""


import os

file_list = []
# write_file_name = 'D:/pytorch/FACT-main/data/datalists/art_painting_train.txt'
# write_file_name = 'D:/pytorch/FACT-main/data/datalists/cartoon_train.txt'
# write_file_name = 'D:/pytorch/FACT-main/data/datalists/photo_train.txt'
write_file_name = 'D:/pytorch/FACT-main/data/datalists/sketch_train.txt'
write_file = open(write_file_name, "w")
number_of_lines=0
class BatchRenamePics(object):
    """
    批量命名目录下的所有图名[.jpg,.png]
    命名格式：1-1,1-2...2-1,2-2...10-1,10-2...eg
    """
    def __init__(self, path):
        # 设置起始路径path
        self.path = path

    def rename(self):
        allfile = os.walk(self.path)
        # j用于计数,统计有多少张照片被重命名
        j = 0
        num = 1
        # 遍历每一层目录,从上到下的顺序
        for dirpath, dirnames, filenames, in allfile:
            # 得到当前文件夹的名字tail
            tail = os.path.split(dirpath)[1]
            # i用于命名
            i = 0
            # 遍历filenames中的每一个文件
            for each in filenames:
                # 如果文件名是以.jpg或者.png结尾则认为是图片,可以自己添加其他格式的照片
                if each.endswith('.jpg') or each.endswith('.png'):
                    i += 1
                    j += 1
                    # 拼接完整的包含路径的文件名
                    scr = os.path.join(dirpath, each)
                    # 拼接新的完整的包含路径的文件名, tail是文件夹的名字
                    if tail=='dog':
                        label_=0
                    elif tail == 'elephant':
                        label_ = 1
                    elif tail == 'giraffe':
                        label_ = 2
                    elif tail == 'guitar':
                        label_ = 3
                    elif tail == 'horse':
                        label_ = 4
                    elif tail == 'house':
                        label_ = 5
                    elif tail == 'person':
                        label_ = 6
                    dst = scr + ' ' + str(label_)
                    # dst = os.path.join(tail + '/' + tail + '_' + str(j) + '.jpg'+' '+ tail)
                    # dst = os.path.join(dirpath, tail + '_' + str(j) + '.jpg' + ' ' + tail)
                    # s = '%04d' % num  # 前面补0占位 以4位数字存储，num为那个数字，从1开始
                    # dst = os.path.join(dirpath, tail + '_' + str(s) + '.jpg'+' '+ tail)
                    num += 1
                    write_name=dst
                    file_list.append(write_name)

                else:
                    continue
        number_of_lines = len(file_list)
        for current_line in range(number_of_lines):
            write_file.write(file_list[current_line] + '\n')
        # 关闭文件
        write_file.close()
        print('累计重命名{}张图片'.format(j))

if __name__ == '__main__':
    # 设置起始路径path
    # path = r'F:/data/Homework3-PACS-master/PACS/art_painting'
    # path = r'F:/data/Homework3-PACS-master/PACS/cartoon'
    # path = r'F:/data/Homework3-PACS-master/PACS/photo'
    path = r'F:/data/Homework3-PACS-master/PACS/sketch'
    # 创建实例对象
    pics = BatchRenamePics(path)
    # 调用实例方法
    pics.rename()







#
# import os #os：操作系统相关的信息模块
# #存放原始图片地址
# data_base_dir = "../my_operations/my_data/"
# # data_base_dir = "./expression_data/my_test/test_image_my/"
# file_list = [] #建立列表，用于保存图片信息
# #读取图片文件，并将图片地址、图片名和标签写到txt文件中
# write_file_name = '../my_operations/my_data/list.txt'
# write_file = open(write_file_name, "w") #以只写方式打开write_file_name文件
# number_of_lines=0
# for file in os.listdir(data_base_dir): #file为current_dir当前目录下图片名
#     if file.endswith(".jpg"): #如果file以jpg结尾
#       write_name = file #图片路径 + 图片名 + 标签
#       file_list.append(write_name) #将write_name添加到file_list列表最后
#       number_of_lines = len(file_list) #列表中元素个数
# #将图片信息写入txt文件中，逐行写入
# for current_line in range(number_of_lines):
#     write_file.write(file_list[current_line] + '\n')
# #关闭文件
# write_file.close()
