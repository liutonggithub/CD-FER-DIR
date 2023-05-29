# -*- coding:utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2019/7/20 20:01

@author: Jock
list.txt:0/0_1.jpg 0 图片路径 表情标签

Angry: 0
Disgust: 1
Fear: 2
Happy: 3
Neutral： 4
Sad： 5
Surprised： 6
"""


import os

file_list = []
# write_file_name = 'D:/pytorch/FACT-main/data/datalists/JAFFE_test.txt'
# write_file_name = 'D:/pytorch/FACT-main/data/datalists/Oulu_CASIA_test.txt'
# write_file_name = 'D:/pytorch/FACT-main/data/datalists/RAF-DB_test.txt'
write_file_name = 'D:/pytorch/FACT-main/data/datalists/SFEW_test.txt'
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
                    if tail=='Angry':
                        label_=0
                    elif tail == 'Disgust':
                        label_ = 1
                    elif tail == 'Fear':
                        label_ = 2
                    elif tail == 'Happy':
                        label_ = 3
                    elif tail == 'Neutral':
                        label_ = 4
                    elif tail == 'Sad':
                        label_ = 5
                    elif tail == 'Surprised':
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

    # path = r'F:/data/FER/JAFFE'
    # path = r'F:/data/FER/Oulu_CASIA'
    # path = r'F:/data/FER/RAF-DB'
    path = r'F:/data/FER/SFEW'
    # 创建实例对象
    pics = BatchRenamePics(path)
    # 调用实例方法
    pics.rename()