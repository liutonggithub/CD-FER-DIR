from PIL import Image
import face_recognition


# 检测单张图片的人脸并保存为原图片名称
def face_detect(image_path):
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(image_path)

    # Find all the faces in the image using a pre-trained convolutional neural network.
    # This method is more accurate than the default HOG model, but it's slower
    # unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
    # this will use GPU acceleration and perform well.
    # See also: find_faces_in_picture.py https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
        #                                                                                             right))

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save(image_path)
        # pil_image.show()


def face_detect_single(image_path):
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(image_path)

    # Find all the faces in the image using a pre-trained convolutional neural network.
    # This method is more accurate than the default HOG model, but it's slower
    # unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
    # this will use GPU acceleration and perform well.
    # See also: find_faces_in_picture.py https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    m=0
    while m < len(face_locations):
        # Print the location of each face in this image
        top, right, bottom, left = face_locations[m]
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left,
                                                                                                    bottom,
                                                                                                    right))
        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]

        pil_image = Image.fromarray(face_image)

        pil_image.save("./expression_data/{}.JPG".format(m))
        m = m+1

        # pil_image.show()

face_detect_single("AF03SUFL.JPG")


# # 检测多张图片中的人脸并保存到一个文件夹中
# from PIL import Image
# import matplotlib.pyplot as plt
#
# filename = (r'./expression_data/MultiPie_train/train_facedetect.txt')

# data_list = []
# with open(filename) as f:
#     for line in f.readlines():
#         line = line.strip('\n')
#
#         # lst = line.split(' ')  # 将str转换为列表
#         data_list.append(line)
#
# k = 0
# while k < len(data_list):
#     path = r'./expression_data/MultiPie_train/{}'.format(data_list[k])
#
#     face_detect(path)
#     k = k + 1


# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @Time    : 2018/1/3 14:12
# # @Author  : He Hangjiang
# # @Site    :
# # @File    : 人脸识别.py
# # @Software: PyCharm
#
# import cv2
# import face_recognition
# import time
#
# timeStart = time.clock()
# #读取图片并定位
# img = face_recognition.load_image_file("azy.jpg")
# face_locations = face_recognition.face_locations(img)
# print(face_locations)
#
# time_1 = time.clock()
# timeRec = time_1 - timeStart
# print("识别时间：",timeRec)
#
# #调用opencv显示人脸
# image = cv2.imread("azy.jpg")
# cv2.imshow("ori",image)
#
# #遍历人脸，并标注
# faceNum = len(face_locations)
# for i in range(faceNum):
#     top = face_locations[i][0]
#     right = face_locations[i][1]
#     bottom = face_locations[i][2]
#     left = face_locations[i][3]
#
#     start = (left,top)
#     end = (right,bottom)
#
#     color = (55,255,155)
#     thickness = 3
#     cv2.rectangle(image,start,end,color,thickness)
#
# cv2.imshow("recognized",image)
#
# time_2 = time.clock()
# timeDraw = time_2 - time_1
# print("画出位置时间：",timeDraw)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
