import copy
import os
import random
from os import path
import cv2 as cv2
import numpy as np


def check_iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    # 判断如果没有相交直接返回 0 相交返回1
    if ax1 >= bx2 or ax2 <= bx1 or ay1 >= by2 or ay2 <= by1:
        return 0
    else:
        return 1


class dataset_formed:
    def __init__(self, dataset_path, dataset_image_folder_name):
        self.dataset_image_part_info = None
        self.dataset_images_shape = None
        self.inserted_images_with_annotations = None
        self.dataset_images_info_with_file_name = None
        self.dataset_path = dataset_path
        self.dataset_image_folder_name = dataset_image_folder_name
        self.dataset_imgs_path_now = path.join(self.dataset_path + "/imgs/" + dataset_image_folder_name)
        self.dataset_annotations_path_now = path.join(self.dataset_path + "/annotations/" + dataset_image_folder_name)
        # print(self.dataset_imgs_path_now)
        # with open(dataset_instances_test, "r", encoding="utf-8") as f:
        #     list_image_info = []
        #     for image_info in f:
        #         image_info = eval(image_info)
        #         list_image_info.append(image_info)
        # # print(list_image_info)
        # # print(list_image_info[0])
        # self.dataset_instances_test = list_image_info
        # print(self.dataset_instances_test)

    def save_result(self, out_put_path):
        i = 0
        annotation_path = path.join(out_put_path + '/annotations')
        annotation_abs_path = path.join(annotation_path + '/' + self.dataset_image_folder_name)
        imgs_path = path.join(out_put_path + '/imgs')
        imgs_abs_path = path.join(imgs_path + '/' + self.dataset_image_folder_name)
        # 创建目录结构
        if not os.path.exists(out_put_path):
            os.makedirs(out_put_path)

            if not os.path.exists(annotation_path):
                os.makedirs(annotation_path)
                if not os.path.exists(annotation_abs_path):
                    os.makedirs(annotation_abs_path)
            if not os.path.exists(imgs_path):
                os.makedirs(imgs_path)
                if not os.path.exists(imgs_abs_path):
                    os.makedirs(imgs_abs_path)

        for inserted_image_with_annotations in self.inserted_images_with_annotations:
            image = inserted_image_with_annotations['image']
            annotations = inserted_image_with_annotations['annotations']
            img_file_name = "img_" + str(i)
            annotation_file_name = "gt_" + img_file_name
            img_file_name = img_file_name + '.' + 'jpg'
            annotation_file_name = annotation_file_name + '.' + 'txt'
            cv2.imwrite(path.join(imgs_abs_path + '/' + img_file_name), image)
            with open(path.join(annotation_abs_path + '/' + annotation_file_name), "w", encoding="utf-8") as f:
                for annotation in annotations:
                    f.write(
                        str(annotation['x0']) + ',' + str(annotation['y0']) + ',' + str(annotation['x1']) + ',' + str(
                            annotation['y1']) + ',' + str(annotation['x2']) + ',' + str(annotation['y2']) + ',' + str(
                            annotation['x3']) + ',' + str(annotation['y3']) + ',' + str(annotation['result']) + '\n')

            i = i + 1

    def load_dataset_images_info(self):
        # 加载数据集的图片信息
        self.dataset_image_part_info = []
        for dataset_image_info in self.dataset_images_info_with_file_name:
            f = path.join(dataset_image_info['file_name'] + '.' + dataset_image_info['file_format'])
            processing_image_path_now = path.join(self.dataset_imgs_path_now + '/' + f)
            processing_image_now = cv2.imread(processing_image_path_now)
            processing_image_now_shape = processing_image_now.shape
            (y, x, a) = processing_image_now_shape
            self.dataset_images_shape = (y, x, a)
            for annotations in dataset_image_info['annotations']:
                # print(annotations)
                temp_image_origin = processing_image_now[int(annotations['y0']):int(annotations['y2']),
                                    # 已完成所需要训练部分摘取
                                    int(annotations['x0']):int(annotations['x2'])]
                position = [int(annotations['x0']), int(annotations['y0'])]
                lenth = [int(annotations['x2']) - int(annotations['x0']),
                         int(annotations['y2']) - int(annotations['y0'])]
                (lena, wida) = lenth
                image_part_info = {'image': temp_image_origin, 'annotations': annotations, 'position': position,
                                   'lenth': lenth, 'square': abs(lena) * abs(wida)}
                self.dataset_image_part_info.append(image_part_info)
        return self.dataset_image_part_info

    def make_random_insert(self, other_image_path_):

        # print(temp_img_part_info)
        # print(self.dataset_images_info_with_file_name)
        file = os.listdir(other_image_path_)
        # print(file)
        insert_images = []
        inserted_images_with_annotations = []
        for insert_image_name in file:
            insert_image = cv2.imread(path.join(other_image_path_ + '/' + insert_image_name))
            insert_images.append(insert_image)
            # insert_image = cv2.resize(insert_image, (1280, 720), interpolation=cv2.INTER_LINEAR)
            # insert_image_shape = insert_image.shape
            # print(insert_image_shape)
        # print(insert_images)
        # 将dataset_image_info_with_file_name中的图片信息提取出来并打包成为只有待增强图像的信息
        self.load_dataset_images_info()
        # print(self.dataset_image_part_info)
        temp_imgs_part_info = sorted(self.dataset_image_part_info, key=lambda x: x['square'],
                                     reverse=True)  # 按照图片的面积进行排序
        # print(temp_imgs_part_info)

        for insert_image in insert_images:
            insert_image = cv2.resize(insert_image, (self.dataset_images_shape[1], self.dataset_images_shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
            new_insert_image = copy.deepcopy(insert_image)  # 图片拷贝
            (y, x, a) = self.dataset_images_shape
            count = 5
            overlay_map = np.zeros((x, y), bool)  # 创建一个覆盖图，0为可用区域1为不可用区域
            annotations = []  # 保存增强后的标注信息列表
            for temp_img_part_info in temp_imgs_part_info:  # 将图片插入到insert_image中
                # print(temp_img_part_info)
                (x0, y0) = temp_img_part_info['position']
                (x1, y1) = temp_img_part_info['lenth']

                # print(y0, x0, y1, x1)
                # try:
                if count == 0:  # count=0时保存前面的图像信息开辟新图像
                    inserted_images_with_annotations.append({'image': new_insert_image, 'annotations': annotations})
                    new_insert_image = copy.deepcopy(insert_image)  # 图片拷贝
                    (y, x, a) = self.dataset_images_shape
                    overlay_map = np.zeros((x, y), bool)  # 创建一个覆盖图，0为可用区域1为不可用区域
                    annotations = []  # 保存增强后的标注信息列表
                    count = 5  # 同一图中至多5次插入图像
                sgn = 0
                for i in range(0, 5):  # 尝试5次找位置插入图像
                    x0 = random.randint(0, x - x1-1)
                    y0 = random.randint(0, y - y1-1)
                    if overlay_map[x0, y0] == False and overlay_map[x0 + x1, y0 + y1] == False and overlay_map[
                        x0, y0 + y1] == False and overlay_map[x0 + x1, y0] == False and count > 0:
                        new_insert_image[y0:y0 + y1, x0:x0 + x1] = temp_img_part_info['image']
                        overlay_map[x0:x0 + x1, y0:y0 + y1] = True
                        annotation = temp_img_part_info['annotations']
                        annotation['x0'] = x0
                        annotation['y0'] = y0
                        annotation['x1'] = x0 + x1
                        annotation['y1'] = y0
                        annotation['x2'] = x0 + x1
                        annotation['y2'] = y0 + y1
                        annotation['x3'] = x0
                        annotation['y3'] = y0 + y1
                        annotations.append(annotation)
                        sgn = 1  # 插入成功
                        break
                if sgn == 1:
                    count = count - 1  # 继续插入下一张图片
                    continue
                if sgn == 0:  # 保存前面的图像信息开辟新图将插不进的这张图插进去
                    inserted_images_with_annotations.append({'image': new_insert_image, 'annotations': annotations})
                    new_insert_image = copy.deepcopy(insert_image)  # 重置新图像拷贝
                    annotations = []  # 重置标注信息
                    overlay_map = np.zeros((x, y), bool)  # 重置覆盖图
                    count = 5  # 重置插入次数
                    # 将插不进的图片插到重置后的覆盖图里
                    new_insert_image[y0:y0 + y1, x0:x0 + x1] = temp_img_part_info['image']
                    overlay_map[x0:x0 + x1, y0:y0 + y1] = True
                    annotation = temp_img_part_info['annotations']
                    annotation['x0'] = x0
                    annotation['y0'] = y0
                    annotation['x1'] = x0 + x1
                    annotation['y1'] = y0
                    annotation['x2'] = x0 + x1
                    annotation['y2'] = y0 + y1
                    annotation['x3'] = x0
                    annotation['y3'] = y0 + y1
                    annotations.append(annotation)
                    count = count - 1
                    continue
        inserted_images_with_annotations.append(
            {'image': new_insert_image, 'annotations': annotations})  # 将最后一张图片保存到队列中
        self.inserted_images_with_annotations = inserted_images_with_annotations
        return self.inserted_images_with_annotations

    def make_insert(self, other_image_path_):
        file = os.listdir(other_image_path_)
        print(file)
        insert_images = []
        inserted_images_with_annotations = []
        for insert_image_name in file:
            insert_image = cv2.imread(path.join(other_image_path_ + '/' + insert_image_name))
            insert_images.append(insert_image)
            # insert_image = cv2.resize(insert_image, (1280, 720), interpolation=cv2.INTER_LINEAR)
            # insert_image_shape = insert_image.shape
            # print(insert_image_shape)
        for dataset_image_info in self.dataset_images_info_with_file_name:
            print(dataset_image_info)
            f = path.join(dataset_image_info['file_name'] + '.' + dataset_image_info['file_format'])
            processing_image_path_now = path.join(self.dataset_imgs_path_now + '/' + f)
            processing_image_now = cv2.imread(processing_image_path_now)
            processing_image_now_shape = processing_image_now.shape
            (y, x, a) = processing_image_now_shape
            for insert_image in insert_images:
                insert_image = cv2.resize(insert_image, (x, y), interpolation=cv2.INTER_LINEAR)
                for annotations in dataset_image_info['annotations']:
                    temp_image_insert = copy.deepcopy(insert_image)
                    temp_image_origin = processing_image_now[int(annotations['y0']):int(annotations['y2']),
                                        # 已完成所需要训练部分摘取
                                        int(annotations['x0']):int(annotations['x2'])]
                    temp_image_insert[int(annotations['y0']):int(annotations['y2']),
                    int(annotations['x0']):int(annotations['x2'])] = temp_image_origin
                    inserted_image_with_annotations = {'image': temp_image_insert, 'annotations': annotations}
                    inserted_images_with_annotations.append(inserted_image_with_annotations)
                    # cv2.rectangle(processing_image_now, (int(annotations['x0']), int(annotations['y0'])),
                    #               (int(annotations['x2']), int(annotations['y2'])), (0, 0, 255), 2)
                    # print()
                    # cv2.imshow("test", temp_image_insert)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
            # print(processing_image_now_shape)
            # print()
        self.inserted_images_with_annotations = inserted_images_with_annotations
        return inserted_images_with_annotations

    def info_open(self):
        file = os.listdir(self.dataset_imgs_path_now)
        list_dict_list_dict_image_info_with_file_name = []
        # print(file)
        for f in file:
            (file_name, file_format) = f.split('.')
            # print(file_name)
            # a=path.join(self.dataset_annotations_path_now+"/gt_"+file_name)
            # print(a)
            with open(path.join(self.dataset_annotations_path_now + "/gt_" + file_name + "." + "txt"), "r",
                      encoding="utf-8-sig") as gt_img:
                list_dict_image_info = []
                for image_info in gt_img:
                    # print(image_info)
                    image_info = image_info.strip('\n')
                    temp_a = image_info.split(',')
                    # print(temp_a)
                    temp_b = ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'result']
                    dict_image_info = dict(zip(temp_b, temp_a))
                    # print(dict_image_info)
                    list_dict_image_info.append(dict_image_info)
            dict_list_dict_image_info_with_file_name = {'file_name': file_name, 'file_format': file_format,
                                                        'annotations': list_dict_image_info}
            # print(dict_list_dict_image_info_with_file_name)
            list_dict_list_dict_image_info_with_file_name.append(dict_list_dict_image_info_with_file_name)
        self.dataset_images_info_with_file_name = list_dict_list_dict_image_info_with_file_name
        # one_image_path = path.join(self.dataset_image_folder_name, f)  # 获得单张图片的图片链接


# dataset_path1 = "icdar2015"
# dataset_image_path1 = "test"
# other_image_path = "insert_image/"
# out_put_path = "/icdar2015_augmented"

dataset_path1 = "toy_dataset"
dataset_image_path1 = "test"
other_image_path = "insert_image/"
out_put_path1 = "toy_dataset_augmented"

test = dataset_formed(dataset_path1, dataset_image_path1)
test.info_open()
# augmented_images = test.make_insert(other_image_path)
test.make_random_insert(other_image_path)
test.save_result(out_put_path1)
# print(augmented_images)
