import copy
import os
from os import path
import cv2 as cv2


class dataset_formed:
    def __init__(self, dataset_path, dataset_image_folder_name):
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
                    f.write(
                        str(annotations['x0']) + ',' + str(annotations['y0']) + ',' + str(annotations['x1']) + ',' + str(
                            annotations['y1']) + ',' + str(annotations['x2']) + ',' + str(annotations['y2']) + ',' + str(
                            annotations['x3']) + ',' + str(annotations['y3']) + ',' + str(annotations['result']) + '\n')

            i = i + 1

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
augmented_images = test.make_insert(other_image_path)
test.save_result(out_put_path1)
# print(augmented_images)
