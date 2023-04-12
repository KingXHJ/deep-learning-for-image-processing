import numpy as np
from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree


class VOCDataSet(Dataset):
    """读取解析PASCAL VOC2007/2012数据集"""

    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        # 增加容错能力
        # 数据根路径
        if "VOCdevkit" in voc_root:
            self.root = os.path.join(voc_root, f"VOC{year}")
        else:
            self.root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        # 图像数据路径
        self.img_root = os.path.join(self.root, "JPEGImages")
        # 标度信息数据路径
        self.annotations_root = os.path.join(self.root, "Annotations")

        # 根据输入参数决定读取 train.txt 或者 val.txt file
        txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        # 分析.txt文件中各个图片的标注信息
        with open(txt_path) as read:
            xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                        for line in read.readlines() if len(line.strip()) > 0]

        self.xml_list = []
        # 检查这些图片的标注信息是否都存在
        for xml_path in xml_list:
            if os.path.exists(xml_path) is False:
                print(f"Warning: not found '{xml_path}', skip this annotation file.")
                continue

            # 检查图片标注信息xml文件中是否有对象
            with open(xml_path) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            if "object" not in data:
                print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                continue

            # 存储xml文件
            self.xml_list.append(xml_path)

        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)

        # 读类别信息文件
        json_file = './pascal_voc_classes.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        with open(json_file, 'r') as f:
            self.class_dict = json.load(f)

        self.transforms = transforms

    # 官方要求实现的两个方法：__len__和__getitem__
    # 返回数据集文件的个数
    def __len__(self):
        return len(self.xml_list)

    # 输入索引值，返回图片和图片信息
    def __getitem__(self, idx):
        # 得到对应的xml文件路径
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        # 通过["annotation"]下标获得xml的子目录
        data = self.parse_xml_to_dict(xml)["annotation"]
        # filename得到图像的具体路径
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)
        # 必须是JPG文件
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        # 保存标注信息
        boxes = []
        labels = []
        iscrowd = [] # 是否有其他目标重叠
        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            # 强制转换成float型，因为模型预测的过程也是float型
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        # 外界传进来的参数
        image_id = torch.tensor([idx])
        # 面积
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # 检测是否需要transforms预处理方法
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    # 优化算法
    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    # 将batch_size*Tuple(image, targets) -> Tuple(batch_size*images), Tuple(batch_size*targets)
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

## 测试样例
# import transforms
# from draw_box_utils import draw_objs
# from PIL import Image
# import json
# import matplotlib.pyplot as plt
# import torchvision.transforms as ts
# import random
#
# # read class_indict
# category_index = {}
# try:
#     json_file = open('./pascal_voc_classes.json', 'r')
#     class_dict = json.load(json_file)
#     category_index = {str(v): str(k) for k, v in class_dict.items()}
# except Exception as e:
#     print(e)
#     exit(-1)
#
# data_transform = {
#     "train": transforms.Compose([transforms.ToTensor(),
#                                  transforms.RandomHorizontalFlip(0.5)]),
#     "val": transforms.Compose([transforms.ToTensor()])
# }
#
# # load train data set
# train_data_set = VOCDataSet(os.getcwd(), "2012", data_transform["train"], "train.txt")
# print(len(train_data_set))
# for index in random.sample(range(0, len(train_data_set)), k=5):
#     img, target = train_data_set[index]
#     img = ts.ToPILImage()(img)
#     plot_img = draw_objs(img,
#                          target["boxes"].numpy(),
#                          target["labels"].numpy(),
#                          np.ones(target["labels"].shape[0]),
#                          category_index=category_index,
#                          box_thresh=0.5,
#                          line_thickness=3,
#                          font='arial.ttf',
#                          font_size=20)
#     plt.imshow(plot_img)
#     plt.show()
