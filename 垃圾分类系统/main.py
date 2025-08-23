import io
import sys
from io import BytesIO
import os
import cv2
import torch
from  PyQt5 import QtGui, QtWidgets
from torch import nn
from torchvision.models import resnet152,resnet50
from torchvision.transforms import transforms
from PIL import Image, ImageQt
import json

from PyQt5.QtCore import Qt, QBuffer
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication

class MainWindow():
    def __init__(self):
        super().__init__()
        # 加载ui文件
        self.window = uic.loadUi("./垃圾识别.ui")
        self.window.setFixedSize(self.window.width(), self.window.height())
        self.window.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        self.lblimg = self.window.ibl_img

        self.data_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # 定义模型等变量
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50()
        # 修改全连接层
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 244)
        # 加载模型文件
        self.model.load_state_dict(torch.load("model_waste_50_best.pth",map_location=torch.device('cpu')))
        self.model = self.model.to(self.device)
        # 自动加载垃圾类别
        train_dir = 'waste_big/train'
        if os.path.exists(train_dir):
            class_names = sorted(os.listdir(train_dir))
            self.wastes = {str(i): name for i, name in enumerate(class_names)}
        else:
            print(f"警告：训练集目录 {train_dir} 不存在！")
            self.wastes = {str(i): f'未知类别_{i}' for i in range(244)}
        # 声明摄像头变量
        self.capture = None
        self.window.pushButton_open.clicked.connect(self.open_capture)
        self.window.pushButton_identify.clicked.connect(self.identify)

    def open_capture(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("无法打开摄像头")
            return

        # 强制设置摄像头分辨率为 640x480
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                print("无法读取帧")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)  # 镜像显示（可选）

            img = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
                               frame.shape[1] * 3, QtGui.QImage.Format_RGB888)

            self.lblimg.setFixedSize(frame.shape[1], frame.shape[0])
            self.lblimg.setPixmap(QtGui.QPixmap.fromImage(img))
            self.lblimg.setAlignment(Qt.AlignCenter)
            self.lblimg.setScaledContents(False)

            QtWidgets.QApplication.processEvents()

    # 将qt像素图转化成pil格式图像
    def convert_qimage_to_pil(self,qimage):
        buffer = QBuffer()
        qimage.save(buffer, 'BMP')
        pil_image = Image.open(io.BytesIO(buffer.data()))
        return pil_image

    def identify(self):
        # 获取当前显示的图像
        pixmap = self.lblimg.pixmap()
        if pixmap is None:
            print("没有图像可供识别")
            return
        # 转换为 PIL 图像
        pil_img = self.convert_qimage_to_pil(pixmap.toImage())  # 注意这里需要 toImage()
        # 图像预处理
        input_tensor = self.data_transforms(pil_img).unsqueeze(0).to(self.device)
        # 模型推理
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            index = torch.argmax(outputs).item()
        # 获取垃圾类别
        result = self.wastes.get(str(index), '未知类别')
        print('识别结果：' + result)
        # 更新UI上的识别结果显示
        self.window.label_result.setText(result)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    mainwindow.window.show()
    sys.exit(app.exec_())
# model.eval()
# if not capture.isOpened():
#     print("无法打开摄像头")
#     exit()
# while True:
#     ret, img = capture.read()
#     if ret == True:
#         cv2.imshow("video-test", img)
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_pil = Image.fromarray(img_rgb)
#         with torch.no_grad():
#             input = data_transforms(img_pil)
#             input = torch.unsqueeze(input, 0)
#             input = input.to(device)
#             y_pre = model(input)
#             index = torch.argmax(y_pre).item()
#             print('识别结果：' + wastes[str(index)])
#         key = cv2.waitKey(0)
#         if key == 27:
#             break
# # 释放摄像头资源
# capture.release()
# cv2.destroyAllWindows()