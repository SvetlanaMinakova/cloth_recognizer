import sys,random
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton,QLabel,QVBoxLayout,QHBoxLayout,QFileDialog,QTableWidget,QTableWidgetItem
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QPixmap,QPainter,QPen,QColor
import selectivesearch
import numpy as np
from skimage import io
from PIL import Image,ImageFile
import CNN_models
from skimage.transform import resize
from itertools import count,izip

class mainWindow(QWidget):
    def __init__(self,sel_search_boxes_num=20,normalization=False):
        super(QWidget,self).__init__()
        self.sel_search_boxes_num = sel_search_boxes_num
        self.img_max_w=500;
        self.img_max_h=500;
        self.img_loaded=False
        self.Network = CNN_models.Network(0)
        self.Network.load_weights('./weights/conv_128_128_sgd_set2_nb64_70ep_4cl_.hdf5')
        #self.Network.evaluate_model('./img_pack')
        self.target_img_pieces_shape=((self.Network.inp_w,self.Network.inp_h))
        self.initUI()

    def initUI(self):
        #main menu btns
        load_btn = QPushButton('Load',self)
        load_btn.clicked.connect(self.load_img)

        draw_btn = QPushButton('Recognize',self)
        draw_btn.clicked.connect(self.recognize)

        close_btn= QPushButton('Quit',self)
        close_btn.clicked.connect(QCoreApplication.instance().quit)

        #recognization results
        self.resultsTable=QTableWidget(self)
        self.resultsTable.setColumnCount(2)
        #img
        self.image = QLabel(self)
        self.image.setObjectName('image')
        self.image.setFixedWidth(self.img_max_w)
        self.image.setFixedHeight(self.img_max_h)

        #layouts
        self.menubox = QHBoxLayout()
        self.centralbox = QHBoxLayout()
        self.windowLayout = QVBoxLayout()
        self.menubox.addWidget(load_btn)
        self.menubox.addWidget(draw_btn)
        self.menubox.addWidget(close_btn)
        self.centralbox.addWidget(self.image)
        self.centralbox.addWidget(self.resultsTable)
        self.windowLayout.addLayout(self.menubox)
        self.windowLayout.addLayout(self.centralbox)
        self.setLayout(self.windowLayout)
        self.show()

    def show_img(self,path):
        self.imgpath=path[0]
        self.pixmap=QPixmap(self.imgpath)
        self.pixmap=self.pixmap.scaledToWidth(self.img_max_w)
        self.pixmap=self.pixmap.scaledToHeight(self.img_max_h)
        self.save_img_to_temp()
        self.img_w=self.pixmap.width()
        self.img_h=self.pixmap.height()
        self.image.setPixmap(self.pixmap)
        self.image.setAlignment(Qt.AlignCenter)

    def save_img_to_temp(self):
        self.pixmap.save('./tmp/tmp.jpg')
        self.pixmap=QPixmap('./tmp/tmp.jpg')


    def drawRects(self):
        self.boxes_colors=[]
        pen = QPen(QColor(224,0,0),2)
        qp=QPainter()
        qp.begin(self.image.pixmap())
        qp.setPen(pen)
        for box in self.boxes:
            qp.drawRect(box[0],box[1],box[2],box[3])
            rnd_color=self.get_random_color()
            pen.setColor(rnd_color)
            self.boxes_colors.append(rnd_color)
        qp.setPen(pen)
        self.image.update()

    def recognize(self):
        if(self.img_loaded):
            self.search_boxes()
            self.pieces_matrixes=self.get_pieces_matrixes(reshape=True,new_shape=self.target_img_pieces_shape)
            self.save_img_pieces()
            self.resultsTable.setRowCount(1)

            predicted_classes = self.Network.predict_classes_as_text(self.pieces_matrixes)
            print (predicted_classes)
            one_class = predicted_classes[0]
            self.resultsTable.setItem(0, 0, QTableWidgetItem(str(self.boxes[0])))
            self.resultsTable.setItem(0,1,QTableWidgetItem(one_class))


    def search_boxes(self):
        self.img_as_matrix= io.imread('./tmp/tmp.jpg',load_func=self.imread_convert)
        self.img_lbl,self.regions = selectivesearch.selective_search(self.img_as_matrix,scale=200,sigma=1.4,min_size=50)
        self.regions.sort(key=lambda  x: x['size'],reverse=True)
        self.boxes = self.get_boxes_witout_duplicates(self.regions)
        self.drawRects()

    def get_boxes_witout_duplicates(self,sorted_regions):
        result_boxes=[]
        if(len(sorted_regions)>0):
            cur_box=(sorted_regions[0])['rect']
            result_boxes.append(cur_box)
            for reg in sorted_regions:
                if(self.boxes_are_different(reg['rect'],cur_box)):
                    result_boxes.append(reg['rect'])
                    cur_box=reg['rect']
                    if(len(result_boxes)>=self.sel_search_boxes_num):
                        return np.asarray(result_boxes)
        return np.asarray(result_boxes)

    def boxes_are_different(self,box1,box2):
        indexes_num=4 #x1,y1,x2,y2
        for i in range(indexes_num):
            if(box1[i]!=box2[i]):
                return True
        return False

    def get_pieces_matrixes(self,reshape=False,new_shape=None):
        pieces_matrixes=[]
        for box in self.boxes:
            piece = self.img_as_matrix[box[1]:(box[1]+box[3]),box[0]:(box[0]+box[2]),:]
            if(reshape):
                piece= resize(piece,new_shape)
            pieces_matrixes.append(piece)
        return np.asarray(pieces_matrixes)

    def imread_convert(self,f):
        return io.imread(f).astype(np.floatX)

    def load_img(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _= QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileName()","","Images (*.jpg *.png)", options=options)
        if(fileName):
            self.show_img(fileName)
            self.img_loaded=True


    def get_random_colors_range(self,len):
        colors_range=[]
        for i in range(len):
             (r,g,b) = np.random.randint(0,255,3)
             color=QColor(r,g,b)
             colors_range.append(color)
        return colors_range


    def get_random_color(self):
        (r,g,b) = np.random.randint(0,255,3)
        return QColor(r,g,b)

    def save_img_pieces(self):
        box_num=0
        for piece in self.pieces_matrixes:
            io.imsave('./tmp/img_pieces/'+str(box_num)+'.jpg',piece)
            box_num +=1

    def save_img_piece(self,box,box_num):
        arrpiece = self.img_as_matrix[box[1]:(box[1]+box[3]),box[0]:(box[0]+box[2]),:]
        io.imsave('./tmp/img_pieces/'+str(box_num)+'.jpg',arrpiece)

    def choose_one_predicted_class(self,predicted_classes):
        rates=[]
        for cl in self.Network.classes:
            rates.append(predicted_classes.count(cl))
        return self.Network.classes[rates.index(max(rates))]


if __name__ == '__main__':
    app=QApplication(sys.argv)
    w = mainWindow(5)
    sys.exit(app.exec_())
