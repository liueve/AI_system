#主視窗與子視窗的介面建立與互動功能
from mainframe_function import mainframe_func
from PyQt5.QtCore import pyqtSlot, pyqtSignal,QCoreApplication,Qt, QSize
from PyQt5 import QtCore,QtWidgets,QtGui
from PyQt5.QtWidgets import QMainWindow,QFrame
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtGui import QPalette, QColor,QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTreeWidgetItem,QLabel,QStackedWidget



# ------------------------------------------------------------------------------------------------------------

#主視窗
class MyMainWindow(QMainWindow, mainframe_func):
    
    def __init__(self):
        super().__init__()

        #設置無邊框
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint) #去除系統邊框
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground) #去除系統預設背景
        #設置邊界範圍
        self.right_frame = QFrame(self)
        self.bottom_frame = QFrame(self)
        self.corner_frame = QFrame(self)
      
        self.model = None
        self.file_paths = []
        self.selected_axis = "Z"
        loadUi(r"Ui.ui", self)
        self.dataframes = {}  # 初始化 dataframes 属性
        self.plot_widgets = [] #用於存放繪製出來的圖表
         # 创建场景对象
        self.calc_data = None
        self.selected_file_name = ""
        self.predicted_label_string = ""
        self.calc_data_B = None
        self.model_name = "" 
        self.images = ""
        self.images_stft = "" #用於丟入CNN的圖片
        self.images_stfted = "" #展示於CNN子視窗的圖片

        # 将场景设置为视图的场景
        self.import_button.clicked.connect(self.open_files)
        self.clear_button.clicked.connect(self.clear_data_output)#點擊'清除'清空時域圖、時頻圖、統計值、下拉選單列表
        self.linechart_button.clicked.connect(self.plot_selected_data)
        self.X_btn.clicked.connect(self.select_x_axis)
        self.Y_btn.clicked.connect(self.select_y_axis)
        self.Z_btn.clicked.connect(self.select_z_axis)
        self.listWidget_1.itemClicked.connect(self.save_item_click)
        self.listWidget_2.itemClicked.connect(self.AI_item_click)
        
        self.titlespace.installEventFilter(self)
        self.corner_frame.installEventFilter(self)
        self.exit_btn.clicked.connect(self.queryExit)
        self.fullscreen_btn.clicked.connect(self.maxOrNormal)
        #在滑鼠移入右側、角落、底下邊界範圍時變更滑鼠樣式
        self.bottom_frame.setCursor(Qt.SizeVerCursor)
        self.corner_frame.setCursor(Qt.SizeFDiagCursor)
        self.right_frame.setCursor(Qt.SizeHorCursor)

        # 记录鼠标按下时的位置
        self.drag_start_position = None

        #拖曳邊界縮放視窗滑鼠事件
        #底下
        self.bottom_frame.mousePressEvent = self.bottomFrameMousePressEvent
        self.bottom_frame.mouseMoveEvent = self.bottomFrameMouseMoveEvent
        #右側
        self.right_frame.mousePressEvent = self.rightFrameMousePressEvent 
        self.right_frame.mouseMoveEvent = self.rightFrameMouseMoveEvent
                

# ------------------------------------------------------------------------------------------------------------
    #<---------------視窗互動事件區塊----------------------->#
    
    #離開主視窗跳出訊息欄
    def queryExit(self):
        res = QtWidgets.QMessageBox.question(self,"Warning","Quit?",QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        if res == QtWidgets.QMessageBox.Yes:
            QCoreApplication.instance().exit()   

    # 切換原大小與最大化視窗
    def maxOrNormal(self):
        if self.isMaximized():
            self.showNormal()
            self.fullscreen_btn.setIcon(QtGui.QIcon(r"button\window-max.png"))
            
        else:
            self.showMaximized()
            self.fullscreen_btn.setIcon(QtGui.QIcon(r"button\window-normal.png"))
    
    #雙擊titlespace進入原大小與最大化視窗功能
    def mouseDoubleClickEvent(self, e):
        if isinstance(self, MyMainWindow):
            if self.childAt(e.pos().x(), e.pos().y()).objectName() == "titlespace":
                if e.button() == QtCore.Qt.LeftButton:
                    self.maxOrNormal()

    
            
#<---------------視窗互動事件區塊----------------------->#

    

    def mousePressEvent(self, event):
        # 记录鼠标按下時的位置和視窗的位置
        if event.button() == Qt.LeftButton and self.titlespace.geometry().contains(event.pos()):
            self.drag_position = event.globalPos() - self.pos()
            event.accept()
        #若鼠標按下位置在corner_frame，則紀錄鼠標位置用於縮放視窗大小
        elif event.button() == Qt.LeftButton and self.corner_frame.geometry().contains(event.pos()):
            self.resize_position = event.globalPos()
            self.start_size = self.size()
            event.accept()

    def mouseMoveEvent(self, event):
        # 拖动窗口
        if hasattr(self, 'drag_position') and event.buttons() == Qt.LeftButton:
            self.setCursor(QtGui.QCursor(QtCore.Qt.ClosedHandCursor)) #1006變換鼠標 劉新增
            self.move(event.globalPos() - self.drag_position)
            event.accept()
        elif hasattr(self, 'resize_position') and event.buttons() == Qt.LeftButton:
            delta = event.globalPos() - self.resize_position
            new_size = self.start_size + QSize(delta.x(), delta.y())
            self.resize(max(new_size.width(), 640), max(new_size.height(), 480))
            event.accept()

    def mouseReleaseEvent(self, event):
        # 清除拖动标志
        if hasattr(self, 'drag_position'):
            self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor)) #1006變換鼠標 劉新增
            del self.drag_position
        elif hasattr(self, 'resize_position'):
            del self.resize_position
            del self.start_size


        
    #<--------------------------------------------------->#
    # 连接事件处理方法
        
    #底下位置拖曳調整視窗大小
    def bottomFrameMousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.globalPos()

    def bottomFrameMouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.drag_start_position:
            global_pos = event.globalPos()
            delta = global_pos - self.drag_start_position
            self.resize(self.width(), self.height() + delta.y())
            self.drag_start_position = global_pos
            event.accept()
    #右側位置拖曳調整視窗大小
    def rightFrameMousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.globalPos()

    def rightFrameMouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.drag_start_position:
            global_pos = event.globalPos()
            delta = global_pos - self.drag_start_position
            self.resize(self.width() + delta.x(), self.height())
            self.drag_start_position = global_pos
            event.accept()

# ------------------------------------------------------------------------------------------------------------
class AI_mouseevent(QWidget):
    def start(self):
        #設置無邊框
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        #退出子視窗
        self.Dialog_exit_btn.clicked.connect(self.close)


    def mousePressEvent(self, event):
        #紀錄點擊鼠標座標與移動座標，生成drag_position(移動距離)
        if event.button() == Qt.LeftButton and self.titlespace.geometry().contains(event.pos()):
            self.drag_position = event.globalPos() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        # 拖动窗口
        if hasattr(self, 'drag_position') and event.buttons() == Qt.LeftButton:
            self.setCursor(QtGui.QCursor(QtCore.Qt.ClosedHandCursor)) #1006變換鼠標 劉新增
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):        
        if hasattr(self, 'drag_position'):#若drag_position存在
            self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor)) #1006變換鼠標 劉新增
            del self.drag_position # 拖動視窗動作結束，清除drag_position

from PyQt5.QtWidgets import QDialog

#RF模型子視窗
class RF_UiWindow(QDialog, AI_mouseevent):
    
    def __init__(self, selected_file_name,predicted_label_string,model_name):
        super().__init__()   
        # 加载UI文件
        loadUi(r"RF_Ui.ui", self)
        #無邊框與退出功能
        self.start()
        print (predicted_label_string)

        #model_name = A模型預測結果
        # 设置filename标签的文本为所选文件名
        self.filename.setText(f"Selected File: {selected_file_name}")
        #print(predicted_label_string)

        #若預測出來的標籤為0%~25%，則文本如下:
        if predicted_label_string == "0%~20%":
            self.circle_color.setStyleSheet("border-radius: 75px;background: rgb(0,0,0);")
            status = "損壞"
            score = "0%~20%"
            comment = "設備已損壞，立即暫停使用"
        elif predicted_label_string == "21%~40%":
            self.circle_color.setStyleSheet("border-radius: 75px;background: rgb(255,0,0);")
            status = "異常"
            score = "21%~40%"
            comment = "設備異常，需停機檢查"
            
        elif predicted_label_string == "41%~60%":
            self.circle_color.setStyleSheet("border-radius: 75px;background: rgb(255,170,0);")
            status = "損耗"
            score = "41%~60%"
            comment ="設備有損耗，需定期保養"
        elif predicted_label_string == "61%~80%":
            self.circle_color.setStyleSheet("border-radius: 75px;background: rgb(0,222,0);")
            status = "正常"
            score ="61%~80%"
            comment ="設備狀況正常，需定期保養"
        elif predicted_label_string == "81%~100%":
            self.circle_color.setStyleSheet("border-radius: 75px;background: rgb(0,170,255);")
            status = "良好"
            score ="81%~100%"
            comment ="設備狀況良好，建議定期保養"
        # 设置状态文本
        self.state.setText(f"狀態: {status}")
        self.score.setText(f"健康程度: {score}")
        self.comment.setText(comment)
         # 设置model_name文本
        self.model_name.setText(f" {model_name}")

# ------------------------------------------------------------------------------------------------------------
#SVM模型視窗
class SVM_UiWindow(QDialog,AI_mouseevent):
    def __init__(self, selected_file_name,predicted_label_string):
        super().__init__()
         # 加载UI文件
        loadUi(r"SVM_Ui.ui", self)
        #無邊框與退出功能
        self.start()

        self.filename.setText(f"Selected File: {selected_file_name}")

        if predicted_label_string == "41%~60%":
            self.circle_color.setStyleSheet("border-radius: 75px;background: rgb(255,0,0);")          
            comment = "設備有損耗，需定期保養"
        elif predicted_label_string == "91%~100%":
            self.circle_color.setStyleSheet("border-radius: 75px;background: rgb(0,222,0);")           
            comment = "設備狀況良好，建議定期保養"

        
        self.comment.setText(comment)

# ------------------------------------------------------------------------------------------------------------
#CNN模型子視窗
class CNN_UiWindow(QDialog,AI_mouseevent):
    
    def __init__(self, selected_file_name, predicted_label_string, images_stfted):
        super().__init__()
        # 加载UI文件
        loadUi(r"CNN_Ui.ui", self)
        #無邊框與退出功能
        self.start()

        self.filename.setText(f"Selected File: {selected_file_name}")
        print(predicted_label_string)

        if predicted_label_string == "0%~20%":
            self.circle_color.setStyleSheet("border-radius: 75px;background: rgb(0,0,0);")
            status = "損壞"
            
            comment = "設備已損壞，立即暫停使用"
        elif predicted_label_string == "21%~40%":
            self.circle_color.setStyleSheet("border-radius: 75px;background: rgb(255,0,0);")
            status = "異常"
            
            comment = "設備異常，需停機檢查"
            
        elif predicted_label_string == "41%~60%":
            self.circle_color.setStyleSheet("border-radius: 75px;background: rgb(255,170,0);")
            status = "損耗"
            
            comment ="設備有損耗，需定期保養"
        elif predicted_label_string == "61%~80%":
            self.circle_color.setStyleSheet("border-radius: 75px;background: rgb(0,222,0);")
            status = "正常"
            
            comment ="設備狀況正常，需定期保養"
        elif predicted_label_string == "81%~100%":
            self.circle_color.setStyleSheet("border-radius: 75px;background: rgb(0,170,255);")
            status = "良好"
            
            comment ="設備狀況良好，建議定期保養"
        # 设置状态文本
        self.state.setText(f"狀態: {status}") 
        self.comment.setText(comment)
         
        
        # 將STFT圖添加到CNN子視窗
        image_page = QWidget(self.stackedWidget)  # 将self.stackedWidget作为父对象
        layout = QVBoxLayout() #新建一個布局，用於添加到stackedWidget裡
        pixmap = QPixmap(images_stfted)# 新建QPixmap物件並設置為STFT圖
        if not pixmap.isNull():
            label = QLabel(self)
            label.setPixmap(pixmap)
            layout.addWidget(label)
        image_page.setLayout(layout)  # 将布局设置给image_page
        self.stackedWidget.addWidget(image_page)  # 添加image_page到stackedWidget3
        self.stackedWidget.setCurrentIndex(self.stackedWidget.indexOf(image_page))  # 显示image_page