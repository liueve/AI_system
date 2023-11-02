#呼叫主視窗，並設定系統字體
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont,QFontDatabase

from ALL_UI import MyMainWindow

#匯入字體庫
font_path = r"GenJyuuGothicX-Medium.ttf"

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 将字体加载到应用程序
    font_id = QFontDatabase.addApplicationFont(font_path)

    if font_id != -1:
        font_family = QFontDatabase.applicationFontFamilies(font_id)
        font = QFont(font_family[0], 12)  # 指定字体大小，这里设置为12
        QApplication.setFont(font)
    # ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------

    main_window = MyMainWindow()
    main_window.show()
    sys.exit(app.exec_())