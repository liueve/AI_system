#主視窗後端功能
from childframes_function import childframe_func

import os
import pandas as pd
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, stft
#import pickle
import matplotlib.pyplot as plt
import seaborn as sns  # 引入Seaborn模組
import mplcyberpunk
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTreeWidgetItem,QStackedWidget
from PyQt5.QtWidgets import QMessageBox
from scipy.signal import hann
from io import BytesIO
from PIL import Image

#繪製圖表使用Cyberpunk渲染
#在这里指定Cyberpunk样式文件的路径
# ------------------------------------------------------------------------------------------------------------
cyberpunk_style_file = r"mplcyberpunk-main\mplcyberpunk\data\cyberpunk.mplstyle"
plt.style.use(cyberpunk_style_file)
# ------------------------------------------------------------------------------------------------------------

#主視窗後端功能
class mainframe_func(childframe_func):
    # ------------------------------------------------------------------------------------------------------------
        
    def on_comboBox_1_currentIndexChanged(self, index):
        # 在comboBox_1的索引变化事件中检查并禁用/启用XY按钮
        selected_file_name = self.comboBox_1.currentText()
        if selected_file_name in self.dataframes:
            selected_df = self.dataframes[selected_file_name]
            if len(selected_df.columns) == 2:
                self.X_btn.setEnabled(False)
                self.Y_btn.setEnabled(False)
            else:
                self.X_btn.setEnabled(True)
                self.Y_btn.setEnabled(True)
            
# ------------------------------------------------------------------------------------------------------------

    def open_files(self):
        self.comboBox_1.clear()
        self.file_paths, _ = QFileDialog.getOpenFileNames(self, "Open Files", "", "All Files (*)")
        if self.file_paths:
            file_count = len(self.file_paths)
            step = 100 / file_count  # 计算每个文件加载的进度步长
            current_progress = 0  # 当前进度

            for file_path in self.file_paths:
                file_extension = os.path.splitext(file_path)[-1][1:]
                if file_extension == "txt":
                    self.txt(file_path)
                elif file_extension == "mat":
                    self.mat(file_path)
                elif file_extension == "csv":
                    self.csv(file_path)
                else:
                    print(f"Unsupported file type: {file_extension}")

                current_progress += step  # 增加当前进度
                self.progressBar_1.setValue(int(current_progress))  # 更新进度条
                QCoreApplication.processEvents()  # 让应用程序处理事件，确保进度条及时更新

            self.progressBar_1.setValue(100)  # 完成后将进度条设置为100%
# ------------------------------------------------------------------------------------------------------------
    #處理.txt檔案
    def txt(self, file_path):
        txt_file_path = file_path
        folder_path = os.path.dirname(txt_file_path)
        data_array = []

        with open(txt_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                numbers = line.strip().split('\t')
                numbers = [float(num) for num in numbers]
                data_array.append(numbers)

        transposed_data = list(zip(*data_array))
        my_csv = {
            "Time": transposed_data[0],
            "X": transposed_data[1],
            "Y": transposed_data[2],
            "Z": transposed_data[3]
        }
        file_name = os.path.splitext(os.path.basename(txt_file_path))[0]
        self.dataframes[file_name] = pd.DataFrame(my_csv)  # 使用 self.dataframes
        self.comboBox_1.addItem(file_name)

    # -----------------------------------------------------------------------------------------------------
    #處理.mat檔案
    def mat(self, file_path):
        mat_data = sio.loadmat(file_path)
        times = mat_data["times"]
        data = mat_data["data"]
        
        time_value = [j[0] for j in times]
        x_value = [x[0] for x in data]
        y_value = [y[1] for y in data]  # 注意这里的索引修改为 [0]
        z_value = [z[2] for z in data]  # 注意这里的索引修改为 [0]

        my_mat = {
            "Time": time_value,
            "X": x_value,
            "Y": y_value,
            "Z": z_value
        }
        #print(mat_data["data"])
        df = pd.DataFrame(my_mat)

        file_name = os.path.splitext(os.path.basename(file_path))[0]
        self.dataframes[file_name] = df
        self.comboBox_1.addItem(file_name)

# -----------------------------------------------------------------------------------------------------
    #處理.csv檔案
    def csv(self, file_path):
        csv_file=open(file_path)
        data = pd.read_csv(csv_file)
        if len(data.columns) == 5:
            data=data.drop(data.columns[-1],axis=1)
            #更改欄位名稱
            columns=["Time","X","Y","Z"]
            #更改欄位名稱
            data.columns.values[:]=columns
            

            file_name = os.path.splitext(os.path.basename(file_path))[0]
            self.dataframes[file_name] = data
            self.comboBox_1.addItem(file_name)
        
        elif len(data.columns) == 1:
            data = pd.read_csv(file_path, sep=';')
            data = data.drop(columns=['DeviceID', 'DeviceName','VESID', 'Type', 'Name', 'Unit', 'SampleRate','Unnamed: 8'])
            time_column = pd.Series([i / 50000 for i in range(len(data))])
            data["Time"] = time_column
            
            
            #print(data.columns)
            
            columns = {"TIME": "Time", data.columns[0]: "Z"}
            data.rename(columns=columns, inplace=True)
            

            file_name = os.path.splitext(os.path.basename(file_path))[0]
            self.dataframes[file_name] = data
            self.comboBox_1.addItem(file_name)
# -----------------------------------------------------------------------------------------------------
    #清除所有顯示數據
    def clear_data_output(self):
        self.comboBox_1.clear() #清空下拉選單
        self.treeWidget.clear() #清空統計值
        self.plot_widgets.clear() #清空圖片列表
        #清空時域圖
        while self.stackedWidget.count() > 0: 
            widget = self.stackedWidget.widget(0)
            self.stackedWidget.removeWidget(widget)
            widget.deleteLater()
        #清空時頻圖
        while self.stackedWidget_2.count() > 0:
            widget = self.stackedWidget_2.widget(0)
            self.stackedWidget_2.removeWidget(widget)
            widget.deleteLater()
        
   
# -----------------------------------------------------------------------------------------------------
    #濾波器
    def high_pass_filter(self, data, cutoff_freq, sampling_rate):
        nyquist_freq = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = butter(1, normal_cutoff, btype='high', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
# -----------------------------------------------------------------------------------------------------
   
    def plot_selected_data(self):
        self.plot_raw_data()
        self.plot_fft_data()
        self.calculate()
        
# ------------------------------------------------------------------------------------------------------------
    
    def plot_raw_data(self):
        selected_file_name = self.comboBox_1.currentText()
        if selected_file_name in self.dataframes:
            selected_df = self.dataframes[selected_file_name]

            # 獲取所選軸的原始數據
            selected_time = selected_df["Time"]
            selected_axis_data = selected_df[self.selected_axis]

            # 創建一個新的widget和布局
            raw_data_widget = QWidget()
            raw_data_layout = QVBoxLayout()
            raw_data_widget.setLayout(raw_data_layout)

            # 創建一個新的Matplotlib圖形對象fig
            fig = plt.figure(figsize=(8, 6))

            # 使用mplcyberpunk風格
            plt.style.use("cyberpunk")

            # 繪製原始數據，而不是經過Hanning窗口處理，並將線條顏色設置為#00BCD4
            plt.plot(selected_time, selected_axis_data, label=f"{self.selected_axis} Axis ", color='#00BCD4')
            plt.xlabel("Time",fontsize=12)
            plt.ylabel(f"{self.selected_axis} Axis",fontsize=12)
            # 設置刻度標籤文字大小
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            plt.legend(fontsize=16)

            # 添加標題
            plt.title(f"{self.selected_axis} {selected_file_name}", fontsize=16)

            # 使用mplcyberpunk風格
            mplcyberpunk.add_glow_effects()

            # 添加圖表到布局
            raw_data_layout.addWidget(fig.canvas)

            # 添加 raw_data_widget 到 stackedWidget
            self.stackedWidget.addWidget(raw_data_widget)

            # 添加繪製的圖表窗口到列表
            self.plot_widgets.append(raw_data_widget)

            # 切換到原始數據圖表
            self.stackedWidget.setCurrentWidget(raw_data_widget)
# ------------------------------------------------------------------------------------------------------------
    def calculate_fft(self, data):
        fft_data = np.fft.fft(data)
        frequencies = np.fft.fftfreq(len(data), d=(data.index[1] - data.index[0]).total_seconds())  # 获取时间间隔
        amplitudes = np.abs(fft_data)

        return data, frequencies, amplitudes  # 返回原始数据、频率和振幅
# ------------------------------------------------------------------------------------------------------------
    def plot_fft_data(self):
        selected_file_name = self.comboBox_1.currentText()
        if selected_file_name in self.dataframes:
            selected_df = self.dataframes[selected_file_name]

            if len(selected_df.columns) == 4 or len(selected_df.columns) == 2:
                # 获取所选轴的数据
                selected_time = selected_df["Time"]
                selected_axis_data = selected_df[self.selected_axis]

                # 获取FFT数据
                fft_data = np.fft.fft(selected_axis_data)
                frequencies = np.fft.fftfreq(len(selected_time), selected_time[1] - selected_time[0])
                amplitudes = np.abs(fft_data)

                # 仅保留正频率分量
                positive_frequencies = frequencies[:len(frequencies) // 2]
                positive_amplitudes = amplitudes[:len(amplitudes) // 2]

                # 创建新的widget和布局
                fft_widget = QWidget()
                fft_layout = QVBoxLayout()
                fft_widget.setLayout(fft_layout)

                # 创建一个新的Matplotlib图形对象fig
                fig = plt.figure(figsize=(8, 6))

                # 使用mplcyberpunk风格
                plt.style.use("cyberpunk")

                # 绘制FFT图像，将线条颜色设置为#00BCD4
                plt.plot(positive_frequencies, positive_amplitudes, label=f"{self.selected_axis} Axis FFT", color='#00BCD4')
                plt.xlabel("Frequency (Hz)",fontsize=12)
                plt.ylabel("Amplitude",fontsize=12)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
        
                plt.legend(fontsize=16)                
                

                # 添加标题
                plt.title(f"{self.selected_axis}  {selected_file_name}", fontsize=16)


                # 使用mplcyberpunk风格
                mplcyberpunk.add_glow_effects()

                # 添加图表到布局
                fft_layout.addWidget(fig.canvas)

                # 将新的widget添加到stackedWidget_2
                self.stackedWidget_2.addWidget(fft_widget)

                # 将绘制的图表窗口添加到列表
                self.plot_widgets.append(fft_widget)

                # 切换到FFT图像
                self.stackedWidget_2.setCurrentWidget(fft_widget)

            else:
                # 如果列数不是4也不是2，给出错误提示
                QMessageBox.critical(self, "Error", "Invalid number of columns in selected DataFrame")
                return

# ------------------------------------------------------------------------------------------------------------

    def select_x_axis(self):
        if self.X_btn.isEnabled():
            self.selected_axis = "X"
            self.plot_selected_data()

    def select_y_axis(self):
        if self.Y_btn.isEnabled():
            self.selected_axis = "Y"
            self.plot_selected_data()

    def select_z_axis(self):
        self.selected_axis = "Z"
        self.plot_selected_data()

# ------------------------------------------------------------------------------------------------------------
    def calculate(self):
        selected_file_name = self.comboBox_1.currentText()
        self.selected_file_name = selected_file_name

        if selected_file_name in self.dataframes:
            selected_df = self.dataframes[selected_file_name]
            #print(selected_df)

            self.calc_csv = {}

            self.treeWidget.clear()  # 清空樹狀結構
            root_item = QTreeWidgetItem(self.treeWidget)
            root_item.setText(0, "Statistics")

            if len(selected_df.columns) == 4: #若數據包含X, Y, Z軸
                #print("YES")
                x_values = selected_df["X"]
                y_values = selected_df["Y"]
                z_values = selected_df["Z"]
                ABS_value_X = [abs(x) for x in x_values]
                ABS_value_Y = [abs(y) for y in y_values]
                ABS_value_Z = [abs(z) for z in z_values]

                hanning_window = hann(len(x_values))
                x_values_windowed = x_values * hanning_window
                y_values_windowed = y_values * hanning_window
                z_values_windowed = z_values * hanning_window

                fft_x = np.fft.fft(x_values_windowed)
                fft_y = np.fft.fft(y_values_windowed)
                fft_z = np.fft.fft(z_values_windowed)

                # 计算X、Y、Z的MEAN、MIN、MAX和RMS值
                # 计算X、Y、Z的统计值
                x_stats = x_values.describe()
                y_stats = y_values.describe()
                z_stats = z_values.describe()

                # 计算RMS
                rms_x = np.sqrt(np.mean(np.square(x_values)))
                rms_y = np.sqrt(np.mean(np.square(y_values)))
                rms_z = np.sqrt(np.mean(np.square(z_values)))

                # 计算Crest
                crest_x = max(np.abs(x_values)) / rms_x
                crest_y = max(np.abs(y_values)) / rms_y
                crest_z = max(np.abs(z_values)) / rms_z

                # 计算Kurtosis和Skewness
                kurtosis_x = x_values.kurtosis()
                kurtosis_y = y_values.kurtosis()
                kurtosis_z = z_values.kurtosis()

                skewness_x = x_values.skew()
                skewness_y = y_values.skew()
                skewness_z = z_values.skew()

                # 计算平均能量
                # 计算FFT之后的"average energy"（不经过高通滤波器）
                energy_x = np.mean(np.square(np.abs(fft_x)))
                energy_y = np.mean(np.square(np.abs(fft_y)))
                energy_z = np.mean(np.square(np.abs(fft_z)))

                # 格式化统计结果字符串 describe()、RMS、Crest、Kurt、Skew、average energy
                stats_data = {
                    "X-axis": {
                        "describe()": x_stats,
                        "RMS": rms_x,
                        "Crest": crest_x,
                        "Kurt": kurtosis_x,
                        "Skew": skewness_x,
                        "average energy": energy_x,
                    },
                    "Y-axis": {
                        "describe()": y_stats,
                        "RMS": rms_y,
                        "Crest": crest_y,
                        "Kurt": kurtosis_y,
                        "Skew": skewness_y,
                        "average energy": energy_y,
                    },
                    "Z-axis": {
                        "describe()": z_stats,
                        "RMS": rms_z,
                        "Crest": crest_z,
                        "Kurt": kurtosis_z,
                        "Skew": skewness_z,
                        "average energy": energy_z
                    }
                }
                self.calc_csv["axis"] = ["X-axis", "Y-axis", "Z-axis"]
                self.calc_csv["count"] = [x_stats["count"], y_stats["count"], z_stats["count"]]
                self.calc_csv["mean"] = [x_stats["mean"], y_stats["mean"], z_stats["mean"]]
                self.calc_csv["std"] = [x_stats["std"], y_stats["std"], z_stats["std"]]
                self.calc_csv["min"] = [x_stats["min"], y_stats["min"], z_stats["min"]]
                self.calc_csv["25%"] = [x_stats["25%"], y_stats["25%"], z_stats["25%"]]
                self.calc_csv["50%"] = [x_stats["50%"], y_stats["50%"], z_stats["50%"]]
                self.calc_csv["75%"] = [x_stats["75%"], y_stats["75%"], z_stats["75%"]]
                self.calc_csv["max"] = [x_stats["max"], y_stats["max"], z_stats["max"]]
                self.calc_csv["RMS"] = [rms_x, rms_y, rms_z]
                self.calc_csv["Crest"] = [crest_x, crest_y, crest_z]
                self.calc_csv["Kurt"] = [kurtosis_x, kurtosis_y, kurtosis_z]
                self.calc_csv["Skew"] = [skewness_x, skewness_y, skewness_z]
                self.calc_csv["average energy"] = [energy_x, energy_y, energy_z]

                


                
                for axis, values in stats_data.items():
                    axis_item = QTreeWidgetItem(root_item)
                    axis_item.setText(0, axis)
                    for stat_name, stat_value in values.items():
                        stat_item = QTreeWidgetItem(axis_item)
                        stat_item.setText(0, f"{stat_name}: {stat_value}")  # 注意這裡的寫法
                        root_item.addChild(axis_item)
                            
                            
            elif len(selected_df.columns) == 2:
                #print("YES 2")

                
                z_values = selected_df["Z"]

                hanning_window = hann(len(z_values))
                z_values_windowed = z_values * hanning_window
                fft_z = np.fft.fft(z_values_windowed)

                z_stats = z_values.describe()

                # 计算RMS
                rms_z = np.sqrt(np.mean(np.square(z_values)))

                # 计算Crest
                crest_z = max(np.abs(z_values)) / rms_z

                # 计算Kurtosis和Skewness
                kurtosis_z = z_values.kurtosis()
                skewness_z = z_values.skew()

                # 计算平均能量
                energy_z = np.mean(np.square(np.abs(fft_z)))

                stats_data = {
                    "Z-axis": {
                        "describe()": z_stats,
                        "RMS": rms_z,
                        "Crest": crest_z,
                        "Kurt": kurtosis_z,
                        "Skew": skewness_z,
                        "average energy": energy_z
                    }
                }
                self.calc_csv["axis"] = ["Z-axis"]
                self.calc_csv["count"] = [z_stats["count"]]
                self.calc_csv["mean"] = [z_stats["mean"]]
                self.calc_csv["std"] = [z_stats["std"]]
                self.calc_csv["min"] = [z_stats["min"]]
                self.calc_csv["25%"] = [z_stats["25%"]]
                self.calc_csv["50%"] = [z_stats["50%"]]
                self.calc_csv["75%"] = [z_stats["75%"]]
                self.calc_csv["max"] = [z_stats["max"]]
                self.calc_csv["RMS"] = [rms_z]
                self.calc_csv["Crest"] = [crest_z]
                self.calc_csv["Kurt"] = [kurtosis_z]
                self.calc_csv["Skew"] = [skewness_z]
                self.calc_csv["average energy"] = [energy_z]

                z_item = QTreeWidgetItem(root_item)
                z_item.setText(0, "Z-axis")
                for stat_name, stat_value in stats_data["Z-axis"].items():
                    stat_item = QTreeWidgetItem(z_item)
                    stat_item.setText(0, f"{stat_name}: {stat_value}")  # 注意這裡的寫法
                root_item.addChild(z_item)
                self.calc_data = pd.DataFrame(self.calc_csv)
                self.calc_data = self.calc_data.drop("axis", axis=1)
                #print(self.calc_data)

                self.calc_data_B = self.calc_data.drop(["std","min", "count", "mean", "25%", "50%", "75%", "max"], axis=1)
                #print(self.calc_data)

        #   -------------時頻圖----存給模型判斷的---------------------------------------------------------------------------------
                fig, ax = plt.subplots(figsize=(256/80, 256/80), dpi=80)
                # Compute STFT
                f, t, Zxx = stft(z_values_windowed, fs=1/(selected_df['Time'][1]-selected_df['Time'][0]), nperseg=1024, noverlap=512, nfft=None)
                ax.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='inferno')

                # 隐藏坐标轴刻度和标签
                ax.set_xticks([])
                ax.set_yticks([])

                # 将STFT图像存储在内存中
                buf_stft = BytesIO()
                plt.savefig(buf_stft, format='png')
                buf_stft.seek(0)
                
                # 将STFT图像保存到文件
                image = selected_file_name + "_stft.png"
                with open(image, 'wb') as f:
                    f.write(buf_stft.read())
                #print(image)
                self.images_stft = image

        #    -------------時頻圖 分隔線-----要顯示在UI上的-------------------------------------------
                # 這邊是用線性分配Y軸的間距  可是會有0和100標籤重疊問題 所以採用下方log scale來分配兼具
            
                fig, ax = plt.subplots(figsize=(400/80, 400/80), dpi=90)
                # Compute STFT
                f, t, Zxx = stft(z_values_windowed, fs=1/(selected_df['Time'][1]-selected_df['Time'][0]), nperseg=1024, noverlap=512, nfft=None)
                ax.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='inferno')

                # 设置Y轴刻度范围
                plt.ylim(0, 25000)  # 将刻度范围调整为包含0和100

                # 设置Y轴刻度标签
                yticks = [0, 100, 1000, 5000, 10000, 15000, 20000, 25000]
                plt.yticks(yticks,fontsize=8)
                
                
                plt.xticks(fontsize=8)
                

                plt.xlabel('Time (s)',fontsize=10)
                plt.ylabel('Frequency (Hz)',fontsize=10)

                pcm = ax.pcolormesh(t, f, 10 * np.log10(np.abs(Zxx)), shading='gouraud', cmap='Blues')  # 使用10 * log10来转换为分贝
                # 添加颜色条
                cbar = plt.colorbar(pcm)
                cbar.set_label('dB', rotation=0)  # 设置颜色条标签，并可以旋转标签
                
                buf_original = BytesIO()
                plt.savefig(buf_original, format='png')
                buf_original.seek(0)
                
                # 将原始图像保存到文件或保留在内存中
                images_stfted = selected_file_name + "_original.png"
                with open(images_stfted, 'wb') as f:
                    f.write(buf_original.read())
                #print(images_stfted)
                self.images_stfted = images_stfted
        
# ------------------------------------------------------------------------------------------------------------
    #儲存資料列表功能
    def save_item_click(self, item):
        selected_text = item.text()

        if selected_text == self.listWidget_1.item(0).text():
            # 第一個選項的功能代碼
            #print("第一個選項被單擊了")
            self.save_raw_data()
        elif selected_text == self.listWidget_1.item(1).text():
            # 第二個選項的功能代碼
            #print("第二個選項被單擊了")
            self.save_fft()
        elif selected_text == self.listWidget_1.item(2).text():
            # 第三個選項的功能代碼
            #print("第三個選項被單擊了")
            self.save_calc_data()
        elif selected_text == self.listWidget_1.item(3).text():
            # 第三個選項的功能代碼
            #print("第三個選項被單擊了")
            self.save_pic()

    def AI_item_click(self, item):
        from ALL_UI import RF_UiWindow,SVM_UiWindow,CNN_UiWindow
    # 獲取被單擊的選項的文本
        selected_text = item.text()
                
        if selected_text == self.listWidget_2.item(0).text():
            self.load_model_randomforest()
            self.model_name = "預測結果"
            model_name = self.model_name
            predicted_label_string = self.predicted_label_string
            selected_file_name = self.selected_file_name
            self.RF_ui = RF_UiWindow(selected_file_name,predicted_label_string,model_name)    
            
            self.RF_ui.show()
            
        elif selected_text == self.listWidget_2.item(1).text():
            self.load_model_SVM()
            model_name = self.model_name
            #self.model_name = "B模型預測結果"
            model_name = self.model_name
            predicted_label_string = self.predicted_label_string
            selected_file_name = self.selected_file_name
            self.svm_ui = SVM_UiWindow(selected_file_name,predicted_label_string)    
            
            self.svm_ui.show()

        elif selected_text == self.listWidget_2.item(2).text():
            self.load_model_CNN()
            model_name = self.model_name
            #self.model_name = "C模型預測結果"
            model_name = self.model_name
            predicted_label_string = self.predicted_label_string
            selected_file_name = self.selected_file_name
            images_stfted = self.images_stfted
            self.CNN_ui = CNN_UiWindow(selected_file_name, predicted_label_string, images_stfted)    
            
            self.CNN_ui.show()
    
    def save_raw_data(self):
        # 获取单击的项目文本
        selected_file_name = self.comboBox_1.currentText()

        # 执行保存操作，与之前的代码相同
        if selected_file_name in self.dataframes:
            selected_df = self.dataframes[selected_file_name]
            save_file_path, _ = QFileDialog.getSaveFileName(None, "Save Raw Data CSV", "", "CSV Files (*.csv)")
            if save_file_path:
                selected_df.to_csv(save_file_path, index=False)
                #print(f"Raw Data CSV 文件已保存至：{save_file_path}")




# ------------------------------------------------------------------------------------------------------------
    def save_fft(self):
        selected_file_name = self.comboBox_1.currentText()
        if selected_file_name in self.dataframes:
            selected_df = self.dataframes[selected_file_name]

            # 计算FFT并创建FFT DataFrame
            fft_data = np.fft.fft(selected_df[self.selected_axis])
            frequencies = np.fft.fftfreq(len(selected_df), selected_df["Time"][1] - selected_df["Time"][0])
            amplitudes = np.abs(fft_data)

            fft_df = pd.DataFrame({"Frequency": frequencies, "Amplitude": amplitudes})

            # 获取用户选择的保存路径和文件名
            save_file_path, _ = QFileDialog.getSaveFileName(self, "Save FFT Data CSV", "", "CSV Files (*.csv)")

            if save_file_path:
                # 将 FFT 数据保存为 CSV 文件
                fft_df.to_csv(save_file_path, index=False)

                print(f"FFT Data CSV 文件已保存至：{save_file_path}")

# ------------------------------------------------------------------------------------------------------------
    def save_calc_data(self):
    # 获取用户选择的保存路径和文件名
        save_file_path, _ = QFileDialog.getSaveFileName(self, "保存统计数据 CSV", "", "CSV 文件 (*.csv)")

        if save_file_path:
            # 将统计数据保存为 CSV 文件，你可以使用 pandas 来完成这个任务
            calc_data = pd.DataFrame(self.calc_csv)
            calc_data.to_csv(save_file_path, index=False)

            print(f"统计数据 CSV 文件已保存至：{save_file_path}")
# ------------------------------------------------------------------------------------------------------------
    def save_pic(self):
        selected_file_name = self.comboBox_1.currentText()
        # 获取用户选择的保存路径和文件名
        save_folder_path = QFileDialog.getExistingDirectory(self, "选择保存文件夹")

        if save_folder_path:
            for idx, plot_widget in enumerate(self.plot_widgets):
                # 假设文件名为 selected_file_name，不包括扩展名
                save_file_name = os.path.splitext(selected_file_name)[0]

                # 将文件名和索引组合在一起
                save_file_path = os.path.join(save_folder_path, f"{save_file_name}_{idx}.png")

                # 获取当前绘图窗口
                current_layout = plot_widget.layout()
                current_canvas = current_layout.itemAt(0).widget()

                # 将绘图保存为图片
                current_canvas.figure.savefig(save_file_path, dpi=300, bbox_inches='tight')

            print(f"已保存 {len(self.plot_widgets)} 张图片至文件夹：{save_folder_path}")
