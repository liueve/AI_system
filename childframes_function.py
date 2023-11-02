import numpy as np
import joblib
import keras
from keras.models import load_model
from scipy.signal import stft
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class childframe_func():
#AI模型區域
# ------------------------------------------------------------------------------------------------------------
    def load_model_randomforest(self):
        loaded_model_filepath = r"random_forest_model_v5.pkl"
        rf_classifier_model = joblib.load(loaded_model_filepath)

        # 預測
        new_label_probabilities = rf_classifier_model.predict_proba(self.calc_data)
        reverse_label_mapping = {0: '0%~20%', 1: '21%~40%',2:'41%~60%',3:'61%~80%',4:'81%~100%'}

        # 找出分數最高的標籤
        predicted_label_index = np.argmax(new_label_probabilities)
        predicted_label = rf_classifier_model.classes_[predicted_label_index]
        # 將索引轉換為字串標籤
        predicted_label_string = reverse_label_mapping[predicted_label]

        #print(f"預測標籤：{predicted_label_string}")
        #print(f"機率：{new_label_probabilities[0, predicted_label_index]:.4f}")
        self.predicted_label_string = predicted_label_string
    # ------------------------------------------------------------------------------------------------------------

    def load_model_SVM(self):
        loaded_model_filepath = r"SVM_model_v3.pkl"
        #模型
        rf_classifier_model = joblib.load(loaded_model_filepath)
        # 預測
        new_label_probabilities = rf_classifier_model.predict_proba(self.calc_data_B)
        
        reverse_label_mapping = {"Abnormal": '41%~60%', "Normal": '91%~100%'}

        # 找出分數最高的標籤
        predicted_label_index = np.argmax(new_label_probabilities)
        predicted_label = rf_classifier_model.classes_[predicted_label_index]
        # 將索引轉換為字串標籤
        predicted_label_string = reverse_label_mapping[predicted_label]

        #print(f"預測標籤：{predicted_label_string}")
        #print(f"機率：{new_label_probabilities[0, predicted_label_index]:.4f}")
        self.predicted_label_string = predicted_label_string


    def load_model_CNN(self):
            
        # 指定模型路径
            model_path = r"CNN_model.h5"

            # 加载模型
            model = load_model(model_path)

            # 指定目标图像尺寸
            target_size = (256, 256)  # 设置为 256x256

            # 加载图像
            img = load_img(self.images_stft, target_size=target_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # 添加批量维度

            # 进行预测
            prediction = model.predict(img_array)

            # 获取预测的标签
            predicted_label = np.argmax(prediction)

            # 将预测标签映射到相应的建议
            if predicted_label == 0:
                suggestion = "21%~40%"
            elif predicted_label == 1:
                suggestion = "61%~80%"
            elif predicted_label == 2:
                suggestion = "41%~60%"
            elif predicted_label == 3:
                suggestion = "0%~20%"
            elif predicted_label == 4:
                suggestion = "80%~100%"
            else:
                suggestion = "未知"

            # 打印结果
            #print(f" Prediction: {predicted_label}, Suggestion: {suggestion}")
            self.predicted_label_string = suggestion