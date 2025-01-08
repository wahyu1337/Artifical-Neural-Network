import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential # Type: ignore
from tensorflow.keras.layers import Dense # Type: ignore
from tensorflow.keras.layers import Input # Type: ignore
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

file_path ='data_padi.csv'
data = pd.read_csv(file_path)

label_encoders = {}
categorical_columns = ['jenis_padi', 'metode_tanam', 'ketersediaan_air', 'serangan_hama', 'cuaca', 'label']

for col in categorical_columns:
  le = LabelEncoder()
  data[col] = le.fit_transform(data[col])
  label_encoders[col] = le

fitur = data.drop(columns=['label'])
target = data['label']

scaler = StandardScaler()
fitur_scaled = scaler.fit_transform(fitur)

fitur_train, fitur_test, target_train, target_test = train_test_split(fitur_scaled, target, test_size=0.2, random_state=42, stratify=target)

# Pemrosesan / Inisialisasi FNN & Layer
model = Sequential([
  Input(shape=(fitur_train.shape[1],)), # input layer
  Dense(256, activation='relu'), # hidden layer 1 dengan 256 neuron
  Dense(128, activation='relu'), # hidden layer 2 dengan 128 neuron
  Dense(len(np.unique(target)), activation='softmax') # output layer dengan jumlah neuron seusai jumlah target
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# proses training
history = model.fit(fitur_train, target_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1) 
# prediksi data uji
target_pred = np.argmax(model.predict(fitur_test), axis=1)
# perhitungan metrik evaluasi
accuracy = accuracy_score(target_test, target_pred)
precision = precision_score(target_test, target_pred, average='weighted')
recall = recall_score(target_test, target_pred, average='weighted')
f1 = f1_score(target_test, target_pred, average='weighted')
print('=================================')
print("Accuracy: "+str(accuracy))
print("Precision: "+str(precision))
print("Recall: "+str(recall))
print("F1 Score: "+str(f1_score))

# form prediksi
def predict_input(model, scaler, label_encoders):
  print("================================")
  print("masukkan nilai untuk fitur berikut")
  input_data = []
  for col in fitur.columns:
    value = input(col+" : ")
    if col in categorical_columns[:-1]:
      value = label_encoders[col].transform([value])[0]
    else:
      value = float(value)
    input_data.append(value)
  input_df = pd.DataFrame([input_data], columns=fitur.columns)
  input_scaled = scaler.transform(input_df)
  prediction = np.argmax(model.predict(input_scaled), axis=1)
  predicted_label = label_encoders['label'].inverse_transform(prediction)[0]
  print("Hasil prediksi: "+predicted_label)
predict_input(model, scaler, label_encoders)