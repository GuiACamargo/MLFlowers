import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Variável para controlar a execução na GPU
run_with_gpu = False

def load_images_and_labels(data_dir, image_size=(64, 64)):
    """
    Carrega imagens e rótulos do diretório especificado.

    Parâmetros:
    data_dir (str): Caminho para o diretório que contém subdiretórios de classes de imagens.
    image_size (tuple): Tamanho para redimensionar as imagens.

    Retorna:
    images (numpy array): Matriz de imagens carregadas e redimensionadas.
    labels (numpy array): Matriz de rótulos correspondentes.
    """
    images = []
    labels = []
    class_names = os.listdir(data_dir)
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                try:
                    # Carrega a imagem e redimensiona para o tamanho especificado
                    image = load_img(image_path, target_size=image_size)
                    # Converte a imagem para um array numpy
                    image = img_to_array(image)
                    # Adiciona a imagem à lista de imagens
                    images.append(image)
                    # Adiciona o nome da classe à lista de rótulos
                    labels.append(class_name)
                except Exception as e:
                    # Imprime erro caso não consiga carregar a imagem
                    print(f"Error loading image {image_path}: {e}")
    return np.array(images), np.array(labels)

# Redirecionar a saída para um arquivo de log para evitar problemas de codificação
import sys
sys.stdout = open('output.log', 'w', encoding='utf-8')

# Carregar o dataset
data_dir = 'flowers'  # Substitua pelo caminho para o diretório das flores
images, labels = load_images_and_labels(data_dir)

# Normalizar as imagens para o intervalo [0, 1]
images = images / 255.0

# Codificar os rótulos de string para inteiros
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Dividir o dataset em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Verificar o shape dos dados carregados e divididos
print(f"Shape das imagens de treinamento: {X_train.shape}")
print(f"Shape das imagens de teste: {X_test.shape}")
print(f"Shape das labels de treinamento: {y_train.shape}")
print(f"Shape das labels de teste: {y_test.shape}")

# Definir a arquitetura da CNN
model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),  # Definindo a camada de entrada com o tamanho da imagem
    layers.Conv2D(32, (3, 3), activation='relu'),  # Primeira camada convolucional com 32 filtros e ativação ReLU
    layers.MaxPooling2D((2, 2)),  # Primeira camada de pooling para reduzir a dimensionalidade espacial
    layers.Conv2D(64, (3, 3), activation='relu'),  # Segunda camada convolucional com 64 filtros e ativação ReLU
    layers.MaxPooling2D((2, 2)),  # Segunda camada de pooling
    layers.Conv2D(64, (3, 3), activation='relu'),  # Terceira camada convolucional com 64 filtros e ativação ReLU
    layers.Flatten(),  # Achatar a entrada para uma dimensão
    layers.Dense(64, activation='relu'),  # Camada totalmente conectada com 64 neurônios e ativação ReLU
    layers.Dropout(0.5),  # Dropout para regularização e evitar overfitting
    layers.Dense(5, activation='softmax')  # Camada de saída com 5 neurônios (uma para cada classe) e ativação softmax
])

# Compilar o modelo com otimizador Adam e função de perda sparse categorical crossentropy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Adicionando Data Augmentation para gerar mais dados de treinamento
datagen = ImageDataGenerator(
    rotation_range=20,  # Rotaciona as imagens aleatoriamente em até 20 graus
    width_shift_range=0.2,  # Desloca aleatoriamente as imagens horizontalmente em até 20% da largura
    height_shift_range=0.2,  # Desloca aleatoriamente as imagens verticalmente em até 20% da altura
    horizontal_flip=True  # Realiza espelhamento horizontal das imagens
)
datagen.fit(X_train)

# Early stopping para evitar overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Contexto para escolher dispositivo (GPU ou CPU)
device = '/GPU:0' if run_with_gpu and tf.test.is_gpu_available() else '/CPU:0'

# Treinando o modelo com data augmentation no dispositivo escolhido
start_time = time.time()
with tf.device(device):
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        epochs=50, validation_data=(X_test, y_test),
                        callbacks=[early_stopping])
training_time = time.time() - start_time
print(f'Tempo de treinamento: {training_time:.2f} segundos')

# Avaliar o modelo no dispositivo escolhido
with tf.device(device):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

def load_and_preprocess_image(image_path, image_size=(64, 64)):
    """
    Carrega e pré-processa uma única imagem para previsão.

    Parâmetros:
    image_path (str): Caminho para a imagem a ser carregada.
    image_size (tuple): Tamanho para redimensionar a imagem.

    Retorna:
    image (numpy array): Imagem pré-processada e redimensionada.
    """
    image = load_img(image_path, target_size=image_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalizar a imagem
    return np.expand_dims(image, axis=0)  # Adicionar uma dimensão extra para corresponder ao formato esperado pelo modelo

# Caminhos das imagens de teste para diferentes classes de flores
path_sunflower = 'flowers/sunflower/29972905_4cc537ff4b_n.jpg'
path_daisy = 'flowers/daisy/525780443_bba812c26a_m.jpg'
path_rose = 'flowers/rose/466486216_ab13b55763.jpg'
path_tulip = 'flowers/tulip/130685040_3c2fcec63e_n.jpg'
path_dandelion = 'flowers/dandelion/160456948_38c3817c6a_m.jpg'

# Carregar e pré-processar as imagens de teste
test_sunflower = load_and_preprocess_image(path_sunflower)
test_daisy = load_and_preprocess_image(path_daisy)
test_rose = load_and_preprocess_image(path_rose)
test_tulip = load_and_preprocess_image(path_tulip)
test_dandelion = load_and_preprocess_image(path_dandelion)

# Fazer previsões usando o modelo treinado
start_time = time.time()
with tf.device(device):
    predictions_sunflower = model.predict(test_sunflower)
    predictions_daisy = model.predict(test_daisy)
    predictions_rose = model.predict(test_rose)
    predictions_tulip = model.predict(test_tulip)
    predictions_dandelion = model.predict(test_dandelion)
prediction_time = time.time() - start_time
print(f'Tempo de previsão: {prediction_time:.2f} segundos')

# Obter as classes previstas com base nas previsões
predicted_class_sunflower = label_encoder.classes_[np.argmax(predictions_sunflower)]
predicted_class_daisy = label_encoder.classes_[np.argmax(predictions_daisy)]
predicted_class_rose = label_encoder.classes_[np.argmax(predictions_rose)]
predicted_class_tulip = label_encoder.classes_[np.argmax(predictions_tulip)]
predicted_class_dandelion = label_encoder.classes_[np.argmax(predictions_dandelion)]

# Mostrar as classes previstas para cada imagem de teste
print(f'O esperado é Sunflower. A imagem é prevista como: {predicted_class_sunflower}')
print(f'O esperado é Daisy. A imagem é prevista como: {predicted_class_daisy}')
print(f'O esperado é Rose. A imagem é prevista como: {predicted_class_rose}')
print(f'O esperado é Tulip. A imagem é prevista como: {predicted_class_tulip}')
print(f'O esperado é Dandelion. A imagem é prevista como: {predicted_class_dandelion}')

# Restaurar a saída padrão do console
sys.stdout.close()
sys.stdout = sys.__stdout__

# Plotar os gráficos de acurácia e perda durante o treinamento e validação
plt.figure(figsize=(12, 4))

# Plotar a acurácia de treinamento e validação
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plotar a perda de treinamento e validação
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
