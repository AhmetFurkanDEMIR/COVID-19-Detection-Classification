#!/usr/bin/env python
# -*- coding: utf-8 -*-



# veri çeşitlendirme
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image

# eniyileme
from keras import optimizers

# CNN katmanları
from keras.layers import Conv2D

# model tanımlamak
from keras.models import Sequential

# enbüyükleri biriktirme
from keras.layers import MaxPooling2D

# çıktı boyutunu düzenleme
from keras.layers import Flatten

# tamamen bağlı katman
from keras.layers import Dense

# iletim sönümü
from keras.layers import Dropout

# kayıpları ve başarımı görselleştirme
import matplotlib.pyplot as plt

from time import sleep as sl

# matris işlemleri
import numpy as np

# klasör ve işletim sistemi
import os, shutil



# --- modeli GPU üzerinden eğitiyoruz --- #

# tensrflow ve kerasın GPU versiyonu gerekli
# Nvdia GPU
# CUDA
# cuDNN
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# cpu üzerinden eğitimi gerçekleştirmek isterseni "-1" yapın
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



# ----------- veri ön işleme (veri klasör yollarını kodda tanımlama) ----------- #


# Test veri seti klasörü
train_dir = os.path.join("Train") 


# Doğrulama veri seti klasörü
validation_dir = os.path.join("Validation")


# Test veri seti klasörü
test_dir = os.path.join("Test")


# Eğitim veri setinde sağlıklı veriler
train_healthy_dir = os.path.join(train_dir,"healthy") 


# Eğitim veri setinde covid-19 veriler
train_infected_dir = os.path.join(train_dir,"infected")


# Doğrulama veri setinde sağlıklı veriler
validation_healthy_dir = os.path.join(validation_dir,"healthy")


# Doğrulama veri setinde covid-19 veriler
validation_infected_dir = os.path.join(validation_dir,"infected")


# Test veri setinde sağlıklı veriler
test_healthy_dir = os.path.join(test_dir,"healthy")


# Test veri setinde covid-19 veriler
test_infected_dir = os.path.join(test_dir,"infected")

os.system("clear")



# --- Veri seti çeşitlendirme --- #


# veri çeşitlendirme
train_datagen = ImageDataGenerator(

      # resim pixellerini 0,1 arasına sıkıştırma
      rescale=1./255,

      # derece cinsinden (0-180) resimlerin rastgele döndürülme açısı
      rotation_range=40,

      # resimlerin yatayda ve dikeyde kaydırılma oranları
      width_shift_range=0.2,

      # resimlerin yatayda ve dikeyde kaydırılma oranları
      height_shift_range=0.2,

      # burkma işlemi
      shear_range=0.2,

      # yakınlaştırma işlemi
      zoom_range=0.2,

      # dikeyde resim döndürme
      horizontal_flip=True,

      # işlemlerden sonra ortaya çıkan  fazla 
      # görüntü noktalarının nasıl doldurulacağını belirler
      fill_mode='nearest')


# test resimlerinde çeşitlendirme yapmıyoruz.
test_datagen = ImageDataGenerator(rescale=1./255)

# çeşitlendirilmiş verileri kullanmak (eğitim)
train_generator = train_datagen.flow_from_directory(

        # hedef dizin
        train_dir,

        # tüm resimler (150x150) olarak boyutlandırılacak
        target_size=(150, 150),

        # yığın boyutu
        batch_size=20,

        # binary_crossentropy kullandığımız için
        # ikili etiketler gerekiyor.
        class_mode='binary')

# çeşitlendirilmiş verileri kullanmak (doğrulama)
validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')

sl(5)



# --- Modelimiz ve sinir ağlarımız --- #


# modelimizi tanımladık
model = Sequential()


# evrişimli sinir ağı,
# alacağı resim boyutu = (150,150,3)
# filitreler 3x3 boyutuna sahip
# 32 boyutlı CNN 
model.add(Conv2D(32, (3,3), activation="elu", input_shape=(150,150,3)))


# en büyükleri biriktirmek.
# her MaxPooling katmanından sonra nitelik haritalarının boyutu yarıya düşer.
# girdi nitelik haritasından pencereler çıkarıp her kanalın en büyük değerini almaktır.
model.add(MaxPooling2D(2,2))

# evrişimli sinir ağı
model.add(Conv2D(64, (3,3), activation="elu"))

# en büyükleri biriktirme
model.add(MaxPooling2D(2,2))

# evrişimli sinir ağı
model.add(Conv2D(128, (3,3), activation="elu"))

# en büyükleri biriktirme
model.add(MaxPooling2D(2,2))

# evrişimli sinir ağı
model.add(Conv2D(256, (3,3), activation="elu"))

# en büyükleri biriktirme
model.add(MaxPooling2D(2,2))

# evrişimli sinir ağı
model.add(Conv2D(256, (3,3), activation="elu"))

# en büyükleri biriktirme
model.add(MaxPooling2D(2,2))


# 3B çıktıları 1B vektörlere düzenler
model.add(Flatten())


# tamamen bağlı katmanlar
model.add(Dense(512, activation="elu"))


# iletim sönümü : 
# modelin aşırı uydurma yapmasını engeller.
# Sinir ağlarının düzleştirilmesinde kullanılır.
# verdiğimiz orana göre elemanları sıfırlar.
model.add(Dropout(0.6))


# fonksiyonu sigmoid olarak kullanarak çıkan değeri [0,1] arasına sıkıştırdık
# çünki ikili sınıflandırma var (hasta, sağlıklı)
model.add(Dense(1,activation="sigmoid"))

# --- modeli derleme ve eğitme --- #


# modeli derleme aşaması
model.compile(loss='binary_crossentropy', # kayıp fonksiyonu

              # eniyileme:
              # ağımızın girdisi olan veri ile oluşturduğu kaybı göz önünde
              # bulundurarak kendisini güncelleme mekanizması
              optimizer=optimizers.RMSprop(lr=2e-5),
              
              # eğitim ve test süresince takip edilecek metrikler.
              metrics=['acc'])


# modeli eğitme aşaması
history = model.fit_generator( # acc, loss, val_acc, val_loss değerlerini history adlı değişkenden alacağız.
      
      # eğitim verileri
      train_generator,
      
      # döngü bitene kadar geçeceği örnek sayısı (alınacak yığın)
      steps_per_epoch=175,

      # döngü sayısı
      epochs=150,

      # doğrulama verisi
      validation_data=validation_generator,

      # doğrulama için yığın sayısı
      validation_steps=75,

      verbose=2)


# modelimizi test için kaydettik.
model.save('asd.h5')




# --- sonuçları görselleştirme --- #


# Eğitim başarım skoru
acc = history.history["acc"]

# doğrulama başarım skoru
val_acc = history.history["val_acc"]

# eğitim kayıp skoru
loss = history.history["loss"]

# doğrulama kayıp skoru
val_loss = history.history["val_loss"]

# epochs sayısına göre grafik çizdireceğiz.
epochs = range(1, len(acc) + 1)

# eğitim başarımını kendine özel çizdirdik.
plt.plot(epochs, acc, "bo", label="Eğitim başarımı")

# doğrulama başarımını kendine özel çizdirdik.
plt.plot(epochs, val_acc, "b", label="Doğrulama başarımı")

# çizdirmemizin başlığı
plt.title("Eğitim ve doğrulama başarımı")

plt.legend()

plt.figure()

# eğitim kaybını kendine özel çizdirdik.
plt.plot(epochs, loss, "bo", label="Eğitim kaybı")

# doğrulama kaybını kendine özel çizdirdik.
plt.plot(epochs, val_loss, "b", label="Doğrulama kaybı")


# çizdirmemizin başlığı
plt.title("Eğitim ve doğrulama kaybı")

plt.legend()

# ekrana çıkartma
plt.show()