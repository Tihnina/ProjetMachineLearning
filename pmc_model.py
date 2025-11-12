# 3_modele_pmc.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Chargement des données
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = data_gen.flow_from_directory(
    "dataset_amphibiens",
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)
val_data = data_gen.flow_from_directory(
    "dataset_amphibiens",
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Modèle PMC
model = Sequential([
    Flatten(input_shape=(128, 128, 3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement sur le modèle
history = model.fit(train_data, validation_data=val_data, epochs=15)

# Sauvegarde du modèle
model.save("models/modele_pmc.keras")
print("model pmc ok")

# Affichage des courbes
plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.title("Courbe d'apprentissage (accuracy)")
plt.xlabel("Épochs")
plt.ylabel("Accuracy")
plt.show()
