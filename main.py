import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

def daten_laden_und_vorbereiten(bilder_pfade, ball_koordinaten, zielgroesse):
    train_images = []
    train_labels = []

    # Lade jedes Bild und füge es zu train_images hinzu
    for bild_pfad in bilder_pfade:
        print(bild_pfad)
        bild = cv2.imread(bild_pfad)
        bild = cv2.cvtColor(bild, cv2.COLOR_BGR2RGB)  # Konvertiere das Bild zu RGB
        bild = cv2.resize(bild, zielgroesse)  # Ändere die Größe des Bildes zu zielgroesse
        bild = img_to_array(bild)  # Konvertiere das Bild zu einem numpy Array
        bild /= 255.0  # Normalisiere das Bild zu Werten zwischen 0 und 1
        train_images.append(bild)

    # Füge die x,y Koordinaten zu train_labels hinzu
    # Achtung: Hier könnte eine Anpassung nötig sein, abhängig davon, wie Ihre Labels skaliert werden sollen.
    for koordinate in ball_koordinaten:
        # Normiere die Koordinaten auf die zielgroesse, wenn notwendig
        x_norm = koordinate[0] / zielgroesse[0]
        y_norm = koordinate[1] / zielgroesse[1]
        train_labels.append([x_norm, y_norm])

    # Konvertiere die Listen in numpy Arrays
    train_images = np.array(train_images, dtype='float32')
    train_labels = np.array(train_labels, dtype='float32')

    return train_images, train_labels


bilder_pfade = ['pictures/DC0824.jpg', 'pictures/DC0624.jpg', 'pictures/DC0524.jpg', 'pictures/DC0424.jpg', 'pictures/DC0724.jpg']
ball_koordinaten = [(2183, 1267), (1665, 1037), (2229, 1853), (2261, 1303), (2495, 1331)]  # (x, y) Koordinaten des Balls

zielgroesse = (4416, 3336)  # Sollte mit der Eingabegröße Ihres Modells übereinstimmen

train_images, train_labels = daten_laden_und_vorbereiten(bilder_pfade, ball_koordinaten, zielgroesse)


modell = Sequential([
    Input(shape=(zielgroesse[0], zielgroesse[1], 3)),
    Conv2D(16, (3, 3), activation='relu'),  # Anzahl der Filter reduziert
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),  # Anzahl der Filter reduziert
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(32, activation='relu'),  # Größe des Dense-Layers reduziert
    Dense(2)
])

# Modell kompilieren
modell.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Trainieren des Modells mit einer kleineren Batch-Größe
batch_groesse = 16  # Batch-Größe reduziert
modell.fit(train_images, train_labels, batch_size=batch_groesse, epochs=10)

# Modell evaluieren
# Hier müssten Sie das Modell mit Testdaten evaluieren

# Modell verwenden
# Um die Koordinaten des Balls auf einem neuen Bild vorherzusagen, können Sie folgendes tun:
print('starten')
vorhersage = modell.predict('pictures/DC0723.jpg')
print('vorhersage', vorhersage)
