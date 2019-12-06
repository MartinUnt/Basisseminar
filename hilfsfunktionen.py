import os
import numpy as np
from tqdm.notebook import tqdm
from random import shuffle
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot

# Artificial data set
#DATA_NO_DEF = "Datasets/Class2"
#DATA_DEF = "Datasets/Class2_def"

# concrete wall data set
DATA_NO_DEF = "Datasets/Concrete_Crack_Binary/Negative"
DATA_DEF = "Datasets/Concrete_Crack_Binary/Positive"
#DATA_NO_DEF = "../../Desktop/Concrete_Crack_Binary/Negative"
#DATA_DEF = "../../Desktop/Concrete_Crack_Binary/Positive"

IMG_SIZE = 100
KERNEL_SIZE = 1
NUMBER_OF_IMAGES_PER_CLASS = 3000

def preprocessData():
    data = []
    data_no_def = []
    data_def = []
    
    # Beschränke die Anzahl von Bildern die eingelesen werden sollen
    no_def_paths_1000 = os.listdir(DATA_NO_DEF)[:NUMBER_OF_IMAGES_PER_CLASS]
    def_paths_1000 = os.listdir(DATA_DEF)[:NUMBER_OF_IMAGES_PER_CLASS]
    
    print("Lese die Bilder ein, welche keine Defekte aufweisen.")
    for img in tqdm(no_def_paths_1000):
        label = 0
        path = os.path.join(DATA_NO_DEF, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #img = img[100:512, 100:512]                 # Alternativ bestimmen Bildausschnitt auswählen
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        #img = cv2.dilate(img, np.ones((KERNEL_SIZE,KERNEL_SIZE), np.uint8), iterations = 2)  # Risse künstlich ausdehnen
        data_no_def.append([img, label]) 
        

    print("Lese die Bilder ein, welche Defekte aufweisen.")
    for img in tqdm(def_paths_1000):
        label = 1
        path = os.path.join(DATA_DEF, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #img = img[100:512, 100:512]                 # Alternativ bestimmen Bildausschnitt auswählen
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        #img = cv2.dilate(img, np.ones((KERNEL_SIZE,KERNEL_SIZE), np.uint8), iterations = 2)  # Risse künstlich ausdehnen
        data_def.append([img, label])
        
    
    #da binary problem von beiden klassen gleich viele bilder nehmen -> im notebook genauer erklären
    data.extend(data_no_def) #[:len(data_def)]
    data.extend(data_def)
    
    shuffle(data)
    
    #split the data into train and test set
    data_train, data_test = train_test_split(data, test_size=0.2)
    
    return data_train, data_test

def prepareXandY(data):
    data_X = np.array([x[0] for x in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    data_y = [x[1] for x in data]
    data_y = np.array(data_y)
    
    return data_X, data_y

def displayImageAndLabelOfData(data, start_index, shape):
    print(f"Lade {shape[0]*shape[1]} Bilder aus dem Datensatz, beginnend bei dem Index ", start_index, "...  (dies kann ein paar Sekunden dauern)")
    
    f, axarr = pyplot.subplots(shape[0], shape[1], figsize=(IMG_SIZE, IMG_SIZE))
    
    cnt = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            axarr[i,j].imshow(data[start_index+cnt][0], cmap=pyplot.cm.binary)
            axarr[i,j].set_title(f"Label: {data[start_index+cnt][1]}", fontsize=100)
            cnt += 1
            
    pyplot.show()  
    
def makePredictionsAndCalculateAccuracy(model, data_X_test, data_y_test):
    predictions = model.predict([tf.cast(data_X_test, tf.float16)])
    predictions = tf.round(predictions)
    correct_predictions = tf.equal(predictions[:, 0], data_y_test)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float16))
    print("")
    print("Die folgenden Wahrheitswerte sagen aus, ob das Netz das korrespondierende Bild aus dem Test-Datensatz richtig klassifiziert hat. True bedeutet dabei, dass das entsprechende Bild richtig identifiziert wurde, bei False handelt es sich um ein falsch klassifiziertes Bild. Bei Letzterem handelt es sich entweder um ein fälschlicherweise als mit Riss identifiziert Bild, oder um ein Bild welches eigentlich einen Riss aufweist, welcher allerdings nicht vom Netz erkannt wurde.")
    print(correct_predictions[:10])
    print("")
    print("Die vom Netz erreichte Genauigkeit auf dem Test-Datensatz beträgt: ")
    tf.print(accuracy)
    
def humanVSmachine(model, rounds):
    for i in range(rounds):
        print("Weißt die im Bild dargestellte Oberfläche einen Defekt auf?")
        pyplot.imshow(data_test[-(rounds+i)][0], cmap=pyplot.cm.binary)
        pyplot.show()
        answer = input("Antworten sie bitte mit Ja oder Nein: ")
        print("")
        
        if answer.lower() == "ja":
            input_answer = 1
        else:
            input_answer = 0
        
        if input_answer == data_test[-(rounds+i)][1]:
            print("Ihre Antwort stimmt mit der des künstlichen neuronalen Netzes überein.")
        else:
            print("Die von Ihnen trainierte künstliche Intelligenz sieht das anders.")
            
        print("")
        print("")