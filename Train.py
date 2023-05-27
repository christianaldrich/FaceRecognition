from math import floor
import os
import cv2 as cv
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score

def load_datasets():
    tr_path = 'dataset'
    tr_name = os.listdir(tr_path)

    # print(tr_name)
    # """
    # load datasets from website
    # preprocess datasets
    # return preprocessed datasets
    # """
    return tr_name, tr_path

def evaluate():

    tr_img = []
    tr_class = []

    face_detect = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    for index, folder_name in enumerate(temp1):
        img_path = temp2 + "/" + folder_name
        # print(os.listdir(os.path.join(os.getcwd(), "photos")))
        
        for img_name in os.listdir(img_path):
            img = cv.imread(f'{temp2}/{folder_name}/{img_name}', 0)
            detected = face_detect.detectMultiScale(img, 1.5, 6)

            if(len(detected) < 1):
                continue
            for img_detected in (detected):
                i, j, k, l = img_detected
                img_crop = img[j:j+l, i:i+k]
                # cv.imshow('img', img_crop)
                # cv.waitKey(10)
                tr_img.append(img_crop)
                tr_class.append(index)

    # print(tr_class)

    face_recog = cv.face.LBPHFaceRecognizer_create()
    face_recog.train(tr_img, np.array(tr_class))

    y_test, y_pred = [], []
    
    # Testing
    test_path = 'test'
    for files in os.listdir(test_path):
        y_test.append(os.path.basename(files))
            
    for index,folder_name in enumerate(os.listdir(test_path)):
        tempp = test_path + '/' + folder_name
             
        for test_name in os.listdir(tempp):
            print(test_name)
            img_bgr = cv.imread(f'{test_path}/{folder_name}/{test_name}')
            img_gray = cv.imread(f'{test_path}/{folder_name}/{test_name}', 0)
            detected = face_detect.detectMultiScale(img_gray, 1.5, 6)
            
            # kalo gada muka, continue aja
            if len(detected) < 1:
                # print('gada muka')
                continue

            for detected_img in detected:
                i, j, k, l = detected_img
                img_crop = img_gray[j:j+l, i:i+k]
                result, confidence = face_recog.predict(img_crop)
                print(confidence)

                img_bgr = cv.rectangle(img_bgr, (i, j), (i+k, j+l), (0, 255, 0), 5)
                # print(result)
                # print(temp1[result])
                # print(confidence)
                # text = f'{temp1[result]} %.2{confidence}%'
                
                # for i in temp1:
                y_pred.append(temp1[result])
                
                text = '%s %.2f'%(temp1[result], confidence)
                img_bgr = cv.putText(img_bgr, text, (50,50),
                                    cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                img_bgr = cv.resize(img_bgr, (500, 350))
                cv.imshow('result', img_bgr)
                # cv.waitKey(0)
        
    # print('Accuracy: %.2f'%(accuracy_score(y_test,y_pred)*100))
    j = 0 
    accuracy = 0
    
    for test in y_test:
        k = 0
        for test1 in y_test:
            if y_pred[j] == y_test[k]:
                print(f'y test: {y_test[k]}, y pred: {y_pred[k]}')
                accuracy +=1
            k+=1
        j+=1

    acc = (accuracy/len(y_test))*100
    print('\n\n\n',acc)
    print(accuracy)
    print(len(y_test), len(y_pred),j)
    print(y_test)
    print(y_pred)
    print(result)
    print(f'Accuracy : {acc}')



# evaluate()
# pass
# """
# evaluate model
# display confusion matrix
# """

if __name__ == "__main__":
    temp1, temp2 = load_datasets()
    evaluate()
