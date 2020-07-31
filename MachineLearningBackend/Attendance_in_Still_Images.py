import cv2
import numpy as np
import os
from datetime import datetime
import face_recognition

path = 'C:/Users/Tavish Chadha/PycharmProjects/FACIALRECOGNITION/ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        # convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print(len(encodeListKnown))

def markAttendance(name):
    with open('Attendance1.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


test_img = face_recognition.load_image_file('C:/Users/Tavish Chadha/PycharmProjects/FACIALRECOGNITION/ImagesBasic/FILMFARE.jpg')
test_img = cv2.resize(test_img,(0,0),None,0.75,0.75)

faces_loc = face_recognition.face_locations(test_img)

encodeIMG = face_recognition.face_encodings(test_img)

for encodeFace, faceLoc in zip(encodeIMG, faces_loc):
    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    print(faceDis)
    matchIndex = np.argmin(faceDis)

    if matches[matchIndex]:
        name = classNames[matchIndex].upper()
        print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1, x2, y2, x1 # multiplied by 4 as, earlier resize factor was 0.25
        cv2.rectangle(test_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(test_img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(test_img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        markAttendance(name)

    if not(matches[matchIndex]):
        print('UNKNOWN')
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # multiplied by 4 as, earlier resize factor was 0.25
        cv2.rectangle(test_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(test_img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(test_img, 'UNKNOWN', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

cv2.imshow('img',test_img)
cv2.waitKey()