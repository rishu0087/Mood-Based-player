from keras.models import model_from_json
from spotify_strike_final import play_song
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np
import cv2
import dlib
import argparse
import os

dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

class FacialExpressionModel(object):
    class_labels = ['Angry','Happy','Neutral','Sad']
    
    def __init__(self, model_json_file, model_weights_file):
            with open(model_json_file, "r") as json_file:
                loaded_model_json = json_file.read() # loaidng the model
                self.loaded_model = model_from_json(loaded_model_json)

            self.loaded_model.load_weights(model_weights_file)
            print("Model loaded from disk")
            self.loaded_model.summary()
            
    def predict_emotion(self, img):
            self.preds = self.loaded_model.predict(img) #[0.9,0.8...]
            return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)] #[0.0 1.0 0.0] 1
        
parser = argparse.ArgumentParser()
parser.add_argument("source") #python fer.py source fps webcam 25
parser.add_argument("fps")
args = parser.parse_args()
cap = cv2.VideoCapture(os.path.abspath(args.source) if not args.source == 'webcam' else 0)
font = cv2.FONT_HERSHEY_SIMPLEX
cap.set(cv2.CAP_PROP_FPS, int(args.fps))

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it  to the format (x, y, w, h) as we would normally do with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def getdata():
    _, frame = cap.read()
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4,3))
    hog_face_detector = dlib.get_frontal_face_detector()
    gray = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    faces = hog_face_detector(gray)
    return faces, frame, gray

def start_app(classifier):
    mood = ''
    while True:
        faces, frame, gray = getdata()
        hog_face_detector = dlib.get_frontal_face_detector()
        rects = hog_face_detector(gray, 1)
        
        for rect in rects:
            (x, y, w, h) = rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
                
                preds = classifier.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
                mood = preds
                class_labels = ['Angry','Happy','Neutral','Sad']
                label=class_labels[preds.argmax()]
                label_position = (x,y)
                cv2.putText(frame,label,label_position,preds,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            else:
                cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

        for face in faces:
            face_landmarks = dlib_facelandmark(gray, face)

            for n in range(0,68):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)

        cv2.imshow("Face Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if cv2.waitKey(1) & 0xFF == ord(' '):
            if(mood!=''):
                print("playing for mood = ",mood)
                play_song(mood)
                
        cv2.imshow("Face Landmarks", frame)
            
    cap.release()
    cv2.destroyAllWindows()
    
def writesome(preds):
    print(preds)
    
if __name__ == '__main__':
    model = FacialExpressionModel("model554.json", "Emotion_detector.h5")
    start_app(model)