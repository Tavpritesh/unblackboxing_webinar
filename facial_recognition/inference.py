import pandas as pd


class FacePredictor(object):
    def __init__(self, face_preprocessor, face_classifier):
        self.face_preprocessor = face_preprocessor
        self.face_classifier = face_classifier
    
    def predict_proba(self, face):
        return NotImplementedError
    
    def predict(self, face):
        return NotImplementedError