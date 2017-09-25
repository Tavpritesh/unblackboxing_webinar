from sklearn.datasets import fetch_lfw_people

from facial_recognition.preprocessing import lfw_train_test_split, FacePreprocessor, tensor2img, img2tensor

NEPTUNE = True
if NEPTUNE:
    DATA_DIR = '/mnt/ml-team/homes/jakub.czakon/.data/scikit_learn_data'
    MODEL_FILEPATH = '/mnt/ml-team/homes/jakub.czakon/.unblackboxing_webinar_data/models/facenet.h5py'
    from facial_recognition.model import FaceClassifierNeptune as FaceClassifier

else:
    DATA_DIR = 'input/scikit_learn_data'
    MODEL_FILEPATH = 'input/models/facenet.h5py'
    from facial_recognition.model import FaceClassifier


if __name__ == '__main__':
    
    lfw_people = fetch_lfw_people(min_faces_per_person=100, resize=1.0, color=True, data_home=DATA_DIR)

    (X_train, y_train), (X_test,y_test) = lfw_train_test_split(lfw_people, train_size=0.8)
    print('Data loaded')
    
    face_prep = FacePreprocessor()
    X_train, y_train = face_prep.fit_transform(X=X_train, y=y_train)
    X_test, y_test = face_prep.transform(X=X_test, y=y_test)
    
    print('Data preprocessed')
    face_classifier = FaceClassifier(input_shape=(125, 94, 3), classes=5, 
                                     model_save_filepath=MODEL_FILEPATH)

    face_classifier.train((X_train, y_train), (X_test,y_test), batch_size=8, epochs=30)
