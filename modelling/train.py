import os 
import pandas
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import sys


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception, ResNet50, InceptionV3, VGG16, VGG19, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential

from sklearn.metrics import multilabel_confusion_matrix, classification_report, confusion_matrix
from sklearn.utils import class_weight

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


def gen_data_dfs(csv, fold):
    train_df = []
    valid_df = []
    test_df = []
    try:
        csv = pandas.read_csv(csv)
    except:
        raise Exception('Could not read csv file')
    
    csv.drop(columns=['Unnamed: 0'], inplace=True)


    seed = 42
    print('Processing csv file...')
    np.random.seed(seed)
    indices = np.random.choice(len(csv), int(0.2*len(csv)), replace=False)
    valid_df = csv.iloc[indices]
    csv.drop(indices, inplace=True)
    csv.reset_index(drop=True, inplace=True)

    # make 5 partitions
    partitions = np.array_split(csv, 5)
    
    test_df = partitions[fold-1]
    partitions.pop(fold-1)
    train_df = pandas.concat(partitions)




    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)


    print('Train: ', len(train_df))
    print('Valid: ', len(valid_df))
    print('Test: ', len(test_df))
    print("Processed csv file")

    return train_df, valid_df, test_df

def train(
            base_path, 
            model_name, epochs=100, 
            save_path="outputs/models", 
            input_shape=(314, 314, 3),
            threshold=10,
            fold=1,
            batch_size=32, 
            lr=0.001,
            class_mode='categorical',
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy'], 
            save_model=True, 
            save_weights=True, 
            save_history=True, 
            verbose=1,
            mode = 'train'
        ):
    


    basemodels = {
        'Xception': Xception,
        'ResNet50': ResNet50,
        'InceptionV3': InceptionV3,
        'VGG16': VGG16,
        'VGG19': VGG19,
        'MobileNetV2': MobileNetV2
    }
    base = basemodels[model_name](weights='imagenet', include_top=False, input_shape=input_shape)

    model = Sequential()
    model.add(base)
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='sigmoid'))
    model.summary()


    optimizer_list = {
        'adam': Adam(lr=lr),
        'sgd': SGD(lr=lr),
        'rmsprop': RMSprop(lr=lr)
    }

    model.compile(
        optimizer=optimizer_list[optimizer],
        loss=loss,
        metrics=metrics
    )

    if mode =="initialize":
        sys.exit(0)


    csv = os.path.join(base_path, 'data/nucls/ind_percs/thresholds', f'perc_{str(threshold).zfill(3)}.csv')
    train_df, valid_df, test_df = gen_data_dfs(csv, fold)


    model_base_path = os.path.join(base_path, 'outputs/models', model_name, f'fold_{fold}')
    os.makedirs(model_base_path, exist_ok=True)

    data_save_path = os.path.join(model_base_path, 'data')
    os.makedirs(data_save_path, exist_ok=True)
    
    train_df.to_csv(os.path.join(data_save_path, 'train.csv'), index=False)
    valid_df.to_csv(os.path.join(data_save_path, 'valid.csv'), index=False)
    test_df.to_csv(os.path.join(data_save_path, 'test.csv'), index=False)
    
    cols = ['perc_0',  'perc_1',  'perc_2',  'perc_3']
    train_df['classes'] = train_df[cols].apply(lambda x: x.tolist(), axis=1)
    valid_df['classes'] = valid_df[cols].apply(lambda x: x.tolist(), axis=1)
    test_df['classes'] = test_df[cols].apply(lambda x: x.tolist(), axis=1)

    train_df['image_name'] = train_df['image_name'].apply(lambda x: x+'.png')

    train_df = train_df[['image_path', 'classes']]

    print(train_df["image_path"].values[0])


    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
    )

    test_datagen = ImageDataGenerator(
        rescale=1./255,
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255,
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col="classes",
        target_size=(input_shape[:2]),
        classes=[0,1,2,3],
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
        seed=42
    )

    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col='image_path',
        y_col="classes",
        target_size=(input_shape[:2]),
        classes=[0,1,2,3],
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
        seed=42
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='image_path',
        y_col="classes",
        target_size=(input_shape[:2]),
        classes=[0,1,2,3],
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
        seed=42
    )


    # save every 5 epochs
    checkpoint = ModelCheckpoint('model{epoch:08d}.h5', period=5) 

    history = model.fit( train_generator, epochs=epochs, validation_data=validation_generator, verbose=verbose,allbacks=[checkpoint])
    model.save(os.path.join(model_base_path, 'model_final.h5'))

    print(history.history)
    json = os.path.join(model_base_path, 'history.json')
    with open(json, 'w+') as f:
        f.write(str(history.history))

    eval = model.evaluate(test_generator)

    preds = model.predict(test_generator)
    np.save(os.path.join(model_base_path, 'preds.npy'), preds)
    np.save(os.path.join(model_base_path, 'labels.npy'), test_generator.labels)

    classification = multilabel_confusion_matrix(test_generator.labels, np.round(preds))
    np.save(os.path.join(model_base_path, 'confusion_matrix.npy'), classification)

    


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--base_path', type=str, default='.')
    argparser.add_argument('--model_name', type=str, default='test')
    argparser.add_argument('--epochs', type=int, default=100)
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--optimizer', type=str, default='adam')
    argparser.add_argument('--loss', type=str, default='categorical_crossentropy')
    argparser.add_argument('--metrics', type=list, default=['accuracy'])
    argparser.add_argument('--class_mode', type=str, default='categorical')
    argparser.add_argument('--save_model', type=bool, default=True)
    argparser.add_argument('--save_weights', type=bool, default=True)
    argparser.add_argument('--save_history', type=bool, default=True)
    argparser.add_argument('--save_path', type=str, default=None)
    argparser.add_argument('--verbose', type=int, default=1)
    argparser.add_argument('--input_shape', type=tuple, default=(314, 314, 3))
    argparser.add_argument('--threshold', type=int, default=10)
    argparser.add_argument('--fold', type=int, default=1)
    argparser.add_argument('--mode', type=str, default='train')
    args = argparser.parse_args()

    train(**vars(args))