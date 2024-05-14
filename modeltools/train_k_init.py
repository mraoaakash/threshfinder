import os
import sys
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import  ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def model(masterpath, model_type, K):
        df_path = os.path.join(masterpath, "model_metadata")

        for fold in range (0,3):
                print("fold", fold)
                df_train = pd.DataFrame(columns=["label", "filename"])
                df_test_path = os.path.join(masterpath, "model_metadata", "model_tr_fold_"+str(fold)+".csv")
                for i in range(0,5):
                        if i == fold:
                                continue
                        df_train_path = os.path.join(masterpath, "model_metadata", "model_tr_fold_"+str(i)+".csv")
                        df_train = pd.concat([df_train, pd.read_csv(df_train_path)], ignore_index=True) 
            

                datagen_train = ImageDataGenerator(rescale = 1.0/255.0,)
                # Training Data
                train_generator = datagen_train.flow_from_dataframe(
                        df_train,
                        x_col = 'filename',
                        y_col = 'label',
                        target_size=(256, 256),
                        batch_size=32,
                        class_mode='categorical',
                        shuffle=False)
                #Validation Data
                datagen_test = ImageDataGenerator(rescale = 1.0/255.0)
                df_test = pd.read_csv(df_test_path)
                valid_generator = datagen_train.flow_from_dataframe(
                        df_test,
                        x_col = 'filename',
                        y_col = 'label',
                        target_size=(256, 256),
                        batch_size=32,
                        class_mode='categorical',
                        shuffle=False)
                print(list(valid_generator.class_indices.keys()))

                # printing dataloader info
                print("train_generator.class_indices", train_generator.class_indices)
                print("valid_generator.class_indices", valid_generator.class_indices)


                model = InceptionV3(
                        weights=None,
                        include_top=False,
                        input_shape=(256,256,3)
                        )
                for layer in model.layers:
                        layer.trainable = True
                x = layers.GlobalAveragePooling2D()(model.output)
                x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
                x = layers.Dense(4, activation = 'softmax',kernel_regularizer=l2(0.01))(x)

                model = Model(model.input, x)

                model.compile(optimizer = RMSprop(learning_rate = 0.0000001), loss = 'categorical_crossentropy', metrics = ['acc'])

                
                if not os.path.exists(f'{masterpath}/models/{model_type}/{model_type}_fold_{fold+1}'):
                        os.makedirs(f'{masterpath}/models/{model_type}/{model_type}_fold_{fold+1}')
                # Model Summary
                # model.summary()

                # Training the model
                print("------------------------------------------")
                print(f'Training the model {model_type}_fold_{fold+1}')
                print("------------------------------------------")
                filepath = f'{masterpath}/models/{model_type}/{model_type}_fold_{fold+1}/model_log'
                if os.path.exists(filepath):
                        os.makedirs(filepath)
                filepath = filepath + "/model-{epoch:02d}.h5"
                callbacks = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10)
                history = model.fit(train_generator, validation_data = valid_generator, epochs=100, callbacks=[callbacks], verbose=1)
                print("------------------------------------------")
                print(f'Training Complete')
                print("------------------------------------------")

                # Saving the model
                model.save(f'{masterpath}/models/{model_type}/{model_type}_fold_{fold+1}/{model_type}_fold_{fold+1}.h5')
                print("------------------------------------------")
                print(f'Model saved')
                print("------------------------------------------")


                #plotting the accuracy and loss
                print("------------------------------------------")
                print(f'Plotting and supplimentary data')
                print("------------------------------------------")
                plt.figure(figsize=(10, 10))
                plt.plot(history.history['acc'], label='Training Accuracy')
                plt.plot(history.history['val_acc'], label='Validation Accuracy')
                plt.title('Training and Validation Accuracy')
                plt.legend(['train', 'test'], loc='upper left')
                plt.tight_layout()
                plt.savefig(f'{masterpath}/models/{model_type}/{model_type}_fold_{fold+1}/Accuracy.jpg')


                # Saving Training History
                hist_df = pd.DataFrame(history.history) 
                # save to json:  
                hist_json_file = f'{masterpath}/models/{model_type}/{model_type}_fold_{fold+1}/history.json' 
                with open(hist_json_file, mode='w') as f:
                        hist_df.to_json(f)
                # or save to csv: 
                hist_csv_file = f'{masterpath}/models/{model_type}/{model_type}_fold_{fold+1}/history.csv'
                with open(hist_csv_file, mode='w') as f:
                        hist_df.to_csv(f)
                        

                # Loading Model for Testing
                loaded_model = load_model(f'{masterpath}/models/{model_type}/{model_type}_fold_{fold+1}/{model_type}_fold_{fold+1}.h5')
                outcomes = loaded_model.predict(valid_generator)
                y_pred = np.argmax(outcomes, axis=1)

                # Computing and saving the confusion matrix
                # confusion matrix
                confusion = confusion_matrix(valid_generator.classes, y_pred)
                plt.figure(figsize=(10, 10))
                sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.tight_layout()
                plt.savefig(f'{masterpath}/models/{model_type}/{model_type}_fold_{fold+1}/Confusion_matrix.jpg')
                conf_df = pd.DataFrame(confusion, index = list(valid_generator.class_indices.keys()), columns = list(valid_generator.class_indices.keys()))
                conf_df.to_csv(f'{masterpath}/models/{model_type}/{model_type}_fold_{fold+1}/Confusion_matrix.csv')



                # Computing and saving the Classification Report
                # classification report
                target_names = list(valid_generator.class_indices.keys())
                report = classification_report(valid_generator.classes, y_pred, target_names=target_names, output_dict=True)
                df = pd.DataFrame(report).transpose()
                df.to_csv(f'{masterpath}/models/{model_type}/{model_type}_fold_{fold+1}/Classification_report.csv')

                print("------------------------------------------")
                print(f'Supplimentary Data Saved')
                print("------------------------------------------")


if __name__ == '__main__':
        args = sys.argv[1:]
        if len(args) == 0:
                print('Please provide the master path model type')
                sys.exit(1)
        masterpath = args[0]
        model_type = args[1]
        K = 0
        model(masterpath, model_type, K)