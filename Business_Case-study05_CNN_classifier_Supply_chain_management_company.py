##---- Importing the libraries---------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score



##-----Import/Download the data--------

cwd = r'D:\IMP\Studies\Machine_Learning\Scaler_Major_Project\Case_Study05_CNN_Classifier_for_Supply_chain_Management_company\multiclass_classifier_data'

training_data_folder = r'D:\IMP\Studies\Machine_Learning\Scaler_Major_Project\Case_Study05_CNN_Classifier_for_Supply_chain_Management_company\multiclass_classifier_data\multiclass_classifier_data\train'
test_data_folder = r'D:\IMP\Studies\Machine_Learning\Scaler_Major_Project\Case_Study05_CNN_Classifier_for_Supply_chain_Management_company\multiclass_classifier_data\multiclass_classifier_data\test'

set_info_list = ['training','validation','test']

# Validate the directory and database
if not os.path.isdir(cwd):
    print(f"Error: The directory '{cwd}' does not exist.")
else:
  # Define the filename
  filename = 'mastersheet.csv'
  file_path = os.path.join(cwd, filename)

  # Check if the file exists in the specified directory
  if not os.path.exists(file_path):
    # Create a new DataFrame 
    df = pd.DataFrame() # Create an empty DataFrame
    
    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False) # Save as 'mastersheet.csv' with no index
    print(f"'{filename}' created successfully in '{cwd}'.")

  data_to_write = []
    
  # training set
  folders_training = [f for f in Path(training_data_folder).iterdir() if f.is_dir() ]

  folder_list_training = ([str(folders_training[i]).split("\\")[-1] for i in range(len(folders_training))])
    
  for i in range(len(folder_list_training)):
    files = [f for f in folders_training[i].iterdir() if f.is_file()]
    for j in range(len(files)):
      data_to_write.append({'filepath':files[j],'set_info': 'training','class_info_ground_truth':folder_list_training[i]})

  # test set
  folders_test = [f for f in Path(test_data_folder).iterdir() if f.is_dir() ]

  folder_list_test = ([str(folders_test[i]).split("\\")[-1] for i in range(len(folders_test))])
    
  for i in range(len(folder_list_test)):
    files = [f for f in folders_test[i].iterdir() if f.is_file()]
    for j in range(len(files)):
      data_to_write.append({'filepath':files[j],'set_info': 'test','class_info_ground_truth':folder_list_training[i]})

  # Create a DataFrame from the data
  df = pd.DataFrame(data_to_write)

  # Write DataFrame to CSV
  df.to_csv(file_path, index=False)

  ##----------------------------------------------------------------------------------
  for i in range(len(set_info_list)):
    set_name = set_info_list[i]
    spec_dir = cwd +'\\'+ 'Processed_files' + '\\' +f"Processed_files_{set_name}"
    globals()[spec_dir]= spec_dir
    os.makedirs(spec_dir, exist_ok=True) # Create the directory if it doesn't exist
    if set_name == 'training':
      train_ds_dir = spec_dir
    if set_name == 'validation':
      val_ds_dir = spec_dir
    if set_name == 'test':
      test_ds_dir = spec_dir

    for j in range(len(folder_list_test)):
      class_dir = spec_dir + '\\'+ folder_list_test[j]
      globals()[class_dir]= class_dir
      os.makedirs(class_dir, exist_ok=True) # Create the directory if it doesn't exist

  ##-----Visualize the data, use the dataset directory to create a list containing all the image paths in the training folder
  plt.figure(figsize=(10,10))
  count = 1
  for i, row in df.sample(9).iterrows():
    #print(row['filepath'].dtype)
    img = cv2.imread(str(row['filepath']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    plt.subplot(3, 3,count )
    plt.imshow(img)
    plt.title(row['class_info_ground_truth'])
    plt.axis('off')
    count += 1
  plt.show()

##----Plot a few of the images of each class to check their dimension

  for i in range(len(folder_list_training)):
    print('!---------------------------------------------!')
    print(f"For the dataset under {folder_list_training[i]} catogory:")
    df_indv_class = df[df['class_info_ground_truth']==folder_list_training[i]]
    df_indv_class_training = df_indv_class[df_indv_class['set_info']=='training']
    df_indv_class_test = df_indv_class[df_indv_class['set_info']=='test']
    print(f'training_samples : {df_indv_class_training.shape[0]}, test_samples: {df_indv_class_test.shape[0]}')
  print('!---------------------------------------------!')

  print("Onion and potato : minor mismatch (potential swap) observed between the given test samples info in problem statement vs actual samples")

##----Verify the count of images in each train and test folder by plotting histogram
  plt.figure(figsize=(10, 5))
  #sns.countplot(data=df, x='class_info_ground_truth')
  #plt.title('Counts of Images per Class')
  
  count_series = df[df['set_info']=='training']['class_info_ground_truth'].value_counts().sort_index(ascending=True) # Count occurrences of each class
  plt.subplot(1,2,1)
  plt.title("training_set_distribution")
  plt.bar(count_series.index, count_series.values, color='skyblue')
  

  # Annotate bar heights
  for i in range(len(count_series)):
    plt.text(i, count_series.values[i], count_series.values[i], ha='center', va='bottom')

  count_series = df[df['set_info']=='test']['class_info_ground_truth'].value_counts().sort_index(ascending=True) # Count occurrences of each class
  plt.subplot(1,2,2)
  plt.title("test_set_distribution")
  plt.bar(count_series.index, count_series.values, color='skyblue')

  # Annotate bar heights
  for i in range(len(count_series)):
    plt.text(i, count_series.values[i], count_series.values[i], ha='center', va='bottom')

  plt.show()

  print('!---------------------------------------------!')
  ##----Split the dataset to a train and validation set
  train_df_initial = df[df['set_info']=='training'].copy()
  test_df = df[df['set_info']=='test'].copy()
  test_df.reset_index(drop=True,inplace=True)

  train_df, val_df = train_test_split(train_df_initial, test_size=0.2, stratify=train_df_initial['class_info_ground_truth'], random_state=42)
  
  # Reset the index for train_df and val_df
  train_df.reset_index(drop=True, inplace=True)
  val_df.reset_index(drop=True, inplace=True)

  print(f"Training set size: {train_df.shape[0]}")
  print(f"Validation set size: {val_df.shape[0]}")
  print(f"Test set size: {test_df.shape[0]}")

  for i in range(len(folder_list_training)):
    print('!---------------------------------------------!')
    print(f"For the dataset under {folder_list_training[i]} catogory:")
    df_indv_class_training = train_df[train_df['class_info_ground_truth']==folder_list_training[i]]
    df_indv_class_val = val_df[val_df['class_info_ground_truth']==folder_list_training[i]]
    print(f'training_samples : {df_indv_class_training.shape[0]}, validation_samples: {df_indv_class_val.shape[0]}')
  print('!---------------------------------------------!')
  

  ##----make sure that each image is square-shaped so that we may resize it to the required dimensions and 
  ## also perform rescaling which will rescale the inputs between 0-1 by dividing each value by 255
  def image_processing_for_square_size(imagepath,size=(224,224)):
    img = cv2.imread(str(imagepath))
    img_size_h = img.shape[0]
    img_size_w = img.shape[1]
    if img_size_h != img_size_w:
      max_dim = max(img_size_w,img_size_h)
      # Create a new square image with padding
      square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
      # Place the original image in the center
      adjust_param_h = (max_dim-img_size_h)//2
      adjust_param_w = (max_dim-img_size_w)//2
      square_img[adjust_param_h:adjust_param_h + img_size_h, adjust_param_w :adjust_param_w + img_size_w] = img
      img = square_img
    img = cv2.resize(img, size) ## Resize the image to the specified size (if needed)
    
    # Rescale pixel values to range [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Convert back to uint8 for saving
    img_to_save = (img * 255).astype(np.uint8)
    
    # Get the original file extension
    file_extension = Path(imagepath).suffix # Get the file extension (e.g., .png, .jpg)
    return img_to_save,file_extension
  
  #--------------------------------------------------------------------
  image_serial= 0
  processed_file_paths = []
  # training set images processing
  
  for i in range(train_df.shape[0]):
    img_1 = cv2.imread(str(train_df['filepath'][i]))
    img_2, original_extension = image_processing_for_square_size(train_df['filepath'][i])
    class_name = train_df['class_info_ground_truth'][i]
    # Construct the output file path Processed_files_
    output_file_path = os.path.join(f'{cwd}\Processed_files\Processed_files_training', f'{class_name}\processed_image_{image_serial + 1}{original_extension}')
   
    # Save the processed image to disk
    cv2.imwrite(output_file_path, img_2)

    #Updating the dataframe with updated filepath
    processed_file_paths.append(output_file_path)
    image_serial += 1

  train_df['processed_filepaths'] = processed_file_paths

  #--------------------------------------------------------------------
  processed_file_paths = []
  # validation set images processing
  
  for i in range(val_df.shape[0]):
    img_1 = cv2.imread(str(val_df['filepath'][i]))
    img_2, original_extension = image_processing_for_square_size(val_df['filepath'][i])
    class_name = val_df['class_info_ground_truth'][i]
    # Construct the output file path
    output_file_path = os.path.join(f'{cwd}\Processed_files\Processed_files_validation', f'{class_name}\processed_image_{image_serial + 1}{original_extension}')
    
    # Save the processed image to disk
    cv2.imwrite(output_file_path, img_2)

    #Updating the dataframe with updated filepath
    processed_file_paths.append(output_file_path)
    image_serial += 1

  val_df['processed_filepaths'] = processed_file_paths

  #--------------------------------------------------------------------
  processed_file_paths = []
  # test set images processing

  for i in range(test_df.shape[0]):
    img_1 = cv2.imread(str(test_df['filepath'][i]))
    img_2, original_extension = image_processing_for_square_size(test_df['filepath'][i])
    class_name = test_df['class_info_ground_truth'][i]
    # Construct the output file path
    output_file_path = os.path.join(f'{cwd}\Processed_files\Processed_files_test', f'{class_name}\processed_image_{image_serial + 1}{original_extension}')
    
    # Save the processed image to disk
    cv2.imwrite(output_file_path, img_2)

    #Updating the dataframe with updated filepath
    processed_file_paths.append(output_file_path)
    image_serial += 1

  test_df['processed_filepaths'] = processed_file_paths
  #--------------------------------------------------------------------   

  # Load the dataset
  batch_size = 64
  img_height = 224
  img_width = 224

  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_ds_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
  )

  # Set up ImageDataGenerator for training data with augmentations
  train_datagen = ImageDataGenerator(
    rotation_range=20, # Randomly rotate images
    width_shift_range=0.2, # Randomly shift images horizontally
    height_shift_range=0.2, # Randomly shift images vertically
    shear_range=0.2, # Shear transformation
    zoom_range=0.2, # Randomly zoom in/out
    horizontal_flip=True, # Randomly flip images
    fill_mode='nearest' # Fill in new pixels created during transformations
  )

  # Load training data
  train_ds = train_datagen.flow_from_directory(
    train_ds_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse', 
    shuffle=True # Shuffle the training data
  )

  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_ds_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
  )

  test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_ds_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
  )

  def create_vgg_resnet_mobilenet_model(num_classes):
    base_model = tf.keras.applications.ResNet50(
      weights='imagenet', # Use pre-trained weights from ImageNet
      include_top=False,  # Exclude the top (classification) layer
      input_shape=(img_height, img_width, 3)
    )
    # Freeze the base model
    base_model.trainable = False

    #for layer in base_model.layers[-4:]:
      #layer.trainable = True

    # Create a new model on top
    model = models.Sequential([
      base_model,
      # Adding additional Conv layers with Batch Normalization
      #layers.Conv2D(256, (3, 3), padding='same', activation='relu'), # Additional Conv Layer
      #layers.BatchNormalization(),
      #layers.Conv2D(256, (3, 3), padding='same', activation='relu'), # Additional Conv Layer
      #layers.BatchNormalization(),
      #layers.MaxPooling2D(pool_size=(2, 2)), # Pooling layer to reduce dimensions
      
      #layers.Conv2D(512, (3, 3), padding='same', activation='relu'), # Additional Conv Layer
      #layers.BatchNormalization(),
      #layers.Conv2D(512, (3, 3), padding='same', activation='relu'), # Additional Conv Layer
      #layers.BatchNormalization(),
      #layers.MaxPooling2D(pool_size=(2, 2)), # Pooling layer to reduce dimensions

      #layers.Flatten(),
      layers.GlobalAveragePooling2D(),
      layers.Dense(256, activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(num_classes, activation='softmax') # Final output layer
    ])

    model.compile(optimizer='adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])
    return model
  
  early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
  model_checkpoint = callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
  tensorboard_callback = callbacks.TensorBoard(log_dir='logs/vgg', histogram_freq=1)

  # Create the VGG/Resnet50/Mobilenet model instance
  try:
    num_classes = len(train_ds.class_names)
  except AttributeError:
    num_classes = len(train_ds.class_indices)

  model = create_vgg_resnet_mobilenet_model(num_classes)

  # Train the model
  history = model.fit(train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=[early_stopping, model_checkpoint, tensorboard_callback])

  ##---Obtain the testing accuracy to see how well your model generalizes.
  # Evaluate the model
  val_loss, val_accuracy = model.evaluate(val_ds)
  print(f'Validation Accuracy: {val_accuracy}')

  test_loss, test_accuracy = model.evaluate(test_ds)
  print(f'Test Accuracy: {test_accuracy}')

  # Generate predictions for the test set
  y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
  y_pred = np.argmax(model.predict(test_ds), axis=-1)

  # Confusion Matrix
  cm = confusion_matrix(y_true, y_pred)

  # Precision
  precision = precision_score(y_true, y_pred, average='weighted')

  # Recall
  recall = recall_score(y_true, y_pred, average='weighted')

  # F1 Score
  f1 = f1_score(y_true, y_pred, average='weighted')

  print(f'Precision: {precision:.4f}')
  print(f'Recall: {recall:.4f}')
  print(f'F1-score: {f1:.4f}')



  # Display Confusion Matrix
  try:
    ConfusionMatrixDisplay(cm, display_labels=train_ds.class_names).plot()
  except AttributeError:
    ConfusionMatrixDisplay(cm, display_labels=train_ds.class_indices).plot()

  plt.title('Confusion Matrix')
  plt.show()