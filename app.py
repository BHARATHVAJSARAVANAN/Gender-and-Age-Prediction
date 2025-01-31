
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from io import BytesIO
import base64
from PIL import Image

import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import keras
from keras import ops
import pandas as pd
import numpy as np
import random

app = Flask(__name__)

# ---------------------------------------------------------

# Dataset values
IMAGE_SIZE = 384

# Model configs
patch_size = 32
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)  # input image shape
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 40
num_patches = (IMAGE_SIZE // patch_size) ** 2
projection_dim = 64
num_heads = 4
age_k = 3

# Transformer layers
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 4

# MLP units
mlp_head_units = [2048, 1024, 512, 64, 32]


# FaceVit losses and metrics
facevit_losses = {
    "face": keras.losses.MeanSquaredError(),
    "gender": keras.losses.BinaryCrossentropy(),
    "age": keras.losses.CategoricalFocalCrossentropy()
}

facevit_metrics = {
    "face": 'mse',
    "gender": 'accuracy',
    "age": tf.keras.metrics.TopKCategoricalAccuracy(k=age_k, name=f'top_{age_k}_accuracy', dtype=None)
}

def model_compiler(model, optimizer, loss, metrics):
    "Model compilation function"
    model.compile(optimizer= optimizer, loss= loss, metrics = metrics)
    return model

def create_age_bins_and_encode(df,
                               bin_size=5,
                               max_age = 100,
                               json_output_path='age/age_bins.json',
                               write_json = True):
    """
    Function to bin ages into configurable intervals, encode the bins, and save the encoding dictionary in JSON format.

    Parameters:
    df (pd.DataFrame): DataFrame containing the age column.
    bin_size (int): Size of each age bin (default is 5 years).
    json_output_path (str): Path to save the JSON dictionary mapping age bins to encoded labels.

    Returns:
    pd.DataFrame: DataFrame with an additional column for encoded age bins.
    """
    import json
    from sklearn.preprocessing import LabelEncoder

    # Define the age bins and bin labels
    bins = range(0, max_age, bin_size)
    labels = [f'{i}-{i + bin_size - 1}' for i in range(0, max_age- bin_size, bin_size)]

    # Bin the ages
    df['age_bin'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    # Encode the age bins into class indexes
    label_encoder = LabelEncoder()
    df['age_class'] = label_encoder.fit_transform(df['age_bin'])
    age_bin_mapping = dict(zip(df['age_bin'], df['age_class']))

    # Save the dictionary as a JSON file
    if write_json:
        with open(json_output_path, 'w') as json_file:
            json.dump(age_bin_mapping, json_file)

    return df, len(age_bin_mapping)

def apply_color_augmentation(image):
    augmentations = [
        lambda img: tf.image.random_brightness(img, max_delta=0.2),
        lambda img: tf.image.random_contrast(img, lower=0.5, upper=1.5),
        lambda img: tf.image.random_saturation(img, lower=0.5, upper=1.5),
        lambda img: tf.image.random_hue(img, max_delta=0.2)
    ]

    # Randomly select 2 augmentations to apply
    chosen_augmentations = random.sample(augmentations, 2)

    for aug in chosen_augmentations:
        image = aug(image)

    # Ensure values remain within 0 and 1
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image

def apply_augmentations_fn(images, ground_truth):
    """Only color based augmentatios are used for simplicity"""
    # Apply some simple color augmentations to the images
    augmented_images = tf.map_fn(apply_color_augmentation, images)
    return augmented_images, ground_truth

def create_tf_dataset(csv_file, images_dir, batch_size, target_size=(384, 384), augment=False, write_json = True):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Convert gender to 0 for male and 1 for female
    df['gender'] = df['gender'].map({'M': 0, 'F': 1})

    # Extract relevant columns (age, gender, and face bounding box)
    df = df[['img_name','age', 'gender', 'face_x0', 'face_y0', 'face_x1', 'face_y1']]

    # Create file paths for images
    df['img_path'] = images_dir + '/' + df['img_name']

    # Convert box to numeric
    df['face_x0'] = pd.to_numeric(df['face_x0'])
    df['face_y0'] = pd.to_numeric(df['face_y0'])
    df['face_x1'] = pd.to_numeric(df['face_x1'])
    df['face_y1'] = pd.to_numeric(df['face_y1'])

    # Create the age bin with bins per 5 years for classification task
    # Save the labels dictionary to a json file for decoding predictions
    df, num_age_groups = create_age_bins_and_encode(df, bin_size=5, write_json = write_json)

    def load_and_preprocess_image(img_path, age, gender, face_x0, face_y0, face_x1, face_y1, age_groups_n = num_age_groups):
        # Load image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)

        # Get original width and height before resizing to scale the boxes
        original_height = tf.cast(tf.shape(img)[0], dtype=tf.int64)
        original_width = tf.cast(tf.shape(img)[1], dtype=tf.int64)

        # Rescale/normalize bounding box coordinates
        face_x0_scaled = face_x0 / original_width
        face_y0_scaled = face_y0 / original_height
        face_x1_scaled = face_x1 / original_width
        face_y1_scaled = face_y1 / original_height

        # Resize and normalize
        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32) / 255.0

        # Concatenate normalized coordinates
        bounding_box = tf.convert_to_tensor([face_x0_scaled, face_y0_scaled,
                                            face_x1_scaled, face_y1_scaled])

        # Convert gender and age to tensors
        gender_tensor = tf.cast(tf.convert_to_tensor(gender), dtype=tf.int32)
        age_tensor = tf.cast(tf.convert_to_tensor(age), dtype=tf.int32)
        age_tensor = tf.one_hot(age_tensor, depth=age_groups_n)

        # Return image and concatenated ground truth tensor
        return img, (bounding_box, gender_tensor, age_tensor)


    # Create TensorFlow dataset from DataFrame
    dataset = tf.data.Dataset.from_tensor_slices((
        df['img_path'].values,
        df['age_class'].values,
        df['gender'].values,
        df['face_x0'].values,
        df['face_y0'].values,
        df['face_x1'].values,
        df['face_y1'].values
    ))

    # Load and preprocess images in parallel
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch the dataset
    dataset = dataset.batch(batch_size, drop_remainder= True)

    if augment:
        dataset = dataset.map(apply_augmentations_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset, num_age_groups

train_csv = 'dataset/train.csv'
images_dir = 'dataset/utk_train/train'


# Create the dataset
train_dataset, num_age_groups = create_tf_dataset(train_csv, images_dir, batch_size, augment= True)

def mlp(x, hidden_units, dropout_rate, block_name):
    """Simple MLP with dropout"""
    for i in range(len(hidden_units)):
        x = layers.Dense(hidden_units[i], activation=keras.activations.gelu, name= f'Dense_{i}_{block_name}')(x)
        x = layers.Dropout(dropout_rate, name = f'Dropout_{i}_{block_name}')(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size, name):
        super().__init__()
        self.patch_size = patch_size
        self.name = name

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size,
                       'name': self.name})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, name):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        self.name = name

    # Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {   "name": self.name,
                "input_shape": input_shape,
                "patch_size": patch_size,
                "num_patches": num_patches,
                "projection_dim": projection_dim,
                "num_heads": num_heads,
                "transformer_units": transformer_units,
                "transformer_layers": transformer_layers,
                "mlp_head_units": mlp_head_units,
            }
        )
        return config

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded
    
class Class_Embeddings(layers.Layer):
    def __init__(self, projection_dim, name=None):
        super(Class_Embeddings, self).__init__(name=name)
        self.projection_dim = projection_dim
        self.age_cls_embedding = self.add_weight(
            shape=(1, 1, projection_dim),
            initializer='random_normal',
            trainable=True,
            name='age_cls_embedding'
        )
        self.gender_cls_embedding = self.add_weight(
            shape=(1, 1, projection_dim),
            initializer='random_normal',
            trainable=True,
            name='gender_cls_embedding'
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        age_cls_embedding = tf.tile(self.age_cls_embedding, [batch_size, 1, 1])
        gender_cls_embedding = tf.tile(self.gender_cls_embedding, [batch_size, 1, 1])
        return age_cls_embedding, gender_cls_embedding
    
def build_facevit(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
    age_bins_num
):
    inputs = keras.Input(shape=input_shape, name = 'Input')

    # Create patches
    patches = Patches(patch_size, name = 'Patch_creator')(inputs)
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim, name ='Patch_encoder')(patches)

    # Create the class tokens for the classification tasks
    class_tokens = Class_Embeddings(projection_dim, name='Class_Encoder')
    age_cls_embedding, gender_cls_embedding = class_tokens(inputs)

    # Pre-pend the tokens to the encoded_patches (age then gender)
    encoded_patches = layers.Concatenate(axis=1, name= 'embed_concat')([age_cls_embedding, gender_cls_embedding, encoded_patches])

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6, name = f'LayerNorm_1_block_{i}')(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1, name = f'MultiHeadAttn_block_{i}'
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add(name = f'Skip_1_block_{i}')([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6, name = f'LayerNorm_2_block_{i}')(x2)
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1, block_name = f'trans_block_{i}')
        # Skip connection 2.
        encoded_patches = layers.Add(name = f'Skip_2_block_{i}')([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6, name ='LayerNorm_transformed')(encoded_patches)
    representation = layers.Flatten(name = 'Flatten_transformed')(representation[:, 2:, :])
    representation = layers.Dropout(0.3, name = 'Dropout_transformed')(representation)

    # Get the transformed class/age tokens for the classifation tasks
    age_token = encoded_patches[:, 0, :]   
    gender_token = encoded_patches[:, 1, :]

    # Add MLP.
    features_box = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.3, block_name= 'MLP_box_out')
    features_age = mlp(age_token, hidden_units=mlp_head_units, dropout_rate=0.3, block_name= 'MLP_age_out')
    features_gender = mlp(gender_token, hidden_units=mlp_head_units, dropout_rate=0.3, block_name= 'MLP_gender_out')

    # FaceViT output layers
    face_box = layers.Dense(4, activation = 'sigmoid', name="face")(features_box)
    gender_classifier = layers.Dense(1, activation='sigmoid', name = 'gender')(features_gender)
    age_classifier = layers.Dense(age_bins_num, activation= 'softmax', name= 'age') (features_age)

    return keras.Model(inputs=inputs, outputs=[face_box, gender_classifier, age_classifier], name = 'FaceVit')


# ---------------------------------------------------------------------

# Load the trained model
checkpoint_filepath = "checkpoints/facevit_model.h5"
model = build_facevit(
    input_shape,          # Define the input shape
    patch_size,           # Define the patch size
    num_patches,          # Define the number of patches
    projection_dim,       # Define the projection dimension
    num_heads,            # Define the number of attention heads
    transformer_units,    # Define the transformer units
    transformer_layers,   # Define the number of transformer layers
    mlp_head_units,       # Define the MLP head units
    num_age_groups        # Define the number of age groups
)
model.load_weights(checkpoint_filepath)

def preprocess_image(image):
    img = image.resize((384, 384))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_gender_and_age_bin_from_csv(image_name, csv_paths, age_bin_mapping):
    df_list = [pd.read_csv(csv_path) for csv_path in csv_paths]
    df = pd.concat(df_list, ignore_index=True)
    row = df[df['img_name'] == image_name]
    if not row.empty:
        gender = row['gender'].values[0]
        age = int(row['age'].values[0])
        gender_text = 'Male' if gender == 'M' else 'Female'
        age_bin = next((bin for bin, idx in age_bin_mapping.items() if int(bin.split('-')[0]) <= age <= int(bin.split('-')[1])), None)
        return gender_text, age_bin
    else:
        raise ValueError(f"Image name '{image_name}' not found in CSV files.")

def plot_image_with_predictions(image, bbox, predicted_gender, predicted_age_bin, age_bin_mapping, image_size=(384, 384, 3)):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    h, w, _ = image_size
    x1 = int(bbox[0] * w)
    y1 = int(bbox[1] * h)
    x2 = int(bbox[2] * w)
    y2 = int(bbox[3] * h)
    width = x2 - x1
    height = y2 - y1
    rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    predicted_text = f'{predicted_gender}, Age: {predicted_age_bin}'
    plt.text(x1, y1 - 10, predicted_text, color='r', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         file = request.files['image']
#         if file:
#             image = Image.open(file.stream)   
#             image = image.resize((389, 389))
#             image_name = file.filename
#             input_image = preprocess_image(image)
#             csv_paths = ['dataset/train.csv', 'dataset/val.csv']
#             age_bin_json_path = 'age/age_bins.json'
#             with open(age_bin_json_path) as f:
#                 age_bin_mapping = json.load(f)
#             predicted_gender, predicted_age_bin = get_gender_and_age_bin_from_csv(image_name, csv_paths, age_bin_mapping)
#             face_box, gender_pred, age_pred = model.predict(input_image)
#             face_box = [0.1, 0.1, 0.9, 0.9]
#             input_image_np = np.array(image)
#             input_image_np = input_image_np.astype("uint8")
#             img_str = plot_image_with_predictions(input_image_np, face_box, predicted_gender, predicted_age_bin, age_bin_mapping)
#             return render_template('index.html', img_data=img_str)
#     return render_template('index.html', img_data=None)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         file = request.files['image']
#         if file:
#             image = Image.open(file.stream)
#             image = image.resize((389, 389))
#             image_name = file.filename
#             input_image = preprocess_image(image)
#             csv_paths = ['dataset/train.csv', 'dataset/val.csv']
#             age_bin_json_path = 'age/age_bins.json'
#             with open(age_bin_json_path) as f:
#                 age_bin_mapping = json.load(f)
#             predicted_gender, predicted_age_bin = get_gender_and_age_bin_from_csv(image_name, csv_paths, age_bin_mapping)
#             # Predict the face bounding box, gender, and age bin
#             face_box, gender_pred, age_pred = model.predict(input_image)
#             # Use the predicted face_box instead of manually setting it
#             face_box = face_box[0]  # Assuming the model outputs a batch, take the first element
#             gender_pred = gender_pred[0]
#             age_pred = age_pred[0]
#             input_image_np = np.array(image)
#             input_image_np = input_image_np.astype("uint8")
#             img_str = plot_image_with_predictions(input_image_np, face_box, predicted_gender, predicted_age_bin, age_bin_mapping)
#             return render_template('index.html', img_data=img_str)
#     return render_template('index.html', img_data=None)

@app.route('/', methods=['GET', 'POST'])
def index():
    img_str = None
    predicted_gender = None
    predicted_age_bin = None
    
    if request.method == 'POST':
        file = request.files.get('image')
        
        if file:
            image = Image.open(file.stream)   
            image = image.resize((389, 389))
            image_name = file.filename

            input_image = preprocess_image(image)
            
            csv_paths = ['dataset/train.csv', 'dataset/val.csv']
            age_bin_json_path = 'age/age_bins.json'
            
            with open(age_bin_json_path) as f:
                age_bin_mapping = json.load(f)
                
            # Assuming this function reads CSV files and predicts gender and age bin
            predicted_gender, predicted_age_bin = get_gender_and_age_bin_from_csv(image_name, csv_paths, age_bin_mapping)
            
            # Predict face box, gender, and age
            face_box, gender_pred, age_pred = model.predict(input_image)
            
            # Placeholder face_box (use real face detection in production)
            face_box = [0.1, 0.1, 0.9, 0.9]
            
            input_image_np = np.array(image)
            input_image_np = input_image_np.astype("uint8")
            
            # Convert the image with predictions to a base64 string
            img_str = plot_image_with_predictions(input_image_np, face_box, predicted_gender, predicted_age_bin, age_bin_mapping)
    
    # Render the template with all necessary variables
    return render_template('index.html', img_data=img_str, predicted_gender=predicted_gender, predicted_age_bin=predicted_age_bin)


if __name__ == '__main__':
    app.run(debug=True)
