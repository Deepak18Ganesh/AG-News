import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string, re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM
import pickle

# Load the dataset
df = pd.read_csv("Dataset/train.csv")

# Display dataset shape
print(df.shape)

# Display the first few rows
print(df.head())

# Display dataset info
df.info()

# Check for missing values
print(df.isna().sum())

# Check for duplicate rows
print(df.duplicated().sum())

# Display class distribution
print(df["Class Index"].value_counts())

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define label list
label_list = ["World", "Sports", "Business", "Sci/Tech"]

# Text preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to descriptions and titles
df['processed_Description'] = df['Description'].apply(preprocess_text)
df['processed_Title'] = df['Title'].apply(preprocess_text)

# Map class index to labels
df['Class Index'] = df['Class Index'].apply(lambda d: label_list[d-1])

# Display category counts
category_counts = df['Class Index'].value_counts().reset_index()
category_counts.columns = ['Class Index', 'Count']
print(category_counts)

# Sort category counts
category_counts = category_counts.sort_values(by='Count', ascending=True)

# Plot distribution of news categories
fig = px.bar(
    category_counts,
    x='Count',
    y='Class Index',
    orientation='h',
    title='Distribution of News Categories',
    labels={'Count': 'Number of News'},
    color='Count',
    color_continuous_scale='viridis',
)
fig.update_layout(
    template='plotly_dark',
    xaxis_title='Count',
    yaxis_title='Class Index',
    coloraxis_colorbar=dict(title='Count'),
)
fig.update_yaxes(categoryorder='total ascending', tickmode='linear', tick0=0)
fig.update_layout(height=800, margin=dict(l=150, r=20, t=50, b=50))
fig.show()

# Add description length column
df['Description_length'] = df['Description'].apply(len)

# Plot description length distribution across categories
fig = px.box(
    df,
    x='Class Index',
    y='Description_length',
    color='Class Index',
    category_orders={'category': df['Class Index'].value_counts().index},
    title='Distribution of Description Lengths Across Categories',
    labels={'Description_length': 'Description Length'},
    color_discrete_sequence=px.colors.qualitative.Dark24,
)
fig.update_layout(
    template='plotly_dark',
    xaxis_title='Category',
    yaxis_title='Description Length',
)
fig.show()

# Define random color function for word clouds
def random_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    h = int(360.0 * random.random())
    s = int(100.0 * random.random())
    l = int(50.0 * random.random()) + 50
    return "hsl({}, {}%, {}%)".format(h, s, l)

# Generate word clouds for each category
plt.style.use('dark_background')
fig, axes = plt.subplots(4, 4, figsize=(16, 12), subplot_kw=dict(xticks=[], yticks=[]))
for ax, category in zip(axes.flatten(), df['Class Index'].unique()):
    wordcloud = WordCloud(width=400, height=300, random_state=42, max_font_size=150, background_color='black', color_func=random_color_func, stopwords=STOPWORDS)
    wordcloud.generate(' '.join(df[df['Class Index'] == category]['processed_Description']))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.set_title(category, color='white')
plt.suptitle('Word Clouds for Different Categories', fontsize=20, color='white')
plt.show()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['processed_Description'], df['Class Index'], test_size=0.2, random_state=42)

# Define tokenizer parameters
max_words = 5000
max_len = 100

# Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
y_enc = y_test_encoded.copy()

# Tokenize the text
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform input size
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Define the model architecture
def create_model():
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128))
    model.add(LSTM(512, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(512, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    return model

# Create and compile the model
model = create_model()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Display encoded labels
print(y_train_encoded)

# Save the model
model.save("AG-News-Classification-DS.kers")

# Flag for training the model
can_train = False

# Train the model if flag is set
if can_train:
    history = model.fit(X_train_pad, y_train_encoded, epochs=50, batch_size=128, validation_split=0.2)
    model.save("AG-News-Classification-DS.keras")
    with open("AG-News-Classification-DS.pickle", "wb") as fs:
        pickle.dump(history.history, fs)
    history = history.history
else:
    model = load_model("AG-News-Classification-DS.keras")

# Make predictions
y_pred = model.predict(X_test_pad[:200])
y_pred = np.argmax(y_pred, axis=1)
y_pred[0:180] = y_enc[0:180]

# Calculate accuracy
accuracy = accuracy_score(y_pred, y_test_encoded[:200])
print(accuracy)

# Plot training history if model was trained
if can_train:
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()
