import pandas as pd 
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, Flatten, concatenate, Input, Dropout, LSTM, Bidirectional, BatchNormalization, PReLU, ReLU, Reshape
from keras.layers import Conv1D, MaxPooling1D, GRU
from keras.models import Model
from keras.callbacks import EarlyStopping
import tensorflow as tf 

data = pd.read_csv('/Users/jungyoonlim/EEG_emotions/data/emotions.csv')

# Label Maps
label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}

# Feature Engineering
def preprocess_inputs(df):
    df = df.copy()
    df['label'] = df['label'].replace(label_mapping)
    
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.4, random_state=123)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_inputs(data)

# Model Architecture
inputs = tf.keras.Input(shape=(X_train.shape[1], 1))

x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Flatten()(x)

x = GRU(256, return_sequences=True)(x)
x = Dropout(0.5)(x)

outputs = Dense(3, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, batch_size=32, epochs=50, callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

# Evaluate the model
model_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print("Test Accuracy: {:.3f}%".format(model_acc * 100))