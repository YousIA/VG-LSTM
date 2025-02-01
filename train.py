from lrSchedule import lr_schedule
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.optimizers import Adam


def train_model(model, train_features, train_labels):
    lr_scheduler = LearningRateScheduler(lr_schedule)
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_features, train_labels, epochs=1000, batch_size=128,
                        callbacks=[lr_scheduler, early_stopping])
    return model, history


def train_model_in(model, train_features, train_labels):
    lr_scheduler = LearningRateScheduler(lr_schedule)
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_features, train_labels, epochs=1000, batch_size=128,
                        callbacks=[lr_scheduler, early_stopping])
    return model, history
