import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from TimingCallback import TimingCallback


def train_model(model, X_train, X_test, Y_train, Y_test, epochs=1000, batch_size=512, validation_split=0.05):
    cb = TimingCallback()
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[cb])
    model.evaluate(X_test, Y_test)
    y_pred = model.predict(X_test)
    pred = np.argmax(y_pred, axis=1)
    pred = np.reshape(pred, (len(pred), 1))
    precision = precision_score(Y_test, pred, average=None)
    recall = recall_score(Y_test, pred, average=None)
    f1score = f1_score(Y_test, pred, average=None)
    Total_training_time=sum(cb.logs)
    report = classification_report(Y_test, pred)
    return precision, recall, f1score, Total_training_time, report
