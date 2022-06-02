import pandas as pd
import matplotlib.pyplot as plt


def load_data(path: str):
    df = pd.read_csv(path, index_col=0)
    return df


def plot_accuracy(df: pd.DataFrame) -> plt:
    graph = plt.figure()
    plt.plot(df['accuracy'])
    plt.plot(df['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    return plt


def plot_loss(df: pd.DataFrame) -> plt:
    graph = plt.figure()
    plt.plot(df['loss'])
    plt.plot(df['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    return plt


if __name__ == '__main__':
    data: pd.DataFrame = load_data("../data/history.csv")
    plot_accuracy(data).savefig("../data/graphs/accuracy.png")
    plot_loss(data).savefig("../data/graphs/loss.png")


