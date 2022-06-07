import pandas as pd
import matplotlib.pyplot as plt


def load_data(path: str):
    df = pd.read_csv(path, index_col=0)
    return df


def plot_accuracy(df: pd.DataFrame) -> plt:
    plt.figure()
    plt.plot(df['accuracy'])
    plt.plot(df['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    return plt


def plot_loss(df: pd.DataFrame) -> plt:
    plt.figure()
    plt.plot(df['loss'])
    plt.plot(df['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    return plt


if __name__ == '__main__':
    accuracy_path = snakemake.output[0]
    loss_path = snakemake.output[1]
    data: pd.DataFrame = load_data(snakemake.input[0])
    plot_accuracy(data).savefig(accuracy_path)
    plot_loss(data).savefig(loss_path)


