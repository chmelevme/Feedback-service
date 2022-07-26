import pandas as pd
from sklearn.preprocessing import LabelEncoder
import click
import pickle


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('label', type=click.STRING)
def encode_label(input_filepath, output_filepath, label):
    df = pd.read_csv(input_filepath)
    le = LabelEncoder()
    le.fit(df[label])
    df[label] = le.transform(df[label])
    df.to_csv(output_filepath, index=False)
    with open('models/label_encoder.sk', 'wb') as f:
        pickle.dump(le, f)


if __name__ == '__main__':
    encode_label()
