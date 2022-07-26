import click
import pandas as pd
from utils import resolve_encodings_and_normalize


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def normalize(input_filepath, output_filepath):
    df = pd.read_csv(input_filepath)
    df['discourse_text'] = df['discourse_text'].apply(lambda x: resolve_encodings_and_normalize(x))
    df['essay_text'] = df['essay_text'].apply(lambda x: resolve_encodings_and_normalize(x))
    df['text'] = df['discourse_type'] + ' ' + df['discourse_text'] + '[SEP]' + df['essay_text']
    df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    normalize()
