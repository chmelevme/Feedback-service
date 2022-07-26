import pandas as pd
import click
from utils import get_essay

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('is_train', type=click.BOOL)
def add_essay(input_filepath, output_filepath, is_train=True):
    df = pd.read_csv(input_filepath)
    df['essay_text'] = df['essay_id'].apply(lambda x: get_essay(x, is_train=is_train))
    df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    add_essay()
