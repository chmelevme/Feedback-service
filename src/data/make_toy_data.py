import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath_train', type=click.Path())
@click.argument('output_filepath_test', type=click.Path())
@click.argument('seed', type=click.INT, default=101)
@click.argument('items_count', type=click.INT, default=4)
def get_toy_data(input_filepath, output_filepath_train, output_filepath_test, seed, items_count):
    df = pd.read_csv(input_filepath)
    train_idx, test_idx = train_test_split(df.essay_id.unique(), test_size=0.2, random_state=seed)
    train_idx = train_idx[:items_count]
    test_idx = test_idx[:items_count]
    train = df[df['essay_id'].isin(train_idx)]
    test = df[df['essay_id'].isin(test_idx)]
    train.to_csv(output_filepath_train, index=False)
    test.to_csv(output_filepath_test, index=False)


if __name__ == '__main__':
    get_toy_data()
