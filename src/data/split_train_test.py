import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath_train', type=click.Path())
@click.argument('output_filepath_test', type=click.Path())
@click.argument('seed', type=click.INT, default=101)
def split_train_test(input_filepath, output_filepath_train, output_filepath_test, seed):
    df = pd.read_csv(input_filepath)
    train_idx, test_idx = train_test_split(df.essay_id.unique(), test_size=0.2, random_state=seed)
    train = df[df['essay_id'].isin(train_idx)]
    test = df[df['essay_id'].isin(test_idx)]
    train.to_csv(output_filepath_train, index=False)
    test.to_csv(output_filepath_test, index=False)


if __name__ == '__main__':
    split_train_test()
