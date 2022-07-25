import click
import pandas as pd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def drop_duplicates(input_filepath, output_filepath):
    df = pd.read_csv(input_filepath)
    df.drop_duplicates('discourse_text')
    df.to_csv(output_filepath)


if __name__ == '__main__':
    drop_duplicates()
