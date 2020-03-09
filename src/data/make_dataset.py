# -*- coding: utf-8 -*-
import click
import logging
from src.data.PrePProcessing import PrePProcessing
from src.features.SelectFeatures import SelectFeatures
from src.data.SelectData import SelectData

from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    # Split data
    sd = SelectData(input_filepath, output_filepath)
    sd.select_data()
    # Pre-processing train data
    train_pp = PrePProcessing(sd.get__training_features(), sd.get__training_target(), "training")
    train_pp.data_info()
    train_pp.clean_data()
    train_pp.pre_processing()

    # # Pre-processing test data
    test_pp = PrePProcessing(sd.get__test_features(), sd.get__test_target(), "test")
    test_pp.pre_processing()


    #  python src/data/make_dataset.py data/raw/bank-tr.csv data/processed

    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
