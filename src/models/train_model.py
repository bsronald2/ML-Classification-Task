# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from src.features.SelectFeatures import SelectFeatures

from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.features.SelectFeatures import SelectFeatures
from src.models.TrainingModel import TrainingModel


@click.command()
@click.argument('features_filepath', type=click.Path(exists=True))
@click.argument('model')
def main(features_filepath, model):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    features = SelectFeatures.load_features(features_filepath)
    tm = TrainingModel(features['features'])
    tm.training_model(model)


    logger.info('Training model')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

