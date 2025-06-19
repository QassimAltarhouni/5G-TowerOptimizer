import pandas as pd

def load_opencellid_data(filepath):
    columns = [
        'radio', 'mcc', 'net', 'area', 'cell', 'unit',
        'lon', 'lat', 'range', 'samples', 'changeable',
        'created', 'updated', 'averageSignal'
    ]
    return pd.read_csv(filepath, header=None, names=columns)
