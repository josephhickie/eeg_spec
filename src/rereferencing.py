import numpy as np


def generate_weighted_sum(reference_channels, dataset, weighting=np.array([0.5, 0.5])):
    assert len(reference_channels) == len(weighting)

    rereference_data = np.array([dataset.get(reference_channel).get('data') for reference_channel in reference_channels])
    weighted_sum = np.sum(rereference_data.T * np.array(weighting), axis=1)


    return weighted_sum


def rereference(reference_channels, dataset, weighting=np.array([0.5, 0.5])):
    reference_data = generate_weighted_sum(reference_channels, dataset, weighting)

    return {channel_name:
                {'data': dataset.get(channel_name).get('data') - reference_data,
                 'type': dataset.get(channel_name).get('type')}
            for channel_name in dataset.keys()}




