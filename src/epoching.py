import numpy as np

def epoch(data, sample_rate, epoch_length, overlap=0.5):

    epoch_samples = int(sample_rate * epoch_length)
    epoch_overlap = int(epoch_samples * overlap)

    data_slices = np.array(
        [
            data[i: i + epoch_samples]
            for i in range(0, len(data) - epoch_samples, epoch_overlap)
        ]
    )

    return data_slices
