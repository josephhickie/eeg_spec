import mne


def fieldtrip_to_dict(filename, channel_names, sample_frequency, channel_types):
    info = mne.create_info(ch_names=channel_names, sfreq=sample_frequency, ch_types=channel_types)

    # Load the EEG data using read_raw_fieldtrip() and pass the info object as an argument
    raw = mne.io.read_raw_fieldtrip(filename, info=info)
    raw_data = raw.get_data()

    data_dict = {channel_name: {'data': data, 'type': channel_type}
            for channel_name, data, channel_type
            in zip(channel_names, raw_data, channel_types)}

    return data_dict



