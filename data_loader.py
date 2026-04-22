import mne
from mne.datasets import eegbci


edf_weights = eegbci.load_data(subjects=[1], runs=[1])

raw = mne.io.read_raw_edf(edf_weights[0], preload=True)


raw.pick_types(eeg=True)


data = raw.get_data()

print(f"Форма матрицы данных: {data.shape}")
# expected output: (64, 15360)
# 64 channels (rows), 15360 time entries (columns)
print(data[0, :10])