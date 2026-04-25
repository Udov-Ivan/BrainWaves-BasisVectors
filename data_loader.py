import mne
from mne.datasets import eegbci


edf_files = eegbci.load_data(subjects=[1], runs=[2]) #resting state (eyes closed)

raw = mne.io.read_raw_edf(edf_files[0], preload=True)


raw.pick_types(eeg=True)


data = raw.get_data()

print(f"Форма матриці даних: {data.shape}")
# expected output: (64, 15360)
# 64 channels (rows), 15360 time entries (columns)
print(data[0, :10])