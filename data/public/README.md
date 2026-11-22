# Public benchmark data

`recofit_subset.pkl.xz` (9.1 MB) holds the 114 strength-exercise recordings that
`python -m vbt.public` benchmarks on, with their values unmodified, taken from:

> Dan Morris, T. Scott Saponas, Andrew Guillory, Ilya Kelner. "RecoFit: Using a Wearable
> Sensor to Find, Recognize, and Count Repetitive Exercises." CHI 2014.
> Data: Microsoft, https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors
> (file `exercise_data.50.0000_singleonly.mat`, SHA-256
> 70eea039362555daccb00065e3775ce37fff421f18d533c60933a9d859f33f8a).

Licence: **Community Data License Agreement – Permissive, Version 2.0**
(https://cdla.dev/permissive-2-0/). This subset is shared under the same licence. The
attribution and licence link are also stored inside the file (`"license"` key).

Format: an xz-compressed pickle, `{"license", "source", "source_sha256", "recs": [...]}`,
one record per recording: subject, activity, true rep count, and the raw 50 Hz
accelerometer and gyroscope matrices. Rebuild it from the full download with
`python -m vbt.public --data <dir outside the repo> --rebuild-cache`.
