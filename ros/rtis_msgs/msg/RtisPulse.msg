# One eRTIS measurement: the microphone signals after PDM demodulation (input of the imaging node).
std_msgs/Header header       # frame_id: the sensor frame
float32 fs                   # Hz (450 kHz)
uint16 n_mics
uint32 n_samples
float32[] data               # n_mics x n_samples, row-major
string array_id              # which array geometry / calibration (data/array.json of the project)
