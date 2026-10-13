# A 2D eRTIS energyscape (horizontal plane): range x azimuth, 10 log10 of the mean envelope power per cell (dB).
std_msgs/Header header       # frame_id: the sensor frame (x forward, y left, z up)
string method                # das | dmas2 | dmas2cf | dmas3cf | deconv
float32[] ranges             # m, cell centres (R)
float32[] azimuths           # rad, positive to the left (D)
float32[] data               # R x D, row-major (range-major), dB
