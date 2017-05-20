# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import BarrowmanFlow as bf

################## User Input ###################
length_body = 2.7 # [m] from nose tip to body. without tail
diameter_body = 0.154 # [m]
length_cg = 0.885 # [m] from nose tip

shape_nose = 'double' # 'ogive' or 'double' or 'parabolic' or 'ellipse'
length_nose = 0.305 # [m]

diameter_tail = 0.1 # [m]
length_tail = 0.1 # [m]

offset_fin = 0.0 # [mm] from body end to fin end
root_chord = 200.0 # [mm]
tip_chord = 90.0 # [mm]
leading_edge_chord = root_chord - tip_chord
span = 130.0 # [mm]
thickness_fin = 2.0 # [mm]
young_modulus = 3.0 # [GPa]
poisson_ratio = 0.3 # [-]
max_altitude = 10000.0 # [m]
#################################################

def mm2m(value):
  return value / 1000.0

offset_fin = mm2m(offset_fin)
root_chord = mm2m(root_chord)
tip_chord = mm2m(tip_chord)
leading_edge_chord = mm2m(leading_edge_chord)
span = mm2m(span)
thickness_fin = mm2m(thickness_fin)

bf.initialize(diameter_body, length_body)
nose = bf.Nose(shape_nose, length_nose)
fin = bf.Fin(root_chord, tip_chord, leading_edge_chord, span, length_body-offset_fin-root_chord)
fin.flutter_speed(young_modulus, poisson_ratio, thickness_fin, max_altitude)
tail = bf.TaperBody(diameter_body, diameter_tail, length_tail, length_body)
stage = bf.integral(length_cg, nose, fin, tail)

print('*=============Result==============*')
print('Length of C.P.:', stage.Lcp, '[m]')
print('Coefficient of Normal Force:', stage.CNa, '[deg^-1]')
print('Coefficient of Pitch Damping Moment:', stage.Cmq, '[-]')
print('Flutter Velocity:', np.max(fin.Vf), '[m/s]')
print('*=================================*')

stage.plot()



