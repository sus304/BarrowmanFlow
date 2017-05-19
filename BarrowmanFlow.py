# -*- coding: utf-8 -*-
import re
import numpy as np
import matplotlib.pyplot as plt

class AeroObj:
  d_body = 0.0
  l_body = 0.0
  Lcg = 0.0

  def __init__(self):
    if AeroObj.d_body == 0.0:
      print('Error:Not initialized. BarrowmanFlow.initialize(...)')
    self.CNa = 0.0
    self.inertia_coefficient = 0.0
    self.Cmq = 0.0
    self.Cnr = 0.0
    self.Lcp = 0.0

class Nose(AeroObj):
  def __init__(self, shape, l_nose):
    super().__init__()
    self.LD = l_nose / AeroObj.d_body

    doublecone_pattern = r'double|double ?cone|cone'
    ogive_pattern = r'ogive|ogive ?cone'
    parabolic_pattern = r'parabolic|ellipse'
    if re.compile(doublecone_pattern, re.IGNORECASE).match(shape):
      CP_coefficient = 2.0 / 3.0
    elif re.compile(ogive_pattern, re.IGNORECASE).match(shape):
      CP_coefficient = 1.0 - ((8.0 * self.LD ** 2 / 3.0) + ((4.0 * self.LD ** 2 - 1.0) ** 2 / 4.0) - (((4.0 * self.LD ** 2 - 1.0) * (4.0 * self.LD ** 2 + 1.0) ** 2 / (16.0 * self.LD)) * np.arcsin(4.0 * self.LD / (4.0 * self.LD ** 2 + 1.0))))
    elif re.compile(parabolic_pattern, re.IGNORECASE).match(shape):
      CP_coefficient =  0.5
    
    self.CNa = 2.0
    self.Lcp = l_nose * CP_coefficient

class TaperBody(AeroObj):
  def __init__(self, d_before, d_after, l_taper, distance_fromNoseTip):
    super().__init__()
    self.d_before = d_before
    self.d_after = d_after
    self.l_taper = l_taper
    self.distance = distance_fromNoseTip

    self.CNa = 2.0 * ((d_after / d_before) ** 2 - 1.0)
    self.Lcp = distance_fromNoseTip + (l_taper / 3.0) * (1.0 + ((1.0 - (d_before / d_after)) / (1.0 - (d_before / d_after) ** 2)))


class Fin(AeroObj):
  def __init__(self, Cr, Ct, Cle, span, distance_fromNoseTip_toRootchordLeadingEdge):
    # input unit [m]
    # Cr:Root Chord
    # Ct:Tip Chord
    # Cle:Leading Edge Chord
    super().__init__()
    self.Cr = Cr
    self.Ct = Ct
    self.Cle = Cle
    self.span = span
    self.distance = distance_fromNoseTip_toRootchordLeadingEdge
    
    if Cle+0.5*Ct == 0.5*Cr:
      mid_chord_line = span
    elif Cle+0.5*Ct > 0.5*Cr:
      mid_chord_line = np.sqrt(span ** 2 + (0.5 * Ct + Cle - 0.5 * Cr) ** 2)
    else:
      mid_chord_line = np.sqrt(span ** 2 + (0.5 * Cr - Cle - 0.5 * Ct) ** 2)

    CNa_single = 16.0 * (span / AeroObj.d_body) ** 2 / (1.0 + np.sqrt(1.0 + (2.0 * mid_chord_line / (Cr + Ct)) ** 2)) # 4fins
    Kfb = 1.0 + 0.5 * AeroObj.d_body / (0.5 * AeroObj.d_body + span) # interference fin and body
    self.CNa = CNa_single * Kfb

    ramda = Ct / Cr
    MAC = 2.0 / 3.0 * Cr * (1 + ramda ** 2 / (1.0 + ramda))
    self.Lcp = self.distance + (Cle * (Cr + 2.0 * Ct) / (3.0 * (Cr + Ct))) + MAC / 4.0

  def flutter_speed(self, young, poisson, thickness, altitude=0.0):
    # ref. NACA Technical Note 4197
    # young:Young`s modulus [GPa]
    # possion:Poisson Ratio
    # thickness:Fin thickness [m]

    def Std_Atmo(altitude):
      # ref. 1976 standard atmosphere
      # ジオポテンシャル高度を基準として標準大気の各層の気温減率から各大気値を算出
      # 高度86 kmまで対応
      # altitude [m]
      R = 287.1
      gamma = 1.4
      Re = 6378.137e3 # Earth Radius [m]
      g0 = 9.80665

      # atmospheric layer
      h_list  = [0.0, 11.0e3, 20.0e3, 32.0e3, 47.0e3, 51.0e3, 71.0e3, 84.852e3] # geopotential height [m]
      TG_list = [-6.5e-3, 0.0, 1.0e-3, 2.8e-3, 0, -2.8e-3, -2.0e-3, 0.0] # Temp. gradient [K/m]
      T_list  = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946] # [K]
      P_list  = [101325.0, 22632.0, 5474.9, 868.02, 110.91, 66.939, 3.9564, 0.3734] # [Pa]

      h = altitude * Re / (Re + altitude) # geometric altitude => geopotential height

      k = 0 # dafault layer
      for i in range(8):
        if h < h_list[i]:
          k = i - 1
          break
        elif h >= h_list[7]:
          k = 7
          break
      
      temperature = T_list[k] + TG_list[k] * (h - h_list[k]) # [K]
      if TG_list[k] == 0.0:
        pressure = P_list[k] * np.exp(g0 / R * (h_list[k] - h) / T_list[k])
      else:
        pressure = P_list[k] * np.power(T_list[k] / temperature, g0 / R / TG_list[k]) # [Pa]
      density = pressure / (R * temperature) # [kg/m^3]
      soundspeed = np.sqrt(gamma * R * temperature) # [m/s]
      
      return temperature, pressure, density, soundspeed

    AR = 2.0 * self.span / (self.Cr + self.Ct) # aspect ratio
    ramda = self.Ct / self.Cr # taper ratio
    shear = young / (2.0 * (1.0 + poisson)) # shear modulus
    self.Vf = [Std_Atmo(alt)[3] * np.sqrt(shear * 10.0 ** 9 / ((1.337 * AR ** 3 * Std_Atmo(alt)[1] * (ramda + 1.0)) / (2.0 * (AR + 2.0) * (thickness / self.Cr) ** 3))) for alt in np.arange(0.0, altitude+10.0, 10.0)]


def initialize(d_body, l_body, Lcg):
  AeroObj.d_body = d_body
  AeroObj.l_body = l_body
  AeroObj.Lcg = Lcg

def integral(*components):
  CNa = 0.0
  Cmq = 0.0
  Cnr = 0.0 # Non Use : damping Roll
  Lcp = 0.0
  for obj in components:
    CNa += obj.CNa
    Lcp += obj.CNa * obj.Lcp
    Cmq -= 4.0 * (0.5 * obj.CNa * ((obj.Lcp - AeroObj.Lcg) / AeroObj.l_body) ** 2)
  Lcp /= CNa

  return CNa, Cmq, Lcp

class Graph:
  # グラフによる機体形状の可視化
  # 単純な単段ロケットのみ対応
  # @ToDo:汎用性を上げる
  def add_point(self, array, x, y):
    return np.vstack((array, np.array([x, y])))

  def add_body_reverse(self, point_list):
    point_parse = point_list[:-1,:]
    for point in point_parse[::-1]:
      point_list = self.add_point(point_list, point[0], point[1] * (-1))
    return point_list

  def add_fin_reverse(self, point_list):
    point_list = self.add_point(point_list, point_list[0,0], point_list[0,1]) # 1st fin close   
    point_2nd = np.array([point_list[0,0], -point_list[0,1]])
    for point in point_list[1:]:
      point_2nd = self.add_point(point_2nd, point[0], point[1] * (-1)) # 2nd fin point
    return point_list, point_2nd

  def __init__(self, d_body, l_body, Lcg, Lcp, *components):
    self.r_body = 0.5 * d_body
    self.l_body = l_body
    

    for obj in components:
      if isinstance(obj, Nose):
        if hasattr(self, 'point_nose'):
          pass
        else:
          self.point_nose = np.array([0.0, 0.0])
        self.point_nose = self.add_point(self.point_nose, obj.LD*d_body, self.r_body)
        self.point_nose = self.add_point(self.point_nose, obj.LD*d_body, 0.0)
        self.point_nose = self.add_body_reverse(self.point_nose)        

      elif isinstance(obj, TaperBody):
        if hasattr(self, 'point_taper'):
          pass
        else:
          self.point_taper = np.array([obj.distance, 0.0])
        self.point_taper = self.add_point(self.point_taper, obj.distance, 0.5*obj.d_before)
        self.point_taper = self.add_point(self.point_taper, obj.distance+obj.l_taper, 0.5*obj.d_after)
        self.point_taper = self.add_point(self.point_taper, obj.distance+obj.l_taper, 0.0)
        self.point_taper = self.add_body_reverse(self.point_taper)

      elif isinstance(obj, Fin):
        if hasattr(self, 'point_fin'):
          pass
        else:
          self.point_fin = np.array([obj.distance, self.r_body])
        self.point_fin = self.add_point(self.point_fin, obj.distance+obj.Cle, self.r_body+obj.span)
        self.point_fin = self.add_point(self.point_fin, obj.distance+obj.Cle+obj.Ct, self.r_body+obj.span)
        self.point_fin = self.add_point(self.point_fin, obj.distance+obj.Cr, self.r_body)
        self.point_fin, self.point_fin_2nd = self.add_fin_reverse(self.point_fin)
  
  def add_body(self):
    start_x = max(self.point_nose[:,0])
    end_x = self.l_body
    self.point_body = np.array([start_x, -self.r_body])
    self.point_body = self.add_point(self.point_body, start_x, self.r_body)
    self.point_body = self.add_point(self.point_body, end_x, self.r_body)
    self.point_body = self.add_point(self.point_body, end_x, -self.r_body)
    self.point_body = self.add_point(self.point_body, start_x, -self.r_body)

  def plot(self):
    plt.close('all')
    plt.figure(0, figsize=(8, 4))
    xmax = 0.0
    ymax = 0.0
    self.add_body()
    for point_list in [self.point_body, self.point_nose, self.point_taper, self.point_fin, self.point_fin_2nd]:
      try:
        plt.plot(point_list[:,0], point_list[:,1], color='black')
        if max(point_list[:,0]) > xmax:
          xmax = max(point_list[:,0])
        if max(point_list[:,1]) > ymax:
          ymax = max(point_list[:,1])
      except:
        pass
    print(xmax, ymax)
    plt.xlim([0.0, np.ceil(xmax+0.5)])
    plt.ylim([np.floor(-ymax-0.2), np.ceil(ymax+0.2)])
    ax = plt.gca()
    aspect = 1.0
    ax.set_aspect(aspect)
    plt.grid()
    plt.show()
