# -*- coding: utf-8 -*-
'''
MIT License
Copyright (c) 2017 Susumu Tanaka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import re
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class Nose:
    def __init__(self, stage, shape, l_nose):
        self.barrowman_method(stage, shape, l_nose)
    
    def barrowman_method(self, stage, shape, l_nose):
        self.Cmq = 0.0
        self.Clp = 0.0
        self.Cnr = 0.0

        self.LD = l_nose / stage.d_body # copy for graph plot
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

    def tm30_method(self, stage, shape, l_nose):
        self.LD = l_nose / stage.d_body
        k = stage.l_body / stage.d_body

        Xcp = 1 - Vol / (area * l_nose)


class TaperBody:
    def __init__(self, stage, d_before, d_after, l_taper, distance_fromNoseTip):
        self.barrowman_method(stage, d_before, d_after, l_taper, distance_fromNoseTip)

    def barrowman_method(self, stage, d_before, d_after, l_taper, distance_fromNoseTip):
        self.Cmq = 0.0
        self.Clp = 0.0
        self.Cnr = 0.0

        self.d_before = d_before # copy for graph plot
        self.d_after = d_after
        self.l_taper = l_taper
        self.distance = distance_fromNoseTip

        self.CNa = 2.0 * ((d_after / d_before) ** 2 - 1.0)
        self.Lcp = distance_fromNoseTip + (l_taper / 3.0) * (1.0 + ((1.0 - (d_before / d_after)) / (1.0 - (d_before / d_after) ** 2)))


class Fin:
    def __init__(self, stage, Cr, Ct, Cle, span, distance_fromNoseTip_toRootchordLeadingEdge):
        self.barrowman_method(stage, Cr, Ct, Cle, span, distance_fromNoseTip_toRootchordLeadingEdge)

        # theta_trailing = np.arctan2(tip + leading - root, span)
        # if -5.0 < np.rad2deg(theta_trailing) < 5.0:
        #     self.tm30_method()
        # else:
        #     self.barrowman_method()
        

    def barrowman_method(self, stage, Cr, Ct, Cle, span, distance_fromNoseTip_toRootchordLeadingEdge):
        self.Cmq = 0.0
        self.Clp = 0.0
        self.Cnr = 0.0

        # input unit [m]
        # Cr:Root Chord
        # Ct:Tip Chord
        # Cle:Leading Edge Chord
        self.Cr = Cr # copy for graph plot
        self.Ct = Ct
        self.Cle = Cle
        self.span = span
        self.distance = distance_fromNoseTip_toRootchordLeadingEdge

        # フィン形状による分岐
        if Cle+0.5*Ct == 0.5*Cr:
            mid_chord_line = span
        elif Cle+0.5*Ct > 0.5*Cr:
            mid_chord_line = np.sqrt(span ** 2 + (0.5 * Ct + Cle - 0.5 * Cr) ** 2)
        else:
            mid_chord_line = np.sqrt(span ** 2 + (0.5 * Cr - Cle - 0.5 * Ct) ** 2)

        CNa_single = 16.0 * (span / stage.d_body) ** 2 / (1.0 + np.sqrt(1.0 + (2.0 * mid_chord_line / (Cr + Ct)) ** 2)) # 4fins
        Kfb = 1.0 + 0.5 * stage.d_body / (0.5 * stage.d_body + span) # interference fin and body
        self.CNa = CNa_single * Kfb

        ramda = Ct / Cr
        MAC = 2.0 / 3.0 * Cr * (1 + ramda ** 2 / (1.0 + ramda)) # Mean Aerodynamic Chord
        self.Lcp = self.distance + (Cle * (Cr + 2.0 * Ct) / (3.0 * (Cr + Ct))) + MAC / 4.0
        self.Clp = -4.0 * 2.0 * (span + 0.5 * stage.d_body) ** 4 / (np.pi * stage.l_body ** 2 * (0.25 * stage.d_body ** 2 * np.pi))

    def tm30_method(self, stage, Cr, Ct, Cle, span, mach):
        def func_CLa_A(betaA, lamda, mach):
            if mach < 1.0:
                betaA_array = np.arange(0.0, 5.1, 0.1)
                lamda025 = [1.57036, 1.55605, 1.54379, 1.53162, 1.51758, 1.50042, 1.48241, 1.46485, 1.44680, 1.42227, 
                1.39319, 1.37207, 1.34670, 1.31996, 1.30313, 1.28074, 1.25601, 1.23624, 1.21747, 1.19771, 1.17764, 1.15872, 
                1.14135, 1.12521, 1.10957, 1.09279, 1.07433, 1.05885, 1.04745, 1.03567, 1.02009, 1.00383, 0.99001, 0.97772, 
                0.96560, 0.95283, 0.93941, 0.92650, 0.91596, 0.90591, 0.89312, 0.87967, 0.86881, 0.85929, 0.84711, 0.83242, 
                0.81711, 0.80313, 0.79238, 0.78680, 0.78831]
                inter = interpolate.interp1d(betaA_array, lamda025, kind='cubic', bounds_error=False, fill_value='extrapolate')
                if min(betaA_array) < betaA < max(betaA_array):
                    pass
                else:
                    print('Attention!! Effective aspect ratio is out of range, drops precision.')
                return inter(betaA)
            else:
                betaA_array = betaA_array = np.arange(0.0, 8.1, 0.1)
                lamda0 = [1.56702, 1.56659, 1.56400, 1.55909, 1.55170, 1.54203, 1.53097, 1.51872, 1.50484, 1.48944, 1.47370, 
                1.45857, 1.44422, 1.43022, 1.41484, 1.39604, 1.37583, 1.35832, 1.34460, 1.33003, 1.31153, 1.29156, 1.27120, 
                1.25377, 1.24348, 1.22980, 1.20873, 1.19278, 1.18078, 1.16792, 1.15308, 1.13689, 1.12001, 1.10311, 1.08699, 
                1.07230, 1.05913, 1.04741, 1.03648, 1.02321, 1.00397, 0.97779, 0.94945, 0.92625, 0.90747, 0.88844, 0.86826, 
                0.85062, 0.83551, 0.82057, 0.80458, 0.78811, 0.77188, 0.75655, 0.74275, 0.73078, 0.71925, 0.70633, 0.69184, 
                0.67778, 0.66577, 0.65563, 0.64674, 0.63841, 0.62981, 0.62008, 0.60934, 0.59870, 0.58929, 0.58121, 0.57380, 
                0.56634, 0.55844, 0.55034, 0.54236, 0.53484, 0.52808, 0.52243, 0.51820, 0.51572, 0.51531]
                lamda025 = [1.57732, 1.57722, 1.58662, 1.60219, 1.62062, 1.64034, 1.66716, 1.69627, 1.72096, 1.74430, 1.75044, 
                1.74865, 1.73997, 1.72417, 1.70424, 1.68281, 1.66099, 1.63754, 1.61009, 1.57938, 1.55235, 1.52530, 1.47978, 
                1.43989, 1.40948, 1.38003, 1.34568, 1.30957, 1.27639, 1.24629, 1.21740, 1.18822, 1.15881, 1.12973, 1.10147, 
                1.07437, 1.04871, 1.02474, 1.00238, 0.98120, 0.96078, 0.94080, 0.92125, 0.90216, 0.88363, 0.86606, 0.84994, 
                0.83503, 0.82006, 0.80378, 0.78688, 0.77185, 0.75921, 0.74729, 0.73435, 0.71931, 0.70544, 0.69455, 0.68404, 
                0.67184, 0.65910, 0.64773, 0.63842, 0.63002, 0.62147, 0.61260, 0.60344, 0.59400, 0.58441, 0.57513, 0.56669, 
                0.55957, 0.55351, 0.54786, 0.54196, 0.53515, 0.52677, 0.51616, 0.50266, 0.48561, 0.46435]
                lamda05 = [1.56084, 1.58408, 1.60385, 1.62677, 1.65945, 1.70528, 1.75740, 1.80196, 1.84772, 1.88648, 1.90172, 
                1.89894, 1.88659, 1.86786, 1.84167, 1.80370, 1.75858, 1.71676, 1.66759, 1.61842, 1.57281, 1.52623, 1.47947, 
                1.44012, 1.40924, 1.38004, 1.34577, 1.30958, 1.27636, 1.24626, 1.21740, 1.18823, 1.15882, 1.12973, 1.10147, 
                1.07437, 1.04870, 1.02474, 1.00238, 0.98120, 0.96078, 0.94080, 0.92125, 0.90216, 0.88363, 0.86606, 0.84994, 
                0.83503, 0.82006, 0.80378, 0.78688, 0.77185, 0.75921, 0.74729, 0.73435, 0.71931, 0.70544, 0.69455, 0.68404, 
                0.67184, 0.65910, 0.64773, 0.63842, 0.63002, 0.62147, 0.61260, 0.60344, 0.59400, 0.58441, 0.57513, 0.56669, 
                0.55957, 0.55351, 0.54786, 0.54196, 0.53515, 0.52677, 0.51616, 0.50266, 0.48561, 0.46435]
                lamda1 = [1.55000, 1.58674, 1.61519, 1.66578, 1.70515, 1.76834, 1.83127, 1.89478, 1.95099, 1.98062, 1.99941, 
                1.99180, 1.96729, 1.93401, 1.89161, 1.82069, 1.76530, 1.68998, 1.63952, 1.57498, 1.51720, 1.46454, 1.41749, 
                1.37325, 1.32615, 1.29077, 1.24999, 1.21516, 1.17876, 1.14777, 1.11759, 1.08517, 1.05627, 1.03425, 1.00451, 
                0.98067, 0.96237, 0.94348, 0.92218, 0.90194, 0.88304, 0.86314, 0.84625, 0.83019, 0.81199, 0.79548, 0.78047, 
                0.76599, 0.75469, 0.74144, 0.71862, 0.70263, 0.69636, 0.69028, 0.67756, 0.65963, 0.64779, 0.64120, 0.63297, 
                0.62242, 0.61273, 0.60460, 0.59635, 0.58655, 0.57584, 0.56573, 0.55769, 0.55194, 0.54714, 0.54188, 0.53546, 
                0.52811, 0.52020, 0.51325, 0.50931, 0.50728, 0.50306, 0.49498, 0.48570, 0.47831, 0.47588]
                CLaA_lists = [lamda0, lamda025, lamda05, lamda1]
                lamda_list = []
                for CLaA_list in CLaA_lists:
                    inter = interpolate.interp1d(betaA_array, CLaA_list, kind='cubic', bounds_error=False, fill_value='extrapolate')
                    lamda_list.append(inter(betaA))
                inter = interpolate.interp1d([0.0, 0.25, 0.5, 1.0], lamda_list, kind='linear', bounds_error=False, fill_value='extrapolate')
                if min(betaA_array) < betaA < max(betaA_array):
                    pass
                else:
                    print('Attention!! Effective aspect ratio is out of range, drops precision.')
                return inter(lamda)
        d = stage.d_body
        r = 0.5 * d  # body radius
        s = span + r
        S = 2.0 * (0.5 * (Cr + Ct) * span)  # area
        A = (2 * span) ** 2 / S  # aspect ratio
        lamda = Ct / Cr  # taper ratio
        # theta_trailing = np.arctan2(tip + leading - root, span)  # 後縁後退角
        # theta_leading = np.arctan2(Cle, span)  # 前縁後退角
        beta = np.sqrt(np.abs(mach ** 2 - 1))
        betaA = beta * A  # effective aspect
        CLa_A = func_CLa_A(betaA, lamda, mach)
        Hw = (8.0 / np.pi ** 2) * ((s**2 / r**2 + r**2 / s**2) * (0.5 * np.arctan(0.5 * (s / r - r / s)) + 0.25 * np.pi) - (s / r - r / s) - 2.0 * np.arctan2(r, s))
        self.CL_fin = Hw * CLa_A
        if mach > 1.0 and Cr/beta > d:
            m = span / Cle
            mbeta = m * beta
            mCr_d1 = m * Cr / d + 1.0
            betad = beta * d
            if mbeta > 1:
                Hb = ((32.0 / np.pi**2) / np.sqrt(mbeta**2 - 1) * (betad / Cr) 
                * (mCr_d1**2 * np.arccos((mbeta + Cr / betad) / mCr_d1) - mbeta**2 * (Cr / betad)**2 * np.arccos(1 / mbeta) 
                + mbeta * (Cr / betad)**2 * np.sqrt(mbeta**2 - 1) * np.arcsin(betad / Cr) - np.sqrt(mbeta**2 - 1) * np.arccosh(Cr / betad)))
                * (s / r - 1) / (beta * CLa_A * A * (lamda + 1))
            elif mbeta < 1:
                Hb = ((64.0 / np.pi**2) * np.sqrt(mbeta) / (mbeta + 1) * (betad / Cr) * (mCr_d1 * np.sqrt((Cr / betad - 1) * mCr_d1) 
                - (Cr / betad)**2 * mbeta**(3/2) + mbeta * (Cr / betad)**2 * (mbeta + 1) * (np.arctan(1 / mbeta) - np.arctan(np.sqrt((Cr / betad - 1) / mCr_d1))) 
                - (mbeta + 1) / np.sqrt(mbeta) * np.arctanh(np.sqrt(mbeta * (Cr / betad - 1) / mCr_d1))))
                * (s / r - 1) / (beta * CLa_A * A * (lamda + 1))
        else:
            Hb = (8.0 / np.pi ** 2) * ((s**2 / r**2 + r**2 / s**2) * (0.25 * np.pi - 0.5 * np.arctan(0.5 * (s / r - r / s))) + (s / r - r / s) + 2.0 * np.arctan2(r, s) - np.pi)
        self.CL_body = Hb * CLa_A
            





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
        self.Vf = Std_Atmo(altitude)[3] * np.sqrt(shear * 10.0 ** 9 / ((1.337 * AR ** 3 * Std_Atmo(altitude)[1] * (ramda + 1.0)) / (2.0 * (AR + 2.0) * (thickness / self.Cr) ** 3)))
        return self.Vf


class Stage:
    def __init__(self, d_body, l_body, Lcg):
        self.d_body = d_body
        self.l_body = l_body
        self.Lcg = Lcg

    def integrate(self, components):
        self.Lcp = 0.0
        self.CNa = 0.0
        self.Cmq = 0.0
        self.Cnr = 0.0
        self.Clp = 0.0
        self.graph_components = []
        
        for obj in components:
            self.CNa += obj.CNa
            self.Lcp += obj.CNa * obj.Lcp
            self.Cmq -= 4.0 * (0.5 * obj.CNa * ((obj.Lcp - self.Lcg) / self.l_body) ** 2)
            self.Clp += obj.Clp
        self.graph_components = components
        self.Lcp /= self.CNa

    def plot(self):
        graph = Graph(self)
        graph.plot()



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

    def __init__(self, stage):
        self.Lcg = stage.Lcg
        self.Lcp = stage.Lcp
        self.point_set = []
        for component in stage.graph_components:
            if isinstance(component, Nose):
                if hasattr(self, 'point_nose'):
                    pass
                else:
                    self.point_nose = np.array([0.0, 0.0])
                self.point_nose = self.add_point(self.point_nose, component.LD*stage.d_body, 0.5*stage.d_body)
                self.point_nose = self.add_point(self.point_nose, component.LD*stage.d_body, 0.0)
                self.point_nose = self.add_body_reverse(self.point_nose)
                self.point_set.append(self.point_nose)

            elif isinstance(component, TaperBody):
                if hasattr(self, 'point_taper'):
                    pass
                else:
                    self.point_taper = np.array([component.distance, 0.0])
                self.point_taper = self.add_point(self.point_taper, component.distance, 0.5*component.d_before)
                self.point_taper = self.add_point(self.point_taper, component.distance+component.l_taper, 0.5*component.d_after)
                self.point_taper = self.add_point(self.point_taper, component.distance+component.l_taper, 0.0)
                self.point_taper = self.add_body_reverse(self.point_taper)
                self.point_set.append(self.point_taper)

            elif isinstance(component, Fin):
                if hasattr(self, 'point_fin'):
                    pass
                else:
                    self.point_fin = np.array([component.distance, 0.5*stage.d_body])
                self.point_fin = self.add_point(self.point_fin, component.distance+component.Cle, 0.5*stage.d_body+component.span)
                self.point_fin = self.add_point(self.point_fin, component.distance+component.Cle+component.Ct, 0.5*stage.d_body+component.span)
                self.point_fin = self.add_point(self.point_fin, component.distance+component.Cr, 0.5*stage.d_body)
                self.point_fin, self.point_fin_2nd = self.add_fin_reverse(self.point_fin)
                self.point_set.append(self.point_fin)
                self.point_set.append(self.point_fin_2nd)

        # add body
        start_x = max(self.point_nose[:,0])
        end_x = stage.l_body
        self.point_body = np.array([start_x, -0.5*stage.d_body])
        self.point_body = self.add_point(self.point_body, start_x, 0.5*stage.d_body)
        self.point_body = self.add_point(self.point_body, end_x, 0.5*stage.d_body)
        self.point_body = self.add_point(self.point_body, end_x, -0.5*stage.d_body)
        self.point_body = self.add_point(self.point_body, start_x, -0.5*stage.d_body)
        self.point_set.append(self.point_body)

    def plot(self):
        plt.close('all')
        plt.figure(0, figsize=(8, 4))
        xmax = 0.0
        ymax = 0.0
        for point_list in self.point_set:
            try:
                plt.plot(point_list[:,0], point_list[:,1], color='black')
                if max(point_list[:,0]) > xmax:
                    xmax = max(point_list[:,0])
                if max(point_list[:,1]) > ymax:
                    ymax = max(point_list[:,1])
            except:
                pass
        plt.plot(self.Lcg, 0.0, 'o', color='black', label='Lcg')
        plt.plot(self.Lcp, 0.0, 'o', color='red', label='Lcp')
        plt.xlim([0.0, np.ceil(xmax)])
        plt.ylim([np.floor(-ymax), np.ceil(ymax)])
        ax = plt.gca()
        aspect = 1.0
        ax.set_aspect(aspect)
        plt.grid()
        plt.legend()
        plt.show()


# Optimize
class FinOptimize:
    def __init__(self):
        pass
    
    def set_param_init(self, param):
        self.param_init = param

    def solve(self, config):
        cons = ({'type':'eq', 'fun': lambda param:self.equality(param, config)}, {'type':'ineq', 'fun': lambda param:self.inequality(param, config)})
        result = minimize(self.cost, self.param_init, args=(config,), constraints=cons, method='SLSQP', options={'maxiter': 100})#, 'disp': True, })
        return result


class Conditions:
    def __init__(self):
        self.value_list = []

    def __call__(self):
        return self.value_list

    def equal(self, target, value):
        self.value_list.append(target - value)

    def upper_bound(self, target, max_value):
        self.value_list.append(max_value - target)

    def lower_bound(self, target, min_value):
        self.value_list.append(target - min_value)

if __name__ == "__main__":
    def eq(s, r):
        return (8.0 / np.pi ** 2) * ((s**2 / r**2 + r**2 / s**2) * (0.5 * np.arctan(0.5 * (s / r - r / s)) + 0.25 * np.pi) - (s / r - r / s) - 2.0 * np.arctan2(r, s))
    r = 0.70
    s_array = np.arange(0.7, 3.5, 0.01)
    rs = r / s_array
    Hw_array = [eq(s, r) for s in s_array]
    
    print(eq(r/0.34, r))
    plt.figure()
    plt.plot(rs, Hw_array)
    plt.xscale('log')
    plt.grid()
    plt.show()
