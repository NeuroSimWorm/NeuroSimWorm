#by HengHuang@2021
from re import T
import pygame
import math 
from pygame.locals import *
from brian2 import *
from Worm import Worm
from WormNet import WormNet
from  myworm import Nacl
import time
from multiprocessing import Process,Queue,Array
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

import collections
import random
import numpy as np
import sys


        
step_size = 0.01
use_defult = False

# each_par= [0.37771727214567363, 0.9866986402776092, -0.8426217364612967, 
#            0.9411787800490856, 1275.30118804425, 1108.3052031230181, 
#            0.5361416230443865, 1.2744932672940195, -58.967636961024255,
#            -56.5039853611961, -64.38890231074765, -56.08220000518486,
#            -17.757977620931342, 6.810994091210887, -19.728963274974376, 
#            4.0213864971883595, 4.046380661893636, 0.7026210583280772, 
#            1.0624693064019084, -55.18996997503564, -62.9006303451024, 
#            -56.553218979388475, -64.92072048829868, 3.3629796060267836, 
#            -18.90466681215912, 2.9889292339794338, 0.6998673977795988, 3.409669080283493]

each_par=  [0.9690260360948741, 0.5701910387724638, -0.9212484885938466,
            0.9794382073450834, 1412.5651386100799, 927.8485512826592, 
            0.5141193028539419, 0.6744308590423316, -64.34486942132935, 
            -55.874782972969115, -64.29088310571387, -58.57985694659874, 
            -16.205403790809214, 7.10071271751076, -21.779596171109006, 
            4.217939774971455, 4.995192727074027, 0.9444623878225684, 
            1.482516119023785, -58.12202509958297, -62.51176098827273, 
            -62.364755279850215, -61.491209864616394, 13.205328077310696,
            -17.147274262970313, 0.28660339303314686, 2.088111466728151, 4.745317091001198]

if not use_defult:
    worm_body_parameters = {
        "head_ms_SMD_gain":each_par[0], #0.4
        "head_ms_RMD_gain":each_par[1], #0.6 
        "vnc_ms_D_gain":each_par[2],
        "vnc_ms_B_gain":each_par[3],
        "vnc_sr_gain":each_par[4],
        "head_sr_gain":each_par[5],
        "muscle_tau":0.1,
        "turn_gain":0.7022261694073677,
        "concentration_th":0.2,#1.5630552610382438,
        "min_turn_time":1.3548880419693887,}

    head_parameters = {
            "SMDD_cm":each_par[6], 
            "SMDV_cm":each_par[6],
            "RMDD_cm":each_par[7],
            "RMDV_cm":each_par[7],

            "SMDD_delta":each_par[8], 
            "SMDV_delta":each_par[8],
            "RMDD_delta":each_par[9],
            "RMDV_delta":each_par[9],

            "SMDD_v":each_par[10], 
            "SMDV_v":each_par[11],
            "RMDD_v":each_par[11],
            "RMDV_v":each_par[10],}

    head_chemical_parameters = {
            "SMDD_to_SMDV":each_par[12], 
            "SMDV_to_SMDD":each_par[12],
            "SMDD_to_RMDV":each_par[13],
            "SMDV_to_RMDD":each_par[13],
            "RMDD_to_RMDV":each_par[14],
            "RMDV_to_RMDD":each_par[14],}

    head_gap_parameters = {
            "SMDD_RMDD": each_par[15], 
            "SMDV_RMDV": each_par[15],
            "RMDD_RMDV": each_par[16],}

    vnc_parameters = {
            "VB_cm":each_par[17], 
            "DB_cm":each_par[17],
            "VD_cm":each_par[18],
            "DD_cm":each_par[18],
            "VB_delta":each_par[19], 
            "DB_delta":each_par[19],
            "VD_delta":each_par[20],
            "DD_delta":each_par[20],

            "VB_v":each_par[21], 
            "DB_v":each_par[21],
            "VD_v":each_par[22],
            "DD_v":each_par[22],
            }

    vnc_chemical_parameters = {
            "DB_to_VD": each_par[23], 
            "VB_to_DD": each_par[23],
            "DB_to_DD": each_par[24],
            "VB_to_VD": each_par[24],}

    vnc_gap_parameters = {
            "DB_DB": each_par[25], 
            "VB_VB": each_par[25],
            "DD_DD": each_par[26],
            "VD_VD": each_par[26],
            "AVB_DB": each_par[27],
            "AVB_VB": each_par[27]}

else:
    worm_body_parameters = None
    head_parameters = None
    head_chemical_parameters = None
    head_gap_parameters = None
    vnc_parameters = None
    vnc_chemical_parameters = None
    vnc_gap_parameters = None
        

use_defult_klinotaxis = False
each_par_klinotaxis = [1.392676408169791, 1.1696825344115496, 0.5365449015516788, 1.0949907049071044, 0.5165577421430498, 1.3565628617070615, -57.59759918320924, -55.60414323583245, -63.84581831516698, -61.72273434465751, -55.38804127601907, -63.86664470192045, 0.12203647168353202, 0.6440665861591697, 1.356307763163932, 0.8434836344560609, -29.14954416686669, -149.81658950680867, 152.2260981053114, 174.1984326345846, 187.64023459982127, 86.29761338233948, -275.3541210805997, 56.735633541829884, 7.724542827345431, 9.634364016354084, 0.7022261694073677, 1.5630552610382438, 1.3548880419693887]

if not use_defult_klinotaxis:
    klinotaxis_parameters = {
            "AIYL_cm":each_par_klinotaxis[0],
            "AIYR_cm":each_par_klinotaxis[1],
            "AIZL_cm":each_par_klinotaxis[2],
            "AIZR_cm":each_par_klinotaxis[3],
            "SMBV_cm":each_par_klinotaxis[4],
            "SMBD_cm":each_par_klinotaxis[5],

            "AIYL_delta":each_par_klinotaxis[6],
            "AIYR_delta":each_par_klinotaxis[7],
            "AIZL_delta":each_par_klinotaxis[8],
            "AIZR_delta":each_par_klinotaxis[9],
            "SMBV_delta":each_par_klinotaxis[10],
            "SMBD_delta":each_par_klinotaxis[11],

            "AIYL_v":-72,
            "AIYR_v":-72,
            "AIZL_v":-72,
            "AIZR_v":-72,
            "SMBV_v":0,
            "SMBD_v":0,

            "ASEL_N":each_par_klinotaxis[12],
            "ASEL_M":each_par_klinotaxis[13],
            "ASEL_v":0.0,

            "ASER_N":each_par_klinotaxis[14],
            "ASER_M":each_par_klinotaxis[15],
            "ASER_v":0.0,
            }

    klinotaxis_chemical_parameters = {
            "ASEL_to_AIYL":each_par_klinotaxis[16],
            "ASEL_to_AIYR":each_par_klinotaxis[17],
            "ASER_to_AIYL":each_par_klinotaxis[18],
            "ASER_to_AIYR":each_par_klinotaxis[19],

            "AIYL_to_AIZL":each_par_klinotaxis[20],
            "AIYR_to_AIZR":each_par_klinotaxis[21],
            "AIZL_to_SMBV":each_par_klinotaxis[22],#-20.0,
            "AIZR_to_SMBD":each_par_klinotaxis[23],#20.0,
            }



    klinotaxis_gap_parameters = {
            "AIYL_AIYR": each_par_klinotaxis[24],
            "AIZL_AIZR": each_par_klinotaxis[25],
            }
else:
    klinotaxis_parameters = None
    klinotaxis_chemical_parameters = None
    klinotaxis_gap_parameters = None

def run_gui():
    # worm_net.run(200*ms)
    trace = []
    trace_step = 30
    # print(nacl.get_concentration_pool())

    pygame.init()
    pygame.display.set_caption("c.elegans")
    screen = pygame.display.set_mode((1800*2, 1200*2)) #1800,1200
    game_clock = pygame.time.Clock()

    # Variable to keep our main loop running
    running = True
    step = 0
    # Our main loop!
    while running:
        step += 1
        #绘制线条
        screen.fill((0,0,0))
        rods_locations,lateral_locations,diagonal_locations = queue.get()
        # start_time = time.time()


        if step % trace_step == 0:
            trace.append(rods_locations[-1][0])
        
        # print(rods_locations)
        for location in rods_locations:
            pygame.draw.line(screen,(255,130,71),location[0],location[1], 2)
        for location in lateral_locations:
            pygame.draw.line(screen,(238,99,99),location[0],location[1], 2)
        for location in diagonal_locations:
            pygame.draw.line(screen,(30,144,255),location[0],location[1], 2)
        
        for t in trace:
            pygame.draw.circle(screen, (255,0,0), (t[0],t[1]), 2) 

        xy_pos = nacl_location
        
        pygame.draw.circle(screen, (50,0,0), (xy_pos[0]*scale+shift[0],xy_pos[1]*scale+shift[1]), r+100+100) 
        pygame.draw.circle(screen, (100,0,0), (xy_pos[0]*scale+shift[0],xy_pos[1]*scale+shift[1]), r+100) 
        pygame.draw.circle(screen, (255,0,0), (xy_pos[0]*scale+shift[0],xy_pos[1]*scale+shift[1]), r) 

        # for x,y,concentration in nacl.get_concentration_pool():
            # pygame.Surface.set_at(screen,(round(x*scale+shift[0]), round(y*scale+shift[1])), (round(255*(concentration/nacl.get_max_concentration())),0,0))
            # print(concentration)

        pygame.display.update()    
        # for loop through the event queue
        for event in pygame.event.get():
            # Check for KEYDOWN event; KEYDOWN is a constant defined in pygame.locals, which we imported earlier
            if event.type == KEYDOWN:
                # If the Esc key has been pressed set running to false to exit the main loop
                if event.key == K_ESCAPE:
                    running = False
            # Check for QUIT event; if QUIT, set running to false
            elif event.type == QUIT:
                running = False
            # handle MOUSEBUTTONUP
            elif event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                # print(pos)
                print((pos[0]-shift[0])/scale,(pos[1]-shift[1])/scale)
                nacl_location[0] = (pos[0]-shift[0])/scale
                nacl_location[1] = (pos[1]-shift[1])/scale
                # print(nacl_location[0],nacl_location[1])
        game_clock.tick(1000)

        # end_time = time.time()
        # print("draw time:",end_time - start_time)

# def plot_state():  
#     plt.ion()
#     plt.figure(1)
#     time = [ ]
#     value = []
#     while True:
#         plt.clf()
#         plt.plot(time,value)
#         # fig.canvas.draw()
#         state_dict = state_queue.get()
#         time.append(state_dict["time"])
#         value.append(state_dict["SMDD"])
#         plt.pause(0.0000001)
#         plt.ioff()

# class WormPlotter():

#     def __init__(self, sampleinterval=0.01, timewindow=10., size=(1000,600)):
#         # Data stuff
#         self._interval = int(sampleinterval*1000)
#         self._bufsize = int(timewindow/sampleinterval)
#         self.databuffer_time = collections.deque([0.0]*self._bufsize, self._bufsize)
#         self.databuffer_y= collections.deque([0.0]*self._bufsize, self._bufsize)
#         self.x = np.zeros(self._bufsize, dtype=np.float) #linspace(-timewindow, 0.0, self._bufsize)
#         self.y = np.zeros(self._bufsize, dtype=np.float)
#         # PyQtGraph stuff
#         self.app = QtGui.QApplication([])
#         pg.setConfigOptions(antialias=True)
#         self.plt = pg.plot(title='c.elegans states')
#         self.plt.resize(*size)
#         self.plt.showGrid(x=True, y=True)
#         self.plt.setLabel('left', 'amplitude', 'V')
#         self.plt.setLabel('bottom', 'time', 's')
#         self.curve = self.plt.plot(self.x, self.y, pen=(255,0,0))
#         # QTimer
#         self.timer = QtCore.QTimer()
#         self.timer.timeout.connect(self.updateplot)
#         self.timer.start(self._interval)

#     def updateplot(self):
#         state_dict = state_queue.get()
#         self.databuffer_y.append(state_dict["SMDV"])
#         self.databuffer_time.append(state_dict["time"])

#         self.x[:] = self.databuffer_time
#         self.y[:] = self.databuffer_y
#         self.curve.setData(self.x, self.y)
#         self.app.processEvents()

#     def run(self):
#         self.app.exec_()

class WormPlotter():

    def __init__(self, sampleinterval=0.01, timewindow=10., size=(1000,600)):
        # Data stuff
        self._interval = int(sampleinterval)
        self._bufsize = int(timewindow/sampleinterval)
        self.databuffer_time = collections.deque([0.0]*self._bufsize, self._bufsize)
        # self.databuffer_y= collections.deque([0.0]*self._bufsize, self._bufsize)
        self.x = np.zeros(self._bufsize, dtype=np.float) #linspace(-timewindow, 0.0, self._bufsize)
        self.y = np.zeros(self._bufsize, dtype=np.float)
        # PyQtGraph stuff
        # self.app = QtGui.QApplication([])
        # pg.setConfigOptions(antialias=True)
        # self.plt = pg.plot(title='c.elegans states')
        # self.plt.resize(*size)
        # self.plt.showGrid(x=True, y=True)
        # self.plt.setLabel('left', 'amplitude', 'V')
        # self.plt.setLabel('bottom', 'time', 's')
        # self.curve = self.plt.plot(self.x, self.y, pen=(255,0,0))

        self.app = pg.mkQApp()
        self.win = pg.GraphicsLayoutWidget(show=True,size=size,title="c.elegans states")

        self.state_to_plot = {"Membrane potential (Head neurons)":["SMDD","SMDV","RMDD","RMDV"],
                              "Nacl sensor neurons":["ASEL","ASER"],
                              "Nacl Klinotaxis outputs to head muscles":["ventral_klinotaxis","dorsal_klinotaxis"]}

        self.state_color = {"SMDD":(138,43,226),"SMDV":(139,0,139),"RMDD":(123,104,238),"RMDV":(0,139,139),
                            "ASEL":(255,127,80),"ASER":(255,69,0),
                            "ventral_klinotaxis": (138,43,226),"dorsal_klinotaxis":(123,104,238),}
        self.plots = []
        self.curve = {}
        self.curve_buffer = {}
        for key in self.state_to_plot.keys():
            ploter = self.win.addPlot(left="amplitude", bottom = "time", title = key)
            ploter.addLegend()
            self.curve[key] = {}
            self.curve_buffer[key] = {}
            for item_name in self.state_to_plot[key]:
                self.curve[key][item_name] = ploter.plot(self.x, self.y, pen=pg.mkPen(self.state_color[item_name],width=2),name=item_name)
                self.curve_buffer[key][item_name]  = collections.deque([0.0]*self._bufsize, self._bufsize)
            self.plots.append(ploter)
            self.win.nextRow()
        # self.win.addLabel(text= "label")

        # QTimer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateplot)
        self.timer.start(self._interval)

    def updateplot(self):
        state_dict = state_queue.get()
        self.databuffer_time.append(state_dict["time"])

        for plot_key,item_dict in self.curve.items():
            for item_key in item_dict.keys():
                self.curve_buffer[plot_key][item_key].append(state_dict[item_key])
                self.curve[plot_key][item_key].setData(self.databuffer_time, self.curve_buffer[plot_key][item_key])
        self.app.processEvents()

    def run(self):
       sys.exit(self.app.exec_()) 

def plot_state():
    m = WormPlotter(sampleinterval=0.05, timewindow=10.)
    m.run()

def run_worm():
    # start_time = time.time()
    while True:
        worm_net.run(100000)
    # end_time = time.time()
    # print(end_time - start_time)

if __name__ == '__main__':
        # scale = 1e6*0.0001
    scale = 1e6*0.5
    shift = (950,500)

    plot_state_flag =  True

    #
    xy_pos = (0.001,0.0009)
    r = 10
    nacl = Nacl(xy_pos[0],xy_pos[1],10000,50*5,r,scale) #x_center,y_center,alpha,peak,r_pixel,scale
    # nacl.update_pool()
    
    if plot_state_flag:
        state_queue = Queue()
        funcs = [run_worm, run_gui, plot_state]
    else:
        state_queue = None
        funcs = [run_worm, run_gui]

    queue = Queue()
    nacl_location = Array('d', range(2))
    nacl_location[0]= xy_pos[0]
    nacl_location[1]= xy_pos[1]

    worm = Worm(step_size = step_size,test=False,parameters=worm_body_parameters,shared_lists=queue,gui_parameters= (scale,shift),nacl_location=nacl_location, state_queue = state_queue)
    worm_net = WormNet(step_size=step_size*ms,worm_entity=worm, nacl_entity = nacl,
                    head_parameters=head_parameters,head_chemical_parameters=head_chemical_parameters,head_gap_parameters=head_gap_parameters,
                    vnc_parameters=vnc_parameters,vnc_chemical_parameters=vnc_chemical_parameters,vnc_gap_parameters=vnc_gap_parameters,
                    klinotaxis_parameters=klinotaxis_parameters,klinotaxis_chemical_parameters=klinotaxis_chemical_parameters,klinotaxis_gap_parameters=klinotaxis_gap_parameters)




    p_lists =  [Process(target=f) for f in funcs]
    for p in p_lists:
        p.start()
    for p in p_lists:
        p.join()
