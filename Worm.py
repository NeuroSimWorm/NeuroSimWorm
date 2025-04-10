#by HengHuang@2021
from math import sqrt
from myworm import StretchReceptor,Muscles,Body
import numpy as np
from matplotlib import pyplot as plt
import math
import time 


class Worm:
    def __init__(self,segment_num = 50, muscle_num = 24, step_size=0.01, parameters=None, test=False, print_info=True, shared_lists = None, gui_parameters= None, nacl_location=None, state_queue = None):

        if parameters is None:
            self.parameters = {
            "head_ms_SMD_gain":0.27246094, #0.4
            "head_ms_RMD_gain":0.69335938, #0.6
            "vnc_ms_D_gain":-0.1,
            "vnc_ms_B_gain":0.6,
            "vnc_sr_gain":1200,
            "head_sr_gain":1200,
            "muscle_tau":0.1,
            "turn_gain":0.5,
            "concentration_th":0.2,
            "min_turn_time":1,
            }
        else:
            self.parameters = parameters
    
        self.segment_num = segment_num
        self.muscle_num = muscle_num
        self.step_size = step_size
        self.test = test
        self.body = Body(segment_num)
        self.sr = StretchReceptor(self.segment_num,7, self.parameters["vnc_sr_gain"],self.parameters["head_sr_gain"]) #numseg #numsr #srvncgain, srheadgain
        self.ms = Muscles(muscle_num, self.parameters["muscle_tau"])
        self.time = 0.0
        self.collect_for_test = []
        self.adjust_time  = 1

        self.ventral_klinotaxis_h = 0 
        self.dorsal_klinotaxis_h = 1
        self.time_h = 0
        self.time_step = 0
        self.aser_h = 0 

        #trace recorded for speed 
        self.ventral_klinotaxis_trace = 0 
        self.dorsal_klinotaxis_trace = 0

        if print_info:
            self.info()

        self.shared_lists = shared_lists
        self.gui_parameters = gui_parameters
        self.nacl_location = nacl_location
        self.state_queue = state_queue

        self.nmj_gain = np.zeros(self.muscle_num)
        for i in range(self.muscle_num):
            self.nmj_gain[i] =  0.7*(1.0 - (((i-1)*0.5)/self.muscle_num))


        #localmotion states
        self.pause = False

    def step(self,neuron_net,nacl):
        #update stretch recepctor 
        for i in range(self.segment_num):
            d = (self.body.get_dorsal_len(i) - self.body.get_rest_len(i))/self.body.get_rest_len(i)
            v = (self.body.get_ventral_len(i) - self.body.get_rest_len(i))/self.body.get_rest_len(i)
            # print(i,d,v)
            self.sr.update_dorsal_input(i, d)
            self.sr.update_ventral_input(i, v)
        self.sr.step()


        if self.test:
            self.manually_set_muscle()
            self.ms.step(self.step_size)
        else:
            #update neuron system 
            self.neuron_net_update(neuron_net,nacl)
            #set muscle input
            self.set_muscles_input(neuron_net)
            self.ms.step(self.step_size)
        #update body
            #first two segments
        self.body.set_dorsal_activation(0, self.ms.get_dorsal_activation(0)/2)
        self.body.set_ventral_activation(0, self.ms.get_ventral_activation(0)/2)
        self.body.set_dorsal_activation(1, self.ms.get_dorsal_activation(0)/2)
        self.body.set_ventral_activation(1, self.ms.get_ventral_activation(0)/2)
            #other segments
        for i in range(2,(self.segment_num-2)):
            mi = int((i-2)/2)
            # print("s_i:",i," mi:",mi,"mi+1:",mi+1)
            # print(self.ms.get_dorsal_activation(mi),self.ms.get_ventral_activation(mi))
            self.body.set_dorsal_activation(i, (self.ms.get_dorsal_activation(mi) + self.ms.get_dorsal_activation(mi+1))/2)
            self.body.set_ventral_activation(i, (self.ms.get_ventral_activation(mi) + self.ms.get_ventral_activation(mi+1))/2)
            #Last two segments
        self.body.set_dorsal_activation(self.segment_num-2, self.ms.get_dorsal_activation(self.muscle_num-1)/2)
        self.body.set_ventral_activation(self.segment_num-2, self.ms.get_ventral_activation(self.muscle_num-1)/2)
        self.body.set_dorsal_activation(self.segment_num-1, self.ms.get_dorsal_activation(self.muscle_num-1)/2)
        self.body.set_ventral_activation(self.segment_num-1, self.ms.get_ventral_activation(self.muscle_num-1)/2)

        self.body.step(self.step_size)
        self.time += self.step_size
        self.time_step += 1
        # print(self.time)

        if self.shared_lists is not None:
            self.shared_lists.put(self.get_locations())
            # print(self.nacl_location[0],self.nacl_location[1])
            nacl.set_xy(self.nacl_location[0],self.nacl_location[1])
        if self.state_queue is not None and self.time_step % 5 == 0:
            self.state_queue.put(self.get_state(neuron_net))

        #test puase
        # if self.time > 2 and self.time < 3:
        #     self.pause = True
        #     self.worm_pause(neuron_net)
        #     print("pause")
        # else:
        #     self.pause = False

            

    def set_muscles_input(self,neuron_net):
        head_muscles_num = 6
        # top_head_num = 2
        head_output = neuron_net.get_all_head_neuron_out()
        dorsalHeadInput = self.parameters["head_ms_SMD_gain"]*head_output[neuron_net.get_head_id("SMDD")] + self.parameters["head_ms_RMD_gain"]*head_output[neuron_net.get_head_id("RMDD")]
        ventralHeadInput = self.parameters["head_ms_SMD_gain"]*head_output[neuron_net.get_head_id("SMDV")] + self.parameters["head_ms_RMD_gain"]*head_output[neuron_net.get_head_id("RMDV")]

        dorsal_klinotaxis = 1
        ventral_klinotaxis = 1
        # klinotaxis_diff = 0

        # ASER_h = neuron_net.ASER.get_v()
        if self.time > self.adjust_time:
            ventral_klinotaxis,dorsal_klinotaxis = neuron_net.get_klinotaxis_output()
            self.ventral_klinotaxis_h, self.dorsal_klinotaxis_h, self.time_h = self.flip(neuron_net, self.ventral_klinotaxis_h, self.dorsal_klinotaxis_h, self.time_h)
            ventral_klinotaxis,dorsal_klinotaxis = self.flip_helper(ventral_klinotaxis, dorsal_klinotaxis, self.ventral_klinotaxis_h, self.dorsal_klinotaxis_h)
            self.aser_h = neuron_net.ASER.get_v()
            # print(ventral_klinotaxis,dorsal_klinotaxis)
            # ASER_h = neuron_net.ASER.get_v() 
            # klinotaxis_diff = dorsal_klinotaxis - ventral_klinotaxis
            # print("dosal ventral:",ventral_klinotaxis,dorsal_klinotaxis)
            # print("oscillation:",dorsalHeadInput,ventralHeadInput)
        
        # print(klinotaxis_diff)

        self.ventral_klinotaxis_trace,self.dorsal_klinotaxis_trace = ventralHeadInput*(1-ventral_klinotaxis),dorsalHeadInput*(1-dorsal_klinotaxis)

        for i in range(head_muscles_num):
            if self.time > self.adjust_time:
                self.ms.update_dorsal_input(i,self.nmj_gain[i]*(dorsalHeadInput + dorsalHeadInput*(1-dorsal_klinotaxis)*self.parameters["turn_gain"]))
                self.ms.update_ventral_input(i,self.nmj_gain[i]*(ventralHeadInput + ventralHeadInput*(1-ventral_klinotaxis)*self.parameters["turn_gain"]))
            else:  
                self.ms.update_dorsal_input(i,self.nmj_gain[i]*dorsalHeadInput)
                self.ms.update_ventral_input(i,self.nmj_gain[i]*ventralHeadInput)  

        # for i in range(self.segment_num):
        #     x,y  = self.body.get_x_y(i)
        #     self.body.set_x_y(i,x+0.000001,y+0.000001)

        # for i in range(head_muscles_num):
        #     self.ms.update_dorsal_input(i,self.nmj_gain[i]*dorsalHeadInput)  
        #     self.ms.update_ventral_input(i,self.nmj_gain[i]*ventralHeadInput) 

        # print(dorsalHeadInput,ventralHeadInput)
        # for i in range(head_muscles_num):
        #     dorsalHeadInput = 2 if math.sin(2*math.pi*self.time) > 0 else 0.1
        #     ventralHeadInput= 2 if math.sin(2*math.pi*self.time+math.pi) > 0 else 0.1
            
        #     self.ms.update_dorsal_input(i,self.nmj_gain[i]*dorsalHeadInput)  
        #     self.ms.update_ventral_input(i,self.nmj_gain[i]*ventralHeadInput)            

        #VNC
        vnc_output = neuron_net.get_all_vnc_neuron_out()
        vnc_start = head_muscles_num
        muscle_per_unit = 3  
        for i in range(vnc_start,self.muscle_num):
            mi = int((i-vnc_start)/muscle_per_unit)
            # print("m_i:",i," vi:",mi)
            dorsalInput = self.parameters["vnc_ms_D_gain"]*vnc_output[neuron_net.get_vnc_id("DD",unit_index=mi)] + self.parameters["vnc_ms_B_gain"]*vnc_output[neuron_net.get_vnc_id("DB",unit_index=mi)]
            ventralInput = self.parameters["vnc_ms_D_gain"]*vnc_output[neuron_net.get_vnc_id("VD",unit_index=mi)] + self.parameters["vnc_ms_B_gain"]*vnc_output[neuron_net.get_vnc_id("VB",unit_index=mi)]
            if self.time > self.adjust_time:
                self.ms.update_dorsal_input(i, self.nmj_gain[i]*(dorsalInput + dorsalInput*(1-dorsal_klinotaxis)*0.5))
                self.ms.update_ventral_input(i, self.nmj_gain[i]*(ventralInput + ventralInput*(1-ventral_klinotaxis)*0.5))
            else:
                self.ms.update_dorsal_input(i, self.nmj_gain[i]*dorsalInput)
                self.ms.update_ventral_input(i, self.nmj_gain[i]*ventralInput)
            # print(i,dorsalInput,ventralInput)

    def worm_pause(self, neuron_net):
        neuron_net.headnet.v_[:] = -0.072
        neuron_net.vncnet.v_[:] = -0.072
    
    def flip(self, neuron_net, ventral_klinotaxis_h, dorsal_klinotaxis_h, time_h):
        #this function will be replaced with real circuit in the future 
        # print(neuron_net.ASER.get_v(), self.aser_h, neuron_net.ASER.get_v() - self.aser_h)
        if (neuron_net.ASER.get_v() - self.aser_h) > self.parameters["concentration_th"] and (self.time - time_h) > self.parameters["min_turn_time"]:
            return dorsal_klinotaxis_h,ventral_klinotaxis_h, self.time
        else:
            return ventral_klinotaxis_h,dorsal_klinotaxis_h, time_h
    
    def flip_helper(self,ventral_klinotaxis, dorsal_klinotaxis, ventral_klinotaxis_h, dorsal_klinotaxis_h):
        if ventral_klinotaxis_h == 1 and dorsal_klinotaxis_h == 0:
            return dorsal_klinotaxis, ventral_klinotaxis
        else:
            return ventral_klinotaxis, dorsal_klinotaxis

            
    def neuron_net_update(self,neuron_net,nacl):
        #nacl
        worm_x = self.body.get_center_x()
        worm_y = self.body.get_center_y()
        concentration = nacl.get_concentration(worm_x,worm_y)
        neuron_net.ASEL.step(concentration)
        neuron_net.ASER.step(concentration)

        # worm_x_head = self.body.x(0)
        # worm_y_head = self.body.y(0)
        # # worm_x_tail = self.body.x(self.segment_num-1)
        # # worm_y_tail = self.body.y(self.segment_num-1)

        # concentration_head = nacl.get_concentration(worm_x_head,worm_y_head)
        # # concentration_tail = nacl.get_concentration(worm_x_tail,worm_y_tail)
        # neuron_net.ASEL.step(concentration_head)
        # neuron_net.ASER.step(concentration_head)

        # neuron_net.set_klinotaxis_oscillation()

        if self.time > self.adjust_time:
            neuron_net.set_klinotaxis_input()
        
        # print(self.time)

        # # nacl_x,nacl_y =  nacl.get_xy()
        # if self.time > self.adjust_time:
        #     print("start:",concentration, neuron_net.ASEL.get_v(),neuron_net.ASER.get_v())
        # else:
        #     print("init:",concentration, neuron_net.ASEL.get_v(),neuron_net.ASER.get_v())
  
        
        # print(concentration,np.sqrt(np.power((worm_x-nacl_x),2) + np.power((worm_y-nacl_y),2)), nacl_x,worm_x,nacl_y,worm_y)
        
        #head
        head_inputs = np.zeros(len(neuron_net.headneuron_index))
        if not self.pause:
            head_inputs[neuron_net.headneuron_index['SMDD']] = self.sr.get_head_dorsal()
            head_inputs[neuron_net.headneuron_index['SMDV']] = self.sr.get_head_ventral()
        neuron_net.set_head_input_faster(head_inputs)

        # neuron_net.set_head_input(self.sr.get_head_dorsal(),self.sr.get_head_ventral())
        # print("head:",self.sr.get_head_dorsal(),self.sr.get_head_ventral())

        # start_time = time.time()
        
        #vnc
        # for i in range(neuron_net.unit_num):
            # neuron_net.set_vnc_input(self.sr.get_vnc_ventral(i),self.sr.get_vnc_dorsal(i),i)

        vnc_inputs = np.zeros(neuron_net.unit_num*neuron_net.neuron_per_unit+1)
        if not self.pause:
            db_index = neuron_net.vncneuron_index['DB']
            vb_index = neuron_net.vncneuron_index['VB']
            for i in range(neuron_net.unit_num):
                unit_jumper = i*neuron_net.neuron_per_unit
                # print(unit_jumper + db_index,unit_jumper + vb_index)
                vnc_inputs[unit_jumper + db_index] =  self.sr.get_vnc_ventral(i) 
                vnc_inputs[unit_jumper + vb_index] =  self.sr.get_vnc_dorsal(i)
        neuron_net.set_vnc_input_faster(vnc_inputs)

        # end_time = time.time()
        # print("vnc time: ",end_time - start_time)

        # start_time = time.time()
        # neuron_net.run()
        # end_time = time.time()
        # print("net time: ",end_time - start_time)

    def get_state(self,worm_net):
        state_dict = {}
        head_v =  worm_net.headnet.v_[:]
        state_dict["time"] = self.time
        #head
        state_dict["SMDD"] = head_v[worm_net.headneuron_index["SMDD"]]
        state_dict["SMDV"] = head_v[worm_net.headneuron_index["SMDV"]]
        state_dict["RMDD"] = head_v[worm_net.headneuron_index["RMDD"]]
        state_dict["RMDV"] = head_v[worm_net.headneuron_index["RMDV"]]
        #nacl sensor
        state_dict["ASEL"] = worm_net.ASEL.get_v()
        state_dict["ASER"] = worm_net.ASER.get_v()
        #nacl klinotaxis outputs
        state_dict["ventral_klinotaxis"] = self.ventral_klinotaxis_trace 
        state_dict["dorsal_klinotaxis"] = self.dorsal_klinotaxis_trace

        return state_dict

    def manually_set_muscle(self):
        lag_duration = 0.1
        # for i in range(1,20):
            # self.ms.set_dorsal_activation(i,0.0)
        # print(self.sr.get_head_dorsal(),self.sr.get_head_ventral())
        for i in range(self.muscle_num):
            ac_value = max(0,np.sin(self.time+i*lag_duration))
            self.ms.set_dorsal_activation(i,ac_value)

    
    def get_locations(self):
        scale = self.gui_parameters[0]
        shift = self.gui_parameters[1]

        rods_locations = []
        for i in range(self.segment_num):
            x =  self.body.x(i)*scale+shift[0]
            y =  self.body.y(i)*scale+shift[1]
            radar_len = self.body.get_r_len(i)*scale
            radian =  self.body.phi(i)
            x_end = x + math.cos(radian)*radar_len
            y_end = y + math.sin(radian)*radar_len
            rods_locations.append([(x,y),(x_end,y_end)])
        lateral_locations = []
        for i in range(len(rods_locations)-1):
            lateral_locations.append([rods_locations[i][0],rods_locations[i+1][0]])
            lateral_locations.append([rods_locations[i][1],rods_locations[i+1][1]])
        diagonal_locations = []
        for i in range(len(rods_locations)-1):
            diagonal_locations.append([rods_locations[i][0],rods_locations[i+1][1]])
            diagonal_locations.append([rods_locations[i][1],rods_locations[i+1][0]])

        return (rods_locations,lateral_locations,diagonal_locations)
        
    
    def info(self):
        segment_num = 50 
        muscle_num = 24
        head_muscle_num = 6
        vnc_muscle_num = muscle_num - head_muscle_num

        vnc_unit_num = 6
        vnc_each_unit_seq_num = 6
        vnc_start = 6

        muscle_to_seg = []
        for i in range(2,(self.segment_num-2)):
            mi = int((i-2)/2)
            muscle_to_seg.append([[mi,mi+1],i])
        print("muscle_to_seg:",muscle_to_seg)
        
        seg_to_vnc = []
        for i in range(vnc_unit_num):
            seg_indexs = []
            for j in range(vnc_each_unit_seq_num):
                seg_index = i*vnc_each_unit_seq_num + vnc_start + j- i*2
                seg_indexs.append(seg_index)
            seg_to_vnc.append([seg_indexs,i])
        print("seg_to_vnc:",seg_to_vnc)

        vnc_to_muscle = []
        vnc_start = head_muscle_num
        muscle_per_unit = 3  
        for i in range(vnc_start,muscle_num):
            mi = int((i-vnc_start)/muscle_per_unit)
            vnc_to_muscle.append([mi,i])
        print("vnc_to_muscle:",vnc_to_muscle)

        vnc_to_segs = []
        for i in range(vnc_unit_num):
            muscle_index = [each[1] for each in vnc_to_muscle if each[0]== i]
            seg_index = [] 
            for each_muscle_index in muscle_index:
                seg_index += [each[1] for each in muscle_to_seg if each_muscle_index in each[0]]
            vnc_to_segs.append([i,seg_index])
        print("vnc_to_segs:",vnc_to_segs)

    def plot_vnc_input(self):
        plt.plot([item[0] for item in self.collect_for_test])
        plt.plot([item[1] for item in self.collect_for_test])
        plt.show() 

        # for i in range(vnc_start,self.muscle_num):
        # mi = int((i-vnc_start)/muscle_per_unit+1)
        

    
