#by HengHuang@2021
from brian2 import *
from brian2.units.allunits import *
from brian2.utils.stringtools import word_substitute
from matplotlib import pyplot as plt
from Worm import Worm
import numpy as np
import random

prefs.codegen.target = 'numpy'
# prefs.logging.file_log = False

eqs = '''
dv/dt = (i_leak+i_act+i_sr*Pamp+i_stim*Pamp+i_gap*Pamp)/c_m :  volt
i_leak = g_m*(v_rest-v) : amp
i_act = (g_act/Psiemens)/(1+exp(-k_act*v_vact))*Pamp: amp
i_out = 1/(1+exp(-k_out*(v/mV-delta))): 1
i_sr : 1
i_stim : 1
i_gap : 1

v_vact = (v-v_act)/mV :1 
#c_m= 1*Pfarad : farad #default 1.00
c_m : farad #default 1.00
v_rest : volt #default -72.00
#g_m = 500.0*Psiemens :siemens #default 500.00
g_m :siemens #default 500.00
#g_act = 50.0*Psiemens : siemens #default:50.0
g_act :siemens #default:50.0
k_act = 1 : 1
k_out = 1 : 1
v_act = -60*mV :volt
delta :1
'''


neuron_chemical_syn_eqs='''
w : 1 # weigths
i_stim_post = w * i_out_pre : 1 (summed)
'''

neuron_gap_syn_eqs='''
w : 1 # weigths
i_gap_post = w * (v_pre - v_post)/mV : 1 (summed)
'''

class NaclSensor():
    def __init__(self, N = 0.5, M = 1.7, v = 0,step_size = 0.01,type="ON"):
        self.v = v
        self.N_size = int(N/step_size)
        self.M_size = int(M/step_size)
        self.trace_size = self.N_size + self.M_size
        self.trace = np.zeros(self.trace_size)
        self.type  = type 
        if type != "ON" and type != "OFF":
            raise Exception('unkown NaCL sensor type')

    def step(self,ct):
        self.v = np.mean(self.trace[self.M_size:]) - np.mean(self.trace[0:self.M_size])
        self.trace[:-1] = self.trace[1:]
        self.trace[-1] = ct
    
    def get_v(self):
        if self.type == "ON":
            return max(0.0,self.v)
        else:
            return max(0.0,-self.v)

class WormNet:
    def __init__(self,method= 'rk2',step_size = 1*ms, worm_entity = None, nacl_entity = None,
                 head_parameters=None, head_chemical_parameters=None, head_gap_parameters = None,
                 vnc_parameters=None, vnc_chemical_parameters=None, vnc_gap_parameters = None,
                 klinotaxis_parameters = None, klinotaxis_chemical_parameters=None,klinotaxis_gap_parameters=None,
                 motor_control_chemical_parameters=None, motor_control_gap_parameters=None,
                 record = False):
        # device.reinit()
        # device.activate()
        self.step_size = step_size
        defaultclock.dt = step_size
        self.net  = Network()
        self.worm_entity = worm_entity
        self.nacl_entity = nacl_entity

        #head parameters 
        if head_parameters is None:
            self.head_parameters = {
                "SMDD_cm":1.0, 
                "SMDV_cm":1.0,
                "RMDD_cm":1.0,
                "RMDV_cm":1.0,

                "SMDD_delta":-60.0, 
                "SMDV_delta":-60.0,
                "RMDD_delta":-60.0,
                "RMDV_delta":-60.0,

                "SMDD_v":-61.0, 
                "SMDV_v":-59.0,
                "RMDD_v":-59.0,
                "RMDV_v":-61.0,
                }
        else:
            self.head_parameters = head_parameters

        if head_chemical_parameters is None:
            self.head_chemical_parameters = {
                "SMDD_to_SMDV":-20.0, 
                "SMDV_to_SMDD":-20.0,
                "SMDD_to_RMDV": 20,
                "SMDV_to_RMDD": 20,
                "RMDD_to_RMDV":-20.0,
                "RMDV_to_RMDD":-20.0,
                }
        else:
            self.head_chemical_parameters = head_chemical_parameters

        if head_gap_parameters is None:
            self.head_gap_parameters = {
                "SMDD_RMDD": 1, 
                "SMDV_RMDV": 1,
                "RMDD_RMDV": 1,
                }
        else:
            self.head_gap_parameters = head_gap_parameters

        #VNC without AVB
        if vnc_parameters is None:
            self.vnc_parameters = {
                "VB_cm":1, 
                "DB_cm":1,
                "VD_cm":1,
                "DD_cm":1,

                "VB_delta":-61, 
                "DB_delta":-61,
                "VD_delta":-61,
                "DD_delta":-61,

                "VB_v":-61, 
                "DB_v":-61,
                "VD_v":-61,
                "DD_v":-61,
                }
        else:
            self.vnc_parameters = vnc_parameters

        if vnc_chemical_parameters is None:
            self.vnc_chemical_parameters = {
                "DB_to_VD": 10.0, 
                "VB_to_DD": 10.0,
                "DB_to_DD": -20,
                "VB_to_VD": -20,
                }
        else:
            self.vnc_chemical_parameters = vnc_chemical_parameters

        if vnc_gap_parameters is None:
            self.vnc_gap_parameters = {
                "DB_DB": 5, 
                "VB_VB": 5,
                "DD_DD": 5,
                "VD_VD": 5,
                "AVB_DB": 5,
                "AVB_VB": 5
                }
        else:
            self.vnc_gap_parameters = vnc_gap_parameters

        #klinotaxis parameters
        if klinotaxis_parameters is None:
            self.klinotaxis_parameters = {
                "AIYL_cm":1,
                "AIYR_cm":1,
                "AIZL_cm":1,
                "AIZR_cm":1,
                "SMBV_cm":1,
                "SMBD_cm":1,

                "AIYL_delta":-60,
                "AIYR_delta":-60,
                "AIZL_delta":-60,
                "AIZR_delta":-60,
                "SMBV_delta":-60,
                "SMBD_delta":-60,

                "AIYL_v":-72,
                "AIYR_v":-72,
                "AIZL_v":-72,
                "AIZR_v":-72,
                "SMBV_v":0,
                "SMBD_v":0,

                "ASEL_N":0.2,#0.3
                "ASEL_M":0.5,
                "ASEL_v":0.0,

                "ASER_N":0.2,
                "ASER_M":0.5,
                "ASER_v":0.0,
                }
        else:
            self.klinotaxis_parameters = klinotaxis_parameters

        if klinotaxis_chemical_parameters is None:
            self.klinotaxis_chemical_parameters = {
                "ASEL_to_AIYL": -400.0,
                "ASEL_to_AIYR": -800.0,
                "ASER_to_AIYL": 800.0,
                "ASER_to_AIYR": 400.0,

                "AIYL_to_AIZL": 200.0,
                "AIYR_to_AIZR": 200.0,
                "AIZL_to_SMBV": -300.0,#-20.0,
                "AIZR_to_SMBD": 300.0,#20.0,
                }
        else:
            self.klinotaxis_chemical_parameters = klinotaxis_chemical_parameters


        if klinotaxis_gap_parameters is None:
            self.klinotaxis_gap_parameters = {
                "AIYL_AIYR": 10.0,
                "AIZL_AIZR": 10.0,
                }
        else:
            self.klinotaxis_gap_parameters = klinotaxis_gap_parameters

        #motor control paramters
        if motor_control_chemical_parameters is None:
            self.motor_control_chemical_parameters = {
                "AIB_to_SAA": 1.0,
                "AIB_to_AIY": 1.0,
                "AIB_to_RIB": 1.0,
                "AIY_to_RIB": 1.0,

                "SMDV_to_RIV": 1.0,
                "RIV_to_SMDV": 1.0,
                "RIV_to_SAA": 1.0,#-20.0,
                "SAA_to_AIB": 1.0,#20.0,
                "SMB_to_SAA": 1.0,
                }
        else:
            self.motor_control_chemical_parameters = motor_control_chemical_parameters

        if motor_control_gap_parameters is None:
            self.motor_control_gap_parameters = {
                "AIB_RIV": 1.0,
                "RIB_SMDV": 1.0,
                "SMDV_RIV": 1.0,
                "SAA_SMB": 1.0,
                }
        else:
            self.motor_control_gap_parameters = motor_control_gap_parameters


        # self.headneuron_index = self.net_reader.neuron_index
        # print(self.headneuron_index)
        # self.headnet = self.create_neuron(self.net_reader.total_nueron_num, eqs, method)
        # self.headsyn_chemi = self.connect_group(self.headnet,self.net_reader.chemical_connections[0],self.net_reader.chemical_connections[1],self.net_reader.chemical_connections[2],neuron_chemical_syn_eqs)
        # self.headsyn_gap = self.connect_group(self.headnet,self.net_reader.gap_connections[0],self.net_reader.gap_connections[1],self.net_reader.gap_connections[2],neuron_gap_syn_eqs)

        #head 
        self.headneuron_index = {"SMDD":0,"SMDV":1,"RMDD":2,"RMDV":3}
        self.headnet = self.create_head_neuron(len(self.headneuron_index), eqs, method)
        head_sources_chemi,head_targets_chemi,head_weights_chemi = self.get_head_chemical_connections(self.headneuron_index)
        head_sources_gap,head_targets_gap,head_weights_gap = self.get_head_gap_connections(self.headneuron_index)
        self.headsyn_chemi = self.connect_group(self.headnet,head_sources_chemi,head_targets_chemi,head_weights_chemi,neuron_chemical_syn_eqs)
        self.headsyn_gap = self.connect_group(self.headnet,head_sources_gap,head_targets_gap,head_weights_gap,neuron_gap_syn_eqs)

        # vnc 
        self.unit_num = 6
        self.vncneuron_index = {"DB":0,"VB":1,"DD":2,"VD":3} 
        self.neuron_per_unit = len(self.vncneuron_index)
        self.vnc_total_num  = self.unit_num * self.neuron_per_unit+1 #plus AVB #AVB is the last neuron
        self.vncnet = self.create_vnc_neuron(self.vnc_total_num, eqs, method)
        vnc_sources_chemi,vnc_targets_chemi,vnc_weights_chemi = self.get_vnc_chemical_connections(self.vncneuron_index)
        self.vncsyn_chemi = self.connect_group(self.vncnet,vnc_sources_chemi,vnc_targets_chemi,vnc_weights_chemi,neuron_chemical_syn_eqs)
        vnc_sources_gap,vnc_targets_gap,vnc_weights_gap = self.get_vnc_gap_connections(self.vncneuron_index)
        self.vncsyn_gap = self.connect_group(self.vncnet,vnc_sources_gap,vnc_targets_gap,vnc_weights_gap,neuron_gap_syn_eqs)

        #klinotaxis circuit
        self.klinotaxis_index = {"AIYL":0, "AIYR":1, "AIZL":2, "AIZR":3, "SMBV":4, "SMBD":5}
        self.klinotaxis_net =  self.create_klinotaxis_neuron(len(self.klinotaxis_index), eqs, method)
        klinotaxis_sources_chemi,klinotaxis_targets_chemi,klinotaxis_weights_chemi = self.get_klinotaxis_chemical_connections(self.klinotaxis_index)
        klinotaxis_sources_gap,klinotaxis_targets_gap,klinotaxis_weights_gap = self.get_klinotaxis_gap_connections(self.klinotaxis_index)
        self.klinotaxis_gap = self.connect_group(self.klinotaxis_net,klinotaxis_sources_gap,klinotaxis_targets_gap,klinotaxis_weights_gap,neuron_gap_syn_eqs)
        self.klinotaxis_chemi = self.connect_group(self.klinotaxis_net,klinotaxis_sources_chemi,klinotaxis_targets_chemi,klinotaxis_weights_chemi,neuron_chemical_syn_eqs)
        
        #motor sequence circuts
        ######
        ######todo
        ######
        self.motor_control_index = {"AIB":0, "RIB":1, "SMDV":2, "RIV":3, "SAA":4, "SMB":5, "AIY":6}
        self.motor_control_net =  self.create_motor_control_neuron(len(self.motor_control_index), eqs, method)
        motor_control_sources_chemi,motor_control_targets_chemi,motor_control_weights_chemi = self.get_motor_control_chemical_connections(self.motor_control_index)
        self.motor_control_chemi = self.connect_group(self.motor_control_net,motor_control_sources_chemi,motor_control_targets_chemi,motor_control_weights_chemi,neuron_chemical_syn_eqs)
        motor_control_sources_gap,motor_control_targets_gap,motor_control_weights_gap = self.get_motor_control_gap_connections(self.motor_control_index)
        self.motor_control_gap = self.connect_group(self.motor_control_net,motor_control_sources_gap,motor_control_targets_gap,motor_control_weights_gap,neuron_gap_syn_eqs)

        #NACL sensor
        self.ASEL = NaclSensor(N=self.klinotaxis_parameters["ASEL_N"],M=self.klinotaxis_parameters["ASEL_M"], v=self.klinotaxis_parameters["ASEL_v"],step_size=step_size/ms,type="ON")
        self.ASER = NaclSensor(N=self.klinotaxis_parameters["ASER_N"],M=self.klinotaxis_parameters["ASER_M"], v=self.klinotaxis_parameters["ASER_v"],step_size=step_size/ms,type="OFF")

        def worm_entity_step():
            self.worm_entity.step(self,self.nacl_entity)

        self.body_op = NetworkOperation(worm_entity_step, dt=self.step_size)
        
        # monitor
        if record:
            self.M = StateMonitor(self.headnet, 'v', record=True)
            self.M_2 = StateMonitor(self.vncnet, 'v', record=True)
            self.net.add([self.headnet, self.headsyn_chemi, self.headsyn_gap, self.vncnet, self.vncsyn_chemi,self.vncsyn_gap, self.klinotaxis_net, self.klinotaxis_chemi, self.klinotaxis_gap,self.body_op, self.motor_control_net,self.motor_control_chemi, self.motor_control_gap, self.M,self.M_2])
        else:
            self.net.add([self.headnet, self.headsyn_chemi, self.headsyn_gap, self.vncnet, self.vncsyn_chemi,self.vncsyn_gap, self.klinotaxis_net, self.klinotaxis_chemi, self.klinotaxis_gap,self.body_op, self.motor_control_net,self.motor_control_chemi, self.motor_control_gap])

    def create_neuron(self,num,eqs,method):
        worm_net = NeuronGroup(num, eqs, method=method)
        for name,index in self.net_reader.neuron_index.items():
            worm_net.c_m[index] = self.net_reader.neuron_paramters[name]["cm"]*Pfarad
            worm_net.delta[index] = self.net_reader.neuron_paramters[name]["delta"] 
            worm_net.v[index] = self.net_reader.neuron_paramters[name]["v"]*mV
            worm_net.g_m = self.net_reader.neuron_paramters[name]["gm"]*Psiemens #i_leak
            worm_net.g_act = self.net_reader.neuron_paramters[name]["gact"]*Psiemens #i_act
            worm_net.v_rest = self.net_reader.neuron_paramters[name]["vrest"]*mV 

        return worm_net

    def create_motor_control_neuron(self,num, eqs, method):
        motor_control_net = NeuronGroup(num, eqs, method=method)
        return motor_control_net

    def get_motor_control_chemical_connections(self,neuron_lists):
        #syns
        connections_m = np.zeros((len(neuron_lists),len(neuron_lists)))
        connections_m[neuron_lists['AIB'],neuron_lists['SAA']] = 1
        connections_m[neuron_lists['AIB'],neuron_lists['AIY']] = 1
        connections_m[neuron_lists['AIB'],neuron_lists['RIB']] = 1
        connections_m[neuron_lists['AIY'],neuron_lists['RIB']] = 1

        connections_m[neuron_lists['SMDV'],neuron_lists['RIV']] = 1
        connections_m[neuron_lists['RIV'],neuron_lists['SMDV']] = 1
        connections_m[neuron_lists['RIV'],neuron_lists['SAA']] = 1
        connections_m[neuron_lists['SAA'],neuron_lists['AIB']] = 1
        connections_m[neuron_lists['SMB'],neuron_lists['SAA']] = 1
        
        sources, targets = connections_m.nonzero()

        #weights
        weights = np.zeros((len(neuron_lists),len(neuron_lists)))
        weights[neuron_lists['AIB'],neuron_lists['SAA']] = self.motor_control_chemical_parameters["AIB_to_SAA"]
        weights[neuron_lists['AIB'],neuron_lists['AIY']] = self.motor_control_chemical_parameters["AIB_to_AIY"]
        weights[neuron_lists['AIB'],neuron_lists['RIB']] = self.motor_control_chemical_parameters["AIB_to_RIB"]
        weights[neuron_lists['AIY'],neuron_lists['RIB']] = self.motor_control_chemical_parameters["AIY_to_RIB"]

        weights[neuron_lists['SMDV'],neuron_lists['RIV']] = self.motor_control_chemical_parameters["SMDV_to_RIV"]
        weights[neuron_lists['RIV'],neuron_lists['SMDV']] = self.motor_control_chemical_parameters["RIV_to_SMDV"]
        weights[neuron_lists['RIV'],neuron_lists['SAA']] = self.motor_control_chemical_parameters["RIV_to_SAA"]
        weights[neuron_lists['SAA'],neuron_lists['AIB']] = self.motor_control_chemical_parameters["SAA_to_AIB"]
        weights[neuron_lists['SMB'],neuron_lists['SAA']] = self.motor_control_chemical_parameters["SMB_to_SAA"]

        return sources,targets,weights

    def get_motor_control_gap_connections(self,neuron_lists):
        #syns
        connections_m = np.zeros((len(neuron_lists),len(neuron_lists)))
        connections_m[neuron_lists['AIB'],neuron_lists['RIV']] = 1
        connections_m[neuron_lists['RIV'],neuron_lists['AIB']] = 1
        connections_m[neuron_lists['RIB'],neuron_lists['SMDV']] = 1
        connections_m[neuron_lists['SMDV'],neuron_lists['RIB']] = 1
        connections_m[neuron_lists['SMDV'],neuron_lists['RIV']] = 1
        connections_m[neuron_lists['RIV'],neuron_lists['SMDV']] = 1
        connections_m[neuron_lists['SAA'],neuron_lists['SMB']] = 1
        connections_m[neuron_lists['SMB'],neuron_lists['SAA']] = 1
        sources, targets = connections_m.nonzero()

        #weights
        weights = np.zeros((len(neuron_lists),len(neuron_lists)))
        weights[neuron_lists['AIB'],neuron_lists['RIV']] = self.motor_control_gap_parameters["AIB_RIV"]
        weights[neuron_lists['RIV'],neuron_lists['AIB']] = self.motor_control_gap_parameters["AIB_RIV"]
        weights[neuron_lists['RIB'],neuron_lists['SMDV']] = self.motor_control_gap_parameters["RIB_SMDV"]
        weights[neuron_lists['SMDV'],neuron_lists['RIB']] = self.motor_control_gap_parameters["RIB_SMDV"]
        weights[neuron_lists['SMDV'],neuron_lists['RIV']] = self.motor_control_gap_parameters["SMDV_RIV"]
        weights[neuron_lists['RIV'],neuron_lists['SMDV']] = self.motor_control_gap_parameters["SMDV_RIV"]
        weights[neuron_lists['SAA'],neuron_lists['SMB']] = self.motor_control_gap_parameters["SAA_SMB"]
        weights[neuron_lists['SMB'],neuron_lists['SAA']] = self.motor_control_gap_parameters["SAA_SMB"]

        return sources,targets,weights

    def create_klinotaxis_neuron(self,num, eqs, method):
        klinotaxis_net = NeuronGroup(num, eqs, method=method)
        # klinotaxis_net.g_act = 0.0*Psiemens #i_act
        klinotaxis_net.g_m = 1200.0*Psiemens #i_leak
        klinotaxis_net.g_act = 100.0*Psiemens #i_act
        klinotaxis_net.v_rest = -72.0*mV 

        klinotaxis_net[self.klinotaxis_index["AIYL"]].g_act = 0.0
        klinotaxis_net[self.klinotaxis_index["AIYR"]].g_act = 0.0
        klinotaxis_net[self.klinotaxis_index["AIZL"]].g_act = 0.0
        klinotaxis_net[self.klinotaxis_index["AIZR"]].g_act = 0.0

        klinotaxis_net[self.klinotaxis_index["SMBV"]].g_m = 500.0*Psiemens
        klinotaxis_net[self.klinotaxis_index["SMBD"]].g_m = 500.0*Psiemens
        klinotaxis_net[self.klinotaxis_index["SMBV"]].v_rest = -55.0*mV
        klinotaxis_net[self.klinotaxis_index["SMBD"]].v_rest = -55.0*mV
        # klinotaxis_net[self.klinotaxis_index["SMBDL"]].v_rest = 0.0*mV
        # klinotaxis_net[self.klinotaxis_index["SMBDR"]].v_rest = 0.0*mV

        for name,index in self.klinotaxis_index.items():
            klinotaxis_net = self.set_neuron_parameters(klinotaxis_net,index,c_m = self.klinotaxis_parameters[name+"_cm"], delta=self.klinotaxis_parameters[name+"_delta"],v=self.klinotaxis_parameters[name+"_v"])

        return klinotaxis_net

    
    def get_klinotaxis_chemical_connections(self,neuron_lists):
        #syns
        connections_m = np.zeros((len(neuron_lists),len(neuron_lists)))
        connections_m[neuron_lists['AIYL'],neuron_lists['AIZL']] = 1
        connections_m[neuron_lists['AIYR'],neuron_lists['AIZR']] = 1
        connections_m[neuron_lists['AIZL'],neuron_lists['SMBV']] = 1
        # connections_m[neuron_lists['AIZL'],neuron_lists['SMBDL']] = 1
        connections_m[neuron_lists['AIZR'],neuron_lists['SMBD']] = 1
        # connections_m[neuron_lists['AIZR'],neuron_lists['SMBDR']] = 1
        sources, targets = connections_m.nonzero()

        #weights
        weights = np.zeros((len(neuron_lists),len(neuron_lists)))
        weights[neuron_lists['AIYL'],neuron_lists['AIZL']] = self.klinotaxis_chemical_parameters["AIYL_to_AIZL"]
        weights[neuron_lists['AIYR'],neuron_lists['AIZR']] = self.klinotaxis_chemical_parameters["AIYR_to_AIZR"]
        weights[neuron_lists['AIZL'],neuron_lists['SMBV']] = self.klinotaxis_chemical_parameters["AIZL_to_SMBV"]
        # weights[neuron_lists['AIZL'],neuron_lists['SMBDL']] = self.klinotaxis_chemical_parameters["AIZL_to_SMBDL"]
        weights[neuron_lists['AIZR'],neuron_lists['SMBD']] = self.klinotaxis_chemical_parameters["AIZR_to_SMBD"]
        # weights[neuron_lists['AIZR'],neuron_lists['SMBDR']] = self.klinotaxis_chemical_parameters["AIZR_to_SMBDR"]

        return sources,targets,weights

    def get_klinotaxis_gap_connections(self,neuron_lists):
        #syns
        connections_m = np.zeros((len(neuron_lists),len(neuron_lists)))
        connections_m[neuron_lists['AIYL'],neuron_lists['AIYR']] = 1
        connections_m[neuron_lists['AIYR'],neuron_lists['AIYL']] = 1
        connections_m[neuron_lists['AIZL'],neuron_lists['AIZR']] = 1
        connections_m[neuron_lists['AIZR'],neuron_lists['AIZL']] = 1
        sources, targets = connections_m.nonzero()

        #weights
        weights = np.zeros((len(neuron_lists),len(neuron_lists)))
        weights[neuron_lists['AIYL'],neuron_lists['AIYR']] = self.klinotaxis_gap_parameters["AIYL_AIYR"]
        weights[neuron_lists['AIYR'],neuron_lists['AIYL']] = self.klinotaxis_gap_parameters["AIYL_AIYR"]
        weights[neuron_lists['AIZL'],neuron_lists['AIZR']] = self.klinotaxis_gap_parameters["AIZL_AIZR"]
        weights[neuron_lists['AIZR'],neuron_lists['AIZL']] = self.klinotaxis_gap_parameters["AIZL_AIZR"]

        return sources,targets,weights


        
    def create_head_neuron(self,num,eqs,method):
        headnet = NeuronGroup(num, eqs, method=method)
        headnet.g_m = 500.0*Psiemens #i_leak
        headnet.g_act = 50.0*Psiemens #i_act
        headnet.v_rest = -72.0*mV 

        headnet = self.set_head_neuron_parameters(headnet,'SMDD',c_m = self.head_parameters["SMDD_cm"], delta=self.head_parameters["SMDD_delta"],v=self.head_parameters["SMDD_v"])
        headnet = self.set_head_neuron_parameters(headnet,'SMDV',c_m = self.head_parameters["SMDV_cm"], delta=self.head_parameters["SMDV_delta"],v=self.head_parameters["SMDV_v"])
        headnet = self.set_head_neuron_parameters(headnet,'RMDD',c_m = self.head_parameters["RMDD_cm"], delta=self.head_parameters["RMDD_delta"],v=self.head_parameters["RMDD_v"])
        headnet = self.set_head_neuron_parameters(headnet,'RMDV',c_m = self.head_parameters["RMDV_cm"], delta=self.head_parameters["RMDV_delta"],v=self.head_parameters["RMDV_v"])

        return headnet
    
    def set_head_neuron_parameters(self,h_net,h_name,c_m,delta,v):
        h_net.c_m[self.headneuron_index[h_name]] = c_m*Pfarad
        h_net.delta[self.headneuron_index[h_name]] = delta
        h_net.v[self.headneuron_index[h_name]] = v*mV
        return h_net
    
    def get_head_chemical_connections(self,headneuron):
        #syns
        head_connections_m = np.zeros((len(headneuron),len(headneuron)))
        head_connections_m[headneuron['SMDD'],headneuron['SMDV']] = 1
        head_connections_m[headneuron['SMDV'],headneuron['SMDD']] = 1
        head_connections_m[headneuron['SMDD'],headneuron['RMDV']] = 1
        head_connections_m[headneuron['SMDV'],headneuron['RMDD']] = 1
        head_connections_m[headneuron['RMDD'],headneuron['RMDV']] = 1
        head_connections_m[headneuron['RMDV'],headneuron['RMDD']] = 1
        sources, targets = head_connections_m.nonzero()

        #weights
        weights = np.zeros((len(headneuron),len(headneuron)))
        weights[headneuron['SMDD'],headneuron['SMDV']] = self.head_chemical_parameters["SMDD_to_SMDV"]
        weights[headneuron['SMDV'],headneuron['SMDD']] = self.head_chemical_parameters["SMDV_to_SMDD"]
        weights[headneuron['SMDD'],headneuron['RMDV']] = self.head_chemical_parameters["SMDD_to_RMDV"]
        weights[headneuron['SMDV'],headneuron['RMDD']] = self.head_chemical_parameters["SMDV_to_RMDD"]
        weights[headneuron['RMDD'],headneuron['RMDV']] = self.head_chemical_parameters["RMDD_to_RMDV"]
        weights[headneuron['RMDV'],headneuron['RMDD']] = self.head_chemical_parameters["RMDV_to_RMDD"]
        return sources,targets,weights

    def get_head_gap_connections(self,headneuron):
        #syns
        head_connections_m = np.zeros((len(headneuron),len(headneuron)))
        head_connections_m[headneuron['SMDD'],headneuron['RMDD']] = 1
        head_connections_m[headneuron['RMDD'],headneuron['SMDD']] = 1
        head_connections_m[headneuron['SMDV'],headneuron['RMDV']] = 1
        head_connections_m[headneuron['RMDV'],headneuron['SMDV']] = 1
        head_connections_m[headneuron['RMDD'],headneuron['RMDV']] = 1
        head_connections_m[headneuron['RMDV'],headneuron['RMDD']] = 1

        sources, targets = head_connections_m.nonzero()

        #weights
        weights = np.zeros((len(headneuron),len(headneuron)))
        weights[headneuron['SMDD'],headneuron['RMDD']] = self.head_gap_parameters["SMDD_RMDD"]
        weights[headneuron['RMDD'],headneuron['SMDD']] = self.head_gap_parameters["SMDD_RMDD"]
        weights[headneuron['SMDV'],headneuron['RMDV']] = self.head_gap_parameters["SMDV_RMDV"]
        weights[headneuron['RMDV'],headneuron['SMDV']] = self.head_gap_parameters["SMDV_RMDV"]
        weights[headneuron['RMDD'],headneuron['RMDV']] = self.head_gap_parameters["RMDD_RMDV"]
        weights[headneuron['RMDV'],headneuron['RMDD']] = self.head_gap_parameters["RMDD_RMDV"]
        return sources,targets,weights

    def set_neuron_parameters(self,net,neuron_index,c_m,delta,v):
        net.c_m[neuron_index] = c_m*Pfarad
        net.delta[neuron_index] = delta
        net.v[neuron_index] = v*mV
        return net
        


    def create_vnc_neuron(self,num,eqs,method):
        vncnet = NeuronGroup(num, eqs, method=method)
        vncnet.delta = -60
        vncnet.g_m = 500.0*Psiemens
        vncnet.g_act = 0.0*Psiemens#50.0*Psiemens #i_act
        vncnet.v_rest = -72.0*mV 

        for i in range(self.unit_num):
            unit_jumper = i*self.neuron_per_unit
            vncnet = self.set_vnc_neuron_parameters(vncnet, unit_jumper + self.vncneuron_index['VB'], c_m = self.vnc_parameters["VB_cm"], delta=self.vnc_parameters["VB_delta"],v=self.vnc_parameters["VB_v"])
            vncnet = self.set_vnc_neuron_parameters(vncnet, unit_jumper + self.vncneuron_index['DB'], c_m = self.vnc_parameters["DB_cm"], delta=self.vnc_parameters["DB_delta"],v=self.vnc_parameters["DB_v"])
            vncnet = self.set_vnc_neuron_parameters(vncnet, unit_jumper + self.vncneuron_index['VD'], c_m = self.vnc_parameters["VD_cm"], delta=self.vnc_parameters["VD_delta"],v=self.vnc_parameters["VD_v"])
            vncnet = self.set_vnc_neuron_parameters(vncnet, unit_jumper + self.vncneuron_index['DD'], c_m = self.vnc_parameters["DD_cm"], delta=self.vnc_parameters["DD_delta"],v=self.vnc_parameters["DD_v"])
        #AVB 
        vncnet.c_m[self.vnc_total_num-1] = 1*Pfarad
        vncnet.v[self.vnc_total_num-1] = -40*mV
        vncnet.g_m[self.vnc_total_num-1] = 0.0*Psiemens
        vncnet.g_act[self.vnc_total_num-1] = 0.0*Psiemens
        # print(self.vncnet.v.shape)

        return vncnet

    def set_vnc_neuron_parameters(self,vncnet,index,c_m,delta,v):
        vncnet.c_m[index] = c_m*Pfarad
        vncnet.delta[index] = delta
        vncnet.v[index] = v*mV
        return vncnet

    def get_vnc_chemical_connections(self,vnc_indexs):
        #syns 
        vnc_connections = np.zeros((self.vnc_total_num,self.vnc_total_num))
        for i in range(self.unit_num):
            unit_jumper = i*self.neuron_per_unit        
            vnc_connections[unit_jumper + vnc_indexs['DB'],unit_jumper + vnc_indexs['VD']] = 1
            vnc_connections[unit_jumper + vnc_indexs['VB'],unit_jumper + vnc_indexs['DD']] = 1
            # vnc_connections[unit_jumper + vnc_indexs['VD'],unit_jumper + vnc_indexs['VB']] = 1
            vnc_connections[unit_jumper + vnc_indexs['DB'],unit_jumper + vnc_indexs['DD']] = 1
            vnc_connections[unit_jumper + vnc_indexs['VB'],unit_jumper + vnc_indexs['VD']] = 1
            # vnc_connections[unit_jumper + vnc_indexs['DD'],unit_jumper + vnc_indexs['DB']] = 1
            # vnc_connections[unit_jumper + vnc_indexs['VD'],unit_jumper + vnc_indexs['VB']] = 1
        sources, targets = vnc_connections.nonzero()
        #weights
        weights = np.zeros((self.vnc_total_num,self.vnc_total_num))
        for i in range(self.unit_num):
            unit_jumper = i*self.neuron_per_unit        
            weights[unit_jumper + vnc_indexs['DB'],unit_jumper + vnc_indexs['VD']] = self.vnc_chemical_parameters["DB_to_VD"]
            weights[unit_jumper + vnc_indexs['VB'],unit_jumper + vnc_indexs['DD']] = self.vnc_chemical_parameters["VB_to_DD"]
            # weights[unit_jumper + vnc_indexs['VD'],unit_jumper + vnc_indexs['VB']] = -10
            weights[unit_jumper + vnc_indexs['DB'],unit_jumper + vnc_indexs['DD']] = self.vnc_chemical_parameters["DB_to_DD"]
            weights[unit_jumper + vnc_indexs['VB'],unit_jumper + vnc_indexs['VD']] = self.vnc_chemical_parameters["VB_to_VD"]
            # weights[unit_jumper + vnc_indexs['DD'],unit_jumper + vnc_indexs['DB']] = -10
            # weights[unit_jumper + vnc_indexs['VD'],unit_jumper + vnc_indexs['VB']] = -10
        return sources,targets,weights

    def get_vnc_gap_connections(self,vnc_indexs):
        #syns 
        vnc_connections = np.zeros((self.vnc_total_num,self.vnc_total_num))
        for i in range(self.unit_num-1):
            pre_unit_jumper = i*self.neuron_per_unit     
            post_unit_jumper = (i+1)*self.neuron_per_unit        
            vnc_connections[pre_unit_jumper + vnc_indexs['DB'],post_unit_jumper + vnc_indexs['DB']] = 1
            vnc_connections[pre_unit_jumper + vnc_indexs['VB'],post_unit_jumper + vnc_indexs['VB']] = 1
            vnc_connections[pre_unit_jumper + vnc_indexs['DD'],post_unit_jumper + vnc_indexs['DD']] = 1
            vnc_connections[pre_unit_jumper + vnc_indexs['VD'],post_unit_jumper + vnc_indexs['VD']] = 1

            vnc_connections[post_unit_jumper + vnc_indexs['DB'],pre_unit_jumper + vnc_indexs['DB']] = 1
            vnc_connections[post_unit_jumper + vnc_indexs['VB'],pre_unit_jumper + vnc_indexs['VB']] = 1
            vnc_connections[post_unit_jumper + vnc_indexs['DD'],pre_unit_jumper + vnc_indexs['DD']] = 1
            vnc_connections[post_unit_jumper + vnc_indexs['VD'],pre_unit_jumper + vnc_indexs['VD']] = 1
        
        #AVB 
        # vnc_connections[vnc_indexs['DB'],self.vnc_total_num-1] = 1
        vnc_connections[self.vnc_total_num-1,vnc_indexs['DB']] = 1
        vnc_connections[self.vnc_total_num-1,vnc_indexs['VB']] = 1
        # vnc_connections[vnc_indexs['VB'],self.vnc_total_num-1] = 1

        # vnc_connections[vnc_indexs['DD'],self.vnc_total_num-1] = 1
        # vnc_connections[self.vnc_total_num-1,vnc_indexs['DD']] = 1
        # vnc_connections[self.vnc_total_num-1,vnc_indexs['VD']] = 1
        # vnc_connections[vnc_indexs['VD'],self.vnc_total_num-1] = 1

        sources, targets = vnc_connections.nonzero()
        #weights
        weights = np.zeros((self.vnc_total_num,self.vnc_total_num))
        for i in range(self.unit_num-1):
            pre_unit_jumper = i*self.neuron_per_unit     
            post_unit_jumper = (i+1)*self.neuron_per_unit         
            weights[pre_unit_jumper + vnc_indexs['DB'],post_unit_jumper + vnc_indexs['DB']] = self.vnc_gap_parameters["DB_DB"]
            weights[pre_unit_jumper + vnc_indexs['VB'],post_unit_jumper + vnc_indexs['VB']] = self.vnc_gap_parameters["VB_VB"]
            weights[pre_unit_jumper + vnc_indexs['DD'],post_unit_jumper + vnc_indexs['DD']] = self.vnc_gap_parameters["DD_DD"]
            weights[pre_unit_jumper + vnc_indexs['VD'],post_unit_jumper + vnc_indexs['VD']] = self.vnc_gap_parameters["VD_VD"]

            weights[post_unit_jumper + vnc_indexs['DB'],pre_unit_jumper + vnc_indexs['DB']] = self.vnc_gap_parameters["DB_DB"]
            weights[post_unit_jumper + vnc_indexs['VB'],pre_unit_jumper + vnc_indexs['VB']] = self.vnc_gap_parameters["VB_VB"]
            weights[post_unit_jumper + vnc_indexs['DD'],pre_unit_jumper + vnc_indexs['DD']] = self.vnc_gap_parameters["DD_DD"]
            weights[post_unit_jumper + vnc_indexs['VD'],pre_unit_jumper + vnc_indexs['VD']] = self.vnc_gap_parameters["VD_VD"]
        
        # weights[vnc_indexs['DB'],self.vnc_total_num-1] = 0.5
        weights[self.vnc_total_num-1,vnc_indexs['DB']] = self.vnc_gap_parameters["AVB_DB"]
        weights[self.vnc_total_num-1,vnc_indexs['VB']] = self.vnc_gap_parameters["AVB_VB"]
        # weights[vnc_indexs['VB'],self.vnc_total_num-1] = 0.5

        # weights[vnc_indexs['DD'],self.vnc_total_num-1] = 0.5
        # weights[self.vnc_total_num-1,vnc_indexs['DD']] = 1*2
        # weights[self.vnc_total_num-1,vnc_indexs['VD']] = 1*2
        # weights[vnc_indexs['VD'],self.vnc_total_num-1] = 0.5

        return sources,targets,weights

    def connect_group(self,neuron_group, sources_index, targets_index, w, syn_eqs):
        s = Synapses(neuron_group, neuron_group, model=syn_eqs)
        s.connect(i=sources_index,j=targets_index)
        for indexs in [i for i in zip(sources_index,targets_index)]:
            s.w[indexs[0],indexs[1]] = w[indexs[0],indexs[1]]
        return s

    def set_head_input(self,headDorsal,headVentral):
        # print(self.headnet[self.headneuron_index['SMDD']].i_stim,self.headnet[self.headneuron_index['SMDV']].i_stim)
        self.headnet[self.headneuron_index['SMDD']].i_sr_ = headDorsal #10  #headDorsal
        self.headnet[self.headneuron_index['SMDV']].i_sr_ = headVentral 
        # print(headDorsal,headVentral)
        # print(self.headnet[self.headneuron_index['SMDD']].i_out,self.headnet[self.headneuron_index['SMDV']].i_out, self.headnet[self.headneuron_index['RMDD']].i_out, self.headnet[self.headneuron_index['RMDV']].i_out)
        # dv/dt = (i_leak+i_act+i_sr*Pamp+i_stim*Pamp+i_gap*Pamp)/c_m :  volt
        #
        # print("<<",self.headnet[self.headneuron_index['SMDD']].i_out,self.headnet[self.headneuron_index['SMDV']].i_out,">>")
        # print("<<",self.headnet[self.headneuron_index['SMDD']].i_leak, self.headnet[self.headneuron_index['SMDD']].i_act ,self.headnet[self.headneuron_index['SMDD']].i_sr,self.headnet[self.headneuron_index['SMDD']].i_stim,self.headnet[self.headneuron_index['SMDD']].i_gap,">>")
    
    def set_head_input_faster(self,inputs):
        self.headnet.i_sr_ = inputs

    def set_vnc_input(self,vncDorsal,vncVentral,unit_index):
        unit_jumper = unit_index*self.neuron_per_unit
        self.vncnet[unit_jumper + self.vncneuron_index['DB']].i_sr_ = vncDorsal 
        self.vncnet[unit_jumper + self.vncneuron_index['VB']].i_sr_ = vncVentral 
        print(unit_jumper + self.vncneuron_index['DB'],unit_jumper + self.vncneuron_index['VB'])
        print(self.vncnet.i_sr_)
        
    def set_vnc_input_faster(self,inputs):
        self.vncnet.i_sr_ = inputs
        # print(self.vncnet.i_sr_)

    def set_klinotaxis_oscillation(self):
        oscillation_v = self.headnet[self.headneuron_index["SMDV"]].i_out_[0]
        oscillation_d = self.headnet[self.headneuron_index["SMDD"]].i_out_[0]
        self.klinotaxis_net[self.klinotaxis_index["SMBVL"]].i_sr_ = (oscillation_v - oscillation_d) * self.klinotaxis_chemical_parameters["SMDV_to_SMBVL"]
        self.klinotaxis_net[self.klinotaxis_index["SMBVR"]].i_sr_ = (oscillation_v  - oscillation_d) * self.klinotaxis_chemical_parameters["SMDV_to_SMBVR"]
        self.klinotaxis_net[self.klinotaxis_index["SMBDL"]].i_sr_ = (-oscillation_v + oscillation_d) * self.klinotaxis_chemical_parameters["SMDV_to_SMBDL"]
        self.klinotaxis_net[self.klinotaxis_index["SMBDR"]].i_sr_ = (-oscillation_v + oscillation_d) * self.klinotaxis_chemical_parameters["SMDV_to_SMBDR"]

        # print(self.klinotaxis_net[self.klinotaxis_index["SMBVL"]].i_sr,self.klinotaxis_net[self.klinotaxis_index["SMBDL"]].i_sr)
        # print(self.klinotaxis_net[self.klinotaxis_index["SMBDL"]].v[0],self.klinotaxis_net[self.klinotaxis_index["SMBVL"]].v[0])

    def set_klinotaxis_input(self):
        asel_value = self.ASEL.get_v()
        aser_value = self.ASER.get_v()
        inputs = np.zeros(len(self.klinotaxis_index))
        inputs[self.klinotaxis_index["AIYL"]] = asel_value*self.klinotaxis_chemical_parameters["ASEL_to_AIYL"] + aser_value*self.klinotaxis_chemical_parameters["ASER_to_AIYL"]
        inputs[self.klinotaxis_index["AIYR"]] = asel_value*self.klinotaxis_chemical_parameters["ASEL_to_AIYR"] + aser_value*self.klinotaxis_chemical_parameters["ASER_to_AIYR"]
        self.klinotaxis_net.i_sr_ = inputs

        # print(self.klinotaxis_net[self.klinotaxis_index["SMBVL"]].i_sr,self.klinotaxis_net[self.klinotaxis_index["SMBVR"]].i_sr,self.klinotaxis_net[self.klinotaxis_index["SMBDL"]].i_sr,self.klinotaxis_net[self.klinotaxis_index["SMBDR"]].i_sr)
        # print(self.klinotaxis_net[self.klinotaxis_index["AIYL"]].v[0], self.klinotaxis_net[self.klinotaxis_index["AIYL"]].i_sr, self.klinotaxis_net[self.klinotaxis_index["AIYR"]].v[0],self.klinotaxis_net[self.klinotaxis_index["AIYR"]].i_sr)
        # print(self.klinotaxis_net[self.klinotaxis_index["AIYL"]].v[0], self.klinotaxis_net[self.klinotaxis_index["AIYL"]].i_out[0], self.klinotaxis_net[self.klinotaxis_index["AIYR"]].v[0], self.klinotaxis_net[self.klinotaxis_index["AIYR"]].i_out[0])
        # print("--------------------------------------------------------------------------------------------------------------------")
        # print("AIY:",self.klinotaxis_net[self.klinotaxis_index["AIYL"]].v_[0], self.klinotaxis_net[self.klinotaxis_index["AIYR"]].v_[0])
        # print("AIZ:",self.klinotaxis_net[self.klinotaxis_index["AIZL"]].v_[0], self.klinotaxis_net[self.klinotaxis_index["AIZR"]].v_[0])
        # print("AIZ_out:",self.klinotaxis_net[self.klinotaxis_index["AIZL"]].i_out_[0], self.klinotaxis_net[self.klinotaxis_index["AIZR"]].i_out_[0])
        # print("SMB:",self.klinotaxis_net[self.klinotaxis_index["SMBV"]].v_[0], self.klinotaxis_net[self.klinotaxis_index["SMBD"]].v_[0])


    def get_klinotaxis_output(self):
        output = self.klinotaxis_net.i_out_[:]
        return (output[self.klinotaxis_index["SMBV"]],
                output[self.klinotaxis_index["SMBD"]])

    def get_all_head_neuron_out(self):
        return self.headnet.i_out_[:]

    def get_head_id(self,neuron_name):
        return self.headneuron_index[neuron_name]

    def get_all_vnc_neuron_out(self):
        return self.vncnet.i_out_[:]

    def get_vnc_id(self,neuron_name,unit_index):
        unit_jumper = unit_index*self.neuron_per_unit
        # print(unit_jumper)
        return unit_jumper + self.vncneuron_index[neuron_name]

    def get_neuron(self,neuron_name,head=False):
        if head:
            return self.headnet[self.headneuron_index[neuron_name]]
        return None

    def run(self,run_time):
        self.net.run(run_time*ms)
        # print(self.klinotaxis_net[self.klinotaxis_index["AIZL"]].i_out,self.klinotaxis_net[self.klinotaxis_index["AIZL"]].v, self.klinotaxis_net[self.klinotaxis_index["AIZR"]].v,self.klinotaxis_net[self.klinotaxis_index["AIZR"]].i_out)

        # print(
        # self.klinotaxis_net[self.klinotaxis_index["AIZL"]].i_stim[0],
        # self.klinotaxis_net[self.klinotaxis_index["AIZR"]].i_stim[0],
        # self.klinotaxis_net[self.klinotaxis_index["SMBVL"]].i_out[0],
        # self.klinotaxis_net[self.klinotaxis_index["SMBVR"]].i_out[0],
        # self.klinotaxis_net[self.klinotaxis_index["SMBDL"]].i_out[0],
        # self.klinotaxis_net[self.klinotaxis_index["SMBDR"]].i_out[0])

    def plot_head(self):
        # self.vncneuron_index = {"DB":0,"VB":1,"DD":2,"VD":3}
        plot(self.M.t/ms, self.M.v[0], label='SMDD')
        plot(self.M.t/ms, self.M.v[1], label='SMDV')
        plot(self.M.t/ms, self.M.v[2], label='RMDD')
        plot(self.M.t/ms, self.M.v[3], label='RMDV')
        ylabel('v')
        legend()
        plt.show()

    def plot_vnc(self):
        # self.vncneuron_index = {"DB":0,"VB":1,"DD":2,"VD":3}
        plot(self.M_2.t/ms, self.M_2.v[0], label='VNC_DB')
        plot(self.M_2.t/ms, self.M_2.v[1], label='VNC_VB')
        plot(self.M_2.t/ms, self.M_2.v[2], label='VNC_DD')
        plot(self.M_2.t/ms, self.M_2.v[3], label='VNC_VD')
        plot(self.M_2.t/ms, self.M_2.v[4], label='VNC_DB_2')
        plot(self.M_2.t/ms, self.M_2.v[5], label='VNC_VB_2')
        plot(self.M_2.t/ms, self.M_2.v[6], label='VNC_DD_2')
        plot(self.M_2.t/ms, self.M_2.v[7], label='VNC_VD_2')
        ylabel('v')
        legend()
        plt.show() 


    def plot_head_vnc(self):
        # self.vncneuron_index = {"DB":0,"VB":1,"DD":2,"VD":3}
        plot(self.M.t/ms, self.M.v[0], label='SMDD')
        plot(self.M.t/ms, self.M.v[1], label='SMDV')
        plot(self.M_2.t/ms, self.M_2.v[0], label='VNC_DB')
        plot(self.M_2.t/ms, self.M_2.v[1], label='VNC_VB')
        plot(self.M_2.t/ms, self.M_2.v[4], label='VNC_DB_2')
        plot(self.M_2.t/ms, self.M_2.v[5], label='VNC_VB_2')
        ylabel('v')
        legend()
        plt.show() 

    def plot_heat_map(self):
        head_labels= [key for key in self.headneuron_index]
        vnc_labels = []
        for vnc_id in range(self.unit_num):
            for key in self.vncneuron_index:
                vnc_labels.append(key+":"+str(vnc_id))
        vnc_labels.append("AVB")
        labels = head_labels +vnc_labels
        # fig, ax = plt.subplots()
        # print(self.M.v_.shape,self.M_2.v_.shape,np.concatenate((self.M.v_,self.M_2.v_)).shape)
        plt.figure(figsize=(100,100))
        plt.imshow(np.concatenate((self.M.v_,self.M_2.v_)))
        # plt.yticks(np.arange(len(labels)),labels)
        # figure.set_yticks(np.arange(len(labels)))
        # figure.set_yticklabels(labels)
        plt.show()
        # plt.savefig("heatmap.jpg")




# step_size = 0.1*ms
# worm_net = WormNet(step_size = step_size)
# worm = Worm(step_size=step_size/ms)

# for i in range(1000):
#     worm.step(worm_net)
#     # print(worm_net.SMDD.i_out,worm_net.SMDV.i_out)

# worm_net.plot()
