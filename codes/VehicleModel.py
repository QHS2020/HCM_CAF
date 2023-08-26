# -*- coding: utf-8 -*-

"""
Containing the models that specity the vehicle motion. 



sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])


---------------------------------------------------------
Get Expected trajectoryes for all possible steering angles:

    reload(VM)
    #expected_trs_for_steers[steeringangle] = {'distances':distances, 'polygons':polygons}
    expected_trs_for_steers = VM.VehicleKineticSolver.GetExpectedTrajectorys_shapely()

    #---------------Affine certain traejctory from polygons_list to affined
    reload(VM)
    TR = expected_trs_for_steers[-0.18795853483015856]
    polygons_list = TR['polygons']
    affined = VM.VehicleKineticSolver.AffineExpectedTrajectory( polygons_list, new_state= (20, 20 , 45*np.pi/180.0))



"""

from RequiredModules import *
import shapely




class TwoDimStochasticIDM():
    """
    The two dimensional stochastic IDM model.
    
    The state model is give by:
        dZ = F(Z)dt + L(Z)dW
    
        
    NOTE the coordinate system: 
    
        that the x is the logitudinal axis and y is the lateral. 
    
        The poisitive axis it downstream. 
    
        The left direction is y-positive. 
        
        If the vehicle turn left the steeer angle is positive. 
    
    
    """
    #vehicle parameters
    veh_paras = {'lf':1.1, 'lr':1.3, 'lF':2.5, 'lR':2.5, 'width':2.0, 'max_steer':70*np.pi/180.0, 'min_steer':-70*np.pi/180.0, 'max_steer_rate':0.4, 'min_steer_rate':-0.4,}
    
    #idm car following parameters
    #   vf unit is m/s; 
    idm_paras = {'idm_vf': 90.0/3.6, 'idm_T':1.5, 'idm_delta':4.0, 'idm_s0':2.0, 'idm_a':3.0, 'idm_b':3.21}
    
    #idm_steer parameters
    idm_steer_paras = {'tau':2.0, }
    @classmethod
    def F(self, STATES, STATES_leader, veh_paras_self = veh_paras, veh_paras_leader = veh_paras, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = 1.0, eta_lat = 1.0, sigma_long = .05, sigma_lat = .05, weight_heading = 1.0, weight_line_CG = 1e-5, sigma_long_drift = 1.0, sigma_lat_drift = 1.0, stochastic_proecess_name = 'OU'):
        """
        THE VEHICLE IS FRONT STEER. 
        
        Return the derivate of the X,Y and PHI. 
        X is the horizontal coordinate
        Y is the vertical coordinate
        PHI is the heading angle. 
        V is the speed. 
        -------------------------------------------
        @input: stochastic_proecess_name
        
            OU, means ornstein ublllk 
            geometric geomtric brownian
            jacobidiffusion
            
            converted
                
                convrt from the 
        
        @input: sigma_long and sigma_lat
            
            the parameters of the sde. 
            
            They are the noise at the longitudinal and lateral dimension. 
            
        @input: STATE_equilibrium
            
            the equilibrium state. 
            
            x_equi,y_equi,phi_equi,v_equi,delta_equi,Z_long_equi,Z_lat_equi = STATE_equilibrium[0],STATE_equilibrium[1],STATE_equilibrium[2],STATE_equilibrium[3],STATE_equilibrium[4],STATE_equilibrium[5],STATE_equilibrium[6]
            
            STATE_equilibrium = np.array([, , , , , 1.0, 1.0])
            
            
        @input: eta_long and eta_lat
        
            the parameters in the systme state equation. 
            
        @input: STATES_leader
        
            the states of the leader. 

            x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
            x and y are coordinate
            
            v is the speed
            
            delta is the steer angle of front wheel
            
            Z_lon and Z_lat is the nosie for longitudinal (x dimension) and lateral (y dimension).
        
        
        @input: STATES
            
            x,y,phi,v,delta,Z_lon,Z_lat=STATES
            
            x and y are coordinate
            
            v is the speed
            
            delta is the steer angle of front wheel
            
            Z_lon and Z_lat is the nosie for longitudinal (x dimension) and lateral (y dimension).
        
        
        
        @input: params
            the parameters of the vehicle. 
        
        
        @input: lf and lr
            unit is meter. 
            lf is the lengh of front. i.e. the distance between front axel to the CG. 
            lr is the rear length, or the ditance betweene rear axel to the CG. 
        
        @OUTPUT: diff_STATES
            len(STATES)=4:
                - X = STATES[0], the X of the CG. X is the horizontal axis. 
                - Y = STATES[1], the Y of the CG
                - PHI = STATES[2], the heading angle. between the vehicle and the X axis. 
                - V = STATES[3]
        """
        #the system  state. 
        x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
        
        
        #
        lr = veh_paras_self.get('lr', 2)
        lf = veh_paras_self.get('lf', 2)
        
        
        #beta, the intermediate parameter
        #print(lr,lf, front_steer_angle, np.tan(front_steer_angle))
        tmp = (lr*np.tan(delta))/(lr+lf)
        beta = np.arctan(tmp)
        
        #################################################
        #_diff means differential , *v/idm_paras['idm_vf']
        # - eta_long*np.tanh(Z_long)
        # - eta_lat*np.tanh(Z_lat)
        #  - eta_long*Z_long
        # - eta_lat*Z_lat
        diff_x = v*np.cos( phi + beta) - eta_long*np.tanh(Z_long)
        diff_y = v*np.sin( phi + beta) - eta_lat*np.tanh(Z_lat)
        diff_phi = v*np.cos(beta)/(lr+lf)*np.tan(delta)
        #
        acce = self.IDM_acce(STATES, STATES_leader, idm_paras = idm_paras, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader)
        #trim the acceleration. 
        diff_v = self.TrimAcce(STATE  = STATES, acce = acce, idm_paras = idm_paras)
        #
        steerrate = self.IDM_steer(STATES, STATES_leader, idm_steer_paras = idm_steer_paras, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader, weight_heading = weight_heading, weight_line_CG = weight_line_CG)
        #trim the steer rate
        diff_delta = self.TrimSteerRate(STATE = STATES, steerrate = steerrate, veh_paras = veh_paras_self)
        #
        #
        if stochastic_proecess_name=='OU':
            #diff_Z_long = -(sigma_long**1.0)*(Z_long**3)
            #diff_Z_lat = -(sigma_lat**1.0)*(Z_lat)
            #print(Z_long, sigma_long_drift)
            diff_Z_long = -sigma_long_drift*Z_long
            diff_Z_lat = -sigma_lat_drift*Z_lat
        elif stochastic_proecess_name=='converted':
            #the converted. 
            diff_Z_long = -(sigma_long**2)*(1-Z_long**2)*Z_long
            diff_Z_lat = -(sigma_lat**2)*(1-Z_lat**2)*Z_lat
        elif stochastic_proecess_name=='geometric':
            #
            diff_Z_long =  -sigma_long_drift*(Z_long)
            diff_Z_lat = -sigma_lat_drift*(Z_lat)
            #
        elif stochastic_proecess_name=='jacobi':
            #
            diff_Z_long = -sigma_long_drift*(Z_long - .0)
            diff_Z_lat = -sigma_lat_drift*(Z_lat - .0)
            
        elif stochastic_proecess_name=='hyperparabolic':
            #
            #diff_Z_long = -sigma_long_drift*(Z_long - .0)
            #diff_Z_lat = -sigma_lat_drift*(Z_lat - .0)
            diff_Z_long =  -Z_long-sigma_long_drift*Z_long
            diff_Z_lat = -Z_lat-sigma_lat_drift*Z_lat
            #print(diff_Z_long, diff_Z_lat)
        elif stochastic_proecess_name=='ROU':
            #ew_state = STATES[-1] + (theta/STATES[-1] -  STATES[-1] )*deltat + sigma*brownian
            diff_Z_long = -sigma_long_drift/Z_long +  Z_long#-sigma_long_drift*(Z_long - .0)
            diff_Z_lat = -sigma_lat_drift/Z_lat +  Z_lat #-sigma_lat_drift*(Z_lat - .0)

        return np.array([diff_x,diff_y,diff_phi,diff_v, diff_delta, diff_Z_long, diff_Z_lat])


    @classmethod
    def F_with_terminal_condition(self, STATES, STATES_leader, terminalcondition, terminalmoment, veh_paras_self = veh_paras, veh_paras_leader = veh_paras, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = 1.0, eta_lat = 1.0, sigma_long = .05, sigma_lat = .05, weight_heading = 1.0, weight_line_CG = 1e-5, sigma_long_drift = 1.0, sigma_lat_drift = 1.0, stochastic_proecess_name = 'OU'):
        """
        This method is used to generate the bridge of the system dynamics. 
        
        Difference between:
            
            - self.F
            - self.F_with_terminal_condition, with two extra args: terminalcondition, terminalmoment
                the former one is a 1d array with length 7, the latter is a float. 
        
        
        THE VEHICLE IS FRONT STEER. 
        
        Return the derivate of the X,Y and PHI. 
        X is the horizontal coordinate
        Y is the vertical coordinate
        PHI is the heading angle. 
        V is the speed. 
        -------------------------------------------
        @input: stochastic_proecess_name
        
            OU, means ornstein ublllk 
            geometric geomtric brownian
            jacobidiffusion
            
            converted
                
                convrt from the 
        
        @input: sigma_long and sigma_lat
            
            the parameters of the sde. 
            
            They are the noise at the longitudinal and lateral dimension. 
            
        @input: STATE_equilibrium
            
            the equilibrium state. 
            
            x_equi,y_equi,phi_equi,v_equi,delta_equi,Z_long_equi,Z_lat_equi = STATE_equilibrium[0],STATE_equilibrium[1],STATE_equilibrium[2],STATE_equilibrium[3],STATE_equilibrium[4],STATE_equilibrium[5],STATE_equilibrium[6]
            
            STATE_equilibrium = np.array([, , , , , 1.0, 1.0])
            
            
        @input: eta_long and eta_lat
        
            the parameters in the systme state equation. 
            
        @input: STATES_leader
        
            the states of the leader. 

            x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
            x and y are coordinate
            
            v is the speed
            
            delta is the steer angle of front wheel
            
            Z_lon and Z_lat is the nosie for longitudinal (x dimension) and lateral (y dimension).
        
        
        @input: STATES
            
            x,y,phi,v,delta,Z_lon,Z_lat=STATES
            
            x and y are coordinate
            
            v is the speed
            
            delta is the steer angle of front wheel
            
            Z_lon and Z_lat is the nosie for longitudinal (x dimension) and lateral (y dimension).
        
        
        
        @input: params
            the parameters of the vehicle. 
        
        
        @input: lf and lr
            unit is meter. 
            lf is the lengh of front. i.e. the distance between front axel to the CG. 
            lr is the rear length, or the ditance betweene rear axel to the CG. 
        
        @OUTPUT: diff_STATES
            len(STATES)=4:
                - X = STATES[0], the X of the CG. X is the horizontal axis. 
                - Y = STATES[1], the Y of the CG
                - PHI = STATES[2], the heading angle. between the vehicle and the X axis. 
                - V = STATES[3]
        """
        #the system  state. 
        x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
        
        
        #
        lr = veh_paras_self.get('lr', 2)
        lf = veh_paras_self.get('lf', 2)
        
        
        #beta, the intermediate parameter
        #print(lr,lf, front_steer_angle, np.tan(front_steer_angle))
        tmp = (lr*np.tan(delta))/(lr+lf)
        beta = np.arctan(tmp)
        
        #################################################
        #_diff means differential , *v/idm_paras['idm_vf']
        # - eta_long*np.tanh(Z_long)
        # - eta_lat*np.tanh(Z_lat)
        #  - eta_long*Z_long
        # - eta_lat*Z_lat
        diff_x = v*np.cos( phi + beta) - eta_long*np.tanh(Z_long)
        diff_y = v*np.sin( phi + beta) - eta_lat*np.tanh(Z_lat)
        diff_phi = v*np.cos(beta)/(lr+lf)*np.tan(delta)
        #
        acce = self.IDM_acce(STATES, STATES_leader, idm_paras = idm_paras, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader)
        #trim the acceleration. 
        diff_v = self.TrimAcce(STATE  = STATES, acce = acce, idm_paras = idm_paras)
        #
        steerrate = self.IDM_steer(STATES, STATES_leader, idm_steer_paras = idm_steer_paras, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader, weight_heading = weight_heading, weight_line_CG = weight_line_CG)
        #trim the steer rate
        diff_delta = self.TrimSteerRate(STATE = STATES, steerrate = steerrate, veh_paras = veh_paras_self)
        #
        #
        if stochastic_proecess_name=='OU':
            #diff_Z_long = -(sigma_long**1.0)*(Z_long**3)
            #diff_Z_lat = -(sigma_lat**1.0)*(Z_lat)
            #print(Z_long, sigma_long_drift)
            diff_Z_long = -sigma_long_drift*Z_long
            diff_Z_lat = -sigma_lat_drift*Z_lat
        elif stochastic_proecess_name=='converted':
            #the converted. 
            diff_Z_long = -(sigma_long**2)*(1-Z_long**2)*Z_long
            diff_Z_lat = -(sigma_lat**2)*(1-Z_lat**2)*Z_lat
        elif stochastic_proecess_name=='geometric':
            #
            diff_Z_long =  -sigma_long_drift*(Z_long)
            diff_Z_lat = -sigma_lat_drift*(Z_lat)
            #
        elif stochastic_proecess_name=='jacobi':
            #
            diff_Z_long = -sigma_long_drift*(Z_long - .0)
            diff_Z_lat = -sigma_lat_drift*(Z_lat - .0)
            
        elif stochastic_proecess_name=='hyperparabolic':
            #
            #diff_Z_long = -sigma_long_drift*(Z_long - .0)
            #diff_Z_lat = -sigma_lat_drift*(Z_lat - .0)
            diff_Z_long =  -Z_long-sigma_long_drift*Z_long
            diff_Z_lat = -Z_lat-sigma_lat_drift*Z_lat
            #print(diff_Z_long, diff_Z_lat)
        elif stochastic_proecess_name=='ROU':
            #ew_state = STATES[-1] + (theta/STATES[-1] -  STATES[-1] )*deltat + sigma*brownian
            diff_Z_long = -sigma_long_drift/Z_long +  Z_long#-sigma_long_drift*(Z_long - .0)
            diff_Z_lat = -sigma_lat_drift/Z_lat +  Z_lat #-sigma_lat_drift*(Z_lat - .0)

        return np.array([diff_x,diff_y,diff_phi,diff_v, diff_delta, diff_Z_long, diff_Z_lat])

    @classmethod
    def F_lyapunov(self, STATES, STATES_leader, veh_paras_self = veh_paras, veh_paras_leader = veh_paras, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = 1.0, eta_lat = 1.0, sigma_long = .05, sigma_lat = .05, weight_heading = 1.0, weight_line_CG = 1e-5, sigma_long_drift = 1.0, sigma_lat_drift = 1.0, stochastic_proecess_name = 'OU'):
        """
        THE VEHICLE IS FRONT STEER. 
        
        Return the derivate of the X,Y and PHI. 
        X is the horizontal coordinate
        Y is the vertical coordinate
        PHI is the heading angle. 
        V is the speed. 
        -------------------------------------------
        @input: sigma_long and sigma_lat
            
            the parameters of the sde. 
            
            They are the noise at the longitudinal and lateral dimension. 
            
        @input: STATE_equilibrium
            
            the equilibrium state. 
            
            x_equi,y_equi,phi_equi,v_equi,delta_equi,Z_long_equi,Z_lat_equi = STATE_equilibrium[0],STATE_equilibrium[1],STATE_equilibrium[2],STATE_equilibrium[3],STATE_equilibrium[4],STATE_equilibrium[5],STATE_equilibrium[6]
            
            STATE_equilibrium = np.array([, , , , , 1.0, 1.0])
            
            
        @input: eta_long and eta_lat
        
            the parameters in the systme state equation. 
            
        @input: STATES_leader
        
            the states of the leader. 

            x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
            x and y are coordinate
            
            v is the speed
            
            delta is the steer angle of front wheel
            
            Z_lon and Z_lat is the nosie for longitudinal (x dimension) and lateral (y dimension).
        
        
        @input: STATES
            
            x,y,phi,v,delta,Z_lon,Z_lat=STATES
            
            x and y are coordinate
            
            v is the speed
            
            delta is the steer angle of front wheel
            
            Z_lon and Z_lat is the nosie for longitudinal (x dimension) and lateral (y dimension).
        
        
        
        @input: params
            the parameters of the vehicle. 
        
        
        @input: lf and lr
            unit is meter. 
            lf is the lengh of front. i.e. the distance between front axel to the CG. 
            lr is the rear length, or the ditance betweene rear axel to the CG. 
        
        @OUTPUT: diff_STATES
            len(STATES)=4:
                - X = STATES[0], the X of the CG. X is the horizontal axis. 
                - Y = STATES[1], the Y of the CG
                - PHI = STATES[2], the heading angle. between the vehicle and the X axis. 
                - V = STATES[3]
        """
        #the system  state. 
        x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
        
        
        #
        lr = veh_paras_self.get('lr', 2)
        lf = veh_paras_self.get('lf', 2)
        
        
        #beta, the intermediate parameter
        #print(lr,lf, front_steer_angle, np.tan(front_steer_angle))
        tmp = (lr*np.tan(delta))/(lr+lf)
        beta = np.arctan(tmp)
        
        #################################################
        #_diff means differential , *v/idm_paras['idm_vf']
        diff_x = -STATES_leader[3] + v*np.cos( phi + beta)
        diff_y = v*np.sin( phi + beta)
        diff_phi = v*np.cos(beta)/(lr+lf)*np.tan(delta)
        #
        acce = self.IDM_acce(STATES, STATES_leader, idm_paras = idm_paras, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader)
        #trim the acceleration. 
        diff_v = self.TrimAcce(STATE  = STATES, acce = acce   - eta_long*Z_long, idm_paras = idm_paras)
        #
        steerrate = self.IDM_steer(STATES, STATES_leader, idm_steer_paras = idm_steer_paras, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader, weight_heading = weight_heading, weight_line_CG = weight_line_CG)
        #trim the steer rate
        diff_delta = self.TrimSteerRate(STATE = STATES, steerrate = steerrate  - eta_lat*Z_lat, veh_paras = veh_paras_self)
        
        #
        if stochastic_proecess_name=='OU':
            #diff_Z_long = -(sigma_long**1.0)*(Z_long**3)
            #diff_Z_lat = -(sigma_lat**1.0)*(Z_lat)
            #print(Z_long, sigma_long_drift)
            diff_Z_long = -sigma_long_drift*Z_long
            diff_Z_lat = -sigma_lat_drift*Z_lat
        elif stochastic_proecess_name=='converted':
            #the converted. 
            diff_Z_long = -(sigma_long**2)*(1-Z_long**2)*Z_long
            diff_Z_lat = -(sigma_lat**2)*(1-Z_lat**2)*Z_lat
        elif stochastic_proecess_name=='geometric':
            #
            diff_Z_long = -sigma_long_drift*Z_long
            diff_Z_lat = -sigma_lat_drift*Z_lat
        elif stochastic_proecess_name=='jacobi':
            #
            diff_Z_long = -sigma_long_drift*Z_long
            diff_Z_lat = -sigma_lat_drift*Z_lat
        
        #the converted. 
        #diff_Z_long = -(sigma_long**2)*(1-Z_long**2)*Z_long + 1/2.0*(sigma_long*(1.0-Z_long**2))**2
        #diff_Z_lat = -(sigma_lat**2)*(1-Z_lat**2)*Z_lat + 1/2.0*(sigma_lat*(1.0-Z_lat**2))**2
        
        return np.array([diff_x,diff_y,diff_phi,diff_v, diff_delta, diff_Z_long, diff_Z_lat])

    
    @classmethod
    def F_BKP(self, STATES, STATES_leader, veh_paras_self = veh_paras, veh_paras_leader = veh_paras, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = 1.0, eta_lat = 1.0, sigma_long = .05, sigma_lat = .05):
        """
        THE VEHICLE IS FRONT STEER. 
        
        Return the derivate of the X,Y and PHI. 
        X is the horizontal coordinate
        Y is the vertical coordinate
        PHI is the heading angle. 
        V is the speed. 
        -------------------------------------------
        @input: sigma_long and sigma_lat
            
            the parameters of the sde. 
            
            They are the noise at the longitudinal and lateral dimension. 
            
            
        @input: eta_long and eta_lat
        
            the parameters in the systme state equation. 
            
        @input: STATES_leader
        
            the states of the leader. 

            x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
            x and y are coordinate
            
            v is the speed
            
            delta is the steer angle of front wheel
            
            Z_lon and Z_lat is the nosie for longitudinal (x dimension) and lateral (y dimension).
        
        
        @input: STATES
            
            x,y,phi,v,delta,Z_lon,Z_lat=STATES
            
            x and y are coordinate
            
            v is the speed
            
            delta is the steer angle of front wheel
            
            Z_lon and Z_lat is the nosie for longitudinal (x dimension) and lateral (y dimension).
        
        
        
        @input: params
            the parameters of the vehicle. 
        
        
        @input: lf and lr
            unit is meter. 
            lf is the lengh of front. i.e. the distance between front axel to the CG. 
            lr is the rear length, or the ditance betweene rear axel to the CG. 
        
        @OUTPUT: diff_STATES
            len(STATES)=4:
                - X = STATES[0], the X of the CG. X is the horizontal axis. 
                - Y = STATES[1], the Y of the CG
                - PHI = STATES[2], the heading angle. between the vehicle and the X axis. 
                - V = STATES[3]
        """
        #the system  state. 
        x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
        
        
        #
        lr = veh_paras_self.get('lr', 2)
        lf = veh_paras_self.get('lf', 2)
        
        
        #beta, the intermediate parameter
        #print(lr,lf, front_steer_angle, np.tan(front_steer_angle))
        tmp = (lr*np.tan(delta))/(lr+lf)
        beta = np.arctan(tmp)
        
        #_diff means differential 
        diff_x = v*np.cos( phi + beta) + eta_long*Z_long
        diff_y = v*np.sin( phi + beta) + eta_lat*Z_lat
        diff_phi = v*np.cos(beta)/(lr+lf)*np.tan(delta)
        #
        acce = self.IDM_acce(STATES, STATES_leader, idm_paras = idm_paras, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader)
        #trim the acceleration. 
        diff_v = self.TrimAcce(STATE  = STATES, acce = acce, idm_paras = idm_paras)
        #
        steerrate = self.IDM_steer(STATES, STATES_leader, idm_steer_paras = idm_steer_paras, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader)
        #trim the steer rate
        diff_delta = self.TrimSteerRate(STATE = STATES, steerrate = steerrate, veh_paras = veh_paras_self)
        #
        diff_Z_long = -(sigma_long**2)*(1-Z_long**2)*Z_long
        diff_Z_lat = -(sigma_lat**2)*(1-Z_lat**2)*Z_lat
        
        return np.array([diff_x,diff_y,diff_phi,diff_v, diff_delta, diff_Z_long, diff_Z_lat])
    
    @classmethod
    def L_lyapunov(self, STATES, eta_long = 1.0, eta_lat = 1.0, sigma_long = .05, sigma_lat = .05, stochastic_proecess_name = 'OU'):
        """
        
        @OUTPUT: array
        
            shape is (7,2).
            
            7 means the dimension of the state. 
            2 measnthat there are two randomness. 
        
        """
        #the system  state. 
        x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]

        if stochastic_proecess_name=='OU':
            tmp_long = sigma_long
            tmp_lat = sigma_lat
            #
        elif stochastic_proecess_name=='converted':
            #the converted. 
            tmp_long =  sigma_long*(1.0-Z_long**2)
            tmp_lat = sigma_lat*(1-Z_lat**2)
            #
        elif stochastic_proecess_name=='geometric':
            #
            tmp_long =  sigma_long*Z_long
            tmp_lat = sigma_lat*Z_lat
            #
        elif stochastic_proecess_name=='jacobi':
            tmp_long =   np.sqrt(sigma_long*(Z_long+.5)*(.5-Z_long))
            tmp_lat = np.sqrt(sigma_lat*(Z_lat+.5)*(.5-Z_lat))

        #
        #tmp_long =  sigma_long*(1.0-Z_long**2)
        #tmp_lat = sigma_lat*(1-Z_lat**2)
        
        array = np.array([[.0, .0, .0, .0, .0, tmp_long, .0], \
                          [.0, .0, .0, .0, .0, .0, tmp_lat]])
        
        
        return array.T

    @classmethod
    def L(self, STATES, eta_long = 1.0, eta_lat = 1.0, sigma_long = .05, sigma_lat = .05, stochastic_proecess_name = 'OU'):
        """
        
        @OUTPUT: array
        
            shape is (7,2).
            
            7 means the dimension of the state. 
            2 measnthat there are two randomness. 
        
        """
        #the system  state. 
        x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
        
        if stochastic_proecess_name=='OU':
            tmp_long = sigma_long
            tmp_lat = sigma_lat
            #
        elif stochastic_proecess_name=='converted':
            #the converted. 
            tmp_long =  sigma_long*(1.0-Z_long**2)
            tmp_lat = sigma_lat*(1-Z_lat**2)
            #
        elif stochastic_proecess_name=='geometric':
            #
            tmp_long =  sigma_long*(Z_long)
            tmp_lat = sigma_lat*(Z_lat)
            #
        elif stochastic_proecess_name=='hyperparabolic':
            #
            tmp_long = sigma_long
            tmp_lat = sigma_lat
            
        elif stochastic_proecess_name=='jacobi':
            #new_state = STATES[-1] - theta*(STATES[-1])*deltat + sigma*np.sqrt((STATES[-1]+.5)*(.5-STATES[-1]))*brownian
            Z_long = max(-.499999, min(Z_long, .4999999))
            tmp_long =   np.sqrt(sigma_long*(Z_long+.5)*(.5-Z_long))
            #print(Z_lat, (.5-Z_lat), (Z_lat+.5))
            Z_lat = max(-.499999, min(Z_lat, .4999999))
            tmp_lat = np.sqrt(sigma_lat*(Z_lat+.5)*(.5-Z_lat))
        elif stochastic_proecess_name=='ROU':
            #
            tmp_long =  sigma_long
            tmp_lat = sigma_lat
            #

        #
        #tmp_long =  sigma_long*(1.0-Z_long**2)
        #tmp_lat = sigma_lat*(1-Z_lat**2)
        
        array = np.array([[.0, .0, .0, .0, .0, tmp_long, .0], \
                          [.0, .0, .0, .0, .0, .0, tmp_lat]])
        
        
        return array.T


    @classmethod
    def L_diff(self, STATES, eta_long = 1.0, eta_lat = 1.0, sigma_long = .05, sigma_lat = .05, stochastic_proecess_name = 'OU'):
        """
        
        @OUTPUT: array
        
            shape is (7,2).
            
            7 means the dimension of the state. 
            2 measnthat there are two randomness. 
        
        """
        #the system  state. 
        x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
        
        if stochastic_proecess_name=='OU':
            tmp_long = sigma_long
            tmp_lat = sigma_lat
            #
        elif stochastic_proecess_name=='converted':
            #the converted. 
            tmp_long =  sigma_long*(1.0-Z_long**2)
            tmp_lat = sigma_lat*(1-Z_lat**2)
            #
        elif stochastic_proecess_name=='geometric':
            #
            tmp_long =  sigma_long*(Z_long)
            tmp_lat = sigma_lat*(Z_lat)
            #
        elif stochastic_proecess_name=='hyperparabolic':
            #
            tmp_long = sigma_long
            tmp_lat = sigma_lat
            
        elif stochastic_proecess_name=='jacobi':
            #new_state = STATES[-1] - theta*(STATES[-1])*deltat + sigma*np.sqrt((STATES[-1]+.5)*(.5-STATES[-1]))*brownian
            Z_long = max(-.499999, min(Z_long, .4999999))
            tmp_long =   np.sqrt(sigma_long*(Z_long+.5)*(.5-Z_long))
            #print(Z_lat, (.5-Z_lat), (Z_lat+.5))
            Z_lat = max(-.499999, min(Z_lat, .4999999))
            tmp_lat = np.sqrt(sigma_lat*(Z_lat+.5)*(.5-Z_lat))
        elif stochastic_proecess_name=='ROU':
            #
            tmp_long =  sigma_long
            tmp_lat = sigma_lat
            #

        #
        #tmp_long =  sigma_long*(1.0-Z_long**2)
        #tmp_lat = sigma_lat*(1-Z_lat**2)
        
        array = np.array([[.0, .0, .0, .0, .0, tmp_long, .0], \
                          [.0, .0, .0, .0, .0, .0, tmp_lat]])
        
        
        return array.T
    
    @classmethod
    def TrimSteerRate(self, STATE, steerrate, veh_paras = veh_paras):
        """
        Change the acceleraiton to make sure that the resulting vehicle constraints would not be violated. 
        
        The following conditions are considered when trimming:
        
            - if the speed reaches the maximum or the minimum, the acceleration is zero
            - if the 
        
        veh_paras = {'lf':1.5, 'lr':1.5, 'lF':2.5, 'lR':2.5, 'width':2.0, 'max_steer':70*np.pi/180.0, 'min_steer':-70*np.pi/180.0, 'max_steer_rate':0.4, 'min_steer_rate':-0.4,}
        
        -----------------------------
        @OUTPUT steer_trimmed
        
            the trimmed acceleration. 
            
            
            
        """
        
        
        
        #the system  state. 
        x,y,phi,v,delta,Z_long,Z_lat = STATE[0],STATE[1],STATE[2],STATE[3],STATE[4],STATE[5],STATE[6]
        
        #
        #if delta<veh_paras['min_steer'] or delta>=veh_paras['max_steer']:
        #    return 0.0

        if delta<veh_paras['min_steer']:
            return 1e-2
        if delta>=veh_paras['max_steer']:
            return -1e-2



        #-------------------------
        return min(veh_paras['max_steer_rate'], max(veh_paras['min_steer_rate'], steerrate))
        
    
    
    @classmethod
    def TrimAcce(self, acce, STATE, idm_paras = idm_paras):
        """
        Change the acceleraiton to make sure that the resulting vehicle constraints would not be violated. 
        
        The following conditions are considered when trimming:
        
            - if the speed reaches the maximum or the minimum, the acceleration is zero
            - if the 
        
        -----------------------------
        @OUTPUT acce_trimmed
        
            the trimmed acceleration. 
            
            
            
        """
        
        #the system  state. 
        x,y,phi,v,delta,Z_long,Z_lat = STATE[0],STATE[1],STATE[2],STATE[3],STATE[4],STATE[5],STATE[6]
        
        #
        #if v<0 or v>=1.3*idm_paras['idm_vf']:
        if v<.0:
            return 1.0
        if v>idm_paras['idm_vf']:
            return -1
            
        #-------------------------
        return min(idm_paras['idm_a'], max(-idm_paras['idm_b'], acce))
        
    
    @classmethod
    def IDM_acce(self, STATES, STATES_leader,  idm_paras = idm_paras,  veh_paras_self = veh_paras, veh_paras_leader = veh_paras):
        """
        IDM formation output, the acceleration. 
        
        Note that the delta_v is defined as v_follower-v_leader. 
        
        @type v: float, unit is km/h
        
        @type vf: km/h.
        
        
        @type T: float.
        @param: T, unit is sec
            Average safe time headway.
            
        @type delta: delta:float
        @param: delta
            parameter in the model.
            
            
        @type s0:float
        @param: s0
            parameter 
        
        @OUTPUT: a
            unit is m/s2.
        """
        #IDM paraser. 
        idm_vf = idm_paras['idm_vf']
        idm_T = idm_paras['idm_T']
        idm_delta = idm_paras['idm_delta']
        idm_s0 = idm_paras['idm_s0']
        idm_a = idm_paras['idm_a']
        idm_b = idm_paras['idm_b']

        #the system  state. 
        x_self,y_self,phi_self,v_self,delta_self,Z_long_self,Z_lat_self = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
        
        #the system  state of the leader.
        x_leader,y_leader,phi_leader,v_leader,delta_leader,Z_long_leader,Z_lat_leader = STATES_leader[0],STATES_leader[1],STATES_leader[2],STATES_leader[3],STATES_leader[4],STATES_leader[5],STATES_leader[6]
        
        #
        #v_self=v/3.6
        #v_leader = v_leader_kmh/3.6
        vf = idm_vf
        
        deltax = np.sqrt((x_self-x_leader)**2 + (y_self - y_leader)**2) - veh_paras_self['lF']/1.0 - veh_paras_leader['lR']/1.0
        
        #
        try:
            s_star = idm_s0+v_self*idm_T+v_self*(v_self - v_leader)/(2.0*np.sqrt(idm_a*idm_b))
            a = 1.0*idm_a*(1-np.power(v_self/vf, idm_delta)-(s_star*s_star)/(deltax*deltax))
        except Exception as e:
            
            print('deltax = ',deltax,', v_self=',v_self,', v_leader',v_leader)
            raise ValueError(e)
            
        return a
    
    
    
    
    
    
    @classmethod
    def random_accelerations(self, amplification = 3.0, ts = np.linspace(0, 300, 300), period_sec = 30):
        """
        
        
        acces  = VM.TwoDimStochasticIDM.random_accelerations(ts = ts)
        """
        return np.array([amplification*np.sin(t/period_sec) for t in ts[1:]])
    
    @classmethod
    def GenerateBrownianPaths(self, ts = np.linspace(0, 300, 300)):
        """
        
        Geneate the brownian paths. 
        
        The generated brownian should be a 2d array. shape is (2, len(ts)-1)
        
        
        brownianpath = self.GenerateBrownianPaths(ts = ts)
        ------------------------------------------------
        
        
        """
        stds = np.sqrt(np.diff(ts))
        means = np.zeros(stds.shape)
        r1 = np.random.normal(loc = means, scale = stds)
        r2 = np.random.normal(loc = means, scale = stds)
        
        #shape is (2, len(ts)-1)
        return np.array([r1,r2])
        
    
    @classmethod
    def GenerateLeaderTrajectories_freedriving(self, ts = np.linspace(0, 300, 300), STATE_init = np.array([.0, .0, .0, .0, .0, .0, .0]), idm_paras  = idm_paras, ):
        """
        Generate the trajectories of the leader that allows the free driving of the following vehicle
        
        Each instance, the state of the vehicle is represented by STATES:
        
            #the system  state. 
            x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
        Note that in the leading trajectory we only consider the longitudinal (x) dimeensnon. 
    
        
        NOTE the coordinate system: 
        
            that the x is the logitudinal axis and y is the lateral. 
        
            The poisitive axis it downstream. 
        
            The left direction is y-positive. 
            
            If the vehicle turn left the steeer angle is positive. 
        
        
        -----------------------------------------------------------------
        
        @input: STATE_init
        
            the initial state. 
        
        @input: ts
        
            the moments. 
        
        @OUTPUT: trajectories
        
            an np.array. 
            
            Shape is (7, moments_N), where 7 is the state number, and moments_N is the number of moments. 
        
            
        
        """
        
        #the system  state. 
        x0,y,phi,v0,delta,Z_long,Z_lat = STATE_init[0],STATE_init[1],STATE_init[2],STATE_init[3],STATE_init[4],STATE_init[5],STATE_init[6]
        
        #the length of accelerations is len(ts-1).
        accelerations = np.array(len(self.random_accelerations(ts = ts))*[idm_paras['idm_vf']])
        
        #IDM paraser. 
        idm_vf = idm_paras['idm_vf']
        idm_T = idm_paras['idm_T']
        idm_delta = idm_paras['idm_delta']
        idm_s0 = idm_paras['idm_s0']
        idm_a = idm_paras['idm_a']
        idm_b = idm_paras['idm_b']
        
        #find the xs
        xs = [x0]
        vs = [v0]
        for deltat,acc0 in zip(np.diff(ts), accelerations):
            STATES_tmp = np.array([xs[-1], 0, phi, vs[-1],  delta,Z_long,Z_lat])
            acc= self.TrimAcce(acc0, STATE = STATES_tmp, idm_paras = idm_paras)
            #
            #print(acc0, acc)
            new_v = min(max(0, vs[-1] + deltat*acc), idm_vf)
            #
            new_x = xs[-1] + deltat*vs[-1]
            
            #
            vs.append(new_v)
            xs.append(new_x)
            
        #
        
        return np.array([xs,[y]*len(ts),[phi]*len(ts),vs,[delta]*len(ts),[Z_long]*len(ts),[Z_lat]*len(ts)])
        
    
    @classmethod
    def GenerateLeaderTrajectories(self, ts = np.linspace(0, 300, 300), STATE_init = np.array([.0, .0, .0, .0, .0, .0, .0]), idm_paras  = idm_paras, amplification_leader = 3.0,  period_sec = 30):
        """
        Generate the trajectories of the leader. 
        
        Each instance, the state of the vehicle is represented by STATES:
        
            #the system  state. 
            x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
        Note that in the leading trajectory we only consider the longitudinal (x) dimeensnon. 
    
        
        NOTE the coordinate system: 
        
            that the x is the logitudinal axis and y is the lateral. 
        
            The poisitive axis it downstream. 
        
            The left direction is y-positive. 
            
            If the vehicle turn left the steeer angle is positive. 
        
        
        -----------------------------------------------------------------
        
        @input: STATE_init
        
            the initial state. 
        
        @input: ts
        
            the moments. 
        
        @OUTPUT: trajectories
        
            an np.array. 
            
            Shape is (7, moments_N), where 7 is the state number, and moments_N is the number of moments. 
        
            
        
        """
        
        #the system  state. 
        x0,y,phi,v0,delta,Z_long,Z_lat = STATE_init[0],STATE_init[1],STATE_init[2],STATE_init[3],STATE_init[4],STATE_init[5],STATE_init[6]
        
        #the length of accelerations is len(ts-1). amplification = 3.0, ts = np.linspace(0, 300, 300), period_sec = 30)
        accelerations = self.random_accelerations(ts = ts, amplification = amplification_leader, period_sec = period_sec)
        
        #IDM paraser. 
        idm_vf = idm_paras['idm_vf']
        idm_T = idm_paras['idm_T']
        idm_delta = idm_paras['idm_delta']
        idm_s0 = idm_paras['idm_s0']
        idm_a = idm_paras['idm_a']
        idm_b = idm_paras['idm_b']
        
        #find the xs
        xs = [x0]
        vs = [v0]
        ACCs = []
        for deltat,acc0 in zip(np.diff(ts), accelerations):
            STATES_tmp = np.array([xs[-1], 0, phi, vs[-1],  delta,Z_long,Z_lat])
            acc= self.TrimAcce(acc0, STATE = STATES_tmp, idm_paras = idm_paras)
            ACCs.append(acc)
            #
            #print(acc0, acc)
            new_v = min(max(0, vs[-1] + deltat*acc), idm_vf)
            #
            new_x = xs[-1] + deltat*vs[-1]
            
            #
            vs.append(new_v)
            xs.append(new_x)
            
        #
        #print('---')
        return ACCs,np.array([xs,[y]*len(ts),[phi]*len(ts),vs,[delta]*len(ts),[Z_long]*len(ts),[Z_lat]*len(ts)])
        
    
    @classmethod
    def EquilibriumHeadway_IDM(self, v,  idm_paras = idm_paras,  veh_paras_self = veh_paras,):
        """
        calculate the equilibrium headway between vehicles in the IDM mode. 
        
            vs = np.linspace(1.0/3.6, 59/3.6, 100)
            gs  = [VM.TwoDimStochasticIDM.EquilibriumHeadway_IDM(v=v) for v in vs]
        
        
        ----------------------------------------------------------
        
        @input: v
        
            unit is m.
            
        @OUTPUT: headway
        
            unit is m. 
        
        """
        
        
        #IDM paraser. 
        idm_vf = idm_paras['idm_vf']
        idm_T = idm_paras['idm_T']
        idm_delta = idm_paras['idm_delta']
        idm_s0 = idm_paras['idm_s0']
        idm_a = idm_paras['idm_a']
        idm_b = idm_paras['idm_b']
        
        if v>idm_vf:
            raise ValueError('speed exceeds the max. ')
        
        
        #unit is m.
        headway_equilibrium = (idm_s0 + v*idm_T)*np.power((1-v/idm_vf), -0.5)
        
        return headway_equilibrium


    @classmethod
    def plot_vehiclestates_xy(self, ts, vehstates_arrays, ax = False, figsize = (5,3), color = 'b', alpha = .4):
        """
        
        @input: vehstates_arrays
        
            a list. 
        
            vehstates_array = vehstates_arrays[0]
        
            a 2d array. (7, N).
            
            7 is the number of vehicle states and N is the length of the moments. 
            
            
            
            
        
        """
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize,)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        for vehstates_array in vehstates_arrays:
        
            
            #the xy
            #ax = axs[0, 0]
            ax.plot(vehstates_array[0, :], vehstates_array[1, :], color = color, alpha = alpha)
            
        
        ax.set_xlabel('x ( m )');ax.set_ylabel('y ( m )'); ax.grid();#ax.set_title('(a)')

        
        ax.grid()
        
        plt.tight_layout()
        return ax

    @classmethod
    def plot_vehiclestates_multi_withxtyt(self, ts, vehstates_arrays, axs = False, figsize = (5,3), alpha = .4,):
        """
        
        @input: vehstates_arrays
        
            a list. 
        
            vehstates_array = vehstates_arrays[0]
        
            a 2d array. (7, N).
            
            7 is the number of vehicle states and N is the length of the moments. 
            
            
            
            
        
        """
        if isinstance(axs, bool):
            fig,axs = plt.subplots(figsize = figsize, nrows = 3, ncols = 2)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        for vehstates_array in vehstates_arrays:
        
            
            #the xy
            ax = axs[0, 0]
            ax.plot(vehstates_array[0, :], vehstates_array[1, :])
            ax.set_xlabel('x ( m )');ax.set_ylabel('y ( m )'); ax.grid();ax.set_title('(a)')
            #ax.set_ylim([-1.8, 1.8])
            
            #
            ax = axs[0, 1]
            ax.plot(ts, vehstates_array[2, :])
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('heading angle (rad) '); ax.grid();ax.set_title('(b)')
            
            #
            ax = axs[1, 0]
            ax.plot(ts, vehstates_array[3, :])
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('speed (m/s)'); ax.grid();ax.set_title('(c)')
            
            #
            ax = axs[1, 1]
            ax.plot(ts, vehstates_array[4, :])
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('Steer (rad)'); ax.grid();ax.set_title('(d)')
            
            #the xy
            ax = axs[2, 0]
            ax.plot(ts, vehstates_array[0, :])
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('x ( m )'); ax.grid();ax.set_title('(e)')
            #the xy
            ax = axs[2, 1]
            ax.plot(ts, vehstates_array[1, :])
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('y ( m )'); ax.grid();ax.set_title('(e)')
        
        axs[0, 0].grid();axs[0, 1].grid();axs[1, 0].grid();axs[1, 1].grid();axs[2,0].grid();axs[2, 1].grid();
        
        
        plt.tight_layout()
        return axs

    @classmethod
    def plot_vehiclestates_multi_im(self, ts, vehstates_arrays, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        @input: vehstates_arrays
        
            a list. 
        
            vehstates_array = vehstates_arrays[0]
        
            a 2d array. (7, N).
            
            7 is the number of vehicle states and N is the length of the moments. 
            
            
            
            
        
        """
        if isinstance(ax, bool):
            fig,axs = plt.subplots(figsize = figsize, nrows = 2, ncols = 2)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        for vehstates_array in vehstates_arrays:
        
            
            #the xy
            ax = axs[0, 0]
            ax.plot(ts, vehstates_array[0, :])
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('x (m) '); ax.grid();ax.set_title('(a)')
            #ax.set_ylim([-1.8, 1.8])
            
            #
            ax = axs[0, 1]
            ax.plot(ts, vehstates_array[1, :])
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('y (m) '); ax.grid();ax.set_title('(b)')
            
            #
            ax = axs[1, 0]
            ax.plot(ts, vehstates_array[3, :])
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('speed (m/s)'); ax.grid();ax.set_title('(c)')
            
            #
            ax = axs[1, 1]
            ax.plot(ts, vehstates_array[4, :])
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('Steer (rad)'); ax.grid();ax.set_title('(d)')
        
        axs[0, 0].grid();axs[0, 1].grid();axs[1, 0].grid();axs[1, 1].grid();
        
        
        plt.tight_layout()
        return axs

    @classmethod
    def plot_vehiclestates_justxy(self,vehstates_arrays, ax = False, figsize = (5,3), alpha = .4, xlabel = 'x ( m )', ylabel = 'y ( m )', title = '(a)'):
        """
        
        @input: vehstates_arrays
        
            a list. 
        
            vehstates_array = vehstates_arrays[0]
        
            a 2d array. (7, N).
            
            7 is the number of vehicle states and N is the length of the moments. 
            
            
            
            
        
        """
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        for vehstates_array in vehstates_arrays:
            #
            ax.plot(vehstates_array[0, :], vehstates_array[1, :])
        #
        ax.set_xlabel(xlabel);ax.set_ylabel(ylabel); ax.grid();ax.set_title(title)
        plt.tight_layout()
        
        return ax




    @classmethod
    def plot_vehiclestates_single_index(self, ts, vehstates_arrays, ax = False, figsize = (5,3), alpha = .4, index_plotted = 1, xlabel = 'x ( m )', ylabel = 'x ( m )', title = 'a'):
        """
        
        @input: vehstates_arrays
        
            a list. 
        
            vehstates_array = vehstates_arrays[0]
        
            a 2d array. (7, N).
            
            7 is the number of vehicle states and N is the length of the moments. 
            
            
            
            
        
        """
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        for vehstates_array in vehstates_arrays:
            #
            ax.plot(ts, vehstates_array[index_plotted, :])
        #
        ax.set_xlabel(xlabel);ax.set_ylabel(ylabel); ax.grid();ax.set_title(title)
        plt.tight_layout()
        
        return ax



    @classmethod
    def plot_vehiclestates_multi(self, ts, vehstates_arrays, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        @input: vehstates_arrays
        
            a list. 
        
            vehstates_array = vehstates_arrays[0]
        
            a 2d array. (7, N).
            
            7 is the number of vehicle states and N is the length of the moments. 
            
            
            
            
        
        """
        if isinstance(ax, bool):
            fig,axs = plt.subplots(figsize = figsize, nrows = 2, ncols = 2)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        for vehstates_array in vehstates_arrays:
        
            
            #the xy
            ax = axs[0, 0]
            ax.plot(vehstates_array[0, :], vehstates_array[1, :])
            ax.set_xlabel('x ( m )');ax.set_ylabel('y ( m )'); ax.grid();ax.set_title('(a)')
            #ax.set_ylim([-1.8, 1.8])
            
            #
            ax = axs[0, 1]
            ax.plot(ts, vehstates_array[2, :])
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('Heading angle (rad) '); ax.grid();ax.set_title('(b)')
            
            #
            ax = axs[1, 0]
            ax.plot(ts, vehstates_array[3, :])
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('Speed (m/s)'); ax.grid();ax.set_title('(c)')
            
            #
            ax = axs[1, 1]
            ax.plot(ts, vehstates_array[4, :])
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('Steer (rad)'); ax.grid();ax.set_title('(d)')
        
        axs[0, 0].grid();axs[0, 1].grid();axs[1, 0].grid();axs[1, 1].grid();
        
        
        plt.tight_layout()
        return axs



    @classmethod
    def plot_vehiclestates_xy_only(self, ts, vehstates_arrays, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        @input: vehstates_arrays
        
            a list. 
        
            vehstates_array = vehstates_arrays[0]
        
            a 2d array. (7, N).
            
            7 is the number of vehicle states and N is the length of the moments. 
            
            
            
            
        
        """
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        for vehstates_array in vehstates_arrays:
        
            
            #the xy
            ax.plot(vehstates_array[0, :], vehstates_array[1, :])
            ax.set_xlabel('x ( m )');ax.set_ylabel('y ( m )'); ax.grid();ax.set_title('(a)')
            #ax.set_ylim([-1.8, 1.8])
            
        
        ax.grid();
        
        
        plt.tight_layout()
        return ax



    @classmethod
    def plot_probabilityevolution_idx(self, ts, vehstates_arrays, figsize = (8,4), ax = False, t_MAX = 50, n_moments_plotted = 4, bins= 20, normalize = False, idx_plotted = 1, x_label = 'y (m)'):
        """
        
        @input: vehstates_arrays
        
            a list. 
        
            vehstates_array = vehstates_arrays[0]
        
            a 2d array. (7, N).
            
            7 is the number of vehicle states and N is the length of the moments. 
        
        @input: t_MAX  and n_moments_plotted
        
            t_MAX is the para that 
            
        """
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #ys shape is (samplepath_N, momnetsN)
        ys = pd.DataFrame([vehstates_array[idx_plotted, :] for vehstates_array in vehstates_arrays])
        #print(ys.shape)
        #
        selected_ts0 = ts[ts<t_MAX]
        #
        selected_idxs = range(0, len(selected_ts0), int(len(selected_ts0)/n_moments_plotted))
        #selected_ts = [selected_ts0[i] for i in selected_idxs]
        
        #print(selected_ts)
        #
        for idx in selected_idxs:
            if idx==0:continue
            #print(idx)
            hs,es0 =np.histogram(ys.iloc[:, idx], bins = bins)
            #
            es = es0[1:]
            if normalize:
                ax.plot(es,hs/sum(hs)/(es[-1]-es[-2]), label = str(int(selected_ts0[idx]*100)/100.0) + ' sec')
            else:
                ax.plot(es,hs, label = str(int(selected_ts0[idx]*100)/100.0) + ' sec')
        
        #
        plt.tight_layout()
        ax.legend()
        ax.set_xlabel(x_label);ax.set_ylabel('Frequencies'); ax.grid()
        return ax

    
    @classmethod
    def plot_variance_idx(self, ts, vehstates_arrays, figsize = (8,4), ax = False, idx_state = 1):
        """
        
        @input: vehstates_arrays
        
            a list. 
        
            vehstates_array = vehstates_arrays[0]
        
            a 2d array. (7, N).
            
            7 is the number of vehicle states and N is the length of the moments. 
        
        @input: t_MAX  and n_moments_plotted
        
            t_MAX is the para that 
            
        """
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #ys shape is (samplepath_N, momnetsN)
        ys = pd.DataFrame([vehstates_array[idx_state, :] for vehstates_array in vehstates_arrays])
        ax.plot(ts, np.std(ys, axis = 0))

        #
        plt.tight_layout()
        #ax.legend()
        ax.set_xlabel('Time ( sec )');ax.set_ylabel('std'); ax.grid()
        return ax

        
        pass
    
    
    
    
    @classmethod
    def plot_probabilityevolution_y(self, ts, vehstates_arrays, figsize = (8,4), ax = False, t_MAX = 50, n_moments_plotted = 4, bins= 20, normalize = False):
        """
        
        @input: vehstates_arrays
        
            a list. 
        
            vehstates_array = vehstates_arrays[0]
        
            a 2d array. (7, N).
            
            7 is the number of vehicle states and N is the length of the moments. 
        
        @input: t_MAX  and n_moments_plotted
        
            t_MAX is the para that 
            
        """
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #ys shape is (samplepath_N, momnetsN)
        ys = pd.DataFrame([vehstates_array[1, :] for vehstates_array in vehstates_arrays])
        #print(ys.shape)
        #
        selected_ts0 = ts[ts<t_MAX]
        #
        selected_idxs = range(0, len(selected_ts0), int(len(selected_ts0)/n_moments_plotted))
        #selected_ts = [selected_ts0[i] for i in selected_idxs]
        
        #print(selected_ts)
        #
        for idx in selected_idxs:
            if idx==0:continue
            #print(idx)
            hs,es0 =np.histogram(ys.iloc[:, idx], bins = bins)
            #
            es = es0[1:]
            if normalize:
                ax.plot(es,hs/sum(hs)/(es[-1]-es[-2]), label = str(int(selected_ts0[idx]*100)/100.0) + ' sec')
            else:
                ax.plot(es,hs, label = str(int(selected_ts0[idx]*100)/100.0) + ' sec')
        
        #
        plt.tight_layout()
        ax.legend()
        ax.set_xlabel('y ( m )');ax.set_ylabel('Frequencies'); ax.grid()
        return ax

        
        pass



    @classmethod
    def plot_vehiclestates(self, ts, vehstates_array, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        @input: vehstates_array
        
            a 2d array. (7, N).
            
            7 is the number of vehicle states and N is the length of the moments. 
            
            
            
            
        
        """
        if isinstance(ax, bool):
            fig,axs = plt.subplots(figsize = figsize, nrows = 2, ncols = 2)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #the xy
        ax = axs[0, 0]
        ax.plot(vehstates_array[0, :], vehstates_array[1, :])
        ax.set_title('(a)')
        ax.set_xlabel('x ( m )');ax.set_ylabel('y ( m )'); ax.grid()
        #ax.set_ylim([-1.8, 1.8])
        
        #
        ax = axs[0, 1]
        ax.plot(ts, vehstates_array[2, :])
        ax.set_xlabel('Time ( sec )');ax.set_ylabel('heading angle (rad) '); ax.grid();ax.set_title('(b)')
        
        #
        ax = axs[1, 0]
        ax.plot(ts, vehstates_array[3, :])
        ax.set_xlabel('Time ( sec )');ax.set_ylabel('speed (m/s)'); ax.grid();ax.set_title('(c)')
        
        #
        ax = axs[1, 1]
        ax.plot(ts, vehstates_array[4, :])
        ax.set_xlabel('Time ( sec )');ax.set_ylabel('Steer angle (rad)'); ax.grid();ax.set_title('(d)')
        
        plt.tight_layout()
        return axs
    
    
    @classmethod
    def plot_ts_idx_state(self, ts, paths, idx_state = 1, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        @input: paths
            
            paths is a list. Each element is for one simulation
            
            sim_res = paths[idx]
            
            sim_res is a  array with shape (7, N), where 7 is the system state and N is the moments number. N = len(ts)
        
        
        """
        
        
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)

        for sim_res in paths:
            ax.plot(ts, sim_res[idx_state,:])
        
        if isinstance(ax, bool):
            ax.set_xlabel('Time (sec)');ax.set_xlabel('y'); 
            ax.grid();
            
            plt.tight_layout()
        
        return ax
    
    @classmethod
    def plot_platoon_3d_speed(self, ts, platoondata_list, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        
        @input: platoondata_list
        
        
            first = platoondata_list[0]
            second = platoondata_list[1]
            ...
            
            first is a list. the number is the paths. 
            
            first_path0 = first[0].
            
            first_path0 is a 2d array with shape (7,N), where N isthe moments number. 
        
        
        """
        
        if isinstance(ax, bool):
            ax = plt.figure().add_subplot(projection='3d')
            #fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #the xy
        for i,veh_in_platoon in enumerate(platoondata_list):
            vehstates_array = veh_in_platoon[0]
            ax.plot(ts, vehstates_array[0, :], label = str(i)+'-th')
            ax.set_title('(a)')
            ax.set_xlabel('time ( sec )');ax.set_ylabel('y ( m )'); ax.grid()
            #ax.set_ylim([-1.8, 1.8])
        ax.legend()
        
        plt.tight_layout()
        return ax
    



    @classmethod
    def plot_platoon_x(self, ts, platoondata_list, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        
        @input: platoondata_list
        
        
            first = platoondata_list[0]
            second = platoondata_list[1]
            ...
            
            first is a list. the number is the paths. 
            
            first_path0 = first[0].
            
            first_path0 is a 2d array with shape (7,N), where N isthe moments number. 
        
        
        """
        
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #the xy
        for i,veh_in_platoon in enumerate(platoondata_list):
            vehstates_array = veh_in_platoon[0]
            ax.plot(ts, vehstates_array[0, :], label = str(i)+'-th')
            ax.set_title('(a)')
            ax.set_xlabel('time ( sec )');ax.set_ylabel('y ( m )'); ax.grid()
            #ax.set_ylim([-1.8, 1.8])
        ax.legend()
        
        plt.tight_layout()
        return ax
    



    @classmethod
    def plot_platoon(self, ts, platoondata_list, axs = False, figsize = (5,3), alpha = .4,):
        """
        
        
        @input: platoondata_list
        
        
            first = platoondata_list[0]
            second = platoondata_list[1]
            ...
            
            first is a list. the number is the paths. 
            
            first_path0 = first[0].
            
            first_path0 is a 2d array with shape (7,N), where N isthe moments number. 
        
        
        """
        
        if isinstance(axs, bool):
            fig,axs = plt.subplots(figsize = figsize, nrows = 2, ncols = 2)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #the xy
        ax = axs[0,0]
        for i,veh_in_platoon in enumerate(platoondata_list):
            vehstates_array = veh_in_platoon[0]
            ax.plot(vehstates_array[0, :], vehstates_array[1, :], label = str(i)+'-th')
            ax.set_title('(a)')
            ax.set_xlabel('x ( m )');ax.set_ylabel('y ( m )'); ax.grid()
            #ax.set_ylim([-1.8, 1.8])
        ax.legend()
        
        #
        ax = axs[0, 1]
        for i,veh_in_platoon in enumerate(platoondata_list):
            vehstates_array = veh_in_platoon[0]
            ax.plot(ts, vehstates_array[2, :], label = str(i)+'-th')
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('heading angle (rad) '); ax.grid();ax.set_title('(b)')
        ax.legend()
        
        #
        ax = axs[1, 0]
        for i,veh_in_platoon in enumerate(platoondata_list):
            vehstates_array = veh_in_platoon[0]
            ax.plot(ts, vehstates_array[3, :], label = str(i)+'-th')
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('speed (m/s)'); ax.grid();ax.set_title('(c)')
        ax.legend()
        
        #
        ax = axs[1, 1]
        for i,veh_in_platoon in enumerate(platoondata_list):
            vehstates_array = veh_in_platoon[0]
            ax.plot(ts, vehstates_array[4, :], label = str(i)+'-th')
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('Steer rate (rad/s)'); ax.grid();ax.set_title('(d)')
        ax.legend()
        
        plt.tight_layout()
        return axs

    
    @classmethod
    def plot_paths(self, paths, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        @input: paths
            
            paths is a list. Each element is for one simulation
            
            sim_res = paths[idx]
            
            sim_res is a  array with shape (7, N), where 7 is the system state and N is the moments number. N = len(ts)
        
        
        """
        
        
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('x');ax.set_ylabel('y'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)

        for sim_res in paths:
            ax = self.plot_path(path = sim_res, ax = ax, figsize = figsize, alpha = alpha,)
        
        #
        ax.set_xlabel('x ( m )');ax.set_ylabel('y ( m )'); ax.grid()
        
            
        plt.tight_layout()
        
        return ax

    @classmethod
    def plot_path_with_vehiclebound(self, path, ax = False, N_vehs_plotted = 30, figsize = (5,3), alpha = .4, veh_paras = veh_paras, facecolor= [0,0.5,0],):
        """
        
        @input: path
            
            path is a  array with shape (7, N), where 7 is the system state and N is the moments number. N = len(ts)
        
        
        """
        import matplotlib
        
        
        if isinstance(ax, bool):
            fig,ax = plt.subplots()
            #fig,ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('x');ax.set_xlabel('y'); 
            ax.grid();
            
            plt.tight_layout()
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)

        xs = path[0, :]
        ys = path[1, :]
        ax.plot(xs, ys)
        
        ######################plot vehi bounds
        lr = veh_paras.get('lr', 2)
        lf = veh_paras.get('lf', 3)
        veh_width = veh_paras.get('width', 2)
        veh_len = veh_paras.get('length', 5)
        
        #-----------------------determine the idx of the ploted samples, in plot_columns_idxs
        plot_columns_idxs = list(range(0, path.shape[1],  int(path.shape[1]/N_vehs_plotted)))
        #
        patches = []
        for idx in plot_columns_idxs:
            #
            x = path[0, idx]
            y = path[1, idx]
            heading = path[2, idx]
            
            #
            rear_right_x,rear_right_y = VehicleKineticSolver.Veicle_Rear_Right(xy_CG = (x,y),w = veh_width, lr =lr, ang= heading)
            
            #print(rear_right_x,rear_right_y)
            #
            rect = matplotlib.patches.Rectangle((rear_right_x,rear_right_y), veh_len, veh_width, heading*180.0/np.pi, alpha = alpha, facecolor=facecolor, ec = 'k')#fill=None, facecolor='none'
            
            ax.add_patch(rect)

        
        return ax
        
    @classmethod
    def plot_state_confidenceinterval(self, ts, vehstates_arrays, figsize = (5,2.5), ax = False, normalize = False, idx_plotted = 3, x_label = 'time (sec)', y_label= 'speed (m/s)', quantiles = [.15, .25], alpha = .3):
        """
        
        @input: vehstates_arrays
        
            a list. 
        
            vehstates_array = vehstates_arrays[0]
        
            a 2d array. (7, N).
            
            7 is the number of vehicle states and N is the length of the moments. 
        
        @input: t_MAX  and n_moments_plotted
        
            t_MAX is the para that 
            
        """
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #ys shape is (samplepath_N, momnetsN)
        ys = pd.DataFrame([vehstates_array[idx_plotted, :] for vehstates_array in vehstates_arrays])
        #print(ys.shape)
        
        #ymean is 1d with shale (momnetsN)
        ymean = np.mean(ys, axis = 0)
        ax.plot(ts, ymean, label = 'mean')
        
        #
        for quantile in quantiles:
            #
            #print(ys.iloc[:,0].shape)
            
            datas1 = [np.quantile(ys.iloc[:,idx_t],.5+quantile) for idx_t in range(len(ts))]
            datas2 = [np.quantile(ys.iloc[:,idx_t], .5-quantile) for idx_t in range(len(ts))]
            
            #
            ax.fill_between(ts, datas1,datas2, alpha = alpha, label = str(quantile))
        
        ax.legend()
        
        ax.set_xlabel(x_label);ax.set_ylabel(y_label); ax.grid()
        return ax
        

    @classmethod
    def plot_path(self, path, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        @input: path
            
            path is a  array with shape (7, N), where 7 is the system state and N is the moments number. N = len(ts)
        
        
        """
        
        
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)

        xs = path[0, :]
        ys = path[1, :]
        
        
        ax.plot(xs, ys)
        
        if isinstance(ax, bool):
            ax.set_xlabel('x');ax.set_xlabel('y'); 
            ax.grid();
            
            plt.tight_layout()
        
        return ax
        
        
        pass
    
    
    @classmethod
    def sim_lc(self, ts, STATES_leader,  STATE_init =  np.array([-30.0, 3.5, .0, .0, .0, .0, .0]), veh_paras_self = veh_paras, veh_paras_leader = veh_paras, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = 1.0, eta_lat = 1.0, sigma_long = .01, sigma_lat = .01, N_paths = 50,  weight_heading = 1.0, weight_line_CG = 0.0):
        """
        
        Simulate the follower trajectories. 
        
        NOTE: 
        
            that the x is the logitudinal axis and y is the lateral. 
        
            The poisitive axis it downstream. 
        
            The left direction is y-positive. 
        
        
        ---------------------------------------
        @input: STATES_leader
        
            an array, which illustrate the movement of the leader 
            
            STATES_leader.shape is (7,N). N ==len(ts)
            
        @input: STATE_init
        
            the initial state of ego vehice. 
            
            
                #the system  state. 
                x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
        --------------------------------------
        @OUTPUT: STATES
        
            an arry. The shape is the same as STATES_leader
            
        
        
        """

        #---------------------------------------
        if len(ts)!=STATES_leader.shape[1]:
            raise ValueError('The input moments N is not equal to the states N')

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape is (2, len(deltats))
            brownianspath =  self.GenerateBrownianPaths(ts = ts)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                x,y,phi,v,delta,Z_long,Z_lat = STATES[-1][0],STATES[-1][1],STATES[-1][2],STATES[-1][3],STATES[-1][4],STATES[-1][5],STATES[-1][6]
                #
                equilibrium = self.EquilibriumHeadway_IDM(v= 16)
                STATE_leader = np.array([STATES[-1][0]+equilibrium, 0, .0, idm_paras['idm_vf'], .0, .0, .0])
                #STATE_leader = STATES_leader[:,idx+1]
                #shape is (2,)
                brownian = brownianspath[:,idx]
                #
                #F is an 1d array with the shape of 7. 
                F = self.F(STATES = STATES[-1], STATES_leader = STATE_leader, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat,  weight_heading = weight_heading, weight_line_CG = weight_line_CG)
                #
                L = self.L(STATES = STATES[-1], eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat)
                #np.multiply(L, brownian)
                new_state = STATES[-1] + F*deltat + np.matmul(L, brownian)
                new_state[3] = max(0,new_state[3]  )
                
                STATES.append(new_state)
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters
        
        pass

    @classmethod
    def OU_with_transform_simulation(self, ts = np.linspace(0, 100, 1000), STATE_init = np.array([.0, .0]), theta = 1.0, sigma = .3, N_paths = 50):
        """
        
        STATES_iters = VM.TwoDimStochasticIDM.OU_simulation()
        
        """

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape i
            
            stds = np.sqrt(np.diff(ts))
            means = np.zeros(stds.shape)
            #brownians shape is (, len(deltats))
            brownianspath = np.random.normal(loc = means, scale = stds)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                brownian = brownianspath[idx]
                new_state0 = STATES[-1][1] - theta*STATES[-1][1]*deltat + sigma*brownian
                
                new_state1 = STATES[-1][0] +  np.tanh(STATES[-1][0])*deltat
                
                STATES.append(np.array([new_state1, new_state0]))
            #
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters

    @classmethod
    def simulation_jacobi(self, ts = np.linspace(0, 100, 1000), STATE_init = .0, theta = 1.0, sigma = .3, N_paths = 50):
        """
        
        STATES_iters = VM.TwoDimStochasticIDM.OU_simulation()
        
        """

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape i
            
            stds = np.sqrt(np.diff(ts))
            means = np.zeros(stds.shape)
            #brownians shape is (, len(deltats))
            brownianspath = np.random.normal(loc = means, scale = stds)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                brownian = brownianspath[idx]
                #
                if STATES[-1]<-0.5 or STATES[-1]>.5:
                    tmp0 = min(.499999, max(-0.499999, STATES[-1]))
                    #tmp = min(.499999, max(-0.499999, sigma*(0.5+STATES[-1])*(0.5-STATES[-1])))
                    #print(STATES[-1], )
                    tmp = sigma*(0.5 + tmp0)*(0.5- tmp0)
                else:
                    tmp = sigma*(0.5+STATES[-1])*(0.5-STATES[-1])
                #print(STATES[-1], tmp)
                new_state = STATES[-1] - theta*(STATES[-1])*deltat + np.sqrt(tmp)*brownian
                
                STATES.append(new_state)
            #
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters,brownianspath
        


    @classmethod
    def geometric_simulation(self, ts = np.linspace(0, 100, 1000), STATE_init = 1.0, theta = 1.0, sigma = .3, N_paths = 50):
        """
        
        STATES_iters = VM.TwoDimStochasticIDM.OU_simulation()
        
        """

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape i
            
            stds = np.sqrt(np.diff(ts))
            means = np.zeros(stds.shape)
            #brownians shape is (, len(deltats))
            brownianspath = np.random.normal(loc = means, scale = stds)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                brownian = brownianspath[idx]
                
                new_state = STATES[-1] + theta*STATES[-1]*deltat + sigma*STATES[-1]*brownian
                
                STATES.append(new_state)
            #
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters,brownianspath
        


    @classmethod
    def simulation_hyperparabolic(self, ts = np.linspace(0, 100, 1000), STATE_init = 2.0, theta = 1.0, sigma = .3, N_paths = 50, alpha = .5, a = 2):
        """
        
        STATES_iters = VM.TwoDimStochasticIDM.OU_simulation()
        
        """

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape i
            
            stds = np.sqrt(np.diff(ts))
            means = np.zeros(stds.shape)
            #brownians shape is (, len(deltats))
            brownianspath = np.random.normal(loc = means, scale = stds)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                brownian = brownianspath[idx]
                
                new_state = STATES[-1] + -(STATES[-1] +  theta/(STATES[-1]))*deltat + sigma*brownian
                
                STATES.append(new_state)
            #
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters
        


    @classmethod
    def simulation_ROU(self, ts = np.linspace(0, 100, 1000), STATE_init = 0, theta = 1.0, sigma = .3, N_paths = 50, alpha = .5):
        """
        
        STATES_iters = VM.TwoDimStochasticIDM.OU_simulation()
        
        """

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape i
            
            stds = np.sqrt(np.diff(ts))
            means = np.zeros(stds.shape)
            #brownians shape is (, len(deltats))
            brownianspath = np.random.normal(loc = means, scale = stds)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                brownian = brownianspath[idx]
                
                new_state = STATES[-1] + (theta/STATES[-1] -  STATES[-1] )*deltat + sigma*brownian
                
                STATES.append(new_state)
            #
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters
        
    


    @classmethod
    def simulation_OU(self, ts = np.linspace(0, 100, 1000), STATE_init = 0, theta = 1.0, sigma = .3, N_paths = 50):
        """
        
        STATES_iters = VM.TwoDimStochasticIDM.OU_simulation()
        
        """

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape i
            
            stds = np.sqrt(np.diff(ts))
            means = np.zeros(stds.shape)
            #brownians shape is (, len(deltats))
            brownianspath = np.random.normal(loc = means, scale = stds)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                brownian = brownianspath[idx]
                
                new_state = STATES[-1] - theta*STATES[-1]*deltat + sigma*brownian
                
                STATES.append(new_state)
            #
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters
        
    
    @classmethod
    def sim_given_brownian(self, ts, STATES_leader,  \
        STATE_init =  np.array([-30.0, 3.5, .0, .0, .0, .0, .0]), \
        states_bridge_paths = [], \
        veh_paras_self = veh_paras, veh_paras_leader = veh_paras, \
        idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, \
        eta_long = 1.0, eta_lat = 1.0, sigma_long = .01, \
        sigma_lat = .01, N_paths = 50,  \
        weight_heading = 1.0, weight_line_CG = 0.0,  \
        sigma_long_drift = 10.0, sigma_lat_drift = 10.0,  \
        stochastic_proecess_name ='geometric'):
        """
        
        Simulate the follower trajectories given the browinan paths. 
        
        NOTE: 
        
            that the x is the logitudinal axis and y is the lateral. 
        
            The poisitive axis it downstream. 
        
            The left direction is y-positive. 
        
        
        ---------------------------------------
        @input: states_bridge_paths
        
            a list containing the pseudo states lists. 
            
            path  = states_bridge_paths[idx]
            
            path shape is (7, len(ts))
            
            It is generated via:
                
                ---------------------------------------------------------
                reload(BG)
                noise_bridge_paths = []
                mus = []
                sigmas = []
                for i in range(100):
                    #
                    path,mu,sigma = BG.BridgeTwoDimIDM.ModifiedDiffusionBridge(ts = ts, initialcondition = initialcondition, \
                                terminalcondition = terminalcondition, eta_long = .1, eta_lat = .1 ,sigma_long_drift = .1, sigma_lat_drift = .1, \
                              sigma_long = .1, sigma_lat = .1, weight_heading = .001, weight_line_CG = .2,stochastic_proecess_name = 'converted', )
                    noise_bridge_paths.append(path)
                ---------------------------------------------------------
            
        @input: STATES_leader
        
            an array, which illustrate the movement of the leader 
            
            STATES_leader.shape is (7,N). N ==len(ts)
            
        @input: STATE_init
        
            the initial state of ego vehice. 
            
            
                #the system  state. 
                x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
        --------------------------------------
        @OUTPUT: STATES
        
            an arry. The shape is the same as STATES_leader
            
        
        
        """

        #---------------------------------------
        if len(ts)!=STATES_leader.shape[1]:
            raise ValueError('The input moments N is not equal to the states N')

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(len(states_bridge_paths)):
            #
            #brownians shape is (2, len(deltats))
            #brownianspath =  states_bridge_paths[iterr][-2:, :]
            brownianspath =  self.GenerateBrownianPaths(ts = ts)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                STATE_leader = STATES_leader[:,idx+1]
                #shape is (2,)
                brownian = brownianspath[:,idx]
                #
                #F is an 1d array with the shape of 7. 
                F = self.F(STATES = STATES[-1], STATES_leader = STATE_leader, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat,  weight_heading = weight_heading, weight_line_CG = weight_line_CG,  sigma_long_drift = sigma_long_drift, sigma_lat_drift = sigma_lat_drift, stochastic_proecess_name  = stochastic_proecess_name)
                #
                L = self.L(STATES = STATES[-1], eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat, stochastic_proecess_name  = stochastic_proecess_name)
                #np.multiply(L, brownian)
                new_state = STATES[-1] + F*deltat + np.matmul(L, brownian)
                new_state[3] = max(0,new_state[3]  )
                
                #
                new_state[-2:] = copy.deepcopy(states_bridge_paths[iterr][-2:, idx])
                
                STATES.append(new_state)
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters
        
        pass

    @classmethod
    def sim_with_terminalcondition_multiplicative(self, ts, STATES_leader,   terminalcondition, terminalmoment, STATE_init =  np.array([-30.0, 3.5, .0, .0, .0, .0, .0]), veh_paras_self = veh_paras, veh_paras_leader = veh_paras, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = 1.0, eta_lat = 1.0, sigma_long = .01, sigma_lat = .01, N_paths = 50,  weight_heading = 1.0, weight_line_CG = 0.0,  sigma_long_drift = 10.0, sigma_lat_drift = 10.0,  stochastic_proecess_name ='geometric'):
        """
        This method is used to generate the bridge of the system dynamics. 
        
        Difference between:
            
            - self.sim, dS = F(S)dt + L(S)dW
            - self.sim_with_terminal_condition, with two extra args: terminalcondition, terminalmoment
                the former one is a 1d array with length 7, the latter is a float. 
                
                dS = F(S)dt + (terminalcondition-S)/(terminalmoment-t) dt  + L(S)dW
        
        
        Simulate the follower trajectories. 
        
        NOTE: 
        
            that the x is the logitudinal axis and y is the lateral. 
        
            The poisitive axis it downstream. 
        
            The left direction is y-positive. 
        
        
        ---------------------------------------
        @input: STATES_leader
        
            an array, which illustrate the movement of the leader 
            
            STATES_leader.shape is (7,N). N ==len(ts)
            
        @input: STATE_init
        
            the initial state of ego vehice. 
            
            
                #the system  state. 
                x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
        --------------------------------------
        @OUTPUT: STATES
        
            an arry. The shape is the same as STATES_leader
            
        
        
        """

        #---------------------------------------
        if len(ts)!=STATES_leader.shape[1]:
            raise ValueError('The input moments N is not equal to the states N')

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape is (2, len(deltats))
            brownianspath =  self.GenerateBrownianPaths(ts = ts)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                STATE_leader = STATES_leader[:,idx+1]
                #shape is (2,)
                brownian = brownianspath[:,idx]
                #
                #F is an 1d array with the shape of 7. 
                F = self.F(STATES = STATES[-1], STATES_leader = STATE_leader, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat,  weight_heading = weight_heading, weight_line_CG = weight_line_CG,  sigma_long_drift = sigma_long_drift, sigma_lat_drift = sigma_lat_drift, stochastic_proecess_name  = stochastic_proecess_name)
                #
                L = self.L(STATES = STATES[-1], eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat, stochastic_proecess_name  = stochastic_proecess_name)
                #np.multiply(L, brownian)
                
                
                #np.matmul(np.matmul(L, L.T), (terminalcondition - STATES[-1])/(((terminalmoment - ts[idx]))**1.0)*deltat)
                #new_state = STATES[-1] + F*deltat  + np.matmul(np.matmul(L, L.T), (terminalcondition - STATES[-1])/(((terminalmoment - ts[idx]))**1.0)*deltat) + np.matmul(L, brownian)
                new_state = STATES[-1] + F*deltat + (terminalcondition - STATES[-1])/(((terminalmoment - ts[idx]))**1.0)*deltat + np.matmul(L, brownian)
                #if terminalmoment > ts[idx+1]:
                #    new_state = STATES[-1] + F*deltat + (terminalcondition-STATES[-1])/(terminalmoment - ts[idx])*deltat + np.matmul(L, brownian)
                #elif terminalmoment == ts[idx+1]:
                #    new_state = STATES[-1] + F*deltat + np.matmul(L, brownian)
                
                new_state[3] = max(0,new_state[3]  )
                
                STATES.append(new_state)
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters
        
        pass
    
    
    @classmethod
    def sim_with_terminalcondition(self, ts, STATES_leader,   terminalcondition, terminalmoment, STATE_init =  np.array([-30.0, 3.5, .0, .0, .0, .0, .0]), veh_paras_self = veh_paras, veh_paras_leader = veh_paras, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = 1.0, eta_lat = 1.0, sigma_long = .01, sigma_lat = .01, N_paths = 50,  weight_heading = 1.0, weight_line_CG = 0.0,  sigma_long_drift = 10.0, sigma_lat_drift = 10.0,  stochastic_proecess_name ='geometric'):
        """
        This method is used to generate the bridge of the system dynamics. 
        
        Difference between:
            
            - self.sim, dS = F(S)dt + L(S)dW
            - self.sim_with_terminal_condition, with two extra args: terminalcondition, terminalmoment
                the former one is a 1d array with length 7, the latter is a float. 
                
                dS = F(S)dt + (terminalcondition-S)/(terminalmoment-t) dt  + L(S)dW
        
        
        Simulate the follower trajectories. 
        
        NOTE: 
        
            that the x is the logitudinal axis and y is the lateral. 
        
            The poisitive axis it downstream. 
        
            The left direction is y-positive. 
        
        
        ---------------------------------------
        @input: STATES_leader
        
            an array, which illustrate the movement of the leader 
            
            STATES_leader.shape is (7,N). N ==len(ts)
            
        @input: STATE_init
        
            the initial state of ego vehice. 
            
            
                #the system  state. 
                x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
        --------------------------------------
        @OUTPUT: STATES
        
            an arry. The shape is the same as STATES_leader
            
        
        
        """

        #---------------------------------------
        if len(ts)!=STATES_leader.shape[1]:
            raise ValueError('The input moments N is not equal to the states N')

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape is (2, len(deltats))
            brownianspath =  self.GenerateBrownianPaths(ts = ts)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                STATE_leader = STATES_leader[:,idx+1]
                #shape is (2,)
                brownian = brownianspath[:,idx]
                #
                #F is an 1d array with the shape of 7. 
                F = self.F(STATES = STATES[-1], STATES_leader = STATE_leader, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat,  weight_heading = weight_heading, weight_line_CG = weight_line_CG,  sigma_long_drift = sigma_long_drift, sigma_lat_drift = sigma_lat_drift, stochastic_proecess_name  = stochastic_proecess_name)
                #
                L = self.L(STATES = STATES[-1], eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat, stochastic_proecess_name  = stochastic_proecess_name)
                #np.multiply(L, brownian)
                
                
                #np.matmul(np.matmul(L, L.T), (terminalcondition - STATES[-1])/(((terminalmoment - ts[idx]))**1.0)*deltat)
                #new_state = STATES[-1] + F*deltat  + np.matmul(np.matmul(L, L.T), (terminalcondition - STATES[-1])/(((terminalmoment - ts[idx]))**1.0)*deltat) + np.matmul(L, brownian)
                new_state = STATES[-1] + F*deltat  + (terminalcondition - STATES[-1])/(((terminalmoment - ts[idx]))**1.0)*deltat + np.matmul(L, brownian)
                #if terminalmoment > ts[idx+1]:
                #    new_state = STATES[-1] + F*deltat + (terminalcondition-STATES[-1])/(terminalmoment - ts[idx])*deltat + np.matmul(L, brownian)
                #elif terminalmoment == ts[idx+1]:
                #    new_state = STATES[-1] + F*deltat + np.matmul(L, brownian)
                
                new_state[3] = max(0,new_state[3]  )
                
                STATES.append(new_state)
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters
        
        pass
    

    
    @classmethod
    def sim(self, ts, STATES_leader,  STATE_init =  np.array([-30.0, 3.5, .0, .0, .0, .0, .0]), veh_paras_self = veh_paras, veh_paras_leader = veh_paras, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = 1.0, eta_lat = 1.0, sigma_long = .01, sigma_lat = .01, N_paths = 50,  weight_heading = 1.0, weight_line_CG = 0.0,  sigma_long_drift = 10.0, sigma_lat_drift = 10.0,  stochastic_proecess_name ='geometric'):
        """
        
        Simulate the follower trajectories. 
        
        NOTE: 
        
            that the x is the logitudinal axis and y is the lateral. 
        
            The poisitive axis it downstream. 
        
            The left direction is y-positive. 
        
        
        ---------------------------------------
        @input: STATES_leader
        
            an array, which illustrate the movement of the leader 
            
            STATES_leader.shape is (7,N). N ==len(ts)
            
        @input: STATE_init
        
            the initial state of ego vehice. 
            
            
                #the system  state. 
                x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
        --------------------------------------
        @OUTPUT: STATES
        
            an arry. The shape is the same as STATES_leader
            
        
        
        """

        #---------------------------------------
        if len(ts)!=STATES_leader.shape[1]:
            raise ValueError('The input moments N is not equal to the states N')

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape is (2, len(deltats))
            brownianspath =  self.GenerateBrownianPaths(ts = ts)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                STATE_leader = STATES_leader[:,idx+1]
                #shape is (2,)
                brownian = brownianspath[:,idx]
                #
                #F is an 1d array with the shape of 7. 
                F = self.F(STATES = STATES[-1], STATES_leader = STATE_leader, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat,  weight_heading = weight_heading, weight_line_CG = weight_line_CG,  sigma_long_drift = sigma_long_drift, sigma_lat_drift = sigma_lat_drift, stochastic_proecess_name  = stochastic_proecess_name)
                #
                L = self.L(STATES = STATES[-1], eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat, stochastic_proecess_name  = stochastic_proecess_name)
                #np.multiply(L, brownian)
                new_state = STATES[-1] + F*deltat + np.matmul(L, brownian)
                new_state[3] = max(0,new_state[3]  )
                
                STATES.append(new_state)
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters
        
        pass
    


    @classmethod
    def sim_backward(self, ts, STATES_leader,  STATE_final =  np.array([-30.0, 3.5, .0, .0, .0, .0, .0]), veh_paras_self = veh_paras, veh_paras_leader = veh_paras, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = 1.0, eta_lat = 1.0, sigma_long = .01, sigma_lat = .01, N_paths = 50,  weight_heading = 1.0, weight_line_CG = 0.0,  sigma_long_drift = 10.0, sigma_lat_drift = 10.0,  stochastic_proecess_name ='geometric'):
        """
        
        Simulate the follower trajectories backward. 
        
        Difference between:
        
            - self.sim, forward in time, initial arg is STATE_init
            - self.sim_backward, terminal condition is given as STATE_final. 
        
        THe backward simulation use the backward Euler marayama schme.e 
        
            S(k)-S(k-1) = F(S)*deltat + L(S)*deltaW
            
            and we have: S(k-1) = S(k) - (F(S)*deltat + L(S)*deltaW)
        
        NOTE: 
        
            that the x is the logitudinal axis and y is the lateral. 
        
            The poisitive axis it downstream. 
        
            The left direction is y-positive. 
        
        
        ---------------------------------------
        @input: STATES_leader
        
            an array, which illustrate the movement of the leader 
            
            STATES_leader.shape is (7,N). N ==len(ts)
            
        @input: STATE_init
        
            the initial state of ego vehice. 
            
            
                #the system  state. 
                x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
        --------------------------------------
        @OUTPUT: STATES
        
            an arry. The shape is the same as STATES_leader, i.e. (7, len(ts)).
            
        
        
        """

        #---------------------------------------
        if len(ts)!=STATES_leader.shape[1]:
            raise ValueError('The input moments N is not equal to the states N')

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape is (2, len(deltats))
            brownianspath =  self.GenerateBrownianPaths(ts = ts)
            
            STATES = [STATE_final]
            for idx in range(len(deltats)):
                #
                deltat = deltats[len(deltats)-1-idx]
                #
                STATE_leader = STATES_leader[:,len(ts)-2-idx]
                #shape is (2,)
                brownian = brownianspath[:,len(ts)-2-idx]
                #
                #F is an 1d array with the shape of 7. 
                F = self.F(STATES = STATES[-1], STATES_leader = STATE_leader, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat,  weight_heading = weight_heading, weight_line_CG = weight_line_CG,  sigma_long_drift = sigma_long_drift, sigma_lat_drift = sigma_lat_drift, stochastic_proecess_name  = stochastic_proecess_name)
                #
                #L shape is (7, 2)
                L = self.L(STATES = STATES[-1], eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat, stochastic_proecess_name  = stochastic_proecess_name)
                #np.multiply(L, brownian)
                
                #1-order Euler scheme
                new_state = STATES[-1] + (F*deltat + np.matmul(L, brownian))
                #new_state = STATES[-1] + F*deltat + np.matmul(L, brownian)
                #
                #new_state = STATES[-1] - (F*deltat + np.matmul(L, brownian) + .5*np.matmul(L, L.T)*(brownian[0]**2 - deltat))
                
                #print(new_state.shape)
                #new_state[3] = max(0,new_state[3]  )
                
                STATES.append(new_state)
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES[::-1]).T)
        
        return STATES_iters
        
        pass
    
    
    @classmethod
    def IDM_steer(self, STATES, STATES_leader, y_expected = .0, phi_expected = 0.0, idm_steer_paras = idm_steer_paras,  veh_paras_self = veh_paras, veh_paras_leader = veh_paras, weight_heading = 1.0, weight_line_CG = 1e-5):
        """
        the steer from the IDM model. 
        
        NOTE: 
        
            that the x is the logitudinal axis and y is the lateral. 
        
            The poisitive axis it downstream. 
        
            The left direction is y-positive. 
        
        ----------------------------------------
        @input: y_expected and phi_expected
        
            the expected lateral location and the heading angle. 
        
        """

        #the system  state. 
        x_self,y_self,phi_self,v_self,delta_self,Z_long_self,Z_lat_self = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
        
        #the system  state of the leader.
        x_leader,y_leader,phi_leader,v_leader,delta_leader,Z_long_leader,Z_lat_leader = STATES_leader[0],STATES_leader[1],STATES_leader[2],STATES_leader[3],STATES_leader[4],STATES_leader[5],STATES_leader[6]
        
        #==========================================
        #==========Three steps:
        #==========calculate the xi_1 and xi_2
        #==========calculate the desired ster
        #==========calculate the steer rate using optimal 
        #calculate the xi_1 and xi_2
        #   xi_1 is the angle difference. 
        xi_1 = phi_expected - phi_self
        #
        #   xi_2 is the angle between the neading angle and the line connecting the CG. 
        #   xi_2 = angle1 - phi_expected, angle1 is the line connecting the CG of two vehicles
        #       
        #       if the vehicle' y is positive, then angle 1 is negatve. 
        #       first calculate the angle of the line of the CG. 
        tmp1 = np.sqrt((x_self-x_leader)**2 + (y_self - y_leader)**2)
        #   the interval of np.arcsin is [-np.pi/2, np.pi/2]
        #
        #print(y_self, y_expected, tmp1, (y_self -y_expected)/tmp1)
        converted = max(-.99999, min(.99999, (y_self -y_expected)/tmp1))
        ##angle1 = np.arcsin((y_self -y_expected)/tmp1)
        angle1 = np.arcsin(converted)
        
        xi_2 = angle1
        #xi_2 = phi_expected - angle1
        #
        #-------
        #desired
        desired_steer = 1.0 - np.exp((weight_heading*xi_1 + weight_line_CG*xi_2)/idm_steer_paras['tau'])
        #
        #--------calculate the steer rate. veh_paras = {'lf':1.5, 'lr':1.5, 'lF':2.5, 'lR':2.5, 'width':2.0, 'max_steer':70*np.pi/180.0, 'min_steer':-70*np.pi/180.0, 'max_steer_rate':0.4, 'min_steer_rate':-0.4,}
        steerrate = veh_paras_self['max_steer_rate']*self.OptimalVelocity(desired_steer)-delta_self
        
        return steerrate
        
    
    @classmethod
    def OptimalVelocity(self, v=2):
        """
        
        """
        return np.tanh(v)
        return np.tanh(v-2.0)- np.tanh(2.0)
        
        pass

    @classmethod
    def Lyapunov_LV(self,STATE, STATE_leader, STATE_equilibrium = np.array([.0, .0, .0, .0 , .0, .0, .0]), veh_paras_self = veh_paras, veh_paras_leader = veh_paras, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = 1.0, eta_lat = 1.0, sigma_long = .05, sigma_lat = .05, weight_heading = 1.0, weight_line_CG = 1e-5, sigma_long_drift = 1.0, sigma_lat_drift = 1.0, stochastic_proecess_name = 'OU'):
        """
        Calculate the function value that is used to test the stability using Lyapunov function. 
        
        LV is defined as 
        
            LV = sum(l_i**2)  + sum_S_*F
            
        ------------------------------------------------------
        @input: sigma_long and sigma_lat
            
            the parameters of the sde. 
            
            They are the noise at the longitudinal and lateral dimension. 
            
            
        @input: eta_long and eta_lat
        
            the parameters in the systme state equation. 
            
        @input: STATES_leader
        
            the states of the leader. 

            x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
            x and y are coordinate
            
            v is the speed
            
            delta is the steer angle of front wheel
            
            Z_lon and Z_lat is the nosie for longitudinal (x dimension) and lateral (y dimension).
        
        
        @input: STATES
            
            x,y,phi,v,delta,Z_lon,Z_lat=STATES
            
            x and y are coordinate
            
            v is the speed
            
            delta is the steer angle of front wheel
            
            Z_lon and Z_lat is the nosie for longitudinal (x dimension) and lateral (y dimension).
        
        
        
        @input: params
            the parameters of the vehicle. 
        
        
        @input: lf and lr
            unit is meter. 
            lf is the lengh of front. i.e. the distance between front axel to the CG. 
            lr is the rear length, or the ditance betweene rear axel to the CG. 
        
        """
        
        #F is an 1d array with the shape of 7. 
        F = self.F_lyapunov(STATES = STATE, STATES_leader = STATE_leader, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat,  weight_heading = weight_heading, weight_line_CG = weight_line_CG, sigma_long_drift = sigma_long_drift, sigma_lat_drift = sigma_lat_drift,  stochastic_proecess_name = stochastic_proecess_name)
        #
        L = self.L_lyapunov(STATES = STATE, eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat, stochastic_proecess_name = stochastic_proecess_name)
        #np.multiply(L, brownian)
        
        return np.dot(STATE[1:]- STATE_equilibrium[1:], F[1:]) + np.sum(np.power(L, 2))/2.0
        
        
        pass





class TwoDimStochasticIDM_SingleTrack():
    """
    The two dimensional stochastic IDM model with single track model. 
    
    The state model is give by:
        dZ = F(Z)dt + L(Z)dW
    
    
    State of the vehicle variable:
    
        STATE = (x, y, delta, v, phi, phi_derivate, beta)
    
        
    NOTE the coordinate system: 
    
        that the x is the logitudinal axis and y is the lateral. 
    
        The poisitive axis it downstream. 
    
        The left direction is y-positive. 
        
        If the vehicle turn left the steeer angle is positive. 
    
    
    """
    #vehicle parameters
    veh_paras = {'lf':1.1, 'lr':1.3, 'lF':2.5, \
                'lR':2.5, 'width':1.6, 'max_steer':55*np.pi/180.0, \
                'min_steer':-55*np.pi/180.0, 'max_steer_rate':0.4, 'min_steer_rate':-0.4, \
                'Iz':1.538, \
                'hcg':0.557, \
                'Rw':0.344, \
                'Iyw':1.7, \
                'Tsb':0.76, \
                'Tse':1, \
                'Csf':20.89, \
                'Csr':20.89, \
                'mu_friction':1.048, \
                'mass':1.225, \
                'lwb':3.6,}
    
    #idm car following parameters
    #   vf unit is m/s; 
    idm_paras = {'idm_vf':60.0/3.6, 'idm_T':1.5, 'idm_delta':4.0, 'idm_s0':2.0, 'idm_a':3.0, 'idm_b':3.21}
    
    #idm_steer parameters
    idm_steer_paras = {'tau':2.0, }
    @classmethod
    def F(self, STATES, STATES_leader, veh_paras_self = veh_paras, veh_paras_leader = veh_paras, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = 1.0, eta_lat = 1.0, sigma_long = .05, sigma_lat = .05, weight_heading = 1.0, weight_line_CG = 1e-5, sigma_long_drift = 1.0, sigma_lat_drift = 1.0, stochastic_proecess_name = 'OU', min_v_avoid_error = 1e-5):
        """
        THE VEHICLE IS FRONT STEER. 
        
        Return:
        
            - diff_x, diff_y, diff_delta, diff_v, diff_phi, diff_phi_2nd, diff_beta, diff_Z_long, diff_Z_lat
        
        Thus the state variable is :
        
            S = (x, y, delta, v, phi, phi_derivate, beta, Z_long, z_lat)
            
            The last two are two dim noise. 
        
        Return the derivate of the X,Y and PHI. 
        X is the horizontal coordinate
        Y is the vertical coordinate
        PHI is the heading angle. 
        V is the speed. 
        -------------------------------------------
        @input: stochastic_proecess_name
        
            OU, means ornstein ublllk 
            geometric geomtric brownian
            jacobidiffusion
            
            converted
                
                convrt from the 
        
        @input: sigma_long and sigma_lat
            
            the parameters of the sde. 
            
            They are the noise at the longitudinal and lateral dimension. 
            
        @input: STATE_equilibrium
            
            the equilibrium state. 
            
            x_equi,y_equi,phi_equi,v_equi,delta_equi,Z_long_equi,Z_lat_equi = STATE_equilibrium[0],STATE_equilibrium[1],STATE_equilibrium[2],STATE_equilibrium[3],STATE_equilibrium[4],STATE_equilibrium[5],STATE_equilibrium[6]
            
            STATE_equilibrium = np.array([, , , , , 1.0, 1.0])
            
            
        @input: eta_long and eta_lat
        
            the parameters in the systme state equation. 
            
        @input: STATES_leader
        
            the states of the leader. 

            x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
            x and y are coordinate
            
            v is the speed
            
            delta is the steer angle of front wheel
            
            Z_lon and Z_lat is the nosie for longitudinal (x dimension) and lateral (y dimension).
        
        
        @input: STATES
            
            x,y,phi,v,delta,Z_lon,Z_lat=STATES
            
            x and y are coordinate
            
            v is the speed
            
            delta is the steer angle of front wheel
            
            Z_lon and Z_lat is the nosie for longitudinal (x dimension) and lateral (y dimension).
        
        
        
        @input: params
            the parameters of the vehicle. 
        
        
        @input: lf and lr
            unit is meter. 
            lf is the lengh of front. i.e. the distance between front axel to the CG. 
            lr is the rear length, or the ditance betweene rear axel to the CG. 
        
        @OUTPUT: diff_STATES
            len(STATES)=4:
                - X = STATES[0], the X of the CG. X is the horizontal axis. 
                - Y = STATES[1], the Y of the CG
                - PHI = STATES[2], the heading angle. between the vehicle and the X axis. 
                - V = STATES[3]
        """
        #the system  state. 
        #x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
        #
        x, y, delta, v, phi, diff_phi, beta, Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6], STATES[7], STATES[8]
        
        #
        lr = veh_paras_self.get('lr', 1.3)
        lf = veh_paras_self.get('lf', 1.2)
        Csf = veh_paras_self.get('Csf', 20.89)
        Csr = veh_paras_self.get('Csr', 20.89)
        hcg = veh_paras_self.get('hcg', 0.557)
        g =  veh_paras_self.get('g', 9.8)
        mu =  veh_paras_self.get('mu_friction', 1.225)
        Iz =  veh_paras_self.get('Iz', 1.538)
        mass = veh_paras_self.get('mass', 1.225)
        lwb = veh_paras_self.get('lwb', 3.6)


        #################################################x, y, delta, v, phi, phi_derivate, beta, Z_long,Z_lat
        #------------------------------------
        #   diff_x and diff_y
        #diff_x = v*np.cos( phi + beta) - eta_long*np.tanh(Z_long)
        #diff_y = v*np.sin( phi + beta) - eta_lat*np.tanh(Z_lat)
        diff_x = v*np.cos( phi + beta) - eta_long*np.tanh(Z_long)
        diff_y = v*np.sin( phi + beta)- eta_lat*np.tanh(Z_lat)
        #------------------------------------
        #   diff_delta
        steerrate = self.IDM_steer_ST(STATES, STATES_leader, idm_steer_paras = idm_steer_paras, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader, weight_heading = weight_heading, weight_line_CG = weight_line_CG)
        #       trim the steer rate
        diff_delta = self.TrimSteerRate(STATE = STATES, steerrate = steerrate, veh_paras = veh_paras_self)
        #------------------------------------
        #
        acce = self.IDM_acce_ST(STATES, STATES_leader, idm_paras = idm_paras, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader)
        #       trim the acceleration. 
        diff_v = self.TrimAcce(STATE  = STATES, acce = acce, idm_paras = idm_paras)
        #------------------------------------
        #diff_phi_new
        if v>=0.1:
            diff_phi_new = copy.deepcopy(diff_phi)
        else:
            #
            diff_phi_new = v*np.cos(beta)/lwb*np.tan(delta)
        #diff_phi_new = self.TrimPhi(diff_phi_new, maxx = .1, minn = -.1)
        
        #------------------------------------
        #diff_beta
        #tmp0 = mu/(v*(lr+lf))
        if v>=0.1:
            tmp0 = mu/(max(v, min_v_avoid_error)*(lr+lf))
            tmp1 = Csf*(g*lr-diff_v*hcg)*delta - (Csr*(g*lf+diff_v*hcg) + Csf*(g*lr-diff_v*hcg))*beta
            #tmp2 = (Csr*(g*lf+acce*hcg)*lr - Csf*(g*lr-acce*hcg)*lf)*diff_phi/v
            tmp2 = (Csr*(g*lf+diff_v*hcg)*lr - Csf*(g*lr-diff_v*hcg)*lf)*diff_phi/v
            diff_beta  = tmp0*(tmp1 + tmp2) - diff_phi
        else:
            tmp1 = 1.0/(1.0+np.power(np.tan(delta)*lr/lwb , 2))
            tmp2 = lr/(lwb*np.power(np.cos(delta), 2))
            tmp3 = diff_delta
            diff_beta = tmp1*tmp2*tmp3

        #------------------------------------
        #diff_diff_phi = 
        #print(v)
        if v>=0.1:
            tmp0 = mu*mass/(Iz*(lf+lr))
            tmp1 = lf*Csf*(g*lr-diff_v*hcg)*delta 
            tmp2 = (lr*Csr*(g*lf+diff_v*hcg) - lf*Csf*(g*lr-diff_v*hcg))*beta
            print(x,y,diff_phi, v)
            tmp3 = (lf*lf*Csf*(g*lr-diff_v*hcg) + lr*lr*Csr*(g*lf+diff_v*hcg))*diff_phi/v
            diff_diff_phi = tmp0*(tmp1+tmp2-tmp3)
        else:
            tmp0 = diff_v
            tmp1 = tmp0*np.cos(beta)*np.tan(delta) - v*np.sin(beta)*np.tan(delta)*diff_beta
            tmp2 = v*np.cos(beta)/(np.power(np.cos(delta) , 2))
            tmp3 = diff_delta
            #
            diff_diff_phi = 1.0/lwb*(tmp1 + tmp2*tmp3)
            
            #tmp3 = (lf*lf*Csf*(g*lr-acce*hcg) + lr*lr*Csr*(g*lf+acce*hcg))*diff_phi/max(v, min_v_avoid_error)
        """
        
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #beta, the intermediate parameter
        #print(lr,lf, front_steer_angle, np.tan(front_steer_angle))
        tmp = (lr*np.tan(delta))/(lr+lf)
        beta = np.arctan(tmp)
        
        #################################################
        #_diff means differential , *v/idm_paras['idm_vf']
        # - eta_long*np.tanh(Z_long)
        # - eta_lat*np.tanh(Z_lat)
        #  - eta_long*Z_long
        # - eta_lat*Z_lat
        diff_x = v*np.cos( phi + beta) - eta_long*np.tanh(Z_long)
        diff_y = v*np.sin( phi + beta) - eta_lat*np.tanh(Z_lat)
        diff_phi = v*np.cos(beta)/(lr+lf)*np.tan(delta)
        #
        acce = self.IDM_acce(STATES, STATES_leader, idm_paras = idm_paras, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader)
        #trim the acceleration. 
        diff_v = self.TrimAcce(STATE  = STATES, acce = acce, idm_paras = idm_paras)
        #
        steerrate = self.IDM_steer(STATES, STATES_leader, idm_steer_paras = idm_steer_paras, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader, weight_heading = weight_heading, weight_line_CG = weight_line_CG)
        #trim the steer rate
        diff_delta = self.TrimSteerRate(STATE = STATES, steerrate = steerrate, veh_paras = veh_paras_self)
        
        """
        #
        #
        if stochastic_proecess_name=='OU':
            #diff_Z_long = -(sigma_long**1.0)*(Z_long**3)
            #diff_Z_lat = -(sigma_lat**1.0)*(Z_lat)
            #print(Z_long, sigma_long_drift)
            diff_Z_long = -sigma_long_drift*Z_long
            diff_Z_lat = -sigma_lat_drift*Z_lat
        elif stochastic_proecess_name=='converted':
            #the converted. 
            diff_Z_long = -(sigma_long**2)*(1-Z_long**2)*Z_long
            diff_Z_lat = -(sigma_lat**2)*(1-Z_lat**2)*Z_lat
        elif stochastic_proecess_name=='geometric':
            #
            diff_Z_long =  -sigma_long_drift*(Z_long)
            diff_Z_lat = -sigma_lat_drift*(Z_lat)
            #
        elif stochastic_proecess_name=='jacobi':
            #
            diff_Z_long = -sigma_long_drift*(Z_long - .0)
            diff_Z_lat = -sigma_lat_drift*(Z_lat - .0)
            
        elif stochastic_proecess_name=='hyperparabolic':
            #
            #diff_Z_long = -sigma_long_drift*(Z_long - .0)
            #diff_Z_lat = -sigma_lat_drift*(Z_lat - .0)
            diff_Z_long =  -Z_long-sigma_long_drift*Z_long
            diff_Z_lat = -Z_lat-sigma_lat_drift*Z_lat
            #print(diff_Z_long, diff_Z_lat)
        elif stochastic_proecess_name=='ROU':
            #ew_state = STATES[-1] + (theta/STATES[-1] -  STATES[-1] )*deltat + sigma*brownian
            diff_Z_long = -sigma_long_drift/Z_long +  Z_long#-sigma_long_drift*(Z_long - .0)
            diff_Z_lat = -sigma_lat_drift/Z_lat +  Z_lat #-sigma_lat_drift*(Z_lat - .0)
        
        #diff_x, diff_y, diff_delta, diff_v, diff_phi, diff_diff_phi, diff_beta, diff_Z_long, diff_Z_lat 
        return np.array([diff_x, diff_y, diff_delta, diff_v, diff_phi_new, diff_diff_phi, diff_beta, diff_Z_long, diff_Z_lat])
        #return np.array([diff_x,diff_y,diff_phi,diff_v, diff_delta, diff_Z_long, diff_Z_lat])

    @classmethod
    def F_lyapunov(self, STATES, STATES_leader, veh_paras_self = veh_paras, veh_paras_leader = veh_paras, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = 1.0, eta_lat = 1.0, sigma_long = .05, sigma_lat = .05, weight_heading = 1.0, weight_line_CG = 1e-5, sigma_long_drift = 1.0, sigma_lat_drift = 1.0, stochastic_proecess_name = 'OU'):
        """
        THE VEHICLE IS FRONT STEER. 
        
        Return the derivate of the X,Y and PHI. 
        X is the horizontal coordinate
        Y is the vertical coordinate
        PHI is the heading angle. 
        V is the speed. 
        -------------------------------------------
        @input: sigma_long and sigma_lat
            
            the parameters of the sde. 
            
            They are the noise at the longitudinal and lateral dimension. 
            
        @input: STATE_equilibrium
            
            the equilibrium state. 
            
            x_equi,y_equi,phi_equi,v_equi,delta_equi,Z_long_equi,Z_lat_equi = STATE_equilibrium[0],STATE_equilibrium[1],STATE_equilibrium[2],STATE_equilibrium[3],STATE_equilibrium[4],STATE_equilibrium[5],STATE_equilibrium[6]
            
            STATE_equilibrium = np.array([, , , , , 1.0, 1.0])
            
            
        @input: eta_long and eta_lat
        
            the parameters in the systme state equation. 
            
        @input: STATES_leader
        
            the states of the leader. 

            x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
            x and y are coordinate
            
            v is the speed
            
            delta is the steer angle of front wheel
            
            Z_lon and Z_lat is the nosie for longitudinal (x dimension) and lateral (y dimension).
        
        
        @input: STATES
            
            x,y,phi,v,delta,Z_lon,Z_lat=STATES
            
            x and y are coordinate
            
            v is the speed
            
            delta is the steer angle of front wheel
            
            Z_lon and Z_lat is the nosie for longitudinal (x dimension) and lateral (y dimension).
        
        
        
        @input: params
            the parameters of the vehicle. 
        
        
        @input: lf and lr
            unit is meter. 
            lf is the lengh of front. i.e. the distance between front axel to the CG. 
            lr is the rear length, or the ditance betweene rear axel to the CG. 
        
        @OUTPUT: diff_STATES
            len(STATES)=4:
                - X = STATES[0], the X of the CG. X is the horizontal axis. 
                - Y = STATES[1], the Y of the CG
                - PHI = STATES[2], the heading angle. between the vehicle and the X axis. 
                - V = STATES[3]
        """
        #the system  state. 
        x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
        
        
        #
        lr = veh_paras_self.get('lr', 2)
        lf = veh_paras_self.get('lf', 2)
        
        
        #beta, the intermediate parameter
        #print(lr,lf, front_steer_angle, np.tan(front_steer_angle))
        tmp = (lr*np.tan(delta))/(lr+lf)
        beta = np.arctan(tmp)
        
        #################################################
        #_diff means differential , *v/idm_paras['idm_vf']
        diff_x = -STATES_leader[3] + v*np.cos( phi + beta)
        diff_y = v*np.sin( phi + beta)
        diff_phi = v*np.cos(beta)/(lr+lf)*np.tan(delta)
        #
        acce = self.IDM_acce(STATES, STATES_leader, idm_paras = idm_paras, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader)
        #trim the acceleration. 
        diff_v = self.TrimAcce(STATE  = STATES, acce = acce   - eta_long*Z_long, idm_paras = idm_paras)
        #
        steerrate = self.IDM_steer(STATES, STATES_leader, idm_steer_paras = idm_steer_paras, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader, weight_heading = weight_heading, weight_line_CG = weight_line_CG)
        #trim the steer rate
        diff_delta = self.TrimSteerRate(STATE = STATES, steerrate = steerrate  - eta_lat*Z_lat, veh_paras = veh_paras_self)
        
        #
        if stochastic_proecess_name=='OU':
            #diff_Z_long = -(sigma_long**1.0)*(Z_long**3)
            #diff_Z_lat = -(sigma_lat**1.0)*(Z_lat)
            #print(Z_long, sigma_long_drift)
            diff_Z_long = -sigma_long_drift*Z_long
            diff_Z_lat = -sigma_lat_drift*Z_lat
        elif stochastic_proecess_name=='converted':
            #the converted. 
            diff_Z_long = -(sigma_long**2)*(1-Z_long**2)*Z_long
            diff_Z_lat = -(sigma_lat**2)*(1-Z_lat**2)*Z_lat
        elif stochastic_proecess_name=='geometric':
            #
            diff_Z_long = -sigma_long_drift*Z_long
            diff_Z_lat = -sigma_lat_drift*Z_lat
        elif stochastic_proecess_name=='jacobi':
            #
            diff_Z_long = -sigma_long_drift*Z_long
            diff_Z_lat = -sigma_lat_drift*Z_lat
        
        #the converted. 
        #diff_Z_long = -(sigma_long**2)*(1-Z_long**2)*Z_long + 1/2.0*(sigma_long*(1.0-Z_long**2))**2
        #diff_Z_lat = -(sigma_lat**2)*(1-Z_lat**2)*Z_lat + 1/2.0*(sigma_lat*(1.0-Z_lat**2))**2
        
        return np.array([diff_x,diff_y,diff_phi,diff_v, diff_delta, diff_Z_long, diff_Z_lat])

    
    @classmethod
    def F_BKP(self, STATES, STATES_leader, veh_paras_self = veh_paras, veh_paras_leader = veh_paras, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = 1.0, eta_lat = 1.0, sigma_long = .05, sigma_lat = .05):
        """
        THE VEHICLE IS FRONT STEER. 
        
        Return the derivate of the X,Y and PHI. 
        X is the horizontal coordinate
        Y is the vertical coordinate
        PHI is the heading angle. 
        V is the speed. 
        -------------------------------------------
        @input: sigma_long and sigma_lat
            
            the parameters of the sde. 
            
            They are the noise at the longitudinal and lateral dimension. 
            
            
        @input: eta_long and eta_lat
        
            the parameters in the systme state equation. 
            
        @input: STATES_leader
        
            the states of the leader. 

            x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
            x and y are coordinate
            
            v is the speed
            
            delta is the steer angle of front wheel
            
            Z_lon and Z_lat is the nosie for longitudinal (x dimension) and lateral (y dimension).
        
        
        @input: STATES
            
            x,y,phi,v,delta,Z_lon,Z_lat=STATES
            
            x and y are coordinate
            
            v is the speed
            
            delta is the steer angle of front wheel
            
            Z_lon and Z_lat is the nosie for longitudinal (x dimension) and lateral (y dimension).
        
        
        
        @input: params
            the parameters of the vehicle. 
        
        
        @input: lf and lr
            unit is meter. 
            lf is the lengh of front. i.e. the distance between front axel to the CG. 
            lr is the rear length, or the ditance betweene rear axel to the CG. 
        
        @OUTPUT: diff_STATES
            len(STATES)=4:
                - X = STATES[0], the X of the CG. X is the horizontal axis. 
                - Y = STATES[1], the Y of the CG
                - PHI = STATES[2], the heading angle. between the vehicle and the X axis. 
                - V = STATES[3]
        """
        #the system  state. 
        x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
        
        
        #
        lr = veh_paras_self.get('lr', 2)
        lf = veh_paras_self.get('lf', 2)
        
        
        #beta, the intermediate parameter
        #print(lr,lf, front_steer_angle, np.tan(front_steer_angle))
        tmp = (lr*np.tan(delta))/(lr+lf)
        beta = np.arctan(tmp)
        
        #_diff means differential 
        diff_x = v*np.cos( phi + beta) + eta_long*Z_long
        diff_y = v*np.sin( phi + beta) + eta_lat*Z_lat
        diff_phi = v*np.cos(beta)/(lr+lf)*np.tan(delta)
        #
        acce = self.IDM_acce(STATES, STATES_leader, idm_paras = idm_paras, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader)
        #trim the acceleration. 
        diff_v = self.TrimAcce(STATE  = STATES, acce = acce, idm_paras = idm_paras)
        #
        steerrate = self.IDM_steer(STATES, STATES_leader, idm_steer_paras = idm_steer_paras, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader)
        #trim the steer rate
        diff_delta = self.TrimSteerRate(STATE = STATES, steerrate = steerrate, veh_paras = veh_paras_self)
        #
        diff_Z_long = -(sigma_long**2)*(1-Z_long**2)*Z_long
        diff_Z_lat = -(sigma_lat**2)*(1-Z_lat**2)*Z_lat
        
        return np.array([diff_x,diff_y,diff_phi,diff_v, diff_delta, diff_Z_long, diff_Z_lat])
    
    @classmethod
    def L_lyapunov(self, STATES, eta_long = 1.0, eta_lat = 1.0, sigma_long = .05, sigma_lat = .05, stochastic_proecess_name = 'OU'):
        """
        
        @OUTPUT: array
        
            shape is (7,2).
            
            7 means the dimension of the state. 
            2 measnthat there are two randomness. 
        
        """
        #the system  state. 
        x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]

        if stochastic_proecess_name=='OU':
            tmp_long = sigma_long
            tmp_lat = sigma_lat
            #
        elif stochastic_proecess_name=='converted':
            #the converted. 
            tmp_long =  sigma_long*(1.0-Z_long**2)
            tmp_lat = sigma_lat*(1-Z_lat**2)
            #
        elif stochastic_proecess_name=='geometric':
            #
            tmp_long =  sigma_long*Z_long
            tmp_lat = sigma_lat*Z_lat
            #
        elif stochastic_proecess_name=='jacobi':
            tmp_long =   np.sqrt(sigma_long*(Z_long+.5)*(.5-Z_long))
            tmp_lat = np.sqrt(sigma_lat*(Z_lat+.5)*(.5-Z_lat))

        #
        #tmp_long =  sigma_long*(1.0-Z_long**2)
        #tmp_lat = sigma_lat*(1-Z_lat**2)
        
        array = np.array([[.0, .0, .0, .0, .0, tmp_long, .0], \
                          [.0, .0, .0, .0, .0, .0, tmp_lat]])
        
        
        return array.T


    @classmethod
    def L(self, STATES, eta_long = 1.0, eta_lat = 1.0, sigma_long = .05, sigma_lat = .05, stochastic_proecess_name = 'OU'):
        """
        
        x, y, delta, v, phi, diff_phi, beta, Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6], STATES[7], STATES[8]
        
        @OUTPUT: array
        
            shape is (7,2).
            
            7 means the dimension of the state. 
            2 measnthat there are two randomness. 
        
        """
        #
        x, y, delta, v, phi, diff_phi, beta, Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6], STATES[7], STATES[8]
        #the system  state. 
        #x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
        
        
        #------------------------------------------
        if stochastic_proecess_name=='OU':
            tmp_long = sigma_long
            tmp_lat = sigma_lat
            #
        elif stochastic_proecess_name=='converted':
            #the converted. 
            tmp_long =  sigma_long*(1.0-Z_long**2)
            tmp_lat = sigma_lat*(1-Z_lat**2)
            #
        elif stochastic_proecess_name=='geometric':
            #
            tmp_long =  sigma_long*(Z_long)
            tmp_lat = sigma_lat*(Z_lat)
            #
        elif stochastic_proecess_name=='hyperparabolic':
            #
            tmp_long = sigma_long
            tmp_lat = sigma_lat
            
        elif stochastic_proecess_name=='jacobi':
            #new_state = STATES[-1] - theta*(STATES[-1])*deltat + sigma*np.sqrt((STATES[-1]+.5)*(.5-STATES[-1]))*brownian
            Z_long = max(-.499999, min(Z_long, .4999999))
            tmp_long =   np.sqrt(sigma_long*(Z_long+.5)*(.5-Z_long))
            #print(Z_lat, (.5-Z_lat), (Z_lat+.5))
            Z_lat = max(-.499999, min(Z_lat, .4999999))
            tmp_lat = np.sqrt(sigma_lat*(Z_lat+.5)*(.5-Z_lat))
        elif stochastic_proecess_name=='ROU':
            #
            tmp_long =  sigma_long
            tmp_lat = sigma_lat
            #

        #
        #tmp_long =  sigma_long*(1.0-Z_long**2)
        #tmp_lat = sigma_lat*(1-Z_lat**2)
        array = np.array([[.0, .0, .0, .0, .0, .0, .0, tmp_long, .0], \
                          [.0, .0, .0, .0, .0, .0, .0, .0, tmp_lat]])
        #array = np.array([[.0, .0, .0, .0, .0, tmp_long, .0], \
        #                  [.0, .0, .0, .0, .0, .0, tmp_lat]])
        
        return array.T

    @classmethod
    def L_BKP(self, STATES, eta_long = 1.0, eta_lat = 1.0, sigma_long = .05, sigma_lat = .05, stochastic_proecess_name = 'OU'):
        """
        
        @OUTPUT: array
        
            shape is (7,2).
            
            7 means the dimension of the state. 
            2 measnthat there are two randomness. 
        
        """
        #the system  state. 
        x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
        
        if stochastic_proecess_name=='OU':
            tmp_long = sigma_long
            tmp_lat = sigma_lat
            #
        elif stochastic_proecess_name=='converted':
            #the converted. 
            tmp_long =  sigma_long*(1.0-Z_long**2)
            tmp_lat = sigma_lat*(1-Z_lat**2)
            #
        elif stochastic_proecess_name=='geometric':
            #
            tmp_long =  sigma_long*(Z_long)
            tmp_lat = sigma_lat*(Z_lat)
            #
        elif stochastic_proecess_name=='hyperparabolic':
            #
            tmp_long = sigma_long
            tmp_lat = sigma_lat
            
        elif stochastic_proecess_name=='jacobi':
            #new_state = STATES[-1] - theta*(STATES[-1])*deltat + sigma*np.sqrt((STATES[-1]+.5)*(.5-STATES[-1]))*brownian
            Z_long = max(-.499999, min(Z_long, .4999999))
            tmp_long =   np.sqrt(sigma_long*(Z_long+.5)*(.5-Z_long))
            #print(Z_lat, (.5-Z_lat), (Z_lat+.5))
            Z_lat = max(-.499999, min(Z_lat, .4999999))
            tmp_lat = np.sqrt(sigma_lat*(Z_lat+.5)*(.5-Z_lat))
        elif stochastic_proecess_name=='ROU':
            #
            tmp_long =  sigma_long
            tmp_lat = sigma_lat
            #

        #
        #tmp_long =  sigma_long*(1.0-Z_long**2)
        #tmp_lat = sigma_lat*(1-Z_lat**2)
        
        array = np.array([[.0, .0, .0, .0, .0, tmp_long, .0], \
                          [.0, .0, .0, .0, .0, .0, tmp_lat]])
        
        
        return array.T
    
    @classmethod
    def TrimSteerRate(self, STATE, steerrate, veh_paras = veh_paras):
        """
        Change the acceleraiton to make sure that the resulting vehicle constraints would not be violated. 
        
        The following conditions are considered when trimming:
        
            - if the speed reaches the maximum or the minimum, the acceleration is zero
            - if the 
        
        veh_paras = {'lf':1.5, 'lr':1.5, 'lF':2.5, 'lR':2.5, 'width':2.0, 'max_steer':70*np.pi/180.0, 'min_steer':-70*np.pi/180.0, 'max_steer_rate':0.4, 'min_steer_rate':-0.4,}
        
        -----------------------------
        @OUTPUT steer_trimmed
        
            the trimmed acceleration. 
            
            
            
        """
        
        
        x, y, delta, v, phi, diff_phi, beta, Z_long,Z_lat = STATE[0],STATE[1],STATE[2],STATE[3],STATE[4],STATE[5],STATE[6], STATE[7], STATE[8]
        #the system  state. 
        #x,y,phi,v,delta,Z_long,Z_lat = STATE[0],STATE[1],STATE[2],STATE[3],STATE[4],STATE[5],STATE[6]
        
        #
        #if delta<veh_paras['min_steer'] or delta>=veh_paras['max_steer']:
        #    return 0.0

        if delta<veh_paras['min_steer']:
            return 1e-1
        if delta>=veh_paras['max_steer']:
            return -1e-1

        #-------------------------
        return min(veh_paras['max_steer_rate'], max(veh_paras['min_steer_rate'], steerrate))
        
    @classmethod
    def TrimPhi(self, phi, maxx = 1.0, minn = -1.0):
        
        
        return min(maxx, max(minn, phi))
    
    @classmethod
    def TrimAcce(self, acce, STATE, idm_paras = idm_paras):
        """
        Change the acceleraiton to make sure that the resulting vehicle constraints would not be violated. 
        
        The following conditions are considered when trimming:
        
            - if the speed reaches the maximum or the minimum, the acceleration is zero
            - if the 
        
        -----------------------------
        @OUTPUT acce_trimmed
        
            the trimmed acceleration. 
            
            
            
        """
        x, y, delta, v, phi, diff_phi, beta, Z_long,Z_lat = STATE[0],STATE[1],STATE[2],STATE[3],STATE[4],STATE[5],STATE[6], STATE[7], STATE[8]
        #the system  state. 
        #x,y,phi,v,delta,Z_long,Z_lat = STATE[0],STATE[1],STATE[2],STATE[3],STATE[4],STATE[5],STATE[6]
        
        #
        #if v<0 or v>=1.3*idm_paras['idm_vf']:
        if v<.0:
            return 1.0
        if v>idm_paras['idm_vf']:
            return -1
            
        #-------------------------
        return min(idm_paras['idm_a'], max(-idm_paras['idm_b'], acce))
        
    
    @classmethod
    def IDM_acce_ST(self, STATES, STATES_leader,  idm_paras = idm_paras,  veh_paras_self = veh_paras, veh_paras_leader = veh_paras):
        """
        IDM formation output, the acceleration. 
        
        Note that the delta_v is defined as v_follower-v_leader. 
        
        @type v: float, unit is km/h
        
        @type vf: km/h.
        
        
        @type T: float.
        @param: T, unit is sec
            Average safe time headway.
            
        @type delta: delta:float
        @param: delta
            parameter in the model.
            
            
        @type s0:float
        @param: s0
            parameter 
        
        @OUTPUT: a
            unit is m/s2.
        """
        #IDM paraser. 
        idm_vf = idm_paras['idm_vf']
        idm_T = idm_paras['idm_T']
        idm_delta = idm_paras['idm_delta']
        idm_s0 = idm_paras['idm_s0']
        idm_a = idm_paras['idm_a']
        idm_b = idm_paras['idm_b']


        #the system  state. 
        #x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
        #
        x_self, y_self, delta_self, v_self, phi_self, diff_phi_self, beta_self, Z_long_self,Z_lat_self = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6], STATES[7], STATES[8]
        #the system  state. 
        #x_self,y_self,phi_self,v_self,delta_self,Z_long_self,Z_lat_self = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
        #the system  state of the leader.
        x_leader,y_leader,delta_leader,v_leader,phi_leader,diff_phi_leader,beta_leader,Z_long_leader,Z_lat_leader = STATES_leader[0],STATES_leader[1],STATES_leader[2],STATES_leader[3],STATES_leader[4],STATES_leader[5],STATES_leader[6],STATES_leader[7],STATES_leader[8]

        #the system  state. 
        #x_self,y_self,phi_self,v_self,delta_self,Z_long_self,Z_lat_self = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
        #the system  state of the leader.
        #x_leader,y_leader,phi_leader,v_leader,delta_leader,Z_long_leader,Z_lat_leader = STATES_leader[0],STATES_leader[1],STATES_leader[2],STATES_leader[3],STATES_leader[4],STATES_leader[5],STATES_leader[6]
        
        #
        #v_self=v/3.6
        #v_leader = v_leader_kmh/3.6
        vf = idm_vf
        
        deltax = np.sqrt((x_self-x_leader)**2 + (y_self - y_leader)**2) - veh_paras_self['lF']/1.0 - veh_paras_leader['lR']/1.0
        
        #
        try:
            s_star = idm_s0+v_self*idm_T+v_self*(v_self - v_leader)/(2.0*np.sqrt(idm_a*idm_b))
            a = 1.0*idm_a*(1-np.power(v_self/vf, idm_delta)-(s_star*s_star)/(deltax*deltax))
        except Exception as e:
            
            print('deltax = ',deltax,', v_self=',v_self,', v_leader',v_leader)
            raise ValueError(e)
            
        return a
    
    
    @classmethod
    def random_accelerations(self, amplification = 3.0, ts = np.linspace(0, 300, 300), period_sec = 30):
        """
        
        
        acces  = VM.TwoDimStochasticIDM.random_accelerations(ts = ts)
        """
        return np.array([amplification*np.sin(t/period_sec) for t in ts[1:]])
    
    @classmethod
    def GenerateBrownianPaths(self, ts = np.linspace(0, 300, 300)):
        """
        
        Geneate the brownian paths. 
        
        The generated brownian should be a 2d array. shape is (2, len(ts)-1)
        
        
        brownianpath = self.GenerateBrownianPaths(ts = ts)
        ------------------------------------------------
        
        
        """
        stds = np.sqrt(np.diff(ts))
        means = np.zeros(stds.shape)
        r1 = np.random.normal(loc = means, scale = stds)
        r2 = np.random.normal(loc = means, scale = stds)
        
        #shape is (2, len(ts)-1)
        return np.array([r1,r2])
        
    
    @classmethod
    def GenerateLeaderTrajectories_freedriving(self, ts = np.linspace(0, 300, 300), STATE_init = np.array([.0, .0, .0, .0, .0, .0, .0]), idm_paras  = idm_paras, ):
        """
        Generate the trajectories of the leader that allows the free driving of the following vehicle
        
        Each instance, the state of the vehicle is represented by STATES:
        
            #the system  state. 
            x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
        Note that in the leading trajectory we only consider the longitudinal (x) dimeensnon. 
    
        
        NOTE the coordinate system: 
        
            that the x is the logitudinal axis and y is the lateral. 
        
            The poisitive axis it downstream. 
        
            The left direction is y-positive. 
            
            If the vehicle turn left the steeer angle is positive. 
        
        
        -----------------------------------------------------------------
        
        @input: STATE_init
        
            the initial state. 
        
        @input: ts
        
            the moments. 
        
        @OUTPUT: trajectories
        
            an np.array. 
            
            Shape is (7, moments_N), where 7 is the state number, and moments_N is the number of moments. 
        
            
        
        """
        
        #the system  state. 
        x0,y,phi,v0,delta,Z_long,Z_lat = STATE_init[0],STATE_init[1],STATE_init[2],STATE_init[3],STATE_init[4],STATE_init[5],STATE_init[6]
        
        #the length of accelerations is len(ts-1).
        accelerations = np.array(len(self.random_accelerations(ts = ts))*[idm_paras['idm_vf']])
        
        #IDM paraser. 
        idm_vf = idm_paras['idm_vf']
        idm_T = idm_paras['idm_T']
        idm_delta = idm_paras['idm_delta']
        idm_s0 = idm_paras['idm_s0']
        idm_a = idm_paras['idm_a']
        idm_b = idm_paras['idm_b']
        
        #find the xs
        xs = [x0]
        vs = [v0]
        for deltat,acc0 in zip(np.diff(ts), accelerations):
            STATES_tmp = np.array([xs[-1], 0, phi, vs[-1],  delta,Z_long,Z_lat])
            acc= self.TrimAcce(acc0, STATE = STATES_tmp, idm_paras = idm_paras)
            #
            #print(acc0, acc)
            new_v = min(max(0, vs[-1] + deltat*acc), idm_vf)
            #
            new_x = xs[-1] + deltat*vs[-1]
            
            #
            vs.append(new_v)
            xs.append(new_x)
            
        #
        
        return np.array([xs,[y]*len(ts),[phi]*len(ts),vs,[delta]*len(ts),[Z_long]*len(ts),[Z_lat]*len(ts)])
        
    
    @classmethod
    def GenerateLeaderTrajectories(self, ts = np.linspace(0, 300, 300), STATE_init = np.array([.0, .0, .0,.0, .0, .0, .0, .0, .0]), idm_paras  = idm_paras, ):
        """
        Generate the trajectories of the leader. 

        Thus the state variable is :
        
            S = (x, y, delta, v, phi, phi_derivate, beta, Z_long, z_lat)


        Each instance, the state of the vehicle is represented by STATES:
            
             x, y, delta, v, phi, diff_phi, beta, Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6], STATES[7], STATES[8]
            
            
        Note that in the leading trajectory we only consider the longitudinal (x) dimeensnon. 
    
        
        NOTE the coordinate system: 
        
            that the x is the logitudinal axis and y is the lateral. 
        
            The poisitive axis it downstream. 
        
            The left direction is y-positive. 
            
            If the vehicle turn left the steeer angle is positive. 
        
        
        -----------------------------------------------------------------
        
        @input: STATE_init
        
            the initial state. 
        
        @input: ts
        
            the moments. 
        
        @OUTPUT: trajectories
        
            an np.array. 
            
            Shape is (7, moments_N), where 7 is the state number, and moments_N is the number of moments. 
        
            
        
        """
        
        #the system  state. 
        #x0,y,phi,v0,delta,Z_long,Z_lat = STATE_init[0],STATE_init[1],STATE_init[2],STATE_init[3],STATE_init[4],STATE_init[5],STATE_init[6]
        x0, y, delta, v0, phi, diff_phi, beta, Z_long,Z_lat = STATE_init[0],STATE_init[1],STATE_init[2],STATE_init[3],STATE_init[4],STATE_init[5],STATE_init[6], STATE_init[7], STATE_init[8]
        
        #
        #the length of accelerations is len(ts-1).
        accelerations = self.random_accelerations(ts = ts)
        
        #IDM paraser. 
        idm_vf = idm_paras['idm_vf']
        idm_T = idm_paras['idm_T']
        idm_delta = idm_paras['idm_delta']
        idm_s0 = idm_paras['idm_s0']
        idm_a = idm_paras['idm_a']
        idm_b = idm_paras['idm_b']
        
        #find the xs
        xs = [x0]
        vs = [v0]
        for deltat,acc0 in zip(np.diff(ts), accelerations):
            STATES_tmp = np.array([xs[-1], 0, delta, vs[-1], phi, diff_phi, beta, Z_long,Z_lat])
            acc = self.TrimAcce(acc0, STATE = STATES_tmp, idm_paras = idm_paras)
            #
            #print(acc0, acc)
            new_v = min(max(0, vs[-1] + deltat*acc), idm_vf)
            #
            new_x = xs[-1] + deltat*vs[-1]
            
            #
            vs.append(new_v)
            xs.append(new_x)
            
        #
        return np.array([xs,[y]*len(ts),[delta]*len(ts),vs,[phi]*len(ts),\
                [diff_phi]*len(ts), [beta]*len(ts),[Z_long]*len(ts),[Z_lat]*len(ts)])
        #return np.array([xs,[y]*len(ts),[phi]*len(ts),vs,[delta]*len(ts),[Z_long]*len(ts),[Z_lat]*len(ts)])
        
    
    @classmethod
    def EquilibriumHeadway_IDM(self, v,  idm_paras = idm_paras,  veh_paras_self = veh_paras,):
        """
        calculate the equilibrium headway between vehicles in the IDM mode. 
        
            vs = np.linspace(1.0/3.6, 59/3.6, 100)
            gs  = [VM.TwoDimStochasticIDM.EquilibriumHeadway_IDM(v=v) for v in vs]
        
        
        ----------------------------------------------------------
        
        @input: v
        
            unit is m.
            
        @OUTPUT: headway
        
            unit is m. 
        
        """
        
        
        #IDM paraser. 
        idm_vf = idm_paras['idm_vf']
        idm_T = idm_paras['idm_T']
        idm_delta = idm_paras['idm_delta']
        idm_s0 = idm_paras['idm_s0']
        idm_a = idm_paras['idm_a']
        idm_b = idm_paras['idm_b']
        
        if v>idm_vf:
            raise ValueError('speed exceeds the max. ')
        
        
        #unit is m.
        headway_equilibrium = (idm_s0 + v*idm_T)*np.power((1-v/idm_vf), -0.5)
        
        return headway_equilibrium
    

    @classmethod
    def plot_vehiclestates_multi(self, ts, vehstates_arrays, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        @input: vehstates_arrays
        
            a list. 
        
            vehstates_array = vehstates_arrays[0]
        
            a 2d array. (7, N).
            
            7 is the number of vehicle states and N is the length of the moments. 
            
            
            
            
        
        """
        if isinstance(ax, bool):
            fig,axs = plt.subplots(figsize = figsize, nrows = 2, ncols = 2)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        for vehstates_array in vehstates_arrays:
        
            
            #the xy
            ax = axs[0, 0]
            ax.plot(vehstates_array[0, :], vehstates_array[1, :])
            ax.set_xlabel('x ( m )');ax.set_ylabel('y ( m )'); ax.grid();ax.set_title('(a)')
            #ax.set_ylim([-1.8, 1.8])
            
            #
            ax = axs[0, 1]
            ax.plot(ts, vehstates_array[2, :])
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('Heading angle (rad) '); ax.grid();ax.set_title('(b)')
            
            #
            ax = axs[1, 0]
            ax.plot(ts, vehstates_array[3, :])
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('Speed (m/s)'); ax.grid();ax.set_title('(c)')
            
            #
            ax = axs[1, 1]
            ax.plot(ts, vehstates_array[4, :])
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('Steer (rad)'); ax.grid();ax.set_title('(d)')
        
        axs[0, 0].grid();axs[0, 1].grid();axs[1, 0].grid();axs[1, 1].grid();
        
        
        plt.tight_layout()
        return axs



    @classmethod
    def plot_probabilityevolution_idx(self, ts, vehstates_arrays, figsize = (8,4), ax = False, t_MAX = 50, n_moments_plotted = 4, bins= 20, normalize = False, idx_plotted = 1, x_label = 'y (m)'):
        """
        
        @input: vehstates_arrays
        
            a list. 
        
            vehstates_array = vehstates_arrays[0]
        
            a 2d array. (9, N).
            
            9 is the number of vehicle states and N is the length of the moments. 
        
        @input: t_MAX  and n_moments_plotted
        
            t_MAX is the para that 
            
        """
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #ys shape is (samplepath_N, momnetsN)
        ys = pd.DataFrame([vehstates_array[idx_plotted, :] for vehstates_array in vehstates_arrays])
        #print(ys.shape)
        #
        selected_ts0 = ts[ts<t_MAX]
        #
        selected_idxs = range(0, len(selected_ts0), int(len(selected_ts0)/n_moments_plotted))
        #selected_ts = [selected_ts0[i] for i in selected_idxs]
        
        #print(selected_ts)
        #
        for idx in selected_idxs:
            if idx==0:continue
            #print(idx)
            hs,es0 =np.histogram(ys.iloc[:, idx], bins = bins)
            #
            es = es0[1:]
            if normalize:
                ax.plot(es,hs/sum(hs)/(es[-1]-es[-2]), label = str(int(selected_ts0[idx]*100)/100.0) + ' sec')
            else:
                ax.plot(es,hs, label = str(int(selected_ts0[idx]*100)/100.0) + ' sec')
        
        #
        plt.tight_layout()
        ax.legend()
        ax.set_xlabel(x_label);ax.set_ylabel('Frequencies'); ax.grid()
        return ax

    
    @classmethod
    def plot_variance_idx(self, ts, vehstates_arrays, figsize = (8,4), ax = False, idx_state = 1):
        """
        
        @input: vehstates_arrays
        
            a list. 
        
            vehstates_array = vehstates_arrays[0]
        
            a 2d array. (7, N).
            
            7 is the number of vehicle states and N is the length of the moments. 
        
        @input: t_MAX  and n_moments_plotted
        
            t_MAX is the para that 
            
        """
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #ys shape is (samplepath_N, momnetsN)
        ys = pd.DataFrame([vehstates_array[idx_state, :] for vehstates_array in vehstates_arrays])
        ax.plot(ts, np.std(ys, axis = 0))

        #
        plt.tight_layout()
        #ax.legend()
        ax.set_xlabel('Time ( sec )');ax.set_ylabel('std'); ax.grid()
        return ax

        
        pass
    
    
    
    
    @classmethod
    def plot_probabilityevolution_y(self, ts, vehstates_arrays, figsize = (8,4), ax = False, t_MAX = 50, n_moments_plotted = 4, bins= 20, normalize = False):
        """
        
        @input: vehstates_arrays
        
            a list. 
        
            vehstates_array = vehstates_arrays[0]
        
            a 2d array. (7, N).
            
            7 is the number of vehicle states and N is the length of the moments. 
        
        @input: t_MAX  and n_moments_plotted
        
            t_MAX is the para that 
            
        """
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #ys shape is (samplepath_N, momnetsN)
        ys = pd.DataFrame([vehstates_array[1, :] for vehstates_array in vehstates_arrays])
        #print(ys.shape)
        #
        selected_ts0 = ts[ts<t_MAX]
        #
        selected_idxs = range(0, len(selected_ts0), int(len(selected_ts0)/n_moments_plotted))
        #selected_ts = [selected_ts0[i] for i in selected_idxs]
        
        #print(selected_ts)
        #
        for idx in selected_idxs:
            if idx==0:continue
            #print(idx)
            hs,es0 =np.histogram(ys.iloc[:, idx], bins = bins)
            #
            es = es0[1:]
            if normalize:
                ax.plot(es,hs/sum(hs)/(es[-1]-es[-2]), label = str(int(selected_ts0[idx]*100)/100.0) + ' sec')
            else:
                ax.plot(es,hs, label = str(int(selected_ts0[idx]*100)/100.0) + ' sec')
        
        #
        plt.tight_layout()
        ax.legend()
        ax.set_xlabel('y ( m )');ax.set_ylabel('Frequencies'); ax.grid()
        return ax

        
        pass



    @classmethod
    def plot_vehiclestates(self, ts, vehstates_array, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        @input: vehstates_array
        
            a 2d array. (9, N).
            
            9 is the number of vehicle states and N is the length of the moments. 
            
            S = (x, y, delta, v, phi, phi_derivate, beta, Z_long, z_lat)
        
        
            
            x, y, delta, v, phi, diff_phi, beta, Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6], STATES[7], STATES[8]
            
        
        """
        if isinstance(ax, bool):
            fig,axs = plt.subplots(figsize = figsize, nrows = 2, ncols = 2)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #the xy
        ax = axs[0, 0]
        ax.plot(vehstates_array[0, :], vehstates_array[1, :])
        ax.set_title('(a)')
        ax.set_xlabel('x ( m )');ax.set_ylabel('y ( m )'); ax.grid()
        #ax.set_ylim([-1.8, 1.8])
        
        #heading angle phi
        ax = axs[0, 1]
        ax.plot(ts, vehstates_array[4, :])
        ax.set_xlabel('Time ( sec )');ax.set_ylabel('heading angle (rad) '); ax.grid();ax.set_title('(b)')
        
        #speed
        ax = axs[1, 0]
        ax.plot(ts, vehstates_array[3, :])
        ax.set_xlabel('Time ( sec )');ax.set_ylabel('speed (m/s)'); ax.grid();ax.set_title('(c)')
        
        #steer angle, delta
        ax = axs[1, 1]
        ax.plot(ts, vehstates_array[2, :])
        ax.set_xlabel('Time ( sec )');ax.set_ylabel('Steer angle (rad)'); ax.grid();ax.set_title('(d)')
        
        plt.tight_layout()
        return axs
    
    
    @classmethod
    def plot_ts_idx_state(self, ts, paths, idx_state = 1, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        @input: paths
            
            paths is a list. Each element is for one simulation
            
            sim_res = paths[idx]
            
            sim_res is a  array with shape (7, N), where 7 is the system state and N is the moments number. N = len(ts)
        
        
        """
        
        
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)

        for sim_res in paths:
            ax.plot(ts, sim_res[idx_state,:])
        
        if isinstance(ax, bool):
            ax.set_xlabel('Time (sec)');ax.set_xlabel('y'); 
            ax.grid();
            
            plt.tight_layout()
        
        return ax
    
    @classmethod
    def plot_platoon_3d_speed(self, ts, platoondata_list, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        
        @input: platoondata_list
        
        
            first = platoondata_list[0]
            second = platoondata_list[1]
            ...
            
            first is a list. the number is the paths. 
            
            first_path0 = first[0].
            
            first_path0 is a 2d array with shape (7,N), where N isthe moments number. 
        
        
        """
        
        if isinstance(ax, bool):
            ax = plt.figure().add_subplot(projection='3d')
            #fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #the xy
        for i,veh_in_platoon in enumerate(platoondata_list):
            vehstates_array = veh_in_platoon[0]
            ax.plot(ts, vehstates_array[0, :], label = str(i)+'-th')
            ax.set_title('(a)')
            ax.set_xlabel('time ( sec )');ax.set_ylabel('y ( m )'); ax.grid()
            #ax.set_ylim([-1.8, 1.8])
        ax.legend()
        
        plt.tight_layout()
        return ax
    



    @classmethod
    def plot_platoon_x(self, ts, platoondata_list, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        
        @input: platoondata_list
        
        
            first = platoondata_list[0]
            second = platoondata_list[1]
            ...
            
            first is a list. the number is the paths. 
            
            first_path0 = first[0].
            
            first_path0 is a 2d array with shape (7,N), where N isthe moments number. 
        
        
        """
        
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #the xy
        for i,veh_in_platoon in enumerate(platoondata_list):
            vehstates_array = veh_in_platoon[0]
            ax.plot(ts, vehstates_array[0, :], label = str(i)+'-th')
            ax.set_title('(a)')
            ax.set_xlabel('time ( sec )');ax.set_ylabel('y ( m )'); ax.grid()
            #ax.set_ylim([-1.8, 1.8])
        ax.legend()
        
        plt.tight_layout()
        return ax
    



    @classmethod
    def plot_platoon(self, ts, platoondata_list, axs = False, figsize = (5,3), alpha = .4,):
        """
        
        
        @input: platoondata_list
        
        
            first = platoondata_list[0]
            second = platoondata_list[1]
            ...
            
            first is a list. the number is the paths. 
            
            first_path0 = first[0].
            
            first_path0 is a 2d array with shape (7,N), where N isthe moments number. 
        
        
        """
        
        if isinstance(axs, bool):
            fig,axs = plt.subplots(figsize = figsize, nrows = 2, ncols = 2)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #the xy
        ax = axs[0,0]
        for i,veh_in_platoon in enumerate(platoondata_list):
            vehstates_array = veh_in_platoon[0]
            ax.plot(vehstates_array[0, :], vehstates_array[1, :], label = str(i)+'-th')
            ax.set_title('(a)')
            ax.set_xlabel('x ( m )');ax.set_ylabel('y ( m )'); ax.grid()
            #ax.set_ylim([-1.8, 1.8])
        ax.legend()
        
        #
        ax = axs[0, 1]
        for i,veh_in_platoon in enumerate(platoondata_list):
            vehstates_array = veh_in_platoon[0]
            ax.plot(ts, vehstates_array[2, :], label = str(i)+'-th')
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('heading angle (rad) '); ax.grid();ax.set_title('(b)')
        ax.legend()
        
        #
        ax = axs[1, 0]
        for i,veh_in_platoon in enumerate(platoondata_list):
            vehstates_array = veh_in_platoon[0]
            ax.plot(ts, vehstates_array[3, :], label = str(i)+'-th')
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('speed (m/s)'); ax.grid();ax.set_title('(c)')
        ax.legend()
        
        #
        ax = axs[1, 1]
        for i,veh_in_platoon in enumerate(platoondata_list):
            vehstates_array = veh_in_platoon[0]
            ax.plot(ts, vehstates_array[4, :], label = str(i)+'-th')
            ax.set_xlabel('Time ( sec )');ax.set_ylabel('Steer rate (rad/s)'); ax.grid();ax.set_title('(d)')
        ax.legend()
        
        plt.tight_layout()
        return axs

    
    @classmethod
    def plot_paths(self, paths, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        @input: paths
            
            paths is a list. Each element is for one simulation
            
            sim_res = paths[idx]
            
            sim_res is a  array with shape (7, N), where 7 is the system state and N is the moments number. N = len(ts)
        
        
        """
        
        
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('x');ax.set_ylabel('y'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)

        for sim_res in paths:
            ax = self.plot_path(path = sim_res, ax = ax, figsize = figsize, alpha = alpha,)
        
        #
        ax.set_xlabel('x ( m )');ax.set_ylabel('y ( m )'); ax.grid()
        
            
        plt.tight_layout()
        
        return ax

    @classmethod
    def plot_path_with_vehiclebound(self, path, ax = False, N_vehs_plotted = 30, figsize = (5,3), alpha = .4, veh_paras = veh_paras, facecolor= [0,0.5,0],):
        """
        
        @input: path
            
            path is a  array with shape (7, N), where 7 is the system state and N is the moments number. N = len(ts)
        
        
        """
        import matplotlib
        
        
        if isinstance(ax, bool):
            fig,ax = plt.subplots()
            #fig,ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('x');ax.set_xlabel('y'); 
            ax.grid();
            
            plt.tight_layout()
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)

        xs = path[0, :]
        ys = path[1, :]
        ax.plot(xs, ys)
        
        ######################plot vehi bounds
        lr = veh_paras.get('lr', 2)
        lf = veh_paras.get('lf', 3)
        veh_width = veh_paras.get('width', 2)
        veh_len = veh_paras.get('length', 5)
        
        #-----------------------determine the idx of the ploted samples, in plot_columns_idxs
        plot_columns_idxs = list(range(0, path.shape[1],  int(path.shape[1]/N_vehs_plotted)))
        #
        patches = []
        for idx in plot_columns_idxs:
            #
            x = path[0, idx]
            y = path[1, idx]
            heading = path[2, idx]
            
            #
            rear_right_x,rear_right_y = VehicleKineticSolver.Veicle_Rear_Right(xy_CG = (x,y),w = veh_width, lr =lr, ang= heading)
            
            #print(rear_right_x,rear_right_y)
            #
            rect = matplotlib.patches.Rectangle((rear_right_x,rear_right_y), veh_len, veh_width, heading*180.0/np.pi, alpha = alpha, facecolor=facecolor, ec = 'k')#fill=None, facecolor='none'
            
            ax.add_patch(rect)

        
        return ax
        
    @classmethod
    def plot_state_confidenceinterval(self, ts, vehstates_arrays, figsize = (5,2.5), ax = False, normalize = False, idx_plotted = 3, x_label = 'time (sec)', y_label= 'speed (m/s)', quantiles = [.15, .25], alpha = .3):
        """
        
        @input: vehstates_arrays
        
            a list. 
        
            vehstates_array = vehstates_arrays[0]
        
            a 2d array. (7, N).
            
            7 is the number of vehicle states and N is the length of the moments. 
        
        @input: t_MAX  and n_moments_plotted
        
            t_MAX is the para that 
            
        """
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #ys shape is (samplepath_N, momnetsN)
        ys = pd.DataFrame([vehstates_array[idx_plotted, :] for vehstates_array in vehstates_arrays])
        #print(ys.shape)
        
        #ymean is 1d with shale (momnetsN)
        ymean = np.mean(ys, axis = 0)
        ax.plot(ts, ymean, label = 'mean')
        
        #
        for quantile in quantiles:
            #
            #print(ys.iloc[:,0].shape)
            
            datas1 = [np.quantile(ys.iloc[:,idx_t],.5+quantile) for idx_t in range(len(ts))]
            datas2 = [np.quantile(ys.iloc[:,idx_t], .5-quantile) for idx_t in range(len(ts))]
            
            #
            ax.fill_between(ts, datas1,datas2, alpha = alpha, label = str(quantile))
        
        ax.legend()
        
        ax.set_xlabel(x_label);ax.set_ylabel(y_label); ax.grid()
        return ax
        

    @classmethod
    def plot_path_idx(self, ts, path, ax = False, figsize = (5,3), alpha = .4, idx = 3):
        """
        
        @input: path
            
            path is a  array with shape (9, N), where 7 is the system state and N is the moments number. N = len(ts)
        
        
        """
        
        
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        ax.plot(ts, path[idx, :])
        
        if isinstance(ax, bool):
            ax.set_xlabel('x');ax.set_xlabel('y'); 
            ax.grid();
            
            plt.tight_layout()
        
        return ax

    @classmethod
    def plot_path(self, path, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        @input: path
            
            path is a  array with shape (9, N), where 7 is the system state and N is the moments number. N = len(ts)
        
        
        """
        
        
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)

        xs = path[-2, :]
        ys = path[-1, :]
        
        
        ax.plot(xs, ys)
        
        if isinstance(ax, bool):
            ax.set_xlabel('x');ax.set_xlabel('y'); 
            ax.grid();
            
            plt.tight_layout()
        
        return ax
    
    
    @classmethod
    def sim_lc(self, ts, STATES_leader,  STATE_init =  np.array([-30.0, 3.5, .0, .0, .0, .0, .0]), veh_paras_self = veh_paras, veh_paras_leader = veh_paras, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = 1.0, eta_lat = 1.0, sigma_long = .01, sigma_lat = .01, N_paths = 50,  weight_heading = 1.0, weight_line_CG = 0.0):
        """
        
        Simulate the follower trajectories. 
        
        NOTE: 
        
            that the x is the logitudinal axis and y is the lateral. 
        
            The poisitive axis it downstream. 
        
            The left direction is y-positive. 
        
        
        ---------------------------------------
        @input: STATES_leader
        
            an array, which illustrate the movement of the leader 
            
            STATES_leader.shape is (7,N). N ==len(ts)
            
        @input: STATE_init
        
            the initial state of ego vehice. 
            
            
                #the system  state. 
                x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
        --------------------------------------
        @OUTPUT: STATES
        
            an arry. The shape is the same as STATES_leader
            
        
        
        """

        #---------------------------------------
        if len(ts)!=STATES_leader.shape[1]:
            raise ValueError('The input moments N is not equal to the states N')

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape is (2, len(deltats))
            brownianspath =  self.GenerateBrownianPaths(ts = ts)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                x,y,phi,v,delta,Z_long,Z_lat = STATES[-1][0],STATES[-1][1],STATES[-1][2],STATES[-1][3],STATES[-1][4],STATES[-1][5],STATES[-1][6]
                #
                equilibrium = self.EquilibriumHeadway_IDM(v= 16)
                STATE_leader = np.array([STATES[-1][0]+equilibrium, 0, .0, idm_paras['idm_vf'], .0, .0, .0])
                #STATE_leader = STATES_leader[:,idx+1]
                #shape is (2,)
                brownian = brownianspath[:,idx]
                #
                #F is an 1d array with the shape of 7. 
                F = self.F(STATES = STATES[-1], STATES_leader = STATE_leader, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat,  weight_heading = weight_heading, weight_line_CG = weight_line_CG)
                #
                L = self.L(STATES = STATES[-1], eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat)
                #np.multiply(L, brownian)
                new_state = STATES[-1] + F*deltat + np.matmul(L, brownian)
                new_state[3] = max(0,new_state[3]  )
                
                STATES.append(new_state)
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters
        
        pass

    @classmethod
    def OU_with_transform_simulation(self, ts = np.linspace(0, 100, 1000), STATE_init = np.array([.0, .0]), theta = 1.0, sigma = .3, N_paths = 50):
        """
        
        STATES_iters = VM.TwoDimStochasticIDM.OU_simulation()
        
        """

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape i
            
            stds = np.sqrt(np.diff(ts))
            means = np.zeros(stds.shape)
            #brownians shape is (, len(deltats))
            brownianspath = np.random.normal(loc = means, scale = stds)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                brownian = brownianspath[idx]
                new_state0 = STATES[-1][1] - theta*STATES[-1][1]*deltat + sigma*brownian
                
                new_state1 = STATES[-1][0] +  np.tanh(STATES[-1][0])*deltat
                
                STATES.append(np.array([new_state1, new_state0]))
            #
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters

    @classmethod
    def simulation_jacobi(self, ts = np.linspace(0, 100, 1000), STATE_init = .0, theta = 1.0, sigma = .3, N_paths = 50):
        """
        
        STATES_iters = VM.TwoDimStochasticIDM.OU_simulation()
        
        """

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape i
            
            stds = np.sqrt(np.diff(ts))
            means = np.zeros(stds.shape)
            #brownians shape is (, len(deltats))
            brownianspath = np.random.normal(loc = means, scale = stds)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                brownian = brownianspath[idx]
                #
                if STATES[-1]<-0.5 or STATES[-1]>.5:
                    tmp0 = min(.499999, max(-0.499999, STATES[-1]))
                    #tmp = min(.499999, max(-0.499999, sigma*(0.5+STATES[-1])*(0.5-STATES[-1])))
                    #print(STATES[-1], )
                    tmp = sigma*(0.5 + tmp0)*(0.5- tmp0)
                else:
                    tmp = sigma*(0.5+STATES[-1])*(0.5-STATES[-1])
                #print(STATES[-1], tmp)
                new_state = STATES[-1] - theta*(STATES[-1])*deltat + np.sqrt(tmp)*brownian
                
                STATES.append(new_state)
            #
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters,brownianspath
        


    @classmethod
    def geometric_simulation(self, ts = np.linspace(0, 100, 1000), STATE_init = 1.0, theta = 1.0, sigma = .3, N_paths = 50):
        """
        
        STATES_iters = VM.TwoDimStochasticIDM.OU_simulation()
        
        """

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape i
            
            stds = np.sqrt(np.diff(ts))
            means = np.zeros(stds.shape)
            #brownians shape is (, len(deltats))
            brownianspath = np.random.normal(loc = means, scale = stds)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                brownian = brownianspath[idx]
                
                new_state = STATES[-1] + theta*STATES[-1]*deltat + sigma*STATES[-1]*brownian
                
                STATES.append(new_state)
            #
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters,brownianspath
        


    @classmethod
    def simulation_hyperparabolic(self, ts = np.linspace(0, 100, 1000), STATE_init = 2.0, theta = 1.0, sigma = .3, N_paths = 50, alpha = .5, a = 2):
        """
        
        STATES_iters = VM.TwoDimStochasticIDM.OU_simulation()
        
        """

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape i
            
            stds = np.sqrt(np.diff(ts))
            means = np.zeros(stds.shape)
            #brownians shape is (, len(deltats))
            brownianspath = np.random.normal(loc = means, scale = stds)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                brownian = brownianspath[idx]
                
                new_state = STATES[-1] + -(STATES[-1] +  theta/(STATES[-1]))*deltat + sigma*brownian
                
                STATES.append(new_state)
            #
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters
        


    @classmethod
    def simulation_ROU(self, ts = np.linspace(0, 100, 1000), STATE_init = 0, theta = 1.0, sigma = .3, N_paths = 50, alpha = .5):
        """
        
        STATES_iters = VM.TwoDimStochasticIDM.OU_simulation()
        
        """

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape i
            
            stds = np.sqrt(np.diff(ts))
            means = np.zeros(stds.shape)
            #brownians shape is (, len(deltats))
            brownianspath = np.random.normal(loc = means, scale = stds)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                brownian = brownianspath[idx]
                
                new_state = STATES[-1] + (theta/STATES[-1] -  STATES[-1] )*deltat + sigma*brownian
                
                STATES.append(new_state)
            #
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters
        
    


    @classmethod
    def simulation_OU(self, ts = np.linspace(0, 100, 1000), STATE_init = 0, theta = 1.0, sigma = .3, N_paths = 50):
        """
        
        STATES_iters = VM.TwoDimStochasticIDM.OU_simulation()
        
        """

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape i
            
            stds = np.sqrt(np.diff(ts))
            means = np.zeros(stds.shape)
            #brownians shape is (, len(deltats))
            brownianspath = np.random.normal(loc = means, scale = stds)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                brownian = brownianspath[idx]
                
                new_state = STATES[-1] - theta*STATES[-1]*deltat + sigma*brownian
                
                STATES.append(new_state)
            #
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters
        
    
    @classmethod
    def sim(self, ts, STATES_leader,  STATE_init =  np.array([-30.0, 3.5, .0, .0, .0, .0, .0, .0, .0]), veh_paras_self = veh_paras, veh_paras_leader = veh_paras, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = 1.0, eta_lat = 1.0, sigma_long = .01, sigma_lat = .01, N_paths = 50,  weight_heading = 1.0, weight_line_CG = 0.0,  sigma_long_drift = 10.0, sigma_lat_drift = 10.0,  stochastic_proecess_name ='OU'):
        """
        
        Simulate the follower trajectories. 
        
        NOTE: 
        
            that the x is the logitudinal axis and y is the lateral. 
        
            The poisitive axis it downstream. 
        
            The left direction is y-positive. 
        
        
        ---------------------------------------
        @input: STATES_leader
        
            an array, which illustrate the movement of the leader 
            
            STATES_leader.shape is (7,N). N ==len(ts)
            
        @input: STATE_init
        
            the initial state of ego vehice. 
            
            
                #the system  state. 
                x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
        --------------------------------------
        @OUTPUT: STATES
        
            an arry. The shape is the same as STATES_leader
            
        
        
        """

        #---------------------------------------
        if len(ts)!=STATES_leader.shape[1]:
            raise ValueError('The input moments N is not equal to the states N')

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape is (2, len(deltats))
            brownianspath =  self.GenerateBrownianPaths(ts = ts)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                STATE_leader = STATES_leader[:,idx+1]
                #shape is (2,)
                brownian = brownianspath[:,idx]
                #
                #F is an 1d array with the shape of 9.
                # np.array([diff_x, diff_y, diff_delta, diff_v, diff_phi, diff_diff_phi, diff_beta, diff_Z_long, diff_Z_lat])
                #       print(STATES[-1].shape, STATE_leader.shape)
                F = self.F(STATES = STATES[-1], STATES_leader = STATE_leader, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat,  weight_heading = weight_heading, weight_line_CG = weight_line_CG,  sigma_long_drift = sigma_long_drift, sigma_lat_drift = sigma_lat_drift, stochastic_proecess_name  = stochastic_proecess_name)
                
                #L shape is 9,2
                L = self.L(STATES = STATES[-1], eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat, stochastic_proecess_name  = stochastic_proecess_name)
                #np.multiply(L, brownian)
                new_state = STATES[-1] + F*deltat + np.matmul(L, brownian)
                #
                new_state[3] = max(.0,new_state[3]  )
                
                STATES.append(new_state)
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters
        
        pass
    
    @classmethod
    def sim_BKP(self, ts, STATES_leader,  STATE_init =  np.array([-30.0, 3.5, .0, .0, .0, .0, .0]), veh_paras_self = veh_paras, veh_paras_leader = veh_paras, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = 1.0, eta_lat = 1.0, sigma_long = .01, sigma_lat = .01, N_paths = 50,  weight_heading = 1.0, weight_line_CG = 0.0,  sigma_long_drift = 10.0, sigma_lat_drift = 10.0,  stochastic_proecess_name ='geometric'):
        """
        
        Simulate the follower trajectories. 
        
        NOTE: 
        
            that the x is the logitudinal axis and y is the lateral. 
        
            The poisitive axis it downstream. 
        
            The left direction is y-positive. 
        
        
        ---------------------------------------
        @input: STATES_leader
        
            an array, which illustrate the movement of the leader 
            
            STATES_leader.shape is (7,N). N ==len(ts)
            
        @input: STATE_init
        
            the initial state of ego vehice. 
            
            
                #the system  state. 
                x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
        --------------------------------------
        @OUTPUT: STATES
        
            an arry. The shape is the same as STATES_leader
            
        
        
        """

        #---------------------------------------
        if len(ts)!=STATES_leader.shape[1]:
            raise ValueError('The input moments N is not equal to the states N')

        #
        STATES_iters = []
        
        #deltats and the Browians
        deltats = np.diff(ts)
        
        #
        #---------------------------------------
        for iterr in range(N_paths):
            #
            #brownians shape is (2, len(deltats))
            brownianspath =  self.GenerateBrownianPaths(ts = ts)
            
            STATES = [STATE_init]
            for idx in range(len(deltats)):
                deltat = deltats[idx]
                #
                STATE_leader = STATES_leader[:,idx+1]
                #shape is (2,)
                brownian = brownianspath[:,idx]
                #
                #F is an 1d array with the shape of 7. 
                F = self.F(STATES = STATES[-1], STATES_leader = STATE_leader, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat,  weight_heading = weight_heading, weight_line_CG = weight_line_CG,  sigma_long_drift = sigma_long_drift, sigma_lat_drift = sigma_lat_drift, stochastic_proecess_name  = stochastic_proecess_name)
                #
                L = self.L(STATES = STATES[-1], eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat, stochastic_proecess_name  = stochastic_proecess_name)
                #np.multiply(L, brownian)
                new_state = STATES[-1] + F*deltat + np.matmul(L, brownian)
                new_state[3] = max(0,new_state[3]  )
                
                STATES.append(new_state)
            #
            #-------------------------------
            STATES_iters.append(np.array(STATES).T)
        
        return STATES_iters
        
        pass
    
    
    @classmethod
    def IDM_steer_ST(self, STATES, STATES_leader, y_expected = .0, phi_expected = 0.0, idm_steer_paras = idm_steer_paras,  veh_paras_self = veh_paras, veh_paras_leader = veh_paras, weight_heading = 1.0, weight_line_CG = 1e-5):
        """
        the steer from the IDM model. 
        
        NOTE: 
        
            that the x is the logitudinal axis and y is the lateral. 
        
            The poisitive axis it downstream. 
        
            The left direction is y-positive. 
        
        ----------------------------------------
        @input: y_expected and phi_expected
        
            the expected lateral location and the heading angle. 
        
        """
        #the system  state. 
        #x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
        #
        x_self, y_self, delta_self, v_self, phi_self, diff_phi_self, beta_self, Z_long_self,Z_lat_self = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6], STATES[7], STATES[8]
        #the system  state. 
        #x_self,y_self,phi_self,v_self,delta_self,Z_long_self,Z_lat_self = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
        
        
        #the system  state of the leader.
        x_leader,y_leader,delta_leader,v_leader,phi_leader,diff_phi_leader,beta_leader,Z_long_leader,Z_lat_leader = STATES_leader[0],STATES_leader[1],STATES_leader[2],STATES_leader[3],STATES_leader[4],STATES_leader[5],STATES_leader[6],STATES_leader[7],STATES_leader[8]
        
        #==========================================
        #==========Three steps:
        #==========calculate the xi_1 and xi_2
        #==========calculate the desired ster
        #==========calculate the steer rate using optimal 
        #calculate the xi_1 and xi_2
        #   xi_1 is the angle difference. 
        xi_1_not_rounded = phi_expected - phi_self
        #
        xi_1 = np.mod(xi_1_not_rounded, 2.0*np.pi)
        #   xi_2 is the angle between the neading angle and the line connecting the CG. 
        #   xi_2 = angle1 - phi_expected, angle1 is the line connecting the CG of two vehicles
        #       
        #       if the vehicle' y is positive, then angle 1 is negatve. 
        #       first calculate the angle of the line of the CG. 
        tmp1 = np.sqrt((x_self-x_leader)**2 + (y_self - y_leader)**2)
        #   the interval of np.arcsin is [-np.pi/2, np.pi/2]
        #print(tmp1)
        angle1 = np.arcsin((y_self -y_expected)/tmp1)
        xi_2 = angle1
        #xi_2 = phi_expected - angle1
        #
        #-------
        #desired
        #print(xi_1, xi_2)
        desired_steer = 1.0 - np.exp((weight_heading*xi_1 + weight_line_CG*xi_2)/idm_steer_paras['tau'])
        #
        #--------calculate the steer rate. veh_paras = {'lf':1.5, 'lr':1.5, 'lF':2.5, 'lR':2.5, 'width':2.0, 'max_steer':70*np.pi/180.0, 'min_steer':-70*np.pi/180.0, 'max_steer_rate':0.4, 'min_steer_rate':-0.4,}
        steerrate = veh_paras_self['max_steer_rate']*self.OptimalVelocity(desired_steer)-delta_self
        
        return steerrate
        
    
    @classmethod
    def OptimalVelocity(self, v=2):
        """
        
        """
        return np.tanh(v)
        return np.tanh(v-2.0)- np.tanh(2.0)
        
        pass

    @classmethod
    def Lyapunov_LV(self,STATE, STATE_leader, STATE_equilibrium = np.array([.0, .0, .0, .0 , .0, .0, .0]), veh_paras_self = veh_paras, veh_paras_leader = veh_paras, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = 1.0, eta_lat = 1.0, sigma_long = .05, sigma_lat = .05, weight_heading = 1.0, weight_line_CG = 1e-5, sigma_long_drift = 1.0, sigma_lat_drift = 1.0, stochastic_proecess_name = 'OU'):
        """
        Calculate the function value that is used to test the stability using Lyapunov function. 
        
        LV is defined as 
        
            LV = sum(l_i**2)  + sum_S_*F
            
        ------------------------------------------------------
        @input: sigma_long and sigma_lat
            
            the parameters of the sde. 
            
            They are the noise at the longitudinal and lateral dimension. 
            
            
        @input: eta_long and eta_lat
        
            the parameters in the systme state equation. 
            
        @input: STATES_leader
        
            the states of the leader. 

            x,y,phi,v,delta,Z_long,Z_lat = STATES[0],STATES[1],STATES[2],STATES[3],STATES[4],STATES[5],STATES[6]
            
            x and y are coordinate
            
            v is the speed
            
            delta is the steer angle of front wheel
            
            Z_lon and Z_lat is the nosie for longitudinal (x dimension) and lateral (y dimension).
        
        
        @input: STATES
            
            x,y,phi,v,delta,Z_lon,Z_lat=STATES
            
            x and y are coordinate
            
            v is the speed
            
            delta is the steer angle of front wheel
            
            Z_lon and Z_lat is the nosie for longitudinal (x dimension) and lateral (y dimension).
        
        
        
        @input: params
            the parameters of the vehicle. 
        
        
        @input: lf and lr
            unit is meter. 
            lf is the lengh of front. i.e. the distance between front axel to the CG. 
            lr is the rear length, or the ditance betweene rear axel to the CG. 
        
        """
        
        #F is an 1d array with the shape of 7. 
        F = self.F_lyapunov(STATES = STATE, STATES_leader = STATE_leader, veh_paras_self = veh_paras_self, veh_paras_leader = veh_paras_leader, idm_paras = idm_paras, idm_steer_paras = idm_steer_paras, eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat,  weight_heading = weight_heading, weight_line_CG = weight_line_CG, sigma_long_drift = sigma_long_drift, sigma_lat_drift = sigma_lat_drift,  stochastic_proecess_name = stochastic_proecess_name)
        #
        L = self.L_lyapunov(STATES = STATE, eta_long = eta_long, eta_lat = eta_lat, sigma_long = sigma_long, sigma_lat = sigma_lat, stochastic_proecess_name = stochastic_proecess_name)
        #np.multiply(L, brownian)
        
        return np.dot(STATE[1:]- STATE_equilibrium[1:], F[1:]) + np.sum(np.power(L, 2))/2.0
        
        
        pass



def TODO():
    """
    
    
    
    - deltaf_sign_detemination_tmp()
    
    """
    
    
    pass




def CG_DISTANCE_T(t_and_deltaf = (10, 30*np.pi/180), v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }):
    """
    @input: forward_backward
        either 1 or -1. 
        
        1 means forward and -1 means backward. 
    
    """
    #----------------------------------------------------------
    t,deltaf = t_and_deltaf[0],t_and_deltaf[1]
    #
    x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
    #
    lr = veh_para.get('lr', 1.7)
    lf = veh_para.get('lf', 1.7)
    lR = veh_para.get('lR', 2)
    lF = veh_para.get('lF', 2)
    w = veh_para.get('w', 2)
    #
    beta = np.arctan((lr*np.tan(deltaf))/(lf+lr))
    
    def fun_to_integrate_CG(t):
        
        phi = phi0 + t*v*np.cos(beta)*np.tan(deltaf)/(lf+lr)
                
        diff_cg_x = v*np.cos(phi+beta)
        diff_cg_y = v*np.sin(phi+beta)
        return np.sqrt(diff_cg_x**2 + diff_cg_y**2)
    
    
    return scipy.integrate.quad(fun_to_integrate_CG, 0, t)[0] 


def CORNER_DISYANCE_t(t_and_deltaf = (10, 30*np.pi/180), v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, corner = 'A'):
    """
    Callback method:
    
        VM.CORNER_DISYANCE_t(t_and_deltaf = (t, 30*np.pi/180), corner = 'D')
    
    Batch callback:
    
        x = [VM.CORNER_DISYANCE_t(t_and_deltaf = (t, 30*np.pi/180), corner = 'D') for t in range(0,30)]
        plt.plot(x)
    
    
    
    
    Get the traveling distance of the 
    
    import scipy.integrate as integrate
    
    import scipy.special as special
    
    result = integrate.quad(lambda x: special.jv(2.5,x), 0, 4.5)
    """
    #----------------------------------------------------------
    t,deltaf = t_and_deltaf[0],t_and_deltaf[1]
    #
    x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
    #
    lr = veh_para.get('lr', 1.7)
    lf = veh_para.get('lf', 1.7)
    lR = veh_para.get('lR', 2)
    lF = veh_para.get('lF', 2)
    w = veh_para.get('w', 2)
    #
    beta = np.arctan((lr*np.tan(deltaf))/(lf+lr))
    
    #----------------------------------------------------------
    
    def fun_to_integrate_CornerA(t):
        """
        
        """
        phi = phi0 + t*v*np.cos(beta)*np.tan(deltaf)/(lf+lr)
    
        diff_xA = v*np.cos(phi+beta)-v*np.tan(deltaf)*np.cos(beta)/(lf+lr)*(lF*np.sin(phi)+w/2.0*np.sin(phi-np.pi/2))
        
        diff_yA = v*np.sin(phi+beta)+v*np.tan(deltaf)*np.cos(beta)/(lf+lr)*(lF*np.cos(phi)+w/2.0*np.cos(phi-np.pi/2))
        
        return np.sqrt(diff_xA**2 + diff_yA**2)
        
        
    def fun_to_integrate_CornerB(t):
        """
        
        """
        phi = phi0 + t*v*np.cos(beta)*np.tan(deltaf)/(lf+lr)
    
        diff_xB = v*np.cos(phi+beta)+v*np.tan(deltaf)*np.cos(beta)/(lf+lr)*(lR*np.sin(phi)-w/2.0*np.sin(phi-np.pi/2))
        
        diff_yB = v*np.sin(phi+beta)-v*np.tan(deltaf)*np.cos(beta)/(lf+lr)*(lR*np.cos(phi)-w/2.0*np.cos(phi-np.pi/2))
        
        return np.sqrt(diff_xB**2 + diff_yB**2)
        
        
    def fun_to_integrate_CornerC(t):
        """
        
        """
        phi = phi0 + t*v*np.cos(beta)*np.tan(deltaf)/(lf+lr)
    
        diff_xC = v*np.cos(phi+beta)+v*np.tan(deltaf)*np.cos(beta)/(lf+lr)*(lR*np.sin(phi)-w/2.0*np.sin(phi+np.pi/2))
        
        diff_yC = v*np.sin(phi+beta)-v*np.tan(deltaf)*np.cos(beta)/(lf+lr)*(lR*np.cos(phi)-w/2.0*np.cos(phi+np.pi/2))
        
        return np.sqrt(diff_xC**2 + diff_yC**2)
        
        
    def fun_to_integrate_CornerD(t):
        """
        
        """
        phi = phi0 + t*v*np.cos(beta)*np.tan(deltaf)/(lf+lr)
    
        diff_xD = v*np.cos(phi+beta)-v*np.tan(deltaf)*np.cos(beta)/(lf+lr)*(lF*np.sin(phi)+w/2.0*np.sin(phi+np.pi/2))
        
        diff_yD = v*np.sin(phi+beta)+v*np.tan(deltaf)*np.cos(beta)/(lf+lr)*(lF*np.cos(phi)+w/2.0*np.cos(phi+np.pi/2))
        
        return np.sqrt(diff_xD**2 + diff_yD**2)
        
        
    if corner=='A':
        
        return scipy.integrate.quad(fun_to_integrate_CornerA, 0, t)[0] 
    elif corner=='B':
        
        return scipy.integrate.quad(fun_to_integrate_CornerB, 0, t)[0] 
    elif corner=='C':
        
        return scipy.integrate.quad(fun_to_integrate_CornerC, 0, t)[0] 
    elif corner=='D':
        
        return scipy.integrate.quad(fun_to_integrate_CornerD, 0, t)[0] 


def CORNER_A_SOLVE_fun(t_and_deltaf, point = (10, 10),v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, forward_backward = 1):
    """
    This method will return the following, given the travelling time and the steer angle:
    
        Coor_A_x - x
        Coor_A_y - y
    
        where (x,y) are given point coordinates; 
    
    THe functions that:
    
        Coor_A_x - x = 0
        Coor_A_y - y = 0
    
    THis function will be used to solve the deltaf, or steering angle of front wheel. 
    
    Where Coor_A_x is the  x coordinate of corner A, and Coor_A_y is the y coordinate of corner A. 
    
    The (x,y) in the equation is given by point (=(x,y)). 
    
    ------------------------------------------
    @input: point
        x,y = point
        
        the given point that the path of corner A will path. 
    
    @input: v
        the default of the vehile. it is the state in the Rajami vehicle model. 
    
    @input: t_and_deltaf
        the two variables that need to solve using scipy.optimize.fsolve
        
        t,deltaf = t_and_deltaf
        
        t is the time that corner pass point and the deltaf (unit is np.pi) is the steering angle that the vehicle use. 
    """
    v = forward_backward*v
    
    
    #CornerA_at_t(self, t_and_deltaf=(50, 1*np.pi/180.0), v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, })
    
    t,deltaf = t_and_deltaf[0],t_and_deltaf[1]
    
    
    x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
    
    #
    lr = veh_para.get('lr', 1.7)
    lf = veh_para.get('lf', 1.7)
    lR = veh_para.get('lR', 2)
    lF = veh_para.get('lF', 2)
    w = veh_para.get('w', 2)
    
    #
    beta = np.arctan((lr*np.tan(deltaf))/(lf+lr))
    
    #phi
    phi = phi0 + t*v*np.cos(beta)*np.tan(deltaf)/(lf+lr)
    
    #CG x and CG y at moment t given delta f.
    cg_x = x0 + (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.sin(phi0 + t*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.sin(phi0 + beta))
    #
    cg_y =  y0 - (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.cos(phi0 + t*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.cos(phi0 + beta))
    
    #
    xA = cg_x + lF*np.cos(phi)+w/2.0*np.cos(phi - np.pi/2)
    yA = cg_y + lF*np.sin(phi)+w/2.0*np.sin(phi - np.pi/2)
    
    #
    return (xA - point[0], yA-point[1])


def CORNER_B_SOLVE_fun(t_and_deltaf, point = (10, 10),v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, forward_backward = 1):
    """
    THe functions that:
        Coor_B_x - x = 0
        Coor_B_y - y = 0
    
    THis function will be used to solve the deltaf, or steering angle of front wheel. 
    
    Where Coor_B_x is the  x coordinate of corner B, and Coor_B_y is the y coordinate of corner B. 
    
    The (x,y) in the equation is given by point (=(x,y)). 
    
    ------------------------------------------
    @input: point
        x,y = point
        
        the given point that the path of corner B will path. 
    
    @input: v
        the default of the vehile. it is the state in the Rajami vehicle model. 
    
    @input: t_and_deltaf
        the two variables that need to solve using scipy.optimize.fsolve
        
        t,deltaf = t_and_deltaf
        
        t is the time that corner pass point and the deltaf (unit is np.pi) is the steering angle that the vehicle use. 
    """
    v = v*forward_backward
    
    #CornerA_at_t(self, t_and_deltaf=(50, 1*np.pi/180.0), v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, })
    
    t,deltaf = t_and_deltaf[0],t_and_deltaf[1]
    
    
    x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
    
    #
    lr = veh_para.get('lr', 1.7)
    lf = veh_para.get('lf', 1.7)
    lR = veh_para.get('lR', 2)
    lF = veh_para.get('lF', 2)
    w = veh_para.get('w', 2)
    
    #
    beta = np.arctan((lr*np.tan(deltaf))/(lf+lr))
    
    #phi
    phi = phi0 + t*v*np.cos(beta)*np.tan(deltaf)/(lf+lr)
    
    #CG x and CG y at moment t given delta f.
    cg_x = x0 + (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.sin(phi0 + t*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.sin(phi0 + beta))
    #
    cg_y =  y0 - (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.cos(phi0 + t*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.cos(phi0 + beta))
    
    #
    xB = cg_x - lR*np.cos(phi)+w/2.0*np.cos(phi - np.pi/2)
    yB = cg_y - lR*np.sin(phi)+w/2.0*np.sin(phi - np.pi/2)
    
    #
    return (xB - point[0], yB-point[1])




def CORNER_C_SOLVE_fun(t_and_deltaf, point = (10, 10),v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, forward_backward = 1):
    """
    THe functions that:
        Coor_B_x - x = 0
        Coor_B_y - y = 0
    
    THis function will be used to solve the deltaf, or steering angle of front wheel. 
    
    Where Coor_B_x is the  x coordinate of corner B, and Coor_B_y is the y coordinate of corner B. 
    
    The (x,y) in the equation is given by point (=(x,y)). 
    
    ------------------------------------------
    @input: point
        x,y = point
        
        the given point that the path of corner B will path. 
    
    @input: v
        the default of the vehile. it is the state in the Rajami vehicle model. 
    
    @input: t_and_deltaf
        the two variables that need to solve using scipy.optimize.fsolve
        
        t,deltaf = t_and_deltaf
        
        t is the time that corner pass point and the deltaf (unit is np.pi) is the steering angle that the vehicle use. 
    """
    v = v*forward_backward
    
    
    #CornerA_at_t(self, t_and_deltaf=(50, 1*np.pi/180.0), v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, })
    
    t,deltaf = t_and_deltaf[0],t_and_deltaf[1]
    
    
    x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
    
    #
    lr = veh_para.get('lr', 1.7)
    lf = veh_para.get('lf', 1.7)
    lR = veh_para.get('lR', 2)
    lF = veh_para.get('lF', 2)
    w = veh_para.get('w', 2)
    
    #
    beta = np.arctan((lr*np.tan(deltaf))/(lf+lr))
    
    #phi
    phi = phi0 + t*v*np.cos(beta)*np.tan(deltaf)/(lf+lr)
    
    #CG x and CG y at moment t given delta f.
    cg_x = x0 + (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.sin(phi0 + t*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.sin(phi0 + beta))
    #
    cg_y =  y0 - (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.cos(phi0 + t*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.cos(phi0 + beta))
    
    #
    xC = cg_x - lR*np.cos(phi)+w/2.0*np.cos(phi + np.pi/2)
    yC = cg_y - lR*np.sin(phi)+w/2.0*np.sin(phi + np.pi/2)
    
    #
    return (xC - point[0], yC-point[1])

def CORNER_D_SOLVE_fun(t_and_deltaf, point = (10, 10),v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, forward_backward = 1):
    """
    THe functions that:
        Coor_A_x - x = 0
        Coor_A_y - y = 0
    
    THis function will be used to solve the deltaf, or steering angle of front wheel. 
    
    Where Coor_A_x is the  x coordinate of corner A, and Coor_A_y is the y coordinate of corner A. 
    
    The (x,y) in the equation is given by point (=(x,y)). 
    
    ------------------------------------------
    @input: point
        x,y = point
        
        the given point that the path of corner A will path. 
    
    @input: v
        the default of the vehile. it is the state in the Rajami vehicle model. 
    
    @input: t_and_deltaf
        the two variables that need to solve using scipy.optimize.fsolve
        
        t,deltaf = t_and_deltaf
        
        t is the time that corner pass point and the deltaf (unit is np.pi) is the steering angle that the vehicle use. 
    """
    v = v*forward_backward
    
    #CornerA_at_t(self, t_and_deltaf=(50, 1*np.pi/180.0), v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, })
    
    t,deltaf = t_and_deltaf[0],t_and_deltaf[1]
    
    
    x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
    
    #
    lr = veh_para.get('lr', 1.7)
    lf = veh_para.get('lf', 1.7)
    lR = veh_para.get('lR', 2)
    lF = veh_para.get('lF', 2)
    w = veh_para.get('w', 2)
    
    #
    beta = np.arctan((lr*np.tan(deltaf))/(lf+lr))
    
    #phi
    phi = phi0 + t*v*np.cos(beta)*np.tan(deltaf)/(lf+lr)
    
    #CG x and CG y at moment t given delta f.
    cg_x = x0 + (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.sin(phi0 + t*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.sin(phi0 + beta))
    #
    cg_y =  y0 - (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.cos(phi0 + t*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.cos(phi0 + beta))
    
    #
    xD = cg_x + lF*np.cos(phi)+w/2.0*np.cos(phi + np.pi/2)
    yD = cg_y + lF*np.sin(phi)+w/2.0*np.sin(phi + np.pi/2)
    
    #
    return (xD - point[0], yD-point[1])



def H_jacobian(x, H):
    """
    the jacobian of observation with respect to the state variable. 
    """
    
    return H






def Hx(x, H):
    """
    function which takes as input the state variable (self.x) along with the optional arguments in hx_args, and returns the measurement that would correspond to that state.
    i.e. the observation equation: 
        z = Hx + noise
    --------------------
    @input: H
        observation matrix. , np.array. 
        Shape is (2, 7)
        
        7 means:
            - four state variables. 
            - three control input variables. 
    
    @input: X
        the state variable. a np.arrray, whose columns number is 1. 
    """
    
    return np.dot(H,x)

def run_ekf(samples, veh_model, F_extended, P_extened, Q_extened, R, H_extended, states_name = ['x', 'y', 'phi', 'v'], a_MAX = 3.5, steer_front_limit = 45*np.pi/180.0):
    """
    
    @input: samples
        samples = (xs, ys)
        
    @input: P_extended
        np.array, the shape is (7,7)
        7 means:
            - four state variables: (x0, y0, phi0, v0)
            - three control inputs: (a, front_steer_angle, rear_steer_angle)
    @input: H_extended
        np.array, the shape is (2,7)
        2 means two observations:
            - x and y
        7 means:
            - four state variables: (x0, y0, phi0, v0)
            - three control inputs: (a, front_steer_angle, rear_steer_angle)
        
    @OUTPUT: 
    
    """
    #initialize. 
    ekf_hdv = EKF_HDV()
    ekf_hdv.R = R
    ekf_hdv._I = np.eye(7)
    
    ts,xs,ys = samples
    
    #===============Initial extened states in X0_extended
    #-------------X0, in (x0, y0, phi0, v0)
    x0 = xs[1];y0 = ys[1]
    phi0 = np.arctan((ys[1]-ys[0])/(xs[1]-xs[0]))
    v0 = np.sqrt((ys[1]-ys[0])**2+(xs[1]-xs[0])**2)/(ts[1]-ts[0])
    #   u0, the initial control input, u0 = (a, front_steer_angle, rear_steer_angle)
    u0 = [0, phi0, 0]
    X0_extended = [x0, y0, phi0, v0] + u0
    
    estimation_res = []
    real_res = []
    #======================================
    P_ex_realtime = P_extened
    X_last = [x0, y0, phi0, v0]
    u_last = u0
    laste_idx = 2
    #For each observation ( x_observation,y_observation)
    for i,(x_observation,y_observation) in enumerate(zip(xs, ys)):
        if i<3:continue
        #if np.sqrt((xs[i]-xs[laste_idx])**2 + (ys[i]-ys[laste_idx])**2)<=.5:
        #   continue
        #-----------Predict step
        #   X is a 1d arra, shape is (4,), 4 is for normal state.
        #   P_new is an array, shape is (4, 4)
        #builtins.tmp = (i, ts[i+2]-ts[i+1], X_last, u_last)
        X_pred,P_new = ekf_hdv.predict(veh_model = veh_model, X0_extended = X_last+u_last, F_extended = F_extended, P_extended = P_ex_realtime, Q_extened = Q_extened, t = ts[i]-ts[laste_idx], states_name = states_name)
        #self.predict(self, veh_model, u, X0_extended, F_extended, P_extended, Q_extened, t = 2,states_name = ['x', 'y', 'phi', 'v']):
        
        #----------update step.
        ekf_hdv.update_ekf(ekf = ekf_hdv, observation = (x_observation,y_observation), H_extended = H_extended)
        
        #
        P_ex_realtime = ekf_hdv.P
        #normalize the input. 
        normalized_x = Normalize_StatesExtended(ekf_hdv.x)
        #ekf_hdv.x = normalized_x
        
        
        X_last = list(normalized_x[:4])
        u_last = list(normalized_x[4:])
        
        print('Iterations------->', i)
        estimation_res.append((ekf_hdv.x[0], ekf_hdv.x[1], ekf_hdv.P))
        laste_idx = i
        builtins.tmp1 = estimation_res
        
    return estimation_res
    

def Normalize_StatesExtended(X, a_MAX = 3.5, delta_f_LIMIT = np.pi/4):
    """
    X[0]--->x
    X[1]--->y
    X[2]--->phi
    X[3]--->v
    X[4]--->a
    X[5]--->front Steer
    X[6]--->Rear Steer. 
    
    """
    X1 = copy.deepcopy(X)
    
    
    #X1[3]------------------------------------
    X1[3] = max(0, X1[3])
    
    #X1[4]
    X1[4] = max(min(X1[4], a_MAX), -a_MAX)
    
    #-X1[5]-------
    if X1[5]>delta_f_LIMIT:
        X1[5]=delta_f_LIMIT
    if X1[5]<-delta_f_LIMIT:
        X1[5] = -delta_f_LIMIT
    
    X1[6] = 0

    return X1

def Evaluate2array(matrix_sympy, values_dict):
    """
    Evaluate the sympy expression and convert it to the array. 
    
    """
    g = matrix_sympy.subs(values_dict)
    
    return np.array(g).astype(np.float64)


def VehicleKineticModelRajamani(self, t, STATES, U, params):
    """
    THE VEHICLE IS FRONT STEER. 
    
    Return the derivate of the X,Y and PHI. 
    X is the horizontal coordinate
    Y is the vertical coordinate
    PHI is the heading angle. 
    -------------------------------------------
    @input: U
        the control input of the model. 
        len(U)=2, 
        front_steer_angle = U[0], domain is [0,2*np.pi]
        
        V = U[1], the speed of the center of gravity of the vehicle, m/s
    
    @input: params
        the parameters of the vehicle. 
    
    
    @input: lf and lr
        unit is meter. 
        lf is the lengh of front. i.e. the distance between front axel to the CG. 
        lr is the rear length, or the ditance betweene rear axel to the CG. 
    
    @OUTPUT: STATES
        len(STATES)=3:
            - X = STATES[0], the X of the CG. X is the horizontal axis. 
            - Y = STATES[1], the Y of the CG
            - PHI = STATES[2], the heading angle. between the vehicle and the X axis. 
    """
    lr = params.get('lr', 2)
    lf = params.get('lf', 2)
    
    #control input
    front_steer_angle = U[0]
    V = U[1]
    
    #states variable. 
    X = STATES[0]
    Y = STATES[1]
    PHI = STATES[2]
    
    #beta, the intermediate parameter
    #print(lr,lf, front_steer_angle, np.tan(front_steer_angle))
    tmp = lr*np.tan(front_steer_angle)/(lr+lf)
    beta = np.arctan(tmp)
    
    #_diff means differential 
    diff_X = V*np.cos( PHI + beta)
    diff_Y = V*np.sin( PHI + beta)
    diff_PHI = V*np.cos(beta)/(lr+lf)*np.tan(front_steer_angle)
        
    return [diff_X,diff_Y,diff_PHI]
    


class CumulativeLine():
    """
    The line that keeps increasing.
    """
    def __init__(self,ts,xs):
        
        #delete the repeat ts, keep the first repeated value.
        idx_es = np.array(ts)>-np.inf
        tmp = np.where(np.diff(ts)==0)[0]
        if len(tmp)>0:
            for i in tmp:idx_es[i]= False
            ts0 = list(np.array(ts)[idx_es])
            xs0 = list(np.array(xs)[idx_es])
        else:
            ts0 = ts
            xs0 = xs
        
        if not np.all(np.diff(ts0) > 0):
            builtins.ts = ts0
            builtins.xs = xs0
            raise ValueError('ts is not strict increasing')
            
        if not np.all(np.diff(xs0) >= 0):
            builtins.ts = ts0
            builtins.xs = xs0
            raise ValueError('xs is not increasing')
        
        self.ts = ts0
        self.xs = xs0
    
    def InsertN(self, l):
        """
        
        """
        if self.xs[-1]<l or self.xs[0]>l:
            raise ValueError('input N is greater than maximal. self.xs[-1] = ', self.xs[-1],', self.xs[0]=', self.xs[0], ', l = ',l)
        
        if l in self.xs:
            return
        
        #insert the t
        #   find the idx of t in self.ts such that
        #   self.ts[idx] < t < self.ts[idx+1]
        idx = np.searchsorted(self.xs,l)-1
        #   compute the N and the CCC
        delta_t = 1.0*self.ts[idx+1] - self.ts[idx]
        delta_x = 1.0*self.xs[idx+1] - self.xs[idx]
        new_t = self.ts[idx]+delta_t*(l-self.xs[idx])/delta_x
        #   insert the t in self.ts self.N and self.CCC
        self.ts.insert(idx+1, new_t)
        self.xs.insert(idx+1, l)
        
    
    def InsertMoment(self,t):
        """
        If t is outside the time domain
            at the left, insert the self.xs[0]
            at the right, insert the self.xs[-1].
        
        """
        if t<self.ts[0]:
            self.ts.insert(0,t)
            self.xs.insert(0,self.xs[0])
            return
        
        if t>self.ts[-1]:
            self.ts.append(t)
            self.xs.append(self.xs[-1])
            return
        
        #more than 1 (include 1) values exist that self.N == n
        if t in self.ts:
            #sum(np.array(self.ts) == t)>=1:
            return
            #if more than one value that N(t) =n
            idx = np.where(np.array(self.N) == n)[0]
        
        #insert the t
        #   find the idx of t in self.ts such that
        #   self.ts[idx] < t < self.ts[idx+1]
        idx = np.searchsorted(self.ts,t)-1
        #   compute the N and the CCC
        delta_t = 1.0*self.ts[idx+1] - self.ts[idx]
        delta_x = 1.0*self.xs[idx+1] - self.xs[idx]
        new_x = self.xs[idx]+delta_x/delta_t*(t-self.ts[idx])
        #   insert the t in self.ts self.N and self.CCC
        self.ts.insert(idx+1, t)
        self.xs.insert(idx+1, new_x)
    
    def Interpolate(self,t):
        """
        If t is outside the time domain
            at the left, return the self.xs[0]
            at the right, return the self.xs[-1].
        other interpolate the value and return it. 
        """
        if t<self.ts[0]:
            return self.xs[0]
        
        if t>self.ts[-1]:
            return self.xs[-1]
        
        #more than 1 (include 1) values exist that self.N == n
        if t in self.ts:
            idx = np.where(np.array(self.ts) == t)[0][0]
            return self.xs[idx]
        
        #insert the t
        #   find the idx of t in self.ts such that
        #   self.ts[idx] < t < self.ts[idx+1]
        idx = np.searchsorted(self.ts,t)-1
        #   compute the N and the CCC
        delta_t = 1.0*self.ts[idx+1] - self.ts[idx]
        delta_x = 1.0*self.xs[idx+1] - self.xs[idx]
        new_x = self.xs[idx]+delta_x/delta_t*(t-self.ts[idx])
        return new_x






class IDM():
    """
    Implementation of the IDM model.
    
    """
    
    
    
    @classmethod
    def tsxs_from_deltats_vs_and_x0_optimizationresult(self, delta_ts, vs, x0, v_final, deltat = 2, T = 3600):
        """
        Using the optimization results for gap adaption. 
        
        @input: v_init, v_final
            both are float. unit is km/h. 
        
        @input x0, float. unit is m. 
            x0 is the initial location. 
        
        @input delta_ts,vs, two arrays with the same length. 
            delta_ts unit is sec
            and vs unit is m/s.
            
        @input: v_final
            the final speed of the vehicle, 
            The final means that 
            
        @input deltat
            the time interval, the unit is sec
        
        @input T
            the horizon of the time. 
            
                    
        """
        #v_init = v_init/3.6
        v_final = v_final/3.6
        
        
        xs0 = [x0] + list(np.multiply(delta_ts, vs))
        ts0 = [0] + list(delta_ts)
        
        ts = list(np.cumsum(ts0))
        xs = list(np.cumsum(xs0))
        
        while ts[-1]<T:
            ts.append(ts[-1]+deltat)
            
            xs.append(xs[-1]+deltat*v_final)
        
        
        return ts,xs
    
    
    
    
    @classmethod
    def tsxs_from_deltats_vs_and_x0(self, delta_ts, vs, x0, deltat = 2, T = 3600):
        """
        construct the xs from vs and x0, 
        
        @input x0, float. unit is m. 
            x0 is the initial location. 
        
        @input delta_ts,vs, two arrays with the same length. 
            delta_ts unit is sec
            and vs unit is m/s.
            
        @input deltat
            the time interval, the unit is sec
        
        @input T
            the horizon of the time. 
            
                    
        """
        xs0 = [x0] + list(np.multiply(delta_ts, vs))
        ts0 = [0] + list(delta_ts)
        
        ts = list(np.cumsum(ts0))
        xs = list(np.cumsum(xs))
        
        while ts[-1]<T:
            ts.append(ts[-1]+deltat)
            
            xs.append(xs[-1]+deltat*vs[-1])
        
        return ts,xs
    
    
    
    @classmethod
    def xs_from_vs_and_x0(self, vs, x0, deltat = 1, T = 3600):
        """
        construct the xs from vs and x0, 
        
        @input x0, float. unit is m. 
            x0 is the initial location. 
        
        @input vs, a list or array.
            The speeds. unit is m/s
            
        @input deltat
            the time interval, the unit is sec
            
            
                    
        """
        xs0 = [v*deltat for v in vs]
        
        xs = [x0] + list(xs0)
        
        return np.cumsum(xs)
    
    

    @classmethod
    def trajectory_follower_acc(self, x0_follower,v0_follower,ts_leading,xs_leading, deltat = 1,tao_delay = .1,vf = 50, speedcontrol_LW = 120, gapcontrol_UP = 100, k_speedcontrol = 0.3, desired_time_gap = 1.64, k2_gapcontrol = 0.23, k3_gapcontrol =0.07):
        """
        ACC mode.
        
        
        Note that the tao should greater enough. 
        
        calculate the trajectory of the following vehicle, given the leading vehicles info (by ts_leading and xs_leading).
        
        It is assumed that at the very begining moment (ts_leading[-1]), the location of the follower is x0_follower with speed v0_follower.
        
        @input ts_leading,xs_leading
            Two list of the same length. 
            ts_leading unit is sec.
            xs_leading unit is m.
        
        @input: x0_follower,v0_follower
            both are float. 
            x0_follower unit is m
            v0_follower unit is m/s
        
        
        @input: tao_delay
            the delay , unit is sec. 
        
        @input: desired_time_gap
            the desied time gap, unit is sec. 
        
        @input: time_step
            the time step of the simulation. unit is sec 
            
        @input: speedcontrol_LW,gapcontrol_UP
            both unit are m. 
            When the gap is greater than speedcontrol_LW, then the speed control mode.
            when the gap is smaller than gapcontrol_UP, then it is gap control model
            when between the two, then remain the previous mode. 
            
        @input: k_speedcontrol = 0.3, k2_gapcontrol = 0.23, k3_gapcontrol =0.07
            the coefficients in the speed control mode and at the gap control mode. 
        
        ------------------------------------------
        @OUTPUT ts_follower,xs_follower
            both are lists. the same length.
            ts_follower unit is sec.
            xs_follower unit is m.
            
        
        ------------------------------------------
        @STEPS:
            - Fill the tao
        """
        v0_follower = v0_follower/3.6
        
        
        #vs_leading corresponds to ts_leading[:-1]
        #vs_leading unit is m/sec.
        #   it is used to interploate the speed at specific moment. 
        vs_leading = np.divide(np.diff(xs_leading),np.diff(ts_leading))
        ts_leading_vs = ts_leading[:-1]
        
        ts_follower = [ts_leading[0], ts_leading[0]+tao_delay]
        xs_follower = [x0_follower, x0_follower+tao_delay*v0_follower]
        vs_follower = [v0_follower, v0_follower]
        """
        ts_follower = [ts_leading[-1]]
        xs_follower = [x0_follower]
        vs_follower = [v0_follower]
        """
        
        #the mode of acc
        lastmode = 'null'
        T_horizon = ts_follower[-1]
        #while idx<=len(ts_leading_vs):
        while T_horizon<ts_leading[-1]+tao_delay:
            
            t = ts_follower[-1]-tao_delay
            
            #calculate the acceleration
            v_leader  = Interpolate(ts_leading_vs,vs_leading, t)
            v_self =  vs_follower[-1]
            x_leader = Interpolate(ts_leading,xs_leading, t)
            x_follower = xs_follower[-1]
            #   acc_mode is either gapcontrolmode or speedcontrolmode
            a,acc_mode  = self.a_ACC_mode(x_leader=x_leader, x_follower=x_follower, v_leader=v_leader, v_follower=v_self, lastmode= lastmode,vf = vf, speedcontrol_LW = speedcontrol_LW, gapcontrol_UP = gapcontrol_UP, k_speedcontrol = k_speedcontrol, desired_time_gap = desired_time_gap, k2_gapcontrol = k2_gapcontrol, k3_gapcontrol =k3_gapcontrol)
            lastmode = acc_mode
            
            #UPDATE vs_follower and xs_follower
            vs_follower.append(vs_follower[-1]+a*deltat)
            xs_follower.append(xs_follower[-1]+1.0*vs_follower[-2]*deltat+a*deltat*deltat/2.0)
            ts_follower.append(ts_follower[-1]+deltat)
            
            T_horizon = T_horizon + deltat
        
        return ts_follower,vs_follower,xs_follower


    @classmethod
    def trajectory_historical_follower_idm_return_increment(self, xs0_follower,ts0_follower,ts_leading,xs_leading, timestep = 2, tao_delay = 1,idm_deltat = 1, idm_vf = 60, idm_T = 1.5, idm_vehicle_length = 5, idm_delta = 4, idm_s0 = 2, idm_a = 1.0,idm_b = 3.21):
        """
        NOTE that the given follower state is for a time interval, rather than single moment. 
        
        Difference between self.trajectory_historical_follower_idm_return_increment() and self.trajectory_historical_follower_idm():
            - self.trajectory_historical_follower_idm_return_increment(), only return increment of ts and xs
            - self.trajectory_historical_follower_idm(), return the whole trajectory, including the given history. 
        
        
        
        calculate the trajectory of the following vehicle, given the leading vehicles info (by ts_leading and xs_leading).
        
        It is assumed that at the very begining moment (ts_leading[-1]), the location of the follower is x0_follower with speed v0_follower.
        
        @input ts_leading,xs_leading
            Two list of the same length. 
            ts_leading unit is sec.
            xs_leading unit is m.
        
        @input: xs0_follower,ts0_follower
            both are lists. If length are both 1, then just give a single point data. . 
            xs0_follower unit is m.
            ts0_follower unit is s.
        
        @input: idm_*
            the parameters in the IDM model. 
        
        ------------------------------------------
        @OUTPUT ts_follower,xs_follower
            both are lists. the same length.
            ts_follower unit is sec.
            xs_follower unit is m.
            
            NOTE THAT:
                ts_follower[0] == ts0_follower[-1]
        
        ------------------------------------------
        @STEPS:
            - Fill the tao
        """
        
        #used to find the idx of the last moment of historycal data. 
        flag_moment  = ts0_follower[-1]
        
        #vs_leading corresponds to ts_leading[:-1]
        #vs_leading unit is m/sec.
        #   it is used to interploate the speed at specific moment. 
        vs_leading = np.divide(np.diff(xs_leading),np.diff(ts_leading))
        ts_leading_vs = ts_leading[:-1]
        
        #initialize xs_follower and ts_follower.
        if ts0_follower[-1]-ts_leading[0]<tao_delay or ts0_follower[-1]-ts0_follower[0]<tao_delay:
            v_temp = (xs0_follower[-1]-xs0_follower[-2])/(ts0_follower[-1]-ts0_follower[-2])
            xs_follower = xs0_follower+[xs0_follower[-1]+v_temp*tao_delay]
            ts_follower = ts0_follower + [ts0_follower[-1]+tao_delay]
        else:
            xs_follower = xs0_follower
            ts_follower = ts0_follower
        
        #idx is the start of the index that ts_leading[idx]>t0+tao_delay
        T_horizon = ts_follower[-1]
        #while idx<=len(ts_leading_vs):
        while T_horizon<ts_leading[-1]+tao_delay:
            t = ts_follower[-1]-tao_delay
            #calculate the acceleration
            #   self.a(self, v_self=10, v_leader = 10, deltax =20, 
            v_leader  = Interpolate(ts_leading_vs,vs_leading, t)
            v_self =  (xs_follower[-1]-xs_follower[-2])/(ts_follower[-1]-ts_follower[-2])
            deltax = Interpolate(ts_leading,xs_leading, t)-Interpolate(ts_follower,xs_follower, t)
            #
            a = self.a(v_self=v_self*3.6, v_leader=v_leader*3.6, deltax=deltax, idm_vf=idm_vf, idm_T = idm_T, vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0=idm_s0, idm_a = idm_a,idm_b = idm_b)
            
            #UPDATE vs_follower and xs_follower
            xs_follower.append(xs_follower[-1]+1.0*v_self*timestep+a*timestep*timestep/2.0)
            ts_follower.append(ts_follower[-1]+timestep)
            
            T_horizon = T_horizon + timestep
        
        #find the idx of flag_moment
        idx_flag_moment = ts_follower.index(flag_moment)
        
        return ts_follower[idx_flag_moment:],xs_follower[idx_flag_moment:]

    @classmethod
    def trajectory_historical_follower_idm(self, xs0_follower,ts0_follower,ts_leading,xs_leading, timestep = 2, tao_delay = 1,idm_deltat = 1, idm_vf = 60, idm_T = 1.5, idm_vehicle_length = 5, idm_delta = 4, idm_s0 = 2, idm_a = 1.0,idm_b = 3.21,):
        """
        NOTE that the given follower state is for a time interval, rather than single moment. 
        
        calculate the trajectory of the following vehicle, given the leading vehicles info (by ts_leading and xs_leading).
        
        It is assumed that at the very begining moment (ts_leading[-1]), the location of the follower is x0_follower with speed v0_follower.
        
        @input ts_leading,xs_leading
            Two list of the same length. 
            ts_leading unit is sec.
            xs_leading unit is m.
        
        @input: xs0_follower,ts0_follower
            both are lists. If length are both 1, then just give a single point data. . 
            xs0_follower unit is m.
            ts0_follower unit is s.
        
        @input: idm_*
            the parameters in the IDM model. 
        
        ------------------------------------------
        @OUTPUT ts_follower,xs_follower
            both are lists. the same length.
            ts_follower unit is sec.
            xs_follower unit is m.
            
        
        ------------------------------------------
        @STEPS:
            - Fill the tao
        """
        
        flag_moment  = ts0_follower[-1]
        
        #vs_leading corresponds to ts_leading[:-1]
        #vs_leading unit is m/sec.
        #   it is used to interploate the speed at specific moment. 
        vs_leading = np.divide(np.diff(xs_leading),np.diff(ts_leading))
        ts_leading_vs = ts_leading[:-1]
        
        #initialize xs_follower and ts_follower.
        if ts0_follower[-1]-ts_leading[0]<tao_delay or ts0_follower[-1]-ts0_follower[0]<tao_delay:
            v_temp = (xs0_follower[-1]-xs0_follower[-2])/(ts0_follower[-1]-ts0_follower[-2])
            xs_follower = xs0_follower+[xs0_follower[-1]+v_temp*tao_delay]
            ts_follower = ts0_follower + [ts0_follower[-1]+tao_delay]
        else:
            xs_follower = xs0_follower
            ts_follower = ts0_follower
        
        #idx is the start of the index that ts_leading[idx]>t0+tao_delay
        T_horizon = ts_follower[-1]
        #while idx<=len(ts_leading_vs):
        while T_horizon<ts_leading[-1]+tao_delay:
            t = ts_follower[-1]-tao_delay
            #calculate the acceleration
            #   self.a(self, v_self=10, v_leader = 10, deltax =20, 
            v_leader  = Interpolate(ts_leading_vs,vs_leading, t)
            v_self =  (xs_follower[-1]-xs_follower[-2])/(ts_follower[-1]-ts_follower[-2])
            deltax = Interpolate(ts_leading,xs_leading, t)-Interpolate(ts_follower,xs_follower, t)
            #
            a = self.a(v_self=v_self*3.6, v_leader=v_leader*3.6, deltax=deltax, idm_vf=idm_vf, idm_T = idm_T, vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0=idm_s0, idm_a = idm_a,idm_b = idm_b)
            
            #UPDATE vs_follower and xs_follower
            xs_follower.append(xs_follower[-1]+1.0*v_self*timestep+a*timestep*timestep/2.0)
            ts_follower.append(ts_follower[-1]+timestep)
            
            T_horizon = T_horizon + timestep
        
        return ts_follower,xs_follower



    @classmethod
    def trajectory_followers_improved_idm(self, ts_leading,xs_leading, locs, speeds, timestep = 2, tao_delay = 1,idm_deltat = 1, idm_vf = 60, idm_T = 1.5, idm_vehicle_length = 5, idm_delta = 4, idm_s0 = 2, idm_a = 1.0,idm_b = 3.21,):
        """
        Construct many followers, given the leading vehicle TR. 
        
        The leading vehicle TR is represented by ts_leading,xs_leading. The two lists are of the same length. 
        
        Initial condition is represented by locs, speeds.
        The two are lists, loc[0] is the 1st veh. THen loc[0]<xs_leading[0]. 
        --------------------------------------------------------
        @input:  locs, speeds
            both are list. 
            speed unit is km/h.. 
        
        
        
        @OUTPUT: tsxs_list
            a list. 
            tsxs_list[idx] = (ts,xs)
        """
        tsxs_list = []
        
        ts_newleading = ts_leading
        xs_newleading = xs_leading
        
        for loc,v in zip(locs,speeds):
            ts,vs,xs = self.trajectory_follower_improved_idm(x0_follower = loc,v0_follower = v,ts_leading = ts_newleading,xs_leading = xs_newleading, timestep = timestep, tao_delay = tao_delay,idm_deltat = idm_deltat, idm_vf = idm_vf, idm_T = idm_T, idm_vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0 = idm_s0, idm_a = idm_a, idm_b = idm_b)
            tsxs_list.append((ts,xs))
            
            
            ts_newleading = ts
            xs_newleading = xs
        
        return tsxs_list
    
    
    
    @classmethod
    def __trajectory_follower_improved_idm0(self, x0_follower,v0_follower,ts_leading,xs_leading, timestep = 2, tao_delay = 1,idm_deltat = 1, idm_vf = 60, idm_T = 1.5, idm_vehicle_length = 5, idm_delta = 4, idm_s0 = 2, idm_a = 1.0,idm_b = 3.21,):
        """
        Note that the tao should greater enough. 
        
        calculate the trajectory of the following vehicle, given the leading vehicles info (by ts_leading and xs_leading).
        
        It is assumed that at the very begining moment (ts_leading[-1]), the location of the follower is x0_follower with speed v0_follower.
        
        @input ts_leading,xs_leading
            Two list of the same length. 
            ts_leading unit is sec.
            xs_leading unit is m.
        
        @input: x0_follower,v0_follower
            both are float. 
            x0_follower unit is m
            v0_follower unit is m/s
        
        @input: idm_*
            the parameters in the IDM model. 
        
        ------------------------------------------
        @OUTPUT ts_follower,xs_follower
            both are lists. the same length.
            ts_follower unit is sec.
            xs_follower unit is m.
            
        
        ------------------------------------------
        @STEPS:
            - Fill the tao
        """
        v0_follower = v0_follower/3.6
        
        #vs_leading corresponds to ts_leading[:-1]
        #vs_leading unit is m/sec.
        #   it is used to interploate the speed at specific moment. 
        vs_leading = np.divide(np.diff(xs_leading),np.diff(ts_leading))
        ts_leading_vs = ts_leading[:-1]
        
        ts_follower = [ts_leading[0], ts_leading[0]+tao_delay]
        xs_follower = [x0_follower, x0_follower+tao_delay*v0_follower]
        vs_follower = [v0_follower, v0_follower]
        """
        ts_follower = [ts_leading[-1]]
        xs_follower = [x0_follower]
        vs_follower = [v0_follower]
        """
        
        #idx is the start of the index that ts_leading[idx]>t0+tao_delay
        T_horizon = ts_follower[-1]
        #while idx<=len(ts_leading_vs):
        while T_horizon<ts_leading[-1]+tao_delay:
            t = ts_follower[-1]-tao_delay
            
            #calculate the acceleration
            #   self.a(self, v_self=10, v_leader = 10, deltax =20, 
            v_leader  = Interpolate(ts_leading_vs,vs_leading, t)
            #v_self =  vs_follower[-1]
            v_self  = Interpolate(ts_follower,vs_follower, t)
            deltax = Interpolate(ts_leading,xs_leading, t)-xs_follower[-1]
            #
            a = self.a(v_self=v_self*3.6, v_leader=v_leader*3.6, deltax=deltax, idm_vf=idm_vf, idm_T = idm_T, vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0=idm_s0, idm_a = idm_a,idm_b = idm_b)
            
            #UPDATE vs_follower and xs_follower
            #Oringinal vs_follower:
            #   vs_follower.append(vs_follower[-1]+a*timestep)
            #       vs_follower.append(max(0.0, vs_follower[-1]+a*timestep))
            #@@@@@@@@@@@! note that the following line is to assure that the speed will be always positive. 
            a = max(a, -vs_follower[-1]/timestep)
            vs_follower.append(vs_follower[-1]+a*timestep)
            xs_follower.append(xs_follower[-1]+1.0*vs_follower[-2]*timestep+a*timestep*timestep/2.0)
            ts_follower.append(ts_follower[-1]+timestep)
            
            T_horizon = T_horizon + timestep
        
        return ts_follower,vs_follower,xs_follower
    
    
    
    
    @classmethod
    def trajectory_follower_improved_idm(self, x0_follower,v0_follower,ts_leading= False,xs_leading = False, T_horizon = 3600, timestep = 2, tao_delay = 1,idm_deltat = 1, idm_vf = 60, idm_T = 1.5, idm_vehicle_length = 5, idm_delta = 4, idm_s0 = 2, idm_a = 1.0,idm_b = 3.21, M_distance = 1e6):
        """
        Note that the tao should greater enough. 
        
        calculate the trajectory of the following vehicle, given the leading vehicles info (by ts_leading and xs_leading).
        
        It is assumed that at the very begining moment (ts_leading[-1]), the location of the follower is x0_follower with speed v0_follower.
        -------------------------------------------------------------
        @input: M_distance
            a number that is greater enough. 
            This is used when ts_leading==False, which means that there is no leading vehicle. 
        
        @input: T_horizon
            the time horizon of the returned trajectory. 
            This come into effect when ts_leading==False. 
        
        @input ts_leading,xs_leading
            Two list of the same length. 
            ts_leading unit is sec.
            xs_leading unit is m.
            
            If they are set to False, it means that there is no leading vehicle. 
        
        @input: x0_follower,v0_follower
            both are float. 
            x0_follower unit is m
            v0_follower unit is m/s
        
        @input: idm_*
            the parameters in the IDM model. 
        
        ------------------------------------------
        @OUTPUT ts_follower,xs_follower
            both are lists. the same length.
            ts_follower unit is sec.
            xs_follower unit is m.
            
        
        ------------------------------------------
        @STEPS:
            - Fill the tao
        """
        v0_follower = v0_follower/3.6
        
        ##################Set the ts_leading,xs_leading when there is no leading vehicle. 
        if ts_leading==False:
            ts_leading = np.linspace(0, T_horizon, int(T_horizon/timestep))
            xs_leading = [M_distance+t*idm_vf/3.6 for t in ts_leading]
        
        ##############################
        #used to interploate the speed
        #vs_leading corresponds to ts_leading[:-1]
        #vs_leading unit is m/sec.
        #   it is used to interploate the speed at specific moment. 
        vs_leading = np.divide(np.diff(xs_leading),np.diff(ts_leading))
        ts_leading_vs = ts_leading[:-1]
        
        ######initialize.
        ts_follower = [ts_leading[0], ts_leading[0]+tao_delay]
        xs_follower = [x0_follower, x0_follower+tao_delay*v0_follower]
        vs_follower = [v0_follower, v0_follower]
        """
        ts_follower = [ts_leading[-1]]
        xs_follower = [x0_follower]
        vs_follower = [v0_follower]
        """
        
        #idx is the start of the index that ts_leading[idx]>t0+tao_delay
        t_horizon = ts_follower[-1]
        #while idx<=len(ts_leading_vs):
        while t_horizon<ts_leading[-1]+tao_delay:
            t = ts_follower[-1]-tao_delay
            
            #calculate the acceleration
            #   self.a(self, v_self=10, v_leader = 10, deltax =20, 
            v_leader  = Interpolate(ts_leading_vs,vs_leading, t)
            #v_self =  vs_follower[-1]
            v_self  = Interpolate(ts_follower,vs_follower, t)
            deltax = Interpolate(ts_leading,xs_leading, t)-xs_follower[-1]
            #
            a = self.a(v_self=v_self*3.6, v_leader=v_leader*3.6, deltax=deltax, idm_vf=idm_vf, idm_T = idm_T, vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0=idm_s0, idm_a = idm_a,idm_b = idm_b)
            
            #UPDATE vs_follower and xs_follower
            #Oringinal vs_follower:
            #   vs_follower.append(vs_follower[-1]+a*timestep)
            #       vs_follower.append(max(0.0, vs_follower[-1]+a*timestep))
            #@@@@@@@@@@@! note that the following line is to assure that the speed will be always positive. 
            a = max(a, -vs_follower[-1]/timestep)
            vs_follower.append(vs_follower[-1]+a*timestep)
            xs_follower.append(xs_follower[-1]+1.0*vs_follower[-2]*timestep+a*timestep*timestep/2.0)
            ts_follower.append(ts_follower[-1]+timestep)
            
            t_horizon = t_horizon + timestep
        
        return ts_follower,vs_follower,xs_follower
    
    
    @classmethod
    def __trajectory_followers_multiple_leaders(self, locs, speeds, t0, tsxs_es_leadings, timestep = 2, tao_delay = 1,idm_deltat = 1, idm_vf = 60, idm_T = 1.5, idm_vehicle_length = 5, idm_delta = 4, idm_s0 = 2, idm_a = 1.0,idm_b = 3.21, a_MAX = 3):
        """
        Get the TR of the followers (multiple), when there are more than two leading vehicles. 
        ----------------------------------------------------------------------
        
        @input:  locs, speeds
            both are float. 
            x0_follower unit is m
            v0_follower unit is veh/h. 
        @input: t0
            the moment corresponds to locs and speeds.
        @input: tsxs_es_leadings
            a list. 
            tsxs_es_leadings[idx] = (ts,xs)
            NOTE THAT IT is assumed the tsxs are ordered in temporal ORDER:
                ts0,xs0 = tsxs_es_leadings[0]
                ts1,xs1 = tsxs_es_leadings[1]
                
                ts0[-1]>ts1[0] AND:
                ts0[-1]<ts1[-1]
                
        @input: idm_*
            the parameters in the IDM model. 
        @OUTPUT: tsxs_list
            tsxs_list[idx] = (ts,xs).
        ------------------------------------------
        @Steps:
            - 
        
        """
        #transition_moments, is a list. 
        #   it is the transitionmoments when the leading vehicle changes. 
        transition_moments = [tsxs[0][-1] for tsxs in tsxs_es_leadings]
        
        #returrned value. 
        tsxs_list = []
        
        #first follower.
        whetherisfirst = True
        for loc,v in zip(locs,speeds):
            if whetherisfirst:
                ts,xs = self.trajectory_follower_two_leaders(x0_follower=loc, v0_follower=v, t0=t0, ts_leading_1st=ts_leading_1st,xs_leading_1st=xs_leading_1st, ts_leading_2nd=ts_leading_2nd, xs_leading_2nd=xs_leading_2nd, timestep = timestep, tao_delay = tao_delay,idm_deltat = idm_deltat, idm_vf = idm_vf, idm_T = idm_T, idm_vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0 = idm_s0, idm_a = idm_a,idm_b = idm_b, a_MAX = a_MAX)
                
                tsxs_list.append((ts,xs))
                
                ts_newleading = ts
                xs_newleading = xs
                
                whetherisfirst = False
            else:
            
                ts,vs,xs = self.trajectory_follower_improved_idm(x0_follower = loc,v0_follower = v,ts_leading = ts_newleading,xs_leading = xs_newleading, timestep = timestep, tao_delay = tao_delay,idm_deltat = idm_deltat, idm_vf = idm_vf, idm_T = idm_T, idm_vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0 = idm_s0, idm_a = idm_a, idm_b = idm_b)
                tsxs_list.append((ts,xs))
                
                ts_newleading = ts
                xs_newleading = xs
        
        return tsxs_list
    
    
    @classmethod
    def trajectory_followers_two_leaders(self, locs, speeds, t0, ts_leading_1st,xs_leading_1st, ts_leading_2nd, xs_leading_2nd, timestep = 2, tao_delay = 1,idm_deltat = 1, idm_vf = 60, idm_T = 1.5, idm_vehicle_length = 5, idm_delta = 4, idm_s0 = 2, idm_a = 1.0,idm_b = 3.21, a_MAX = 3):
        """
        Get the TR of the followers (multiple), when there are two leading vehicles. 
        The difference between:
            - self.trajectory_follower_two_leaders()
            - self.trajectory_follower_two_leaders_transitionasreactiondelay()
            - self.trajectory_followers_two_leaders()
        The second one treat the transition as reaction delay. 
        The firstone only get the TR of one vehicle (its state is specidied as x0_follower, v0_follower)
        The last one get the TRs of multiple followers. 
        
        The followers firstly follower leader 1 (ts_leading_1st,xs_leading_1st) and then follow leader 2 (ts_leading_2nd, xs_leading_2nd). 
        
        #def trajectory_two_leaders(self, xs0_follower,ts0_follower,ts_leading_1st,xs1_leading_1st, ts_leading_2nd,xs1_leading_2nd, timestep = 2,, tao_delay = 1,idm_deltat = 1, idm_vf = 60, idm_T = 1.5, idm_vehicle_length = 5, idm_delta = 4, idm_s0 = 2, idm_a = 1.0,idm_b = 3.21):
        
        
        Using the IDM to get the trajectory of the follower. There are two leaders. 
            - ts_leading_1st,xs1_leading_1st. The time and spatial of the 1st vehicle;
            - ts_leading_2nd,xs1_leading_2nd. The time and spatial of the 2nd vehicle. 
        
        Note that, the ts_leading_2nd[0] must be greater than ts_leading_1st[0]+tao_delay
        ----------------------------------------------------------------------
        
        @input:  locs, speeds
            both are float. 
            x0_follower unit is m
            v0_follower unit is veh/h. 
        @input: t0
            the moment of x0_follower,t0_follower. 
        
        
        @input: idm_*
            the parameters in the IDM model. 
        @OUTPUT: tsxs_list
            tsxs_list[idx] = (ts,xs).
        ------------------------------------------
        @Steps:
            - 
        
        """
        tsxs_list = []
        
        #first follower.
        
        whetherisfirst = True
        for loc,v in zip(locs,speeds):
            if whetherisfirst:
                ts,xs = self.trajectory_follower_two_leaders(x0_follower=loc, v0_follower=v, t0=t0, ts_leading_1st=ts_leading_1st,xs_leading_1st=xs_leading_1st, ts_leading_2nd=ts_leading_2nd, xs_leading_2nd=xs_leading_2nd, timestep = timestep, tao_delay = tao_delay,idm_deltat = idm_deltat, idm_vf = idm_vf, idm_T = idm_T, idm_vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0 = idm_s0, idm_a = idm_a,idm_b = idm_b, a_MAX = a_MAX)
                
                tsxs_list.append((ts,xs))
                
                ts_newleading = ts
                xs_newleading = xs
                
                whetherisfirst = False
            else:
            
                ts,vs,xs = self.trajectory_follower_improved_idm(x0_follower = loc,v0_follower = v,ts_leading = ts_newleading,xs_leading = xs_newleading, timestep = timestep, tao_delay = tao_delay,idm_deltat = idm_deltat, idm_vf = idm_vf, idm_T = idm_T, idm_vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0 = idm_s0, idm_a = idm_a, idm_b = idm_b)
                tsxs_list.append((ts,xs))
                
                ts_newleading = ts
                xs_newleading = xs
        
        return tsxs_list

        
        #The 
        if (t0<ts_leading_1st[0]) or (t0>ts_leading_1st[-1]) or (ts_leading_2nd[0]>ts_leading_1st[-1]) or (ts_leading_2nd[-1]<ts_leading_1st[-1]):
            raise ValueError("fsdfsdfsdf", ts_leading_1st[0], ts_leading_1st[-1], ts_leading_2nd[0], ts_leading_2nd[-1])
            
        #initialize the retuened value
        ts_follower = [t0, t0+tao_delay]
        xs_follower = [x0_follower, x0_follower+tao_delay*v0_follower/3.6]
        #vs_follower = [v0_follower, v0_follower]
        
        #Variables used to interploate the speed of leading vehicles. 
        #vs_leading_** corresponds to ts_leading[:-1]
        #vs_leading unit is m/sec.
        #   it is used to interploate the speed at specific moment. 
        vs_leading_1st = np.divide(np.diff(xs_leading_1st),np.diff(ts_leading_1st))
        ts_leading_vs_1st = ts_leading_1st[:-1]
        vs_leading_2nd = np.divide(np.diff(xs_leading_2nd),np.diff(ts_leading_2nd))
        ts_leading_vs_2nd = ts_leading_2nd[:-1]
        
        T_horizon = ts_follower[-1]
        while T_horizon<=ts_leading_1st[-1]+tao_delay:
            #the moment when the 
            t = ts_follower[-1]-tao_delay
            #calculate the acceleration
            #   self.a(self, v_self=10, v_leader = 10, deltax =20, 
            v_leader  = Interpolate(ts_leading_vs_1st,vs_leading_1st, t)
            v_self =  (xs_follower[-1]-xs_follower[-2])/(ts_follower[-1]-ts_follower[-2])
            deltax = abs(Interpolate(ts_leading_1st,xs_leading_1st, t)-Interpolate(ts_follower,xs_follower, t))
            
            #unit is m/s2.
            a = self.a(v_self=v_self*3.6, v_leader=v_leader*3.6, deltax=deltax, idm_vf=idm_vf, idm_T = idm_T, vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0=idm_s0, idm_a = idm_a,idm_b = idm_b)
            
            #UPDATE vs_follower and xs_follower
            #@@@@@@@@@@@! note that the following line is to assure that the speed will be always positive. 
            a = max(a, -1.0*v_self/timestep)
            xs_follower.append(xs_follower[-1]+1.0*v_self*timestep+a*timestep*timestep/2.0)
            ts_follower.append(ts_follower[-1]+timestep)
            
            T_horizon = T_horizon + timestep
            
        while T_horizon<=ts_leading_2nd[-1]+tao_delay:
            t = ts_follower[-1]-tao_delay
            #calculate the acceleration
            #   self.a(self, v_self=10, v_leader = 10, deltax =20, 
            v_leader  = Interpolate(ts_leading_vs_2nd,vs_leading_2nd, t)
            v_self =  (xs_follower[-1]-xs_follower[-2])/(ts_follower[-1]-ts_follower[-2])
            deltax = Interpolate(ts_leading_2nd,xs_leading_2nd, t)-Interpolate(ts_follower,xs_follower, t)
            #
            a = self.a(v_self=v_self*3.6, v_leader=v_leader*3.6, deltax=deltax, idm_vf=idm_vf, idm_T = idm_T, vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0=idm_s0, idm_a = idm_a,idm_b = idm_b)
            
            #UPDATE vs_follower and xs_follower
            #@@@@@@@@@@@! note that the following line is to assure that the speed will be always positive.
            a = max(a, -1.0*v_self/timestep)
            xs_follower.append(xs_follower[-1]+1.0*v_self*timestep+a*timestep*timestep/2.0)
            ts_follower.append(ts_follower[-1]+timestep)
            
            T_horizon = T_horizon + timestep
            
        return ts_follower,xs_follower
    
    
    
    @classmethod
    def trajectory_followers_multiple_leaders(self, locs, speeds, tsxs_leadings, t0= 0, timestep = 2, tao_delay = 1,idm_deltat = 1, idm_vf = 60, idm_T = 1.5, idm_vehicle_length = 5, idm_delta = 4, idm_s0 = 2, idm_a = 1.0,idm_b = 3.21):
        """
        Get the TR of the followers, when there are two leading vehicles. 
        The difference between:
        ----------------------------------------------------------------------
        @input: t0
            the moment of x0_follower,t0_follower. 
            
        @input:  locs, speeds
            both are list. 
            speed unit is km/h.. 
        
            Note that the loc[0] is the first follower of tsxs_leadings
        
        @input: tsxs_leadings
            a list which represents the leading vehicles. 
            tsxs_leadings[idx]  = (ts,xs)
            
            Note that the temporal of TRs in tsxs_leadings is in order. And it assumed that the TRs in tsxs_leadings is sorted in downstream order. 
                ts0,xs0 = tsxs_leadings[0]
                ts1,xs1 = tsxs_leadings[1]
                
                ts0[-1]>ts1[0] AND:
                ts0[-1]<ts1[-1]
        
        @input: idm_*
            the parameters in the IDM model. 
        ------------------------------------------
        @Steps:
            - 
        
        """
        #returrned value. 
        tsxs_list = []
        
        #first follower.
        whetherisfirst = True
        for loc,v in zip(locs,speeds):
            if whetherisfirst:
                ts,xs = self.trajectory_follower_multi_leaders(x0_follower = loc, v0_follower = v, tsxs_leadings = tsxs_leadings, t0= t0, timestep = timestep, tao_delay = tao_delay, idm_deltat = idm_deltat, idm_vf = idm_vf, idm_T = idm_T, idm_vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0 = idm_s0, idm_a = idm_a, idm_b = idm_b)
                
                tsxs_list.append((ts,xs))
                
                ts_newleading = ts
                xs_newleading = xs
                
                whetherisfirst = False
            else:
            
                ts,vs,xs = self.trajectory_follower_improved_idm(x0_follower = loc,v0_follower = v,ts_leading = ts_newleading,xs_leading = xs_newleading, timestep = timestep, tao_delay = tao_delay,idm_deltat = idm_deltat, idm_vf = idm_vf, idm_T = idm_T, idm_vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0 = idm_s0, idm_a = idm_a, idm_b = idm_b)
                tsxs_list.append((ts,xs))
                
                ts_newleading = ts
                xs_newleading = xs
        
        return tsxs_list
    
    @classmethod
    def trajectory_follower_multi_leaders(self, x0_follower, v0_follower, tsxs_leadings, t0= 0, timestep = 2, tao_delay = 1,idm_deltat = 1, idm_vf = 60, idm_T = 1.5, idm_vehicle_length = 5, idm_delta = 4, idm_s0 = 2, idm_a = 1.0,idm_b = 3.21, a_MAX = 3):
        """
        Get the TR of the follower, when there are two leading vehicles. 
        The difference between:
            - self.trajectory_follower_two_leaders()
            - self.trajectory_follower_two_leaders_transitionasreactiondelay()
        The latter one treat the transition as reaction delay. 
        
        The followers firstly follower leader 1 (ts_leading_1st,xs_leading_1st) and then follow leader 2 (ts_leading_2nd, xs_leading_2nd). 
        
        #def trajectory_two_leaders(self, xs0_follower,ts0_follower,ts_leading_1st,xs1_leading_1st, ts_leading_2nd,xs1_leading_2nd, timestep = 2,, tao_delay = 1,idm_deltat = 1, idm_vf = 60, idm_T = 1.5, idm_vehicle_length = 5, idm_delta = 4, idm_s0 = 2, idm_a = 1.0,idm_b = 3.21):
        
        
        Using the IDM to get the trajectory of the follower. There are two leaders. 
            - ts_leading_1st,xs1_leading_1st. The time and spatial of the 1st vehicle;
            - ts_leading_2nd,xs1_leading_2nd. The time and spatial of the 2nd vehicle. 
        
        Note that, the ts_leading_2nd[0] must be greater than ts_leading_1st[0]+tao_delay
        ----------------------------------------------------------------------
        
        @input: x0_follower,t0_follower
            both are float. 
            x0_follower unit is m
            v0_follower unit is veh/h. 
        @input: t0
            the moment of x0_follower,t0_follower. 
        
        @input: tsxs_leadings
            a list. 
            tsxs_leadings[idx]  = (ts,xs)
            
            Note that the temporal of TRs in tsxs_leadings is in order. 
                ts0,xs0 = tsxs_leadings[0]
                ts1,xs1 = tsxs_leadings[1]
                
                ts0[-1]>ts1[0] AND:
                ts0[-1]<ts1[-1]
        
        @input: idm_*
            the parameters in the IDM model. 
        ------------------------------------------
        @Steps:
            - 
        
        """
        
        #initialize the retuened value
        ts_follower = [t0, t0+tao_delay]
        xs_follower = [x0_follower, x0_follower+tao_delay*v0_follower/3.6]
        #vs_follower = [v0_follower, v0_follower]
        
        ######################################
        #Used to interploate the speed, which is stored in tsvs_leadings
        #   tsvs_leadings[idx] = (ts,xs)
        tsvs_leadings = []
        for TR in tsxs_leadings:
            ts,xs = TR
            vs = np.divide(np.diff(xs),np.diff(ts))
            tsvs_leadings.append((ts[1:], vs))
        
        #The transition moments of the multiple leading trajetories. 
        transitionmoments = [TR[0][-1] for TR in tsxs_leadings]
        
        #find the idx of the leading vehicle. The idx is in tsxs_leadings
        #   np.searchsorted([0,1,2,3,4,5],1) return 1
        #   np.searchsorted([0,1,2,3,4,5],1.9) return 2
        idx_leading_1st = np.searchsorted(transitionmoments,ts_follower[-1])
        t_horizon = ts_follower[-1]
        for idx_leading in range(idx_leading_1st, len(tsxs_leadings)):
            ts_leading,xs_leading = tsxs_leadings[idx_leading]
            while t_horizon<=ts_leading[-1]:
                #the moment when the 
                t = ts_follower[-1]-tao_delay
                v_leader  = Interpolate(tsvs_leadings[idx_leading][0],tsvs_leadings[idx_leading][1], t)
                v_self =  (xs_follower[-1]-xs_follower[-2])/(ts_follower[-1]-ts_follower[-2])
                deltax = abs(Interpolate(ts_leading,xs_leading, t)-Interpolate(ts_follower,xs_follower, t))
                #unit is m/s2.
                a = self.a(v_self=v_self*3.6, v_leader=v_leader*3.6, deltax=deltax, idm_vf=idm_vf, idm_T = idm_T, vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0=idm_s0, idm_a = idm_a,idm_b = idm_b)
                
                #UPDATE vs_follower and xs_follower
                #@@@@@@@@@@@! note that the following line is to assure that the speed will be always positive. 
                a = max(a, -1.0*v_self/timestep)
                xs_follower.append(xs_follower[-1]+1.0*v_self*timestep+a*timestep*timestep/2.0)
                ts_follower.append(ts_follower[-1]+timestep)
                
                t_horizon = t_horizon + timestep

        return ts_follower,xs_follower
    
    @classmethod
    def trajectory_follower_two_leaders(self, x0_follower, v0_follower, t0, ts_leading_1st,xs_leading_1st, ts_leading_2nd, xs_leading_2nd, timestep = 2, tao_delay = 1,idm_deltat = 1, idm_vf = 60, idm_T = 1.5, idm_vehicle_length = 5, idm_delta = 4, idm_s0 = 2, idm_a = 1.0,idm_b = 3.21, a_MAX = 3):
        """
        Get the TR of the follower, when there are two leading vehicles. 
        The difference between:
            - self.trajectory_follower_two_leaders()
            - self.trajectory_follower_two_leaders_transitionasreactiondelay()
        The latter one treat the transition as reaction delay. 
        
        The followers firstly follower leader 1 (ts_leading_1st,xs_leading_1st) and then follow leader 2 (ts_leading_2nd, xs_leading_2nd). 
        
        #def trajectory_two_leaders(self, xs0_follower,ts0_follower,ts_leading_1st,xs1_leading_1st, ts_leading_2nd,xs1_leading_2nd, timestep = 2,, tao_delay = 1,idm_deltat = 1, idm_vf = 60, idm_T = 1.5, idm_vehicle_length = 5, idm_delta = 4, idm_s0 = 2, idm_a = 1.0,idm_b = 3.21):
        
        
        Using the IDM to get the trajectory of the follower. There are two leaders. 
            - ts_leading_1st,xs1_leading_1st. The time and spatial of the 1st vehicle;
            - ts_leading_2nd,xs1_leading_2nd. The time and spatial of the 2nd vehicle. 
        
        Note that, the ts_leading_2nd[0] must be greater than ts_leading_1st[0]+tao_delay
        ----------------------------------------------------------------------
        
        @input: x0_follower,t0_follower
            both are float. 
            x0_follower unit is m
            v0_follower unit is veh/h. 
        @input: t0
            the moment of x0_follower,t0_follower. 
        
        
        @input: idm_*
            the parameters in the IDM model. 
        ------------------------------------------
        @Steps:
            - 
        
        """
        #The 
        if (t0<ts_leading_1st[0]) or (t0>ts_leading_1st[-1]) or (ts_leading_2nd[0]>ts_leading_1st[-1]) or (ts_leading_2nd[-1]<ts_leading_1st[-1]):
            raise ValueError("fsdfsdfsdf", ts_leading_1st[0], ts_leading_1st[-1], ts_leading_2nd[0], ts_leading_2nd[-1])
            
        #initialize the retuened value
        ts_follower = [t0, t0+tao_delay]
        xs_follower = [x0_follower, x0_follower+tao_delay*v0_follower/3.6]
        #vs_follower = [v0_follower, v0_follower]
        
        #Variables used to interploate the speed of leading vehicles. 
        #vs_leading_** corresponds to ts_leading[:-1]
        #vs_leading unit is m/sec.
        #   it is used to interploate the speed at specific moment. 
        vs_leading_1st = np.divide(np.diff(xs_leading_1st),np.diff(ts_leading_1st))
        ts_leading_vs_1st = ts_leading_1st[:-1]
        vs_leading_2nd = np.divide(np.diff(xs_leading_2nd),np.diff(ts_leading_2nd))
        ts_leading_vs_2nd = ts_leading_2nd[:-1]
        
        T_horizon = ts_follower[-1]
        while T_horizon<=ts_leading_1st[-1]+tao_delay:
            #the moment when the 
            t = ts_follower[-1]-tao_delay
            #calculate the acceleration
            #   self.a(self, v_self=10, v_leader = 10, deltax =20, 
            v_leader  = Interpolate(ts_leading_vs_1st,vs_leading_1st, t)
            v_self =  (xs_follower[-1]-xs_follower[-2])/(ts_follower[-1]-ts_follower[-2])
            deltax = abs(Interpolate(ts_leading_1st,xs_leading_1st, t)-Interpolate(ts_follower,xs_follower, t))
            
            #unit is m/s2.
            a = self.a(v_self=v_self*3.6, v_leader=v_leader*3.6, deltax=deltax, idm_vf=idm_vf, idm_T = idm_T, vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0=idm_s0, idm_a = idm_a,idm_b = idm_b)
            
            #UPDATE vs_follower and xs_follower
            #@@@@@@@@@@@! note that the following line is to assure that the speed will be always positive. 
            a = max(a, -1.0*v_self/timestep)
            xs_follower.append(xs_follower[-1]+1.0*v_self*timestep+a*timestep*timestep/2.0)
            ts_follower.append(ts_follower[-1]+timestep)
            
            T_horizon = T_horizon + timestep
            
        while T_horizon<=ts_leading_2nd[-1]+tao_delay:
            t = ts_follower[-1]-tao_delay
            #calculate the acceleration
            #   self.a(self, v_self=10, v_leader = 10, deltax =20, 
            v_leader  = Interpolate(ts_leading_vs_2nd,vs_leading_2nd, t)
            v_self =  (xs_follower[-1]-xs_follower[-2])/(ts_follower[-1]-ts_follower[-2])
            deltax = Interpolate(ts_leading_2nd,xs_leading_2nd, t)-Interpolate(ts_follower,xs_follower, t)
            #
            a = self.a(v_self=v_self*3.6, v_leader=v_leader*3.6, deltax=deltax, idm_vf=idm_vf, idm_T = idm_T, vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0=idm_s0, idm_a = idm_a,idm_b = idm_b)
            
            #UPDATE vs_follower and xs_follower
            #@@@@@@@@@@@! note that the following line is to assure that the speed will be always positive.
            a = max(a, -1.0*v_self/timestep)
            xs_follower.append(xs_follower[-1]+1.0*v_self*timestep+a*timestep*timestep/2.0)
            ts_follower.append(ts_follower[-1]+timestep)
            
            T_horizon = T_horizon + timestep
            
        return ts_follower,xs_follower
    
    
    
    
    
    
    
    @classmethod
    def trajectory_follower_two_leaders_transitionasreactiondelay(self, xs0_follower, ts0_follower,ts_leading_1st,xs_leading_1st, ts_leading_2nd, xs_leading_2nd, timestep = 2, tao_delay = 1,idm_deltat = 1, idm_vf = 60, idm_T = 1.5, idm_vehicle_length = 5, idm_delta = 4, idm_s0 = 2, idm_a = 1.0,idm_b = 3.21):
        """
        Get the TR of the follower, when there are two leading vehicles. 
        The transition among the two leaders are treated using reaction delay. 
        
        
        
        #def trajectory_two_leaders(self, xs0_follower,ts0_follower,ts_leading_1st,xs1_leading_1st, ts_leading_2nd,xs1_leading_2nd, timestep = 2,, tao_delay = 1,idm_deltat = 1, idm_vf = 60, idm_T = 1.5, idm_vehicle_length = 5, idm_delta = 4, idm_s0 = 2, idm_a = 1.0,idm_b = 3.21):
        
        
        Using the IDM to get the trajectory of the follower. There are two leaders. 
            - ts_leading_1st,xs1_leading_1st. The time and spatial of the 1st vehicle;
            - ts_leading_2nd,xs1_leading_2nd. The time and spatial of the 2nd vehicle. 
        
        Note that, the ts_leading_2nd[0] must be greater than ts_leading_1st[0]+tao_delay
        ----------------------------------------------------------------------
        
        @input: xs0_follower,ts0_follower
            both are list. 
            x0_follower unit is m
            v0_follower unit is m/s
        
        @input: idm_*
            the parameters in the IDM model. 
        ------------------------------------------
        @Steps:
            - 
        
        """
        #
        if ts_leading_2nd[0]<(ts_leading_1st[0]+tao_delay):
            raise ValueError("fsdfsdfsdf", ts_leading_1st[0], ts_leading_2nd[0])
        
        #Variables used to interploate the speed of leading vehicles. 
        #vs_leading_** corresponds to ts_leading[:-1]
        #vs_leading unit is m/sec.
        #   it is used to interploate the speed at specific moment. 
        vs_leading_1st = np.divide(np.diff(xs_leading_1st),np.diff(ts_leading_1st))
        ts_leading_vs_1st = ts_leading_1st[:-1]
        vs_leading_2nd = np.divide(np.diff(xs_leading_2nd),np.diff(ts_leading_2nd))
        ts_leading_vs_2nd = ts_leading_2nd[:-1]
        
        #initialize xs_follower and ts_follower.
        if ts0_follower[-1]-ts0_follower[0]<tao_delay:
            #v_temp is the speed of the follower at the begining. 
            v_temp = (xs0_follower[-1]-xs0_follower[-2])/(ts0_follower[-1]-ts0_follower[-2])
            xs_follower = xs0_follower+[xs0_follower[-1]+v_temp*tao_delay]
            ts_follower = ts0_follower + [ts0_follower[-1]+tao_delay]
        else:
            xs_follower = xs0_follower
            ts_follower = ts0_follower
        
        T_horizon = ts_follower[-1]
        while T_horizon<ts_leading_2nd[0]+tao_delay:
            t = ts_follower[-1]-tao_delay
            #calculate the acceleration
            #   self.a(self, v_self=10, v_leader = 10, deltax =20, 
            v_leader  = Interpolate(ts_leading_vs_1st,vs_leading_1st, t)
            v_self =  (xs_follower[-1]-xs_follower[-2])/(ts_follower[-1]-ts_follower[-2])
            deltax = Interpolate(ts_leading_1st,xs_leading_1st, t)-Interpolate(ts_follower,xs_follower, t)
            #
            a = self.a(v_self=v_self*3.6, v_leader=v_leader*3.6, deltax=deltax, idm_vf=idm_vf, idm_T = idm_T, vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0=idm_s0, idm_a = idm_a,idm_b = idm_b)
            
            #UPDATE vs_follower and xs_follower
            xs_follower.append(xs_follower[-1]+1.0*v_self*timestep+a*timestep*timestep/2.0)
            ts_follower.append(ts_follower[-1]+timestep)
            
            T_horizon = T_horizon + timestep
            
        while T_horizon<ts_leading_2nd[-1]+tao_delay:
            t = ts_follower[-1]-tao_delay
            #calculate the acceleration
            #   self.a(self, v_self=10, v_leader = 10, deltax =20, 
            v_leader  = Interpolate(ts_leading_vs_2nd,vs_leading_2nd, t)
            v_self =  (xs_follower[-1]-xs_follower[-2])/(ts_follower[-1]-ts_follower[-2])
            deltax = Interpolate(ts_leading_2nd,xs_leading_2nd, t)-Interpolate(ts_follower,xs_follower, t)
            #
            a = self.a(v_self=v_self*3.6, v_leader=v_leader*3.6, deltax=deltax, idm_vf=idm_vf, idm_T = idm_T, vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0=idm_s0, idm_a = idm_a,idm_b = idm_b)
            
            #UPDATE vs_follower and xs_follower
            xs_follower.append(xs_follower[-1]+1.0*v_self*timestep+a*timestep*timestep/2.0)
            ts_follower.append(ts_follower[-1]+timestep)
            
            T_horizon = T_horizon + timestep
            
        return ts_follower,xs_follower


    @classmethod
    def trajectory_follower(self, x0_follower,v0_follower,ts_leading,xs_leading, tao_delay = 1,idm_deltat = 1, idm_vf = 60, idm_T = 1.5, idm_vehicle_length = 5, idm_delta = 4, idm_s0 = 2, idm_a = 1.0,idm_b = 3.21,):
        """
        Note that the tao should greater enough. 
        
        calculate the trajectory of the following vehicle, given the leading vehicles info (by ts_leading and xs_leading).
        
        It is assumed that at the very begining moment (ts_leading[-1]), the location of the follower is x0_follower with speed v0_follower.
        
        @input ts_leading,xs_leading
            Two list of the same length. 
            ts_leading unit is sec.
            xs_leading unit is m.
        
        @input: x0_follower,v0_follower
            both are float. 
            x0_follower unit is m
            v0_follower unit is m/s
        
        @input: idm_*
            the parameters in the IDM model. 
        
        ------------------------------------------
        @OUTPUT ts_follower,xs_follower
            both are lists. the same length.
            ts_follower unit is sec.
            xs_follower unit is m.
            
        
        ------------------------------------------
        @STEPS:
            - Fill the tao
        """
        v0_follower = v0_follower/3.6
        
        #vs_leading corresponds to ts_leading[:-1]
        #vs_leading unit is m/sec.
        #   it is used to interploate the speed at specific moment. 
        vs_leading = np.divide(np.diff(xs_leading),np.diff(ts_leading))
        ts_leading_vs = ts_leading[:-1]
        
        ts_follower = [ts_leading[0], ts_leading[0]+tao_delay]
        xs_follower = [x0_follower, x0_follower+tao_delay*v0_follower]
        vs_follower = [v0_follower, v0_follower]
        """
        ts_follower = [ts_leading[-1]]
        xs_follower = [x0_follower]
        vs_follower = [v0_follower]
        """
        
        #idx is the start of the index that ts_leading[idx]>t0+tao_delay
        T_horizon = ts_follower[-1]
        #while idx<=len(ts_leading_vs):
        while T_horizon<ts_leading[-1]+tao_delay:
            t = ts_follower[-1]+tao_delay
            
            #calculate the acceleration
            #   self.a(self, v_self=10, v_leader = 10, deltax =20, 
            v_leader  = Interpolate(ts_leading_vs,vs_leading, t)
            v_self =  vs_follower[-1]
            deltax = Interpolate(ts_leading,xs_leading, t)-xs_follower[-1]
            #
            a = self.a(v_self=v_self*3.6, v_leader=v_leader*3.6, deltax=deltax, idm_vf=idm_vf, idm_T = idm_T, vehicle_length = idm_vehicle_length, idm_delta = idm_delta, idm_s0=idm_s0, idm_a = idm_a,idm_b = idm_b)
            
            #UPDATE vs_follower and xs_follower
            vs_follower.append(vs_follower[-1]+a*tao_delay)
            xs_follower.append(xs_follower[-1]+1.0*vs_follower[-2]*tao_delay+a*tao_delay*tao_delay/2.0)
            ts_follower.append(ts_follower[-1]+tao_delay)
            
            T_horizon = T_horizon + tao_delay
        
        return ts_follower,vs_follower,xs_follower
    
    @classmethod
    def a_ACC_mode(self, x_leader, x_follower, v_leader, v_follower,lastmode='nan', vf = 40, speedcontrol_LW = 120, gapcontrol_UP = 100, k_speedcontrol = 0.3, desired_time_gap = 1.64, k2_gapcontrol = 0.23, k3_gapcontrol =0.07):
        """
        The acceleration of the adaptive cruise control. The ref is 
            Porfyri, Kallirroi N, Evangelos Mintsis, and Evangelos Mitsakis 2018Assessment of ACC and CACC Systems Using SUMO. EPiC Series in Engineering 2: 82–93.
        The coordinate increase downstream. 
        
        @input: v_leader, v_follower
            unit is km/h. 
        
        @input: tao_delay
            the delay , unit is sec. 
        
        @input: desired_time_gap
            the desied time gap, unit is sec. 
        
        @input: time_step
            the time step of the simulation. unit is sec 
            
        @input: speedcontrol_LW,gapcontrol_UP
            both unit are m. 
            When the gap is greater than speedcontrol_LW, then the speed control mode.
            when the gap is smaller than gapcontrol_UP, then it is gap control model
            when between the two, then remain the previous mode. 
            
        @input: k_speedcontrol = 0.3, k2_gapcontrol = 0.23, k3_gapcontrol =0.07
            the coefficients in the speed control mode and at the gap control mode. 
        
        
        
        """
        if lastmode=='speedcontrolmode' or abs(x_leader-x_follower)>=speedcontrol_LW:
            #speed control mode.
            return k_speedcontrol*(vf/3.6-v_follower),'speedcontrolmode'
        elif lastmode=='gapcontrolmode' or x_leader-x_follower<gapcontrol_UP:
            return k2_gapcontrol*(x_leader-x_follower-desired_time_gap*v_follower/3.6)+k3_gapcontrol*(v_leader-v_follower),'gapcontrolmode'
    
    @classmethod
    def generate_past_reaction_duration_tsxs(self, ts, xs, tao_delay  =2, direction  = 1):
        """
        Generate the ts_reac and xs_reac, that satisties:
            - ts_reac[-2] = ts_reac[-1]-tao_delay
        
        @input: direction
            either 1 or -1. 
            '1' means the x increase downstream.
        
        @input: ts, xs
            both are list or array. 
            Unit are sec and m. 
        
        
        """
        v_tmp = (xs[-1]-xs[-2])/(ts[-1]-ts[-2])
        
        ts_reac = [ts[-1]-tao_delay, ts[-1]]
        xs_reac = [xs[-1]-direction*v_tmp*tao_delay, xs[-1]]
        
        return ts_reac,xs_reac
    
    @classmethod
    def generate_reaction_tsxs(self, t0, v0, x0, tao_delay = 2):
        """
        
        Generate the ts and xs during the reaction time of driver. 
        
        @input: t0,v0,x0
            All are floats. 
            the initial moment, speed and location. 
            Units are sec, km/h and m respectively. 
        
        @input: tao_delay
            the reaction delay., unit is sec.
        """
        ts = [t0, t0+tao_delay]
        xs = [x0, x0+ v0/3.6*tao_delay]
        
        return ts,xs
    
    
    @classmethod
    def extend_tsxs_horizon(self, ts, xs, horizon = 500):
        """
        The ts[1]-ts[0] may not greater enough, the 
        
        """
        
        
        
        pass
    
    
    @classmethod
    def a(self, v_self=10, v_leader = 10, deltax =20, idm_vf=60, idm_T = 1.5, vehicle_length = 5, idm_delta = 4, idm_s0=2, idm_a = 1.0,idm_b = 3.21, ):
        """
        IDM formation output, the acceleration. 
        
        Note that the delta_v is defined as v_follower-v_leader. 
        
        @type v: float, unit is km/h
        
        @type vf: km/h.
        
        
        @type T: float.
        @param: T, unit is sec
            Average safe time headway.
            
        @type delta: delta:float
        @param: delta
            parameter in the model.
            
            
        @type s0:float
        @param: s0
            parameter 
        
        @OUTPUT: a
            unit is m/s2.
        """
        
        
        
        
        
        v_self=v_self/3.6
        v_leader = v_leader/3.6
        vf = idm_vf/3.6
        
        builtins.tmp = v_self,v_leader
        
        try:
            s_star = idm_s0+v_self*idm_T+v_self*(v_self - v_leader)/(2.0*np.sqrt(idm_a*idm_b))
            a = 1.0*idm_a*(1-np.power(v_self/vf, idm_delta)-(s_star*s_star)/(deltax*deltax))
        except Exception as e:
            
            print('deltax = ',deltax,', v_self=',v_self,', v_leader',v_leader)
            raise ValueError(e)
            
        return a
        
    
    @classmethod
    def equilibrim_k_from_v(self, v=10, vf=60, T = 1.5, vehicle_length = 5, delta = 4, s0=2):
        """
        NOTE that in the originla IDM equation, the s0 is calculated which take the vehicle length into account. 
        Compute the equilibrium state k from the v, using IDM model. 
        
        @type v: float, unit is km/h
        
        @type vf: km/h.
        
        
        @type T: float.
        @param: T, unit is sec
            Average safe time headway.
            
        @type delta: delta:float
        @param: delta
            parameter in the model.
            
            
        @type s0:float
        @param: s0
            parameter 
            
        @OUTPUT: k
            density. unit is veh/m.!!!!!!!!!!!!
            
        
        """
        v=v/3.6
        vf = vf/3.6
        
        if v>vf:raise ValueError("v should be smaller than vf.")
        
        return 1.0*np.sqrt(1.0-np.power(v/vf,delta))/((s0+vehicle_length)+v*T)
    
   
   



class DataProcess():
    """
    
    """
    
    
    
    @classmethod
    def DataFilter(self, ts,xs, ys, delta_L = .5):
        """
        Filter the data, the relative distance is greater than delta_L,  then it will be recorded
        
        """
        
        ts1 = [ts[0]]
        xs1 = [xs[0]]
        ys1 = [ys[0]]
        for t,x,y in zip(ts[1:],xs[1:],ys[1:]):
            delta = np.sqrt((x-xs1[-1])**2 + (y - ys1[-1])**2)
            if delta>=delta_L:
                ts1.append(t)
                xs1.append(x)
                ys1.append(y)
        return ts1,xs1,ys1
        

from filterpy.kalman import ExtendedKalmanFilter as EKF
class EKF_HDV(EKF):
    """
    extend kalman filter for HDV.
    FIrst filter and then predict. 
    
    The state variables X are:
        x,y,phi,v
    The control input are:
        a,front_steer,rear_steer
    The observation Z are:
        x,y
        
    
    """
    def __init__(self, ):
        """
        X_derivate = Fx+ Bu

        Z = Hx
        
        """
        pass
        
    
    def update_ekf(self, ekf, observation, H_extended):
        """
        @input: observation
            (x_observation,y_observation)
        """
        
        ekf.update(z = observation, HJacobian = H_jacobian, Hx = Hx, args = (H_extended), hx_args=(H_extended))
        

    def predict(self, veh_model, X0_extended, F_extended, P_extended, Q_extened, t = 2,states_name = ['x', 'y', 'phi', 'v'], n_timesteps_divided = 10):
        """
        Change the following:
            - self.x
            - self.P
        
        
        Predict the vehicle state at moment t(suppose that the current time is 0), given initial state X0 and the control input u. 
        --------------------------------
        @input: t
            unit is sec
            
        @Input veh_model
            control.iosys.NonlinearIOSystem
            Can be constructed via
                - veh_model = VM.VehicleKineticSolver.control_model_Rajamani()
        @input: states_name
            a list containing the str of the name, used in the jacobi evaluation. 
            
        @input: u, 
            a list. 
            the control input. 
            u = (a, steer_front, steer_rear)
        
        @input: X0_extended
            a list. 
            the initial state. 
            X0_extended = [x0, y0, phi0, v0, a, front_steer, rear_steer]
            X0 = [x0, y0, phi0, v0] is the veh_model state and 
            u = 
        
        @input: 
            np.array. 
            THe covariance matrix (of the states, i.e. X) at past time step. 
        
        @input: Q
            np.array, the state variance matrix. 
            
        
        @input: F_jacobi, type is sympy.matrices.dense.MutableDenseMatrix.
            X_derivate =  Fx+BU
            F is the jacobian matrix. 
            
            F.subs(dict_) can evaluate the matrix
            and 
        --------------------------
        
        @output: X,P_new
            the shape is (num_states, 1)
            Usually X = [x,y,phi,v]
            
        """
        #=========================Predict the new state
        #estimation moments.
        T = np.linspace(0, t, n_timesteps_divided)  
        #T = np.array([0 ,t])
        X0 = X0_extended[:4]
        u = X0_extended[4:]
        #convert the u into inputs of the model
        inputs = np.array([u for i in range(n_timesteps_divided)]).T
        #   predict, y shape is an array, shape is (len(X0),2). 
        #builtins.tmp = T, inputs, X0
        t, y = VehicleKineticSolver.TR_solover(veh_model, T, inputs, X0)
        
        #=========================Update the covariance matrix, and get P_new, 
        #   the data for subs. 
        subs = {states_name[i]:y[i,1] for i,_ in enumerate(states_name)}
        subs['a'] = u[0];subs['delta_f'] = u[1];subs['delta_r'] = u[2];
        F = F_extended.subs(subs)
        #   np.array. 
        #builtins.tmp = F,subs
        F1 = np.array(F).astype(np.float64)
        
        builtins.tmp = F1,P_extended,F1.T,Q_extened
        P_new = F1 @ P_extended @ F1.T + Q_extened
        
        #
        self.x = list(y[:,1]) + u
        self.P = P_new
        
        return self.x,self.P

    
    def Predict_BKP(self, veh_model, u, X0, F_jacobi, P, Q, t = 2,states_name = ['x', 'y', 'phi', 'v']):
        """
        Predict the vehicle state at moment t(suppose that the current time is 0), given initial state X0 and the control input u. 
        --------------------------------
        @input: t
            unit is sec
            
        @Input veh_model
            control.iosys.NonlinearIOSystem
            Can be constructed via
                - veh_model = VM.VehicleKineticSolver.control_model_Rajamani()
        @input: states_name
            a list containing the str of the name, used in the jacobi evaluation. 
            
        @input: u, 
            a list. 
            the control input. 
            u = (a, steer_front, steer_rear)
        
        @input: X0
            a list. 
            the initial state. 
            X0 = [x0, y0, phi0, v0]
        
        @input: 
            np.array. 
            THe covariance matrix (of the states, i.e. X) at past time step. 
        
        @input: Q
            np.array, the state variance matrix. 
            
        
        @input: F_jacobi, type is sympy.matrices.dense.MutableDenseMatrix.
            X_derivate =  Fx+BU
            F is the jacobian matrix. 
            
            F.subs(dict_) can evaluate the matrix
            and 
        --------------------------
        
        @output: X
            the shape is (num_states, 1)
            Usually X = [x,y,phi,v]
            
        """
        #------------------------Predict the new state
        #estimation moments. 
        T = [0 ,t]
        #convert the u into inputs of the model
        inputs = np.array([u for i in range(2)]).T
        t, y = VehicleKineticSolver.TR_solover(veh_model, T, inputs, X0)
        
        #------------------------Update the covariance matrix, and get P_new
        F = F_jacobi.subs({states_name[i]:y[i] for i in enumerate(states_name)})
        P_new = F @ P @ F.T + Q
        
        return y[:,1],P_new
    
    
class ExpectedTrajectorys():
    """
    Call back method:
    
    
    reload(VM)
    expectedTRs_instances = VM.ExpectedTrajectorys(）
    
    #expected_tr[steerangle] = {}..
    expected_tr = GetExpectedTrajectoriesGivenVehicetype( vehicle_type = "vehicle.audi.a2")
    
    
    
    """
    
    @classmethod
    def AffineExpectedTrajectory(self, polygons_list, new_state= (0, 0 , 90*np.pi/180.0)):
        """
        Move and rotate the expected trajectory, whihc is represented by a list of polygons. 
        Using: 
            - shapely.affinity.translate(geom, xoff=0.0, yoff=0.0)
            - shapely.affinity.rotate(geom, angle, origin='center', use_radians=False)
        
        polygons_list[0] is the start pose of the vehicle. The heading angle of the start pose should be 90degree and the start location (CG location) should be (0, 0). 
        

        polygons_list can be obtained via:
            reload(VM)
            res = VM.VehicleKineticSolver.GetExpectedTrajectory_shapely(steering_angle = 10*np.pi/180.0)
            polygons_list = res['polygons']
        
        -----------------------------------------
        @input: polygons_list
            expected trajectory. The start pose should 
        
        @input: new_state
            x,y,phi=new_state
            
            THe phi is the heading angle which is defined for the angle between head and x-axis. unit is np.pi. 
        
        @output: 
            affined_polygons
            
            len(affined_polygons) = len(polygons_list)
        -----------------------------------------
            
        """
        #phi should be unit in np.pi
        x,y,phi = new_state[0],new_state[1],new_state[2]
        
        #
        affined_polygons = []
        
        #
        for p in polygons_list:
            #
            #p1 = shapely.affinity.translate(p, xoff=x, yoff= y)
            
            #
            translated = shapely.affinity.translate(p, xoff=x, yoff= y)
            #-90 is because that the default start pose is is 90 degree. 
            p2 = shapely.affinity.rotate(translated, angle = phi*180/np.pi-90, origin=(x,y))
            
            affined_polygons.append(p2)
        
        return affined_polygons
    
    def plot_TR_nearest_angle_with_affine(self, angle,  new_state= (0, 0 , 90*np.pi/180.0)):
        """
        @input: angle
            unit is np.pi
            a dict. 
        """
        fig,ax = plt.subplots()
        
        nearestangle = self.NearestSteeringAngle(angle = angle)
        TR = self.expected_trajectorys_steers[nearestangle]
        
        affinedTR = self.affine2newstate(angle = angle, new_state= new_state)
        
        
        for polygon in TR['polygons']:
            x,y = polygon.exterior.xy
            ax.plot(x,y)
        xs = [state[0] for state in TR['states']]
        ys = [state[1] for state in TR['states']]
        ax.plot(xs,ys)
        
        for polygon in affinedTR['polygons']:
            x,y = polygon.exterior.xy
            ax.plot(x,y)
        xs = [state[0] for state in affinedTR['states']]
        ys = [state[1] for state in affinedTR['states']]
        ax.plot(xs,ys)
        
        
        ax.axis('equal')
        
        return ax
    
    
    def plot_TR_nearest_angle(self, angle):
        """
        @input: angle
            unit is np.pi
            a dict. 
        """
        fig,ax = plt.subplots()
        
        nearestangle = self.NearestSteeringAngle(angle = angle)
        TR = self.expected_trajectorys_steers[nearestangle]
        
        
        for polygon in TR['polygons']:
            x,y = polygon.exterior.xy
            ax.plot(x,y)
            
        xs = [state[0] for state in TR['states']]
        ys = [state[1] for state in TR['states']]
        
        ax.plot(xs,ys)
        ax.axis('equal')
        
        return ax
    
    
    @classmethod
    def plot_CG_trajectory(self, CG_line_coors, N_interval = 10, type_id = "vehicle.audi.a2"):
        """
        
        """
        
        keys = sorted(CG_line_coors[type_id].keys())[::N_interval]
        
        
        fig,ax = plt.subplots()
        
        for k in keys:
            
            xs = [i[0] for i in CG_line_coors[type_id][k]]
            ys = [i[1] for i in CG_line_coors[type_id][k]]
            
            ax.plot(xs, ys)
            pass
        
        ax.grid()
        
        pass
    
    
    
    
    
    @classmethod
    def RelativePose(self, veh1_state, veh2_state):
        """
        Calculate the relative pose of veh2_state relative to the vehicle 1. 
        
        
        The new axis locates at CG of vehicle 1 and the y-axis keep alligned with longitudinal middle line 
        
        
        ----------------------------
        @input: veh1_state
            veh1_state = (x1,y1,phi1), phi unit is np.pi. 
            
        
        @OUTPUT:
            vehicle2staterelative  = (new_x, new_y, new_phi)
            
        ---------------------------------------------
        
        
        
        """
        #the state decompose. 
        x1,y1,phi1 = veh1_state[0],veh1_state[1],veh1_state[2]
        x2,y2,phi2 = veh2_state[0],veh2_state[1],veh2_state[2]
        
        R = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        
        #the angle of line (x1,y1)---(x2,y2) in angle_line
        #   unit is degree
        #   angle_line is in Rajami coordinate. 
        #   arctan2 return value within [-pi.pi, np.pi]:
        #       x = np.array([-1, +1, +1, -1])
        #       y = np.array([-1, -1, +1, +1])
        #       np.arctan2(y, x) * 180 / np.pi
        #       array([-135.,  -45.,   45.,  135.])
        degree_relative0 = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
        if degree_relative0<0:
            angle_line = 360 + degree_relative0
        else:
            angle_line = degree_relative0
        
        #--------------------the angle of (x1,y1)---(x2,y2) in new axis
        angle_line_new_axis = angle_line + (90.0 - phi1*180/np.pi)
        new_x,new_y = R*np.cos(angle_line_new_axis*np.pi/180.0),R*np.sin(angle_line_new_axis*np.pi/180.0)
        
        new_phi = phi2 + (np.pi/2 - phi1)
        
        
        return new_x,new_y,new_phi
    
    
    
    @classmethod
    def plot_TR(self, TR):
        """
        @input: TR
            a dict. 
            keys include ['distances', 'polygons', 'states']
        """
        fig,ax = plt.subplots()
        
        for polygon in TR['polygons']:
            x,y = polygon.exterior.xy
            ax.plot(x,y)
            
        xs = [state[0] for state in TR['states']]
        ys = [state[1] for state in TR['states']]
        
        ax.plot(xs,ys)
        ax.axis('equal')
        
        return ax
    
    @classmethod
    def affine2newstate_classmethod(self, TR, new_state= (0, 0 , 90*np.pi/180.0)):
        """
        affine the self.expected_trajectorys_steers, a dict. 
            self.expected_trajectorys_steers[certain_angle].keys() = ['distances', 'polygons', 'states']
        
        Move and rotate the expected trajectory, whihc is represented by a list of polygons. 
        Using: 
            - shapely.affinity.translate(geom, xoff=0.0, yoff=0.0)
            - shapely.affinity.rotate(geom, angle, origin='center', use_radians=False)
        
        polygons_list[0] is the start pose of the vehicle. The heading angle of the start pose should be 90degree and the start location (CG location) should be (0, 0). 
        

        polygons_list can be obtained via:
            reload(VM)
            res = VM.VehicleKineticSolver.GetExpectedTrajectory_shapely(steering_angle = 10*np.pi/180.0)
            polygons_list = res['polygons']
        
        -----------------------------------------
        @input: TR
            a dict. 
            
            TR['polygons'] = a list of polygon
            TR['states'] = a list of state. Each state is [x,y,phi], phi unit is np.pi 
            TR['distances'] = a list of float. 
        
        @input: new_state
            x,y,phi=new_state
            
            THe phi is the heading angle which is defined for the angle between head and x-axis. unit is np.pi. 
        
        @output: affinedTR
            {'polygons':affined_polygons_list, 'states':affined_states_list, 'distances':copy.deepcopy(self.expected_trajectorys_steers[nearestangle]['distances'])}
            
        -----------------------------------------
        """
        #
        polygon_list = TR['polygons']
        states_list = TR['states']
        
        #   affined_polygons_list 
        affined_polygons_list = self.AffineExpectedTrajectory(polygons_list = polygon_list, new_state= new_state)
        
        #   affined state list.np.array(states_list).T shape is (3, N)
        xy_array_new = self.Affine2dTR(xy_array = np.array(states_list).T, new_state= new_state)
        #print('xy_array_new shape', xy_array_new.shape)
        #
        affined_states_list = [(xy_array_new[0,i],xy_array_new[1,i],xy_array_new[2,i]) for i in range(xy_array_new.shape[1])]
        
        #
        return {'polygons':affined_polygons_list, 'states':affined_states_list, 'distances':copy.deepcopy(TR['distances'])}
    
    @classmethod
    def affine_polygon(self, polygonshapely, new_state= (0, 0 , 90*np.pi/180.0)):
        """
        just affine single polygon. 
        """
        
        #phi should be unit in np.pi
        x,y,phi = new_state[0],new_state[1],new_state[2]
        
        
        translated = shapely.affinity.translate(polygonshapely, xoff=x, yoff= y)
        affined = shapely.affinity.rotate(translated, angle = phi*180/np.pi-90, origin=(x,y))
        
        
        return affined
    
    
    def affine2newstate(self, angle, new_state= (0, 0 , 90*np.pi/180.0)):
        """
        affine the self.expected_trajectorys_steers, a dict. 
            self.expected_trajectorys_steers[certain_angle].keys() = ['distances', 'polygons', 'states']
        
        Move and rotate the expected trajectory, whihc is represented by a list of polygons. 
        Using: 
            - shapely.affinity.translate(geom, xoff=0.0, yoff=0.0)
            - shapely.affinity.rotate(geom, angle, origin='center', use_radians=False)
        
        polygons_list[0] is the start pose of the vehicle. The heading angle of the start pose should be 90degree and the start location (CG location) should be (0, 0). 
        

        polygons_list can be obtained via:
            reload(VM)
            res = VM.VehicleKineticSolver.GetExpectedTrajectory_shapely(steering_angle = 10*np.pi/180.0)
            polygons_list = res['polygons']
        
        -----------------------------------------
        @input: angle
            unit is np.pi. 
        
        @input: new_state
            x,y,phi=new_state
            
            THe phi is the heading angle which is defined for the angle between head and x-axis. unit is np.pi. 
        
        @output: affinedTR
            {'polygons':affined_polygons_list, 'states':affined_states_list, 'distances':copy.deepcopy(self.expected_trajectorys_steers[nearestangle]['distances'])}
            
        -----------------------------------------
        """
        #find the nearest angle. nearestangle unit is np.pi
        nearestangle = self.NearestSteeringAngle( angle )
        
        #
        polygon_list = self.expected_trajectorys_steers[nearestangle]['polygons']
        states_list = self.expected_trajectorys_steers[nearestangle]['states']
        
        #   affined_polygons_list 
        affined_polygons_list = self.AffineExpectedTrajectory(polygons_list = polygon_list, new_state= new_state)
        
        #   affined state list.np.array(states_list).T shape is (3, N)
        xy_array_new = self.Affine2dTR(xy_array = np.array(states_list).T, new_state= new_state)
        #print('xy_array_new shape', xy_array_new.shape)
        #
        affined_states_list = [(xy_array_new[0,i],xy_array_new[1,i],xy_array_new[2,i]) for i in range(xy_array_new.shape[1])]
        
        #
        return {'polygons':affined_polygons_list, 'states':affined_states_list, 'distances':copy.deepcopy(self.expected_trajectorys_steers[nearestangle]['distances'])}

    @classmethod
    def Affine_CG(self, CG_line, new_state= (10, 20 , 30*np.pi/180.0)):
        """
        affine the cg line. 
        
        Callback method:
        
            fig,ax = plt.subplots()
            array = np.array(expected_TR_class.CG_line_coors['vehicle.audi.a2'][0]).T
            ax.plot(array[0, :], array[1,:])

            array = VM.ExpectedTrajectorys.Affine_CG(expected_TR_class.CG_line_coors['vehicle.audi.a2'][0])
            ax.plot(array[0, :], array[1,:])
                    
        ---------------------------------------------------
        @input: CG_line
            a list of xy. 
            
            CG_line = [(x1,y1), (x2,y2)...]
            
            CG_line = expected_TR_class.CG_line_coors['vehicle.audi.a2'][0]
            
            np.array(expected_TR_class.CG_line_coors['vehicle.audi.a2'][0]).shape = (N, 2)
        ----------------------------------------------------
        
        """
        #xy_array shape is (2, N)
        xy_array = np.array(CG_line).T
        
        #phi should be unit in np.pi
        xy_array_new = copy.deepcopy(xy_array)
        
        #rotated angle, unit is degree. -90 is because that defaule benchmark angle is 90 degree. 
        theta_roted = (new_state[2]*180/np.pi-90)*np.pi/180.0
        c,s = np.cos(theta_roted), np.sin(theta_roted)
        rotete_matrix = np.array(((c, -s), (s, c)))
        xy_array_new1 = np.matmul(rotete_matrix, xy_array_new[[0,1],:])
        
        #translate
        xy_array_new[0,:] = xy_array_new1[0,:]+new_state[0]
        xy_array_new[1,:] = xy_array_new1[1,:]+new_state[1]
        #xy_array_new[2,:] = copy.deepcopy(xy_array[2]-np.pi/2 + new_state[2])
        
        return xy_array_new
        


    @classmethod
    def Affine2dTR(self, xy_array, new_state= (0, 0 , 90*np.pi/180.0)):
        """
        Move and rotate the expected trajectory, whihc is represented by a list of polygons. 
        Using: 
            - shapely.affinity.translate(geom, xoff=0.0, yoff=0.0)
            - shapely.affinity.rotate(geom, angle, origin='center', use_radians=False)
        
        polygons_list[0] is the start pose of the vehicle. The heading angle of the start pose should be 90degree and the start location (CG location) should be (0, 0). 
        

        polygons_list can be obtained via:
            reload(VM)
            res = VM.VehicleKineticSolver.GetExpectedTrajectory_shapely(steering_angle = 10*np.pi/180.0)
            polygons_list = res['polygons']
        
        -----------------------------------------
        @input: xy_array
            an array. 
            shape is 2*N.
            
            THe 1st row is x and 2nd row is y. 
        
        @input: new_state
            x,y,phi=new_state
            
            THe phi is the heading angle which is defined for the angle between head and x-axis. unit is np.pi. 
        
        @output: 
            affined_polygons
            
            len(affined_polygons) = len(polygons_list)
        -----------------------------------------
            
        """
        #phi should be unit in np.pi
        xy_array_new = copy.deepcopy(xy_array)
        
        #rotated angle, unit is degree. -90 is because that 
        theta_roted = (new_state[2]*180/np.pi-90)*np.pi/180.0
        c,s = np.cos(theta_roted), np.sin(theta_roted)
        rotete_matrix = np.array(((c, -s), (s, c)))
        xy_array_new1 = np.matmul(rotete_matrix, xy_array_new[[0,1],:])
        
        #translate
        xy_array_new[0,:] = xy_array_new1[0,:]+new_state[0]
        xy_array_new[1,:] = xy_array_new1[1,:]+new_state[1]
        xy_array_new[2,:] = copy.deepcopy(xy_array[2]-np.pi/2 + new_state[2])
        
        return xy_array_new
    
    def GetExpectedTrajectoriesGivenVehicetype(self, vehicle_type = "vehicle.audi.a2"):
        """
        @OUTPUT: 
         expected_tr_for_vtype
            a dict. 
            expected_tr_for_vtype.keys() = [steer1, steer2,.....]
            
            steers unit is np.pi. 
        """
        if vehicle_type in self.expected_trajectorys_steers.keys():
            return self.expected_trajectorys_steers[vehicle_type]
        else:
            return self.expected_trajectorys_steers["vehicle.audi.a2"]
        
    def GetExpectedTrajectoriesGivenVehicetype_angle(self, deltaf = 0.0, vehicle_type = "vehicle.audi.a2"):
        """
        @input: deltaf
            the steering angle of the front wheel.
            
            Unit is np.pi. 
        @output: expectedTR
            a dict
            expectedTR.keys() = ['polygons', 'distances']
            
        """
        if vehicle_type in self.expected_trajectorys_steers.keys():
            #find the nearest angle. nearestangle unit is np.pi
            nearestangle = self.NearestSteeringAngle(deltaf )
            return self.expected_trajectorys_steers[vehicle_type][nearestangle]
        else:
            #find the nearest angle. nearestangle unit is np.pi
            nearestangle = self.NearestSteeringAngle(deltaf )
            return self.expected_trajectorys_steers["vehicle.audi.a2"][nearestangle]
        
    def generate_expectedTRs4AllSteers(self, max_steering_angle = 70*np.pi/180, N_steering_angles = 300, X0 = [0, 0, 90/180*np.pi],  default_v = 10, delta_t = 0.05, T_horizon = 5, distance_threshold = 30, T_horizon_incremental = 3, veh_para1= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2}, mirror = True, vehicle_type = "vehicle.audi.a2"):
        """
        -------------------------------------
        @input: T_horizon_incremental
            if the distance is not long enough, the T_horizon will be increased and trajectory will be calculated again by extending the 
            
        @input: X0
            x,y, phi = X0
            
            phi is the heading angle, unit is np.pi
            
        @input: max_steering_angle and N_steering_angles
        
            max_steering_angle is the max steering angle of front wheel for bicycle model.  unit is np.pi. 
            
            N_steering_angles is the number of discretization of the max_steering_angle.
            
        @input: distance_threshold
            the generated trajectory length threshold. 
            
        @input: mirror
            mirror about the y-axis 
            
            Note that when steering angle is positive, then turn left, else turn right. 
        
        """
        
        #returned value. 
        expected_trajectorys_steers =  {}
        
        veh_model = VehicleKineticSolver.control_model_Rajamani_using_v(params = veh_para1)
        
        for steering_angle in np.linspace(0, max_steering_angle, N_steering_angles):
            
            #-------------obatain distances and polygons_veh
            distances_veh1= []
            while len(distances_veh1)==0 or distances_veh1[-1]<distance_threshold:
                if len(distances_veh1)>0:
                    T_horizon = T_horizon + T_horizon_incremental
                
                T = np.linspace(0, T_horizon, int(T_horizon/delta_t)+1)
                #   the last 0 is for steering angle of the rear wheel.  
                inputs = np.array([[default_v, steering_angle, 0] for i in range(len(T))]).T
                #   t and y1 are np.array. 
                #       t.shape is (len(T),) and y1 shape is (3, len(T))
                t, y1 = VehicleKineticSolver.TR_solover(veh_model, T, inputs, X0)
                distances_veh1 = [0]+list(np.cumsum(np.sqrt(np.diff(y1[0,:])**2 + np.diff(y1[1,:])**2)))
                #steps2: obtain the shapelys of the polygons in polygons_veh1, a list. 
                #   shapelys_polygons is multipolygon instance. 
            
            #find the index in distances_veh1 that distances_veh1[idx]-distance_threshold is minimal
            idx = np.where(np.array(distances_veh1)>=distance_threshold)[0][0]
            
            #
            distances = distances_veh1[:idx]
            polygons_veh = VehicleKineticSolver.shapelys_given_states(TR = y1[:,:idx], veh_para = veh_para1)
            #
            expected_trajectorys_steers[steering_angle] = {'distances':distances, 'polygons':polygons_veh}
            
            #mirror the resulf for -steering_angle
            if mirror:
                polygons_veh2 = [shapely.affinity.scale(p, xfact = -1, origin = (0, 0)) for p in polygons_veh]
                expected_trajectorys_steers[-steering_angle] = {'distances':distances, 'polygons':polygons_veh2}
            
            #
            #expected_trajectorys_steers[steering_angle]['states'] = [X0, X1, X2,...]
            expected_trajectorys_steers[steering_angle]['states'] = [list(y1[:,i]) for i in range(idx)]
            
            if mirror:
                #   the state is the same while y is mirrot about y-axis and phi is changed
                expected_trajectorys_steers[-steering_angle]['states'] = [[-y1[0,i],y1[1,i],np.pi-y1[2,i]] for i in range(idx)]
        
        #
        self.expected_trajectorys_steers[vehicle_type] = expected_trajectorys_steers
        
        
        pass
    
    
    
    
    
    def __init__1(self, max_steering_angle = 70*np.pi/180, N_steering_angles = 300, X0 = [0, 0, 90/180*np.pi],  default_v = 10, delta_t = 0.05, T_horizon = 5, distance_threshold = 30, T_horizon_incremental = 3, veh_para1= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2}, mirror = True):
        """
        Generate the expected trajectory (represented by a list of distance and a list of shapely).
        
        Different from __init__:
            __init__1() consider the terminal angle of the trajectory. 
        
        X0 = [0, 0, 90/180*np.pi] means the heading angle is 90 degree. 
        ---------------
        
        --------------
        
            
        Expected trajectory for specific steering_angle can naively obtained via:
        
            X0 = state_veh1#x,y,phi
            T = np.linspace(0, T_horizon, int(T_horizon/delta_t)+1) 
            inputs = np.array([[default_v, steering_angle1, 0] for i in range(len(T))]).T
            t, y1 = VM.VehicleKineticSolver.TR_solover(veh_model, T, inputs, X0)
            distances_veh1 = [0]+list(np.cumsum(np.sqrt(np.diff(y1[0,:])**2 + np.diff(y1[1,:])**2)))
            #steps2: obtain the shapelys of the polygons in polygons_veh1, a list. 
            #   shapelys_polygons is multipolygon instance. 
            polygons_veh1 = VM.VehicleKineticSolver.shapelys_given_states(TR = y1, veh_para = veh_para1)
            
        -------------------------------------
        @input: T_horizon_incremental
            if the distance is not long enough, the T_horizon will be increased and trajectory will be calculated again by extending the 
            
        @input: X0
            x,y, phi = X0
            
            phi is the heading angle, unit is np.pi
            
        @input: max_steering_angle and N_steering_angles
        
            max_steering_angle is the max steering angle of front wheel for bicycle model.  unit is np.pi. 
            
            N_steering_angles is the number of discretization of the max_steering_angle.
            
        @input: distance_threshold
            the generated trajectory length threshold. 
            
        @input: mirror
            mirror about the y-axis 
            
            Note that when steering angle is positive, then turn left, else turn right. 
            
            
        @OUTPUT: expected_trajectorys_steers
            a dict. expected_trajectorys_steers keys are steer angle. 
            
            expected_trajectorys_steers[steer_angle] = {'distances':distances, 'polygons':polygons_veh, 'states':states_list}
            
            states_list = [X0, X1， X2,...], X0 = [x,y,phi], phi unit is np.pi
            
            distances,polygons_veh both are list. 
            distances is a list of float, represent the travelling distacne of the CG from X0, and 
            polygons_veh1 is a list of shapely polygon. 
            
        """
        self.vehicle_type = ["vehicle.audi.a2", "vehicle.audi.tt", "vehicle.carlamotors.carlacola", "vehicle.jeep.wrangler_rubicon", "vehicle.chevrolet.impala", "vehicle.mini.cooperst", "vehicle.audi.etron", "vehicle.mercedes-benz.coupe", "vehicle.bmw.grandtourer", "vehicle.toyota.prius", "vehicle.citroen.c3", "vehicle.mustang.mustang", "vehicle.tesla.model3", "vehicle.tesla.cybertruck", "vehicle.volkswagen.t2", "vehicle.lincoln.mkz2017", "vehicle.seat.leon", "vehicle.nissan.patrol", "vehicle.nissan.micra"]
        
        
        #returned value. 
        expected_trajectorys_steers =  {}
        
        veh_model = VehicleKineticSolver.control_model_Rajamani_using_v(params = veh_para1)
        
        for steering_angle in np.linspace(0, max_steering_angle, N_steering_angles):
            
            #-------------obatain distances and polygons_veh
            distances_veh1= []
            while len(distances_veh1)==0 or distances_veh1[-1]<distance_threshold:
                if len(distances_veh1)>0:
                    T_horizon = T_horizon + T_horizon_incremental
                
                T = np.linspace(0, T_horizon, int(T_horizon/delta_t)+1)
                #   the last 0 is for steering angle of the rear wheel.  
                inputs = np.array([[default_v, steering_angle, 0] for i in range(len(T))]).T
                #   t and y1 are np.array. 
                #       t.shape is (len(T),) and y1 shape is (3, len(T))
                t, y1 = VehicleKineticSolver.TR_solover(veh_model, T, inputs, X0)
                distances_veh1 = [0]+list(np.cumsum(np.sqrt(np.diff(y1[0,:])**2 + np.diff(y1[1,:])**2)))
                #steps2: obtain the shapelys of the polygons in polygons_veh1, a list. 
                #   shapelys_polygons is multipolygon instance. 
            
            #find the index in distances_veh1 that distances_veh1[idx]-distance_threshold is minimal
            idx = np.where(np.array(distances_veh1)>=distance_threshold)[0][0]
            
            #
            distances = distances_veh1[:idx]
            polygons_veh = VehicleKineticSolver.shapelys_given_states(TR = y1[:,:idx], veh_para = veh_para1)
            #
            expected_trajectorys_steers[steering_angle] = {'distances':distances, 'polygons':polygons_veh}
            
            #mirror the resulf for -steering_angle
            if mirror:
                polygons_veh2 = [shapely.affinity.scale(p, xfact = -1, origin = (0, 0)) for p in polygons_veh]
                expected_trajectorys_steers[-steering_angle] = {'distances':distances, 'polygons':polygons_veh2}
            
            #
            #expected_trajectorys_steers[steering_angle]['states'] = [X0, X1, X2,...]
            expected_trajectorys_steers[steering_angle]['states'] = [list(y1[:,i]) for i in range(idx)]
            
            if mirror:
                #   the state is the same while y is mirrot about y-axis and phi is changed
                expected_trajectorys_steers[-steering_angle]['states'] = [[-y1[0,i],y1[1,i],np.pi-y1[2,i]] for i in range(idx)]
            
        self.expected_trajectorys_steers = expected_trajectorys_steers
        
        self.expected_trajectorys_steers[ "vehicle.audi.a2"] = copy.deepcopy(expected_trajectorys_steers)
    
    def __init__(self, max_steering_angle = 70*np.pi/180, N_steering_angles = 300, X0 = [0, 0, 90/180*np.pi],  default_v = 10, delta_t = 0.05, T_horizon = 5, distance_threshold = 30, notvalid_phi_threshold_LWUP = (np.pi, 2.0*np.pi), T_horizon_incremental = 3, lflrlFlR_vtypes_as_key = {"vehicle.audi.a2":{'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2}}, mirror = True):
        """
        Generate the expected trajectory (represented by a list of distance and a list of shapely).
        
        KEY ATTRS:
            - self.max_steering_angle, float, unit is np.pi
            - self.delta_steer_angles, float, unit is np.pi
        
            ForwardTrejctory
                - self.expected_trajectorys_steers, a dict. self.expected_trajectorys_steers["vehicle.audi.a2"][steer] = {'polygons':polygonslist, 'distances':distances_list, }
                - self.united_expected_trajectorys[vtype][steer]=polygon, used to check the colision
                - self.CG_line_coors, self.CG_line_coors[vtype][steer] = [(x1,y1), (x2,y2), ....]
                - self.UNITEDPOLYGON, self.UNITEDPOLYGON[vtype] = Polygon, ALL trajectories, 
            
            Backward trajectory:
                
                - self.backward_attrs
                    = self.backward_attrs['united_expected_trajectorys'][vtype][steer] = shapelygon
                    = 
            
        X0 = [0, 0, 90/180*np.pi] means the heading angle is 90 degree. 
        ---------------
        
        --------------
        
            
        Expected trajectory for specific steering_angle can naively obtained via:
        
            X0 = state_veh1#x,y,phi
            T = np.linspace(0, T_horizon, int(T_horizon/delta_t)+1) 
            inputs = np.array([[default_v, steering_angle1, 0] for i in range(len(T))]).T
            t, y1 = VM.VehicleKineticSolver.TR_solover(veh_model, T, inputs, X0)
            distances_veh1 = [0]+list(np.cumsum(np.sqrt(np.diff(y1[0,:])**2 + np.diff(y1[1,:])**2)))
            #steps2: obtain the shapelys of the polygons in polygons_veh1, a list. 
            #   shapelys_polygons is multipolygon instance. 
            polygons_veh1 = VM.VehicleKineticSolver.shapelys_given_states(TR = y1, veh_para = veh_para1)
            
        -------------------------------------
        @input: T_horizon_incremental
            if the distance is not long enough, the T_horizon will be increased and trajectory will be calculated again by extending the 
        
        @input: notvalid_phi_threshold_LWUP
            the phi interval which is not considered. 
            
            Default is (np.pi, 2.0*np.pi), because the initial state is 90degree. 
        
        @input: X0
            x,y, phi = X0
            
            phi is the heading angle, unit is np.pi
            
        @input: max_steering_angle and N_steering_angles
        
            max_steering_angle is the max steering angle of front wheel for bicycle model.  unit is np.pi. 
            
            N_steering_angles is the number of discretization of the max_steering_angle.
            
        @input: distance_threshold
            the generated trajectory length threshold. 
            
        @input: mirror
            mirror about the y-axis 
            
            Note that when steering angle is positive, then turn left, else turn right. 
            
            
        @OUTPUT: expected_trajectorys_steers
            a dict. expected_trajectorys_steers keys are steer angle. 
            
            expected_trajectorys_steers[steer_angle] = {'distances':distances, 'polygons':polygons_veh, 'states':states_list}
            
            states_list = [X0, X1， X2,...], X0 = [x,y,phi], phi unit is np.pi
            
            distances,polygons_veh both are list. 
            distances is a list of float, represent the travelling distacne of the CG from X0, and 
            polygons_veh1 is a list of shapely polygon. 
            
        """
        #unit is pi
        self.max_steering_angle = max_steering_angle
        #   unit is angle. 
        self.delta_steer_angles = max_steering_angle*1.0/N_steering_angles
        #self.N_steering_angles = N_steering_angles
        
        #self.backward_attrs['united_expected_trajectorys'][vtype][steer] = polygon.
        self.backward_attrs = {'united_expected_trajectorys':{}, 'CG_line_coors':{}}
        
        #union later in this method, which means unite the polygons along the trajectory together. 
        self.union_or_not = True
        self.vehicle_type = ["vehicle.audi.a2", "vehicle.audi.tt", "vehicle.carlamotors.carlacola", "vehicle.jeep.wrangler_rubicon", "vehicle.chevrolet.impala", "vehicle.mini.cooperst", "vehicle.audi.etron", "vehicle.mercedes-benz.coupe", "vehicle.bmw.grandtourer", "vehicle.toyota.prius", "vehicle.citroen.c3", "vehicle.mustang.mustang", "vehicle.tesla.model3", "vehicle.tesla.cybertruck", "vehicle.volkswagen.t2", "vehicle.lincoln.mkz2017", "vehicle.seat.leon", "vehicle.nissan.patrol", "vehicle.nissan.micra"]
        
        #returned value. 
        expected_trajectorys_steers =  {}
        #-----------------------------
        for vtype in self.vehicle_type:
            #init
            expected_trajectorys_steers[vtype] =  {}
            
            #get veh para
            if vtype in lflrlFlR_vtypes_as_key.keys():
                veh_para = lflrlFlR_vtypes_as_key[vtype]
            else:
                veh_para = lflrlFlR_vtypes_as_key["vehicle.audi.a2"]
            
            #get vehicle model
            veh_model = VehicleKineticSolver.control_model_Rajamani_using_v(params = veh_para)
        
            for steering_angle in np.arange(0, max_steering_angle, self.delta_steer_angles):#np.linspace(0, max_steering_angle, N_steering_angles):
                
                #-------------obatain distances and polygons_veh, in distances_veh1 and polygons_veh
                distances_veh1= []
                while len(distances_veh1)==0 or distances_veh1[-1]<distance_threshold:
                    if len(distances_veh1)>0:
                        T_horizon = T_horizon + T_horizon_incremental
                    
                    T = np.linspace(0, T_horizon, int(T_horizon/delta_t)+1)
                    #   the last 0 is for steering angle of the rear wheel.  
                    inputs = np.array([[default_v, steering_angle, 0] for i in range(len(T))]).T
                    #   t and y1 are np.array. 
                    #       t.shape is (len(T),) and y1 shape is (3, len(T))
                    t, y1 = VehicleKineticSolver.TR_solover(veh_model, T, inputs, X0)
                    distances_veh1 = [0]+list(np.cumsum(np.sqrt(np.diff(y1[0,:])**2 + np.diff(y1[1,:])**2)))
                    #steps2: obtain the shapelys of the polygons in polygons_veh1, a list. 
                    #   shapelys_polygons is multipolygon instance. 
                
                #find the index in distances_veh1 that distances_veh1[idx]-distance_threshold is minimal
                idx = np.where(np.array(distances_veh1)>=distance_threshold)[0][0]
                #   find the idx that the phi is not valid, checked by the arg notvalid_phi_threshold_LWUP
                idx_phivalid = np.inf
                for idx0,phi in enumerate(y1[2,:]):
                    if phi>=notvalid_phi_threshold_LWUP[0] and phi<=notvalid_phi_threshold_LWUP[1]:
                        idx = min(idx, idx0)
                        break
                
                #
                distances = distances_veh1[:idx]
                polygons_veh = VehicleKineticSolver.shapelys_given_states(TR = y1[:,:idx], veh_para = lflrlFlR_vtypes_as_key["vehicle.audi.a2"])
                #
                expected_trajectorys_steers[vtype][steering_angle] = {'distances':distances, 'polygons':polygons_veh}
                
                #mirror the resulf for -steering_angle
                if mirror:
                    polygons_veh2 = [shapely.affinity.scale(p, xfact = -1, origin = (0, 0)) for p in polygons_veh]
                    expected_trajectorys_steers[vtype][-steering_angle] = {'distances':distances, 'polygons':polygons_veh2}
                
                #
                #expected_trajectorys_steers[vtype][steering_angle]['states'] = [X0, X1, X2,...]
                expected_trajectorys_steers[vtype][steering_angle]['states'] = [list(y1[:,i]) for i in range(idx)]
                
                if mirror:
                    #   the state is the same while y is mirrot about y-axis and phi is changed
                    expected_trajectorys_steers[vtype][-steering_angle]['states'] = [[-y1[0,i],y1[1,i],np.pi-y1[2,i]] for i in range(idx)]
                
        self.expected_trajectorys_steers = expected_trajectorys_steers
        
        #------------------------union the polygons in self.united_expected_trajectorys_steers
        #   shapely.ops.unary_union(expected_trajectorys_steers[vtype][steering_angle]['polygons']) will return a polygon.
        self.united_expected_trajectorys = {}
        self.UNITEDPOLYGON = {}
        for vtype in self.expected_trajectorys_steers.keys():
            self.united_expected_trajectorys[vtype] = {}
            for steer in self.expected_trajectorys_steers[vtype].keys():
                self.united_expected_trajectorys[vtype][steer] = shapely.ops.unary_union(self.expected_trajectorys_steers[vtype][steer]['polygons'])
            self.UNITEDPOLYGON[vtype] = shapely.ops.unary_union(self.united_expected_trajectorys[vtype].values())
        #self.
                
        #----------------------set the self.CG_line_coors, self.CG_line_coors[vtype][steer] = [(x1,y1), (x2,y2), ....]
        self.CG_line_coors = {}
        for vtype in self.expected_trajectorys_steers.keys():
            self.CG_line_coors[vtype] = {}
            for steer in self.expected_trajectorys_steers[vtype].keys():
                self.CG_line_coors[vtype][steer] = [(s[0],s[1]) for s in self.expected_trajectorys_steers[vtype][steer]['states']]
                
        #------------------------------------self.backward_attrs['united_expected_trajectorys'][vtype][steer] = polygon.
        #   affine_polygon(self, polygonshapely, new_state= (0, 0 , 90*np.pi/180.0)):
        for vtype in self.united_expected_trajectorys.keys():
            #---------------------CG_line_coors
            self.backward_attrs['CG_line_coors'][vtype] = {}
            for steer in self.united_expected_trajectorys[vtype].keys():
                #[(x1,y1), (x2,y2), ....]
                CG_line_coors = self.CG_line_coors[vtype][steer]
                self.backward_attrs['CG_line_coors'][vtype][steer] = [(xy[0],-xy[1]) for xy in CG_line_coors]
            #--------------------------------------
            self.backward_attrs['united_expected_trajectorys'][vtype] = {}
            for steer in self.united_expected_trajectorys[vtype].keys():
                polygonshapely = self.united_expected_trajectorys[vtype][steer]
                self.backward_attrs['united_expected_trajectorys'][vtype][steer] = self.affine_polygon(polygonshapely= polygonshapely, new_state= (0, 0 , 270*np.pi/180.0))
        

    def __init__BKPSUCCESS(self, max_steering_angle = 70*np.pi/180, N_steering_angles = 300, X0 = [0, 0, 90/180*np.pi],  default_v = 10, delta_t = 0.05, T_horizon = 5, distance_threshold = 30, T_horizon_incremental = 3, lflrlFlR_vtypes_as_key = {"vehicle.audi.a2":{'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2}}, mirror = True):
        """
        Generate the expected trajectory (represented by a list of distance and a list of shapely).
        
        KEY ATTRS:
            - self.expected_trajectorys_steers, a dict. self.expected_trajectorys_steers["vehicle.audi.a2"][steer] = {'polygons':polygonslist, 'distances':distances_list, }
            - self.united_expected_trajectorys_vtypes_steers, united_expected_trajectorys_vtypes_steers["vehicle.audi.a2"][steer]=polygon
        
        X0 = [0, 0, 90/180*np.pi] means the heading angle is 90 degree. 
        ---------------
        
        --------------
        
            
        Expected trajectory for specific steering_angle can naively obtained via:
        
            X0 = state_veh1#x,y,phi
            T = np.linspace(0, T_horizon, int(T_horizon/delta_t)+1) 
            inputs = np.array([[default_v, steering_angle1, 0] for i in range(len(T))]).T
            t, y1 = VM.VehicleKineticSolver.TR_solover(veh_model, T, inputs, X0)
            distances_veh1 = [0]+list(np.cumsum(np.sqrt(np.diff(y1[0,:])**2 + np.diff(y1[1,:])**2)))
            #steps2: obtain the shapelys of the polygons in polygons_veh1, a list. 
            #   shapelys_polygons is multipolygon instance. 
            polygons_veh1 = VM.VehicleKineticSolver.shapelys_given_states(TR = y1, veh_para = veh_para1)
            
        -------------------------------------
        @input: T_horizon_incremental
            if the distance is not long enough, the T_horizon will be increased and trajectory will be calculated again by extending the 
            
        @input: X0
            x,y, phi = X0
            
            phi is the heading angle, unit is np.pi
            
        @input: max_steering_angle and N_steering_angles
        
            max_steering_angle is the max steering angle of front wheel for bicycle model.  unit is np.pi. 
            
            N_steering_angles is the number of discretization of the max_steering_angle.
            
        @input: distance_threshold
            the generated trajectory length threshold. 
            
        @input: mirror
            mirror about the y-axis 
            
            Note that when steering angle is positive, then turn left, else turn right. 
            
            
        @OUTPUT: expected_trajectorys_steers
            a dict. expected_trajectorys_steers keys are steer angle. 
            
            expected_trajectorys_steers[steer_angle] = {'distances':distances, 'polygons':polygons_veh, 'states':states_list}
            
            states_list = [X0, X1， X2,...], X0 = [x,y,phi], phi unit is np.pi
            
            distances,polygons_veh both are list. 
            distances is a list of float, represent the travelling distacne of the CG from X0, and 
            polygons_veh1 is a list of shapely polygon. 
            
        """
        
        #union later in this method, which means unite the polygons along the trajectory together. 
        self.union_or_not = True
        self.vehicle_type = ["vehicle.audi.a2", "vehicle.audi.tt", "vehicle.carlamotors.carlacola", "vehicle.jeep.wrangler_rubicon", "vehicle.chevrolet.impala", "vehicle.mini.cooperst", "vehicle.audi.etron", "vehicle.mercedes-benz.coupe", "vehicle.bmw.grandtourer", "vehicle.toyota.prius", "vehicle.citroen.c3", "vehicle.mustang.mustang", "vehicle.tesla.model3", "vehicle.tesla.cybertruck", "vehicle.volkswagen.t2", "vehicle.lincoln.mkz2017", "vehicle.seat.leon", "vehicle.nissan.patrol", "vehicle.nissan.micra"]
        
        
        #returned value. 
        expected_trajectorys_steers =  {}
        #veh_para1= lflrlFlR_vtypes_as_key[veh_type] for instance, veh_para1= lflrlFlR_vtypes_as_key["vehicle.audi.a2"]
        veh_model = VehicleKineticSolver.control_model_Rajamani_using_v(params = lflrlFlR_vtypes_as_key["vehicle.audi.a2"])
        
        for steering_angle in np.linspace(0, max_steering_angle, N_steering_angles):
            
            
            #-------------obatain distances and polygons_veh
            distances_veh1= []
            while len(distances_veh1)==0 or distances_veh1[-1]<distance_threshold:
                if len(distances_veh1)>0:
                    T_horizon = T_horizon + T_horizon_incremental
                
                T = np.linspace(0, T_horizon, int(T_horizon/delta_t)+1)
                #   the last 0 is for steering angle of the rear wheel.  
                inputs = np.array([[default_v, steering_angle, 0] for i in range(len(T))]).T
                #   t and y1 are np.array. 
                #       t.shape is (len(T),) and y1 shape is (3, len(T))
                t, y1 = VehicleKineticSolver.TR_solover(veh_model, T, inputs, X0)
                distances_veh1 = [0]+list(np.cumsum(np.sqrt(np.diff(y1[0,:])**2 + np.diff(y1[1,:])**2)))
                #steps2: obtain the shapelys of the polygons in polygons_veh1, a list. 
                #   shapelys_polygons is multipolygon instance. 
            
            #find the index in distances_veh1 that distances_veh1[idx]-distance_threshold is minimal
            idx = np.where(np.array(distances_veh1)>=distance_threshold)[0][0]
            
            #
            distances = distances_veh1[:idx]
            polygons_veh = VehicleKineticSolver.shapelys_given_states(TR = y1[:,:idx], veh_para = lflrlFlR_vtypes_as_key["vehicle.audi.a2"])
            #
            expected_trajectorys_steers[steering_angle] = {'distances':distances, 'polygons':polygons_veh}
            
            #mirror the resulf for -steering_angle
            if mirror:
                polygons_veh2 = [shapely.affinity.scale(p, xfact = -1, origin = (0, 0)) for p in polygons_veh]
                expected_trajectorys_steers[-steering_angle] = {'distances':distances, 'polygons':polygons_veh2}
            
            #
            #expected_trajectorys_steers[steering_angle]['states'] = [X0, X1, X2,...]
            expected_trajectorys_steers[steering_angle]['states'] = [list(y1[:,i]) for i in range(idx)]
            
            if mirror:
                #   the state is the same while y is mirrot about y-axis and phi is changed
                expected_trajectorys_steers[-steering_angle]['states'] = [[-y1[0,i],y1[1,i],np.pi-y1[2,i]] for i in range(idx)]
            
        self.expected_trajectorys_steers = expected_trajectorys_steers
        
        self.expected_trajectorys_steers[ "vehicle.audi.a2"] = copy.deepcopy(expected_trajectorys_steers)
        
        
        #------------------------union the 
        #   shapely.ops.unary_union(tmp['polygons'])
        
        
    def NearestSteeringAngle(self, angle, ):
        """
        Find the angle in angles that is most near angle.
        ----------------------------------------------
        @input: angle
            unit is np.pi, it is in rajamani coordinate. 
        
        @input: angles
            a list of angle. 
        
        @OUTPUT:    
            a value in angles that is most near given angle. 
        """
        angles = sorted(self.expected_trajectorys_steers["vehicle.audi.a2"].keys())
        
        difference = np.abs(np.array(angles)-angle)
        idx = np.where(difference== min(difference))[0][0]
        
        return angles[idx]
        


class VehicleKineticSolver():
    """
    The kinetic model of the Rajamani book , at page 27.
    
    
    """
    
    @classmethod
    def a_b_c_vehboundary(self, veh_state = (0 ,0, np.pi/2) , veh_para= {'lR':3, 'lF':2, 'w':2}):
        """
        calculate the function of each boundary of the vehicle. 
        
        Callback method:
            
            (a1,b1,c1),(a2,b2,c2),(a3,b3,c3),(a4,b4,c4) = self.a_b_c_vehboundary( veh_state = (0 ,0, np.pi/2) , veh_para= {'lR':3, 'lF':2, 'w':2})
            
        -----------------------------------
        @input: veh_state
            
            veh_state is either a list of three element
            
            or array with shape (3, N)
            
            
            
        @output: (a1,b1,c1),(a2,2,c2),(a3,b3,c3),(a4,b4,c4)
        
            they correspond to the following four lines: AB BC CD DA
            
            a1 is a float, if veh_state is a list. 
            
            a1 is a 1D array of length N, if veh_state is the array of (3, N)
            
        
        """
        w = veh_para.get('w', 2)
        lF = veh_para.get('lF', 2)
        lR = veh_para.get('lR', 2)

        x,y,phi = veh_state[0],veh_state[1],veh_state[2]
        
        
        #--------------------AB
        a1,b1 = np.cos(phi+np.pi/2),np.sin(phi+np.pi/2)
        corsspoint = x+w/2.0*np.cos(phi-np.pi/2.0),y+w/2.0*np.sin(phi-np.pi/2.0)
        #   solve c1
        c1 = a1*corsspoint[0] + b1*corsspoint[1]
        
        #-------------------BC
        a2,b2 = np.cos(phi),np.sin(phi)
        corsspoint = x-lR*np.cos(phi),y-lR*np.sin(phi)
        c2 = a2*corsspoint[0] + b2*corsspoint[1]
        
        #CD
        a3,b3 = np.cos(phi+np.pi/2),np.sin(phi+np.pi/2)
        corsspoint = x+w/2.0*np.cos(phi+np.pi/2.0),y+w/2.0*np.sin(phi+np.pi/2.0)
        #c3
        c3 = a3*corsspoint[0] + b3*corsspoint[1]
        
        #DA
        a4,b4 = np.cos(phi),np.sin(phi)
        corsspoint = x+lF*np.cos(phi),y+lF*np.sin(phi)
        c4 = a4*corsspoint[0] + b4*corsspoint[1]
        
        #
        return (a1,b1,c1),(a2,b2,c2),(a3,b3,c3),(a4,b4,c4)
        
    
    
    
    @classmethod
    def LimitStates(self, X, phi_limit = 0, ):
        """
        
        """
        
        
        
        pass
    
    @classmethod
    def InnerVehBoundary_points(self, veh_state = (0 ,0, np.pi/2) , points = np.array([[10], [10]]), points4cal_dis = False, veh_para= {'lR':3, 'lF':2, 'w':2},  area_buffer = 1e-2):
        """
        TEST OK. 
        
        Determine whether point is within the vehicle boundary. 
        
        If yes, find the distance. the distaance is the 
        
        @input: points4cal_dis
            
            the points used to calculate the distance. 
            
            should be the same as points. 
            
        @input: points
            shape is (2, N)
            
            0th row is the x and 1st row is y. 
        
        @output: invehicle_ornot,distance
            invehicle_ornot
                a array of False or True. lengh is the same as points.shape[1].
                
                If True, if means inside the vehicle boundary. 
            
            distance
                a float. the distance outside the vehicle. 
        """
        w = veh_para.get('w', 2)
        lF = veh_para.get('lF', 2)
        lR = veh_para.get('lR', 2)
        veh_area = w*(lR+lF)
        
        #abc1,abc2,abc3,abc4 = abc_es
        #   abc1 = a1,b1,c1, which means the line a1*x+b1*y = c1#
        abc_es = self.a_b_c_vehboundary(veh_state =veh_state , veh_para= veh_para)
        lengths_corres_abc_es = lF+lR,w,lF+lR,w
        #calculate
        
        #areas_triangle is a list. areas_triangle[idx] = np.array, shape is (N,), where N=points.shape[1]
        areas_triangle = [.5*l*np.abs(abc[0]*points[0] + abc[1]*points[1] - abc[2])/np.sqrt(abc[0]**2+abc[1]**2) for abc,l in zip(abc_es,lengths_corres_abc_es)]
        
        #invehicle_ornot is an 1D array of True or False. lenth is N. 
        #   np.array(areas_triangle).shape = (4, N), where N=points.shape[1]
        #builtins.tmp = np.array(areas_triangle).sum(axis = 0)
        invehicle_ornot =  np.array(areas_triangle).sum(axis = 0) <= (veh_area+area_buffer)
        
        #in_veh_idxs is an array of idx.
        in_veh_idxs = np.where(invehicle_ornot)[0]
        
        #print(invehicle_ornot, veh_area)
        if len(in_veh_idxs)==0:
            
            return False,0
        else:
            
            if isinstance(points4cal_dis,bool):
                xs = points[0,:in_veh_idxs[0]]
                ys = points[1,:in_veh_idxs[0]]
                return True,np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2))
            else:
                xs = points4cal_dis[0,:in_veh_idxs[0]]
                ys = points4cal_dis[1,:in_veh_idxs[0]]
                return True,np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2))
        
    @classmethod
    def InnerVehBoundary_singlepoint(self, veh_state = (0 ,0, np.pi/2) , point = (10, 10), veh_para= {'lR':3, 'lF':2, 'w':2}):
        """
        TEST OK.
        Determine whether point is within the vehicle boundary. 
        
        @input: 
        
        """
        w = veh_para.get('w', 2)
        lF = veh_para.get('lF', 2)
        lR = veh_para.get('lR', 2)
        veh_area = w*(lR+lF)
        
        
        #abc1,abc2,abc3,abc4 = abc_es
        #   abc1 = a1,b1,c1, which means the line a1*x+b1*y = c1#
        abc_es = self.a_b_c_vehboundary(veh_state =veh_state , veh_para= veh_para)
        lengths_corres_abc_es = lF+lR,w,lF+lR,w
        #calculate
        
        areas_triangle = [.5*l*np.abs(abc[0]*point[0] + abc[1]*point[1] - abc[2])/np.sqrt(abc[0]**2+abc[1]**2) for abc,l in zip(abc_es,lengths_corres_abc_es)]

            
        if sum(areas_triangle)>veh_area:
            return False
        else:
            return True
    
    
    @classmethod
    def FourCornersCoor(self, x=0 , y = 0, ang = 30*np.pi/180.0, veh_para= {'lR':3, 'lF':2, 'w':2}):
        """
        Compute the four corners of the vehicle, given the x,y and phi of the CG. 
        Callback method:
            - 
        ------------------------------------------------
        @input: x and y
            the coor of the CG
        @input: ang
            the heading angle 
            the heading angle of the vehicle. 
            unit is degree*np.pi/180. 
            
        @input: veh_para
            a dict containing the vehicle parameter. 
            
            lR = veh_para['lR']
            LF = veh_para['lF']
            w = veh_para['w']
            
        --------------------------------------------------
        @OUTPUT: ((x0_right_2,y0_right_2), (x0_right_1, y0_right_1), (x0_left_1,y0_left_1), (x0_left_2,y0_left_2))
        
            the order: front-right, rear-right, rear-left, front-left
        """
        lR = veh_para['lR']
        lF = veh_para['lF']
        w = veh_para['w']
        
        #--------------------------rear left--
        #the coordinate of the point of left boundary, CG-this_point is parallel with front and rear bump. 
        x0_left = x + w/2.0*np.cos(ang+90.0/180*np.pi)
        y0_left = y + w/2.0*np.sin(ang+90.0/180*np.pi)
        #
        x0_left_1 = x0_left - lR*np.cos(ang)
        y0_left_1 = y0_left - lR*np.sin(ang)
        
        #--------------------------rear right
        #the coordinate of the point of right boundary, CG-this_point is parallel with front and rear bump. 
        x0_right = x + w/2.0*np.cos(ang+270.0/180*np.pi)
        y0_right = y + w/2.0*np.sin(ang+270.0/180*np.pi)
        #
        x0_right_1 = x0_right - lR*np.cos(ang)
        y0_right_1 = y0_right - lR*np.sin(ang)
        
        #---------------------front left
        x0_left = x + w/2.0*np.cos(ang+90.0/180*np.pi)
        y0_left = y + w/2.0*np.sin(ang+90.0/180*np.pi)
        #
        x0_left_2 = x0_left + lF*np.cos(ang)
        y0_left_2 = y0_left + lF*np.sin(ang)
        
        #---------------------front right
        x0_right = x + w/2.0*np.cos(ang+270.0/180*np.pi)
        y0_right = y + w/2.0*np.sin(ang+270.0/180*np.pi)
        #
        x0_right_2 = x0_right + lF*np.cos(ang)
        y0_right_2 = y0_right + lF*np.sin(ang)
        
        #the order: front-right, rear-right, rear-left, front-left
        return ((x0_right_2,y0_right_2), (x0_right_1, y0_right_1), (x0_left_1,y0_left_1), (x0_left_2,y0_left_2))
    
    @classmethod
    def DistanceVeh2Curve(self, xsys = ((0, 1), (0,1)), X1 = [0,0, 30*np.pi/180], veh_params = {}):
        """
        Calculate the minimal distance of vehicle (indicated by X1) and the curve, which is indicated by xsys = (xs, ys). xs and ys are list. 
        
        @inputL veh_params
            the physical parameters of the vehicles. 
        """
        lr = params.get('lr', 2)
        lf = params.get('lf', 2)
        lR = params.get('lr', 2)
        lF = params.get('lf', 2)
                
        
        
        pass
    
    
    
    @classmethod
    def Lock_free_distance(self, ):
        """
        
        """
        
        
        
        
        pass

    @classmethod
    def CornerB_y_given_x(self, deltaf,veh_state_init  = (0, 0, 90*np.pi/180.0), x = 0, veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }):
        """
        The y coor of corner A given the x. 
        
        There are four corners of a vehicle, ABCD, They are:
            - front-right, rear-right, rear-left, front-left
        
        @input: deltaf
            the steering angle of the front wheel. 
            Unit is np.pi.
        
        
        @OUTPUT; y
            a float. 
            
            (x,y) will be on the trajectory of corner A. 
        -------------------------------
        
        
        
        """
        
        
        
        
        pass

    @classmethod
    def CornerC_y_given_x(self, deltaf,veh_state_init  = (0, 0, 90*np.pi/180.0), x = 0, veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }):
        """
        The y coor of corner A given the x. 
        
        There are four corners of a vehicle, ABCD, They are:
            - front-right, rear-right, rear-left, front-left
        
        @input: deltaf
            the steering angle of the front wheel. 
            Unit is np.pi.
        
        
        @OUTPUT; y
            a float. 
            
            (x,y) will be on the trajectory of corner A. 
        -------------------------------
        
        
        
        """
        
        
        
        
        pass
    
    
    
    
    @classmethod
    def CornerD_y_given_x(self, deltaf,veh_state_init  = (0, 0, 90*np.pi/180.0), x = 0, veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }):
        """
        The y coor of corner A given the x. 
        
        There are four corners of a vehicle, ABCD, They are:
            - front-right, rear-right, rear-left, front-left
        
        @input: deltaf
            the steering angle of the front wheel. 
            Unit is np.pi.
        
        
        @OUTPUT; y
            a float. 
            
            (x,y) will be on the trajectory of corner A. 
        -------------------------------
        
        
        
        """
        
        
        
        
        pass

    @classmethod
    def y_drivate_with_respect_to_x(self,x ,deltaf, phi0,lf, lr, x0):
        """
        Integrand for self.CGy_Given_x.
        
        @input: x0
            the initial state of x. 
        
        """
        #   print(lr,lf, front_steer_angle, np.tan(front_steer_angle))
        tmp = (lr*np.tan(deltaf) + lf*np.tan(0))/(lr+lf)
        beta = np.arctan(tmp)
        
        #
        tmp1= (x-x0)*np.cos(beta)*np.tan(deltaf)/(lf+lr) + np.sin(phi0+beta)
        return np.tan(np.arcsin(tmp1))
        
        
        pass

    
    @classmethod
    def y_integrand_given_x(self,x ,deltaf, phi0,lf, lr, x0):
        """
        Integrand for self.CGy_Given_x.
        
        @input: x0
            the initial state of x. 
        
        """
        #   print(lr,lf, front_steer_angle, np.tan(front_steer_angle))
        tmp = (lr*np.tan(deltaf) + lf*np.tan(0))/(lr+lf)
        beta = np.arctan(tmp)
        
        #
        tmp1= (x-x0)*np.cos(beta)*np.tan(deltaf)/(lf+lr) + np.sin(phi0+beta)
        return np.tan(np.arcsin(tmp1))
    
    @classmethod
    def debug_arcsin(self,x ,deltaf, lf = 2, lr = 2, x0 = 0, phi0 = 90*np.pi/180.0):
        """
        Integrand for self.CGy_Given_x.
        
        @input: x0
            the initial state of x. 
        
        """
        #   print(lr,lf, front_steer_angle, np.tan(front_steer_angle))
        tmp = (lr*np.tan(deltaf) + lf*np.tan(0))/(lr+lf)
        beta = np.arctan(tmp)
        
        #
        tmp1= (x-x0)*np.cos(beta)*np.tan(deltaf)/(lf+lr) + np.sin(phi0+beta)
        return tmp1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    @classmethod
    def deltaf_sign_detemination_tmp(self, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, point = (10, 10),):
        """
        Determine the deltaf will be positive or negative. 
        Turn left means positive and turn right means negative. 
        
        
        Thus if point locates on the left of veh_state_init, return 'left' or 'right'
        
        Principle: 
            roate the angle, and check the sign of the y. 
            
            
        
        
        
        -------------------------------------------
        @OUTPUT: left_or_right
            a str. either 'left' or 'right'
        
            middle means the given point locates . 
        -----------------------------
        
        
        """
        #--------------
        x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
        x,y = point
        
        #rotate the line (x0,y0)--->(x,y)
        #   first step: first get the angle of this line, in angle_line, unit is degree, [0, 360]. 
        #   second step: 
        #relative degree which is represented by angle_line, unit is degree, [0, 360]. 
        #   angle_line is in Rajami coordinate. 
        #   arctan2 return value within [-pi.pi, np.pi]:
        #       x = np.array([-1, +1, +1, -1])
        #       y = np.array([-1, -1, +1, +1])
        #       np.arctan2(y, x) * 180 / np.pi
        #       array([-135.,  -45.,   45.,  135.])
        degree_relative0 = np.arctan2(y-y0, x-x0) * 180 / np.pi
        if degree_relative0<0:
            angle_line = 360 + degree_relative0
        else:
            angle_line = degree_relative0

        #-----------------------------
        #angle_after_rotation unit is degree. 
        angle_after_rotation = angle_line - phi0 * 180 / np.pi
        
        #-----------------------------
        new_y = np.sin(angle_after_rotation*np.pi/180.0)
        
        if new_y>0:
            return 'left'
        elif new_y==0:
            return 'middle'
        else:
            return 'right'
        
        
        
        
        
        
        
        
        
        
        #print(phi0 * 180 / np.pi, np.arctan2(y1-y0, x1-x0),degree_relative0,  angle_line)
        #--------------make sure that relative_angle is within [0, 180]
        #WRONG:  relative_angle = np.abs(phi0 * 180 / np.pi-angle_line) if np.abs(phi0 * 180 / np.pi -angle_line)<=180 else 180-np.abs(phi0 * 180 / np.pi -angle_line)
        if np.abs(phi0 * 180 / np.pi -angle_line)<=180:
            relative_angle = np.abs(phi0 * 180 / np.pi-angle_line)
        else:
            relative_angle = 360-np.abs(phi0 * 180 / np.pi -angle_line)
        
        #
        if relative_angle>ahead_angle_degree_limit:
            #at behind
            return False
        else:
            #at front. 
            
            
            pass
            
        #front-right, rear-right, rear-left, front-left
        #pointA,pointB,pointC,pointD = self.FourCornersCoor(x=x0, y = y0, ang = phi0, veh_para= veh_para)
        
        #



    @classmethod
    def SteerAngleGivenCornerAndPathPoint(self, point = (10, 10), v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, init_solution0 = (10, -10*np.pi/180), corner = 'A', forward_backward = 1, ):
        """
        find the steer angle (unit is np.pi) that makes the front right point traverse certain point. 
        
        the trajectory of corner A is given in self.CornerA_at_t().
        
        
        
        -------------------------------------------------
        @input: point
            the point that front right corner pass. 
            
        @input: veh_state_init
        
            x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
            
            phi0 unit is np.pi. 
        
        @input: init_solution0
        
            the init solution. 
        
            t_solution, deltaf_solution = init_solution0
            
            unit of deltaf is np.pi.
            
            The sign of deltaf_solution will be adjusted. 
            
        @OUTPUT: t,deltaf,distance
            
            
            
            float, unit is np.pi. 
        
        ----------------------------------------------
        
        Using sympy to solve the CG_x,CG_y,phi:
            
            v, phi, beta, deltaf, lf ,lr ,w, xa, x0,y0, x ,t, phi0,derivate= sympy.symbols('v phi beta deltaf lf lr w xa x0 y0 x t phi0 derivate')
            beta = sympy.atan(lr*sympy.tan(deltaf)/(lf+lr))

            #--------------------
            derivate_phi = v*sympy.tan(deltaf)*sympy.cos(beta)/(lf+lr)
            derivate_x = v*sympy.cos(phi+beta)
            derivate_y = v*sympy.sin(phi+beta)

            phi =  phi0 + sympy.integrate(derivate_phi, (t, 0, t))
            x = x0 + sympy.integrate(derivate_x, (t, 0, t))
            y = y0 + sympy.integrate(derivate_y, (t, 0, t))

        
        """
        #----------------------------check the sign of deltaf of the initial solution.
        #   if turn left, positive; if turn right, negative. 
        x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
        pointA,pointB,pointC,pointD = self.FourCornersCoor(x=x0, y = y0, ang = phi0, veh_para= veh_para)
        
        if corner=='A':
            
            #--------------determine left or right, to determine the sign of initial solution.     
            #   left_or_right is either 'left' or 'right'
            left_or_right = self.deltaf_sign_detemination_tmp(veh_state_init  = (pointA[0], pointA[1], phi0), veh_para= veh_para, point = point)
            
            #----initial solution
            if left_or_right=='left':
                init_solution = (init_solution0[0], np.abs(init_solution0[1]))
            elif left_or_right=='middle':
                distance  = np.sqrt((pointA[0] - point[0])**2 + (pointA[1] - point[1])**2)
                t = distance/v
                return t,0.0,distance
            else:
                init_solution = (init_solution0[0], -1*np.abs(init_solution0[1]))
            
            # CORNER_A_SOLVE_fun(t_and_deltaf, point = (10, 10),v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, forward_backward = 1):
            args = (point, v, veh_state_init, veh_para, forward_backward)
            #   z = (t,deltaf)
            z = scipy.optimize.fsolve(CORNER_A_SOLVE_fun, init_solution, args)
            
            #find the distance that travelled. 
            #CORNER_DISYANCE_t(t_and_deltaf = (10, 30*np.pi/180), v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, corner = 'A')
            #distancetravelled = CORNER_DISYANCE_t(t_and_deltaf = z, v = v, veh_state_init  = veh_state_init, veh_para = veh_para, corner = 'A', )
            distancetravelled = CG_DISTANCE_T(t_and_deltaf = z, v = v, veh_state_init  = veh_state_init, veh_para= veh_para)
            
            #z = (t,deltaf)
            return z[0],z[1],distancetravelled
        
        elif corner=='B':
            #--------------determine left or right, to determine the sign of initial solution.   
            #   left_or_right is either 'left' or 'right'
            left_or_right = self.deltaf_sign_detemination_tmp(veh_state_init  = (pointB[0], pointB[1], phi0), veh_para= veh_para, point = point)
            
            #----initial solution
            if left_or_right=='left':
                init_solution = (init_solution0[0], np.abs(init_solution0[1]))
            elif left_or_right=='middle':
                
                distance  = np.sqrt((pointB[0] - point[0])**2 + (pointB[1] - point[1])**2)
                t = 1.0*distance/v
                return t,0.0,distance
                
                return 0.0
            else:
                init_solution = (init_solution0[0], -1*np.abs(init_solution0[1]))
            
            #CORNER_A_SOLVE_fun(t_and_deltaf, point = (10, 10),v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, })
            args = (point, v, veh_state_init, veh_para, forward_backward)
            z = scipy.optimize.fsolve(CORNER_B_SOLVE_fun, init_solution, args)
            
            #find the distance that travelled. 
            #CORNER_DISYANCE_t(t_and_deltaf = (10, 30*np.pi/180), v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, corner = 'A')
            #distancetravelled = CORNER_DISYANCE_t(t_and_deltaf = z, v = v, veh_state_init  = veh_state_init, veh_para = veh_para, corner = 'B', )
            distancetravelled = CG_DISTANCE_T(t_and_deltaf = z, v = v, veh_state_init  = veh_state_init, veh_para= veh_para)
            
            #z = (t,deltaf)
            return z[0],z[1],distancetravelled
            
        elif corner=='C':
            
            #   left_or_right is either 'left' or 'right'
            left_or_right = self.deltaf_sign_detemination_tmp(veh_state_init  = (pointC[0], pointC[1], phi0), veh_para= veh_para, point = point)
            
            #----initial solution
            if left_or_right=='left':
                init_solution = (init_solution0[0], np.abs(init_solution0[1]))
            elif left_or_right=='middle':
                
                distance  = np.sqrt((pointC[0] - point[0])**2 + (pointC[1] - point[1])**2)
                t = 1.0*distance/v
                return t,0.0,distance
                
                
                return 0.0
            else:
                init_solution = (init_solution0[0], -1*np.abs(init_solution0[1]))
            
            #CORNER_A_SOLVE_fun(t_and_deltaf, point = (10, 10),v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, })
            args = (point, v, veh_state_init, veh_para, forward_backward)
            z = scipy.optimize.fsolve(CORNER_C_SOLVE_fun, init_solution, args)
            
            #find the distance that travelled. 
            #CORNER_DISYANCE_t(t_and_deltaf = (10, 30*np.pi/180), v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, corner = 'A')
            #distancetravelled = CORNER_DISYANCE_t(t_and_deltaf = z, v = v, veh_state_init  = veh_state_init, veh_para = veh_para, corner = 'C', )
            distancetravelled = CG_DISTANCE_T(t_and_deltaf = z, v = v, veh_state_init  = veh_state_init, veh_para= veh_para)
            
            #z = (t,deltaf)
            return z[0],z[1],distancetravelled
            
        elif corner=='D':
            
            #   left_or_right is either 'left' or 'right'
            left_or_right = self.deltaf_sign_detemination_tmp(veh_state_init  = (pointD[0], pointD[1], phi0), veh_para= veh_para, point = point)
            
            #----initial solution
            if left_or_right=='left':
                init_solution = (init_solution0[0], np.abs(init_solution0[1]))
            elif left_or_right=='middle':
                
                
                distance  = np.sqrt((pointD[0] - point[0])**2 + (pointD[1] - point[1])**2)
                t = 1.0*distance/v
                return t,0.0,distance
                
                return 0.0
            else:
                init_solution = (init_solution0[0], -1*np.abs(init_solution0[1]))
            
            #CORNER_A_SOLVE_fun(t_and_deltaf, point = (10, 10),v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, })
            args = (point, v, veh_state_init, veh_para, forward_backward)
            z = scipy.optimize.fsolve(CORNER_D_SOLVE_fun, init_solution, args)
            
            #find the distance that travelled. 
            #CORNER_DISYANCE_t(t_and_deltaf = (10, 30*np.pi/180), v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, corner = 'A')
            #distancetravelled = CORNER_DISYANCE_t(t_and_deltaf = z, v = v, veh_state_init  = veh_state_init, veh_para = veh_para, corner = 'D', )
            distancetravelled = CG_DISTANCE_T(t_and_deltaf = z, v = v, veh_state_init  = veh_state_init, veh_para= veh_para)
            
            #z = (t,deltaf)
            return z[0],z[1],distancetravelled
            

    @classmethod
    def SteerAngleGivenCornerAndPathPoint_BKP_SUCCESS(self, point = (10, 10),v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, init_solution0 = (10, -10*np.pi/180), corner = 'A'):
        """
        find the steer angle (unit is np.pi) that makes the front right point traverse certain point. 
        
        the trajectory of corner A is given in self.CornerA_at_t().
        
        
        
        -------------------------------------------------
        @input: point
            the point that front right corner pass. 
            
        @input: veh_state_init
        
            x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
            
            phi0 unit is np.pi. 
        
        @input: init_solution0
        
            the init solution. 
        
            t_solution, deltaf_solution = init_solution0
            
            unit of deltaf is np.pi.
            
            The sign of deltaf_solution will be adjusted. 
            
        @OUTPUT: steerangle
            float, unit is np.pi. 
        
        ----------------------------------------------
        
        Using sympy to solve the CG_x,CG_y,phi:
            
            v, phi, beta, deltaf, lf ,lr ,w, xa, x0,y0, x ,t, phi0,derivate= sympy.symbols('v phi beta deltaf lf lr w xa x0 y0 x t phi0 derivate')
            beta = sympy.atan(lr*sympy.tan(deltaf)/(lf+lr))

            #--------------------
            derivate_phi = v*sympy.tan(deltaf)*sympy.cos(beta)/(lf+lr)
            derivate_x = v*sympy.cos(phi+beta)
            derivate_y = v*sympy.sin(phi+beta)

            phi =  phi0 + sympy.integrate(derivate_phi, (t, 0, t))
            x = x0 + sympy.integrate(derivate_x, (t, 0, t))
            y = y0 + sympy.integrate(derivate_y, (t, 0, t))

        
        """
        #----------------------------check the sign of deltaf of the initial solution.
        #   if turn left, positive; if turn right, negative. 
        x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
        pointA,pointB,pointC,pointD = self.FourCornersCoor(x=x0, y = y0, ang = phi0, veh_para= veh_para)
        
        
        if corner=='A':
                
            #   left_or_right is either 'left' or 'right'
            left_or_right = self.deltaf_sign_detemination_tmp(veh_state_init  = (pointA[0], pointA[1], phi0), veh_para= veh_para, point = point)
            
            
            #----initial solution
            if left_or_right=='left':
                init_solution = (init_solution0[0], np.abs(init_solution0[1]))
            elif left_or_right=='middle':
                return 0.0
            else:
                init_solution = (init_solution0[0], -1*np.abs(init_solution0[1]))
            
            #CORNER_A_SOLVE_fun(t_and_deltaf, point = (10, 10),v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, })
            args = (point, v, veh_state_init, veh_para)
            #   z = (t,deltaf)
            z = scipy.optimize.fsolve(CORNER_A_SOLVE_fun, init_solution, args)
            
            #find the distance that travelled. 
            
            
            #z = (t,deltaf)
            return z
        
        elif corner=='B':
            
            #   left_or_right is either 'left' or 'right'
            left_or_right = self.deltaf_sign_detemination_tmp(veh_state_init  = (pointB[0], pointB[1], phi0), veh_para= veh_para, point = point)
            
            #----initial solution
            if left_or_right=='left':
                init_solution = (init_solution0[0], np.abs(init_solution0[1]))
            elif left_or_right=='middle':
                return 0.0
            else:
                init_solution = (init_solution0[0], -1*np.abs(init_solution0[1]))
            
            #CORNER_A_SOLVE_fun(t_and_deltaf, point = (10, 10),v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, })
            args = (point, v, veh_state_init, veh_para)
            z = scipy.optimize.fsolve(CORNER_B_SOLVE_fun, init_solution, args)
            
            #z = (t,deltaf)
            return z
            
        elif corner=='C':
            
            #   left_or_right is either 'left' or 'right'
            left_or_right = self.deltaf_sign_detemination_tmp(veh_state_init  = (pointC[0], pointC[1], phi0), veh_para= veh_para, point = point)
            
            #----initial solution
            if left_or_right=='left':
                init_solution = (init_solution0[0], np.abs(init_solution0[1]))
            elif left_or_right=='middle':
                return 0.0
            else:
                init_solution = (init_solution0[0], -1*np.abs(init_solution0[1]))
            
            #CORNER_A_SOLVE_fun(t_and_deltaf, point = (10, 10),v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, })
            args = (point, v, veh_state_init, veh_para)
            z = scipy.optimize.fsolve(CORNER_C_SOLVE_fun, init_solution, args)
            
            #z = (t,deltaf)
            return z
            
        elif corner=='D':
            
            #   left_or_right is either 'left' or 'right'
            left_or_right = self.deltaf_sign_detemination_tmp(veh_state_init  = (pointD[0], pointD[1], phi0), veh_para= veh_para, point = point)
            
            #----initial solution
            if left_or_right=='left':
                init_solution = (init_solution0[0], np.abs(init_solution0[1]))
            elif left_or_right=='middle':
                return 0.0
            else:
                init_solution = (init_solution0[0], -1*np.abs(init_solution0[1]))
            
            #CORNER_A_SOLVE_fun(t_and_deltaf, point = (10, 10),v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, })
            args = (point, v, veh_state_init, veh_para)
            z = scipy.optimize.fsolve(CORNER_D_SOLVE_fun, init_solution, args)
            
            #z = (t,deltaf)
            return z
            



    @classmethod
    def SteerAngleGiven_CornerA_PathPoint(self, point = (10, 10),v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, init_solution0 = (10, 0*np.pi/180)):
        """
        find the steer angle (unit is np.pi) that makes the front right point traverse certain point. 
        
        the trajectory of corner A is given in self.CornerA_at_t().
        
        
        
        -------------------------------------------------
        @input: point
            the point that front right corner pass. 
            
        @input: veh_state_init
        
            x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
            
            phi0 unit is np.pi. 
        
        @OUTPUT: steerangle
            float, unit is np.pi. 
        
        ----------------------------------------------
        
        Using sympy to solve the CG_x,CG_y,phi:
            
            v, phi, beta, deltaf, lf ,lr ,w, xa, x0,y0, x ,t, phi0,derivate= sympy.symbols('v phi beta deltaf lf lr w xa x0 y0 x t phi0 derivate')
            beta = sympy.atan(lr*sympy.tan(deltaf)/(lf+lr))

            #--------------------
            derivate_phi = v*sympy.tan(deltaf)*sympy.cos(beta)/(lf+lr)
            derivate_x = v*sympy.cos(phi+beta)
            derivate_y = v*sympy.sin(phi+beta)

            phi =  phi0 + sympy.integrate(derivate_phi, (t, 0, t))
            x = x0 + sympy.integrate(derivate_x, (t, 0, t))
            y = y0 + sympy.integrate(derivate_y, (t, 0, t))

        
        """
        #----------------------------check the sign of deltaf of the initial solution.
        #   if turn left, positive; if turn right, negative. 
        x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
        pointA,pointB,pointC,pointD = self.FourCornersCoor(x=x0, y = y0, ang = phi0, veh_para= veh_para)
        #   left_or_right is either 'left' or 'right'
        left_or_right = self.deltaf_sign_detemination_tmp(veh_state_init  = (pointA[0], pointA[1], phi0), veh_para= veh_para, point = point)
        
        #----initial solution
        if left_or_right=='left':
            init_solution = (init_solution0[0], np.abs(init_solution0[1]))
        else:
            init_solution = (init_solution0[0], -1*np.abs(init_solution0[1]))


        #CORNER_A_SOLVE(t_and_deltaf, point = (10, 10),v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }):
        
        args = (point, v, veh_state_init, veh_para)
        z = scipy.optimize.fsolve(CORNER_A_SOLVE, init_solution, args)
        
        #z = (t,deltaf)
        return z
        
        
    
    
    @classmethod
    def CGy_Given_x(self, deltaf  = 10*np.pi/180.0, veh_state_init  = (0, 0, 90*np.pi/180.0), x = 10, veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, N_x = 100):
        """
        Given the control input, deltaf and v
        Calculate the y given x. 
        ------------------------------------
        @input: deltaf
            the angle of front wheel. 
        
        """
        #the derivates for each x
        drivates = []
        #   the integral.
        to_sums = []
        
        #@@@@@@@@@@@@@@@@
        debug_arcsins = []
        
        
        #
        x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
        #
        lr = veh_para.get('lr', 2)
        lf = veh_para.get('lf', 2)
        
        #beta, the intermediate parameter
        #   print(lr,lf, front_steer_angle, np.tan(front_steer_angle))
        #tmp = (lr*np.tan(deltaf) + lf*np.tan(0))/(lr+lf)
        #beta = np.arctan(tmp)
        
        #
        xs= np.linspace(x0, x, N_x)
        
        to_sums = []
        for x_a,x_b in zip(xs[:-1], xs[1:]):
            drivate = self.y_drivate_with_respect_to_x(x=x_a, deltaf=deltaf, phi0 = phi0,lf=lf, lr=lr, x0=x0)
            tmp = (x_b-x_a)*drivate
            debug_arcsin = self.debug_arcsin(x=x_a, deltaf=deltaf, phi0 = phi0,lf=lf, lr=lr, x0=x0)
            
            to_sums.append(tmp)
            drivates.append(drivate)
            debug_arcsins.append(debug_arcsin)
            
        #return to_sums
        return y0+sum(to_sums),drivates,debug_arcsins
        
        
        
        #control input
        V = U[0]
        front_steer_angle = U[1]
        rear_steer_angle = U[2]
        
        #states variable. 
        X = STATES[0]
        Y = STATES[1]
        PHI = STATES[2]
        

        
        #_diff means differential 
        diff_X = V*np.cos( PHI + beta)
        diff_Y = V*np.sin( PHI + beta)
        diff_PHI = V*np.cos(beta)/(lr+lf)*np.tan(front_steer_angle)
        return [diff_X,diff_Y,diff_PHI]

    
    
    
    
    @classmethod
    def CG_and_phi_at_t(self, t= 50, deltaf = 1*np.pi/180.0, v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }):
        """
        Calculate the CG (both x and y) at moment t. 
        ----------------------------------
        @input: deltaf
        
            unit is np.pi. 
        
        @input: v
            unit is m/s. 
        
        @input: deltaf
            turn left correspond to positive. 
        
        
        """
        x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
        
        #
        lr = veh_para.get('lr', 1.7)
        lf = veh_para.get('lf', 1.7)
        lR = veh_para.get('lR', 2)
        lF = veh_para.get('lF', 2)
        w = veh_para.get('w', 2)
        
        #
        beta = np.arctan((lr*np.tan(deltaf))/(lf+lr))
        
        phi = phi0 + t*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr)
        
        cg_x = x0 + (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.sin(phi0 + t*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.sin(phi0 + beta))
        
        cg_y =  y0 - (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.cos(phi0 + t*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.cos(phi0 + beta))
        
        return (cg_x,cg_y, phi)
    
    @classmethod
    def CG_until_t(self, t = 20, N_t = 200, deltaf = 10*np.pi/180.0, v = 4, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, phi_change_not_exceed = np.pi/2, truncate_distance = False):#t= 50, deltaf = 1*np.pi/180.0,
        """
        Calculate the corners trajectory until t. 
        
        Callbackmethod: 
        
            cg_x_es, cg_y_es, phi_es = self.CG_until_t(t = 20, N_t = 200, deltaf = 10*np.pi/180.0, v = 4, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, phi_change_not_exceed = np.pi/2, truncate_distance = False)
            
        ----------------------------------
        @input: phi_change_not_exceed
            limit the change of the traejctories. 
        
        @input: truncate_distance
            
            truncate the distance. 
            
        @OUTPUT: (cg_x_es, cg_y_es, phi_es)
        
            tuple of array. 
        """
        
        ts = np.linspace(0, t, N_t)
        
        x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
        
        #
        lr = veh_para.get('lr', 1.7)
        lf = veh_para.get('lf', 1.7)
        lR = veh_para.get('lR', 2)
        lF = veh_para.get('lF', 2)
        w = veh_para.get('w', 2)
        
        if deltaf==0:
            phi_es  = phi0 + ts*0
            cg_x_es = x0 + ts*v*np.cos(phi0)
            cg_y_es = y0 + ts*v*np.sin(phi0)

        else:
        
            #
            beta = np.arctan((lr*np.tan(deltaf))/(lf+lr))
            
            #----------------------------
            #phis is a np.array, shape is the same as ts. 
            phi_es  = phi0 + ts*v*np.cos(beta)*np.tan(deltaf)/(lf+lr)
            #-------------------truncate the ts, to make sure that the change of phi should not exceed phi_change_not_exceed.
            if phi_change_not_exceed!=False:
                #------------------------phi_change domain is [0, 2*np.pi]
                phi_change =  np.mod(phi_es -phi0, 2*np.pi)
                #-------------------------invalid_idxs is an array
                invalid_idxs = np.where((phi_change<=2*np.pi-phi_change_not_exceed) & (phi_change>=phi_change_not_exceed))[0]
                
                if len(invalid_idxs)>0:
                    ts = np.linspace(0, ts[invalid_idxs[0]-1], N_t)
                    #ts  = ts[:invalid_idxs[0]]
                    
                    #ts = ts[(phi_change <= phi_change_not_exceed) | (phi_change >= 2*np.pi-phi_change_not_exceed)]
                    phi_es  = phi0 + ts*v*np.cos(beta)*np.tan(deltaf)/(lf+lr)
            
            
            #cg_x_es and cg_y_es
            cg_x_es = x0 + (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.sin(phi0 + ts*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.sin(phi0 + beta))
            #
            cg_y_es =  y0 - (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.cos(phi0 + ts*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.cos(phi0 + beta))
        
        if isinstance(truncate_distance, bool):
            
            #cg_x_es i array. 
            return (cg_x_es, cg_y_es, phi_es)
        else:
            #distances length is len(cg_x_es)-1
            tmp = np.sqrt(np.diff(cg_x_es)**2+np.diff(cg_y_es)**2)
            distances = np.cumsum(tmp)
            
            #find the idx
            if distances[-1]<=truncate_distance:
                return (cg_x_es, cg_y_es, phi_es)
            
            #find the idx that 
            #   np.where(np.linspace(0,10,100)<2)[0] is an array of idx. 
            idx = np.where(distances<=truncate_distance)[0][-1]
            return (cg_x_es[:idx+2], cg_y_es[:idx+2], phi_es[:idx+2])
            
            pass
        
        
        #cg_x_es i array. 
        return (cg_x_es, cg_y_es, phi_es)
    
        
    
    
    @classmethod
    def CG_at_t(self, t= 50, deltaf = 1*np.pi/180.0, v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }):
        """
        Calculate the CG (both x and y) at moment t. 
        ----------------------------------
        @input: deltaf
        
            unit is np.pi. 
        
        @input: v
            unit is m/s. 
        
        @input: deltaf
            turn left correspond to positive. 
        
        
        """
        x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
        
        #
        lr = veh_para.get('lr', 1.7)
        lf = veh_para.get('lf', 1.7)
        lR = veh_para.get('lR', 2)
        lF = veh_para.get('lF', 2)
        w = veh_para.get('w', 2)
        
        #
        beta = np.arctan((lr*np.tan(deltaf))/(lf+lr))
        
        
        cg_x = x0 + (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.sin(phi0 + t*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.sin(phi0 + beta))
        
        cg_y =  y0 - (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.cos(phi0 + t*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.cos(phi0 + beta))
        
        return (cg_x,cg_y)
    
    
    @classmethod
    def ThreeDistances_and_d1sd2s(self, egostate  = (0, 0, 90*np.pi/180.0), targetstate = (10, 20, 90*np.pi/180.0), deltaf_ego = 10*np.pi/180.0, deltaf_target =  10*np.pi/180.0, v = 4,  veh_para_ego= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, veh_para_target= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, t = 20, N_t = 300, plot = False, phi_change_not_exceed = 100*np.pi/180.0, area_buffer = 1e-2, forward_backward = 1, buffer_evasion_after_exacttouch = .1):#t= 50, deltaf = 1*np.pi/180.0,
        """
        TEST OK. 
        
        Calculate the three distances, i.e. permittedtraveldistance,evasioncondition,evasiondistance, given the states of two vehicles. 
        
        and d1s d2s. 
        
        Callback method:
        
            blockornot,d1s,d2s,permittedtraveldistance,evasioncondition,evasiondistance = VM.VehicleKineticSolver.ThreeDistances_and_d1sd2s(egostate  = (0, 0, 90*np.pi/180.0), targetstate = (10, 20, 90*np.pi/180.0), deltaf_ego = 10*np.pi/180.0, deltaf_target =  10*np.pi/180.0, v = 4,  veh_para_ego= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, veh_para_target= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, t= 20, N_t = 300, plot = False, phi_change_not_exceed = np.pi/2, area_buffer = 1e-2)
            
        
        ------------------------------------------
        @input: buffer_evasion_after_exacttouch
            
            Formally is it hard to defind the evasiondistance. 
            
            Thus when the target vehicle doesnot intersect with the ego vehicle trajectory, the evasiondistance is calculated as the last distance (it is travel distance which is blocked) plus a smaller buffer, which is given by buffer_evasion_after_exacttouch
            
        @input: forward_backward
            it is either 1 or -1. 
            
            1 means forward and -1 means backward, 
            
        @input: t N_t
            the default time the vehicle travels.
        
            N_t is the number of discretization of the time.
        
        @input: area_buffer
            used to buffer the colision check using area method. 
        
        @output: blockornot,d1s,d2s,permittedtraveldistance,evasioncondition,evasiondistance
            blockornot = False
            d1s = []
            d2s = []
            permittedtraveldistance = False
            evasioncondition = False
            evasiondistance = False
            
        -------------------------------------------
        @METHOD:
            - 
            - 
            
            
            - 
            
        """
        #-------------returned values. 
        blockornot = False
        d1s = []
        d2s = []
        permittedtraveldistance = False
        evasioncondition = False
        evasiondistance = False
        
        #-------------Determine blockornot.
        #get ego trajectories. 
        #   cg_xs_ego is array. 
        cg_xs_ego,cg_ys_ego,_ = self.CG_until_t(t = t, N_t = N_t, deltaf = deltaf_ego, v = v*forward_backward, veh_state_init  = egostate, veh_para= veh_para_ego, phi_change_not_exceed = phi_change_not_exceed)
        #   distances_ego is a 1D array. distance that ego travel, 
        distances_ego = np.cumsum(np.sqrt(np.diff(cg_xs_ego)**2+ np.diff(cg_ys_ego)**2))
        #   (xsA,ysA),(xsB,ysB),(xsC,ysC),(xsD,ysD)  = four_trs, xsA is an array. 
        four_trs = self.Corners_xsys_Until_t(t = t, N_t = N_t, deltaf = deltaf_ego, v = v*forward_backward, veh_state_init  = egostate, veh_para = veh_para_ego, plot = False, phi_change_not_exceed = phi_change_not_exceed)#t= 50, deltaf = 1*np.pi/180.0,
        #       
        #---------------------Get the touch and permittedtraveldistance
        blockornot  = False#indicate whether the two vehicles is in block relationship. 
        permittedtraveldistance = np.inf
        for tr in list(four_trs)+[(cg_xs_ego,cg_ys_ego)]:
            #xs is an array. 
            xs,ys = tr
            #   np.array([xs,ys]).shape is (2,N)
            #   touch is a bool, dis is a float. 
            touch0,dis0 = self.InnerVehBoundary_points(veh_state = targetstate , points = np.array([xs,ys]), points4cal_dis = np.array([cg_xs_ego,cg_ys_ego]), veh_para= veh_para_target, area_buffer = area_buffer)
            if touch0:
                blockornot = touch0
                permittedtraveldistance = min(permittedtraveldistance, dis0)
        
        #--------------------------
        if not blockornot:
            return blockornot,d1s,d2s,permittedtraveldistance,evasioncondition,evasiondistance
        
        #----------------------target pose----------------
        #   cg_xs_target is an array. 
        cg_xs_target, cg_ys_target, phi_es_target = self.CG_until_t(t = t, N_t = N_t, deltaf = deltaf_target, v = v*forward_backward, veh_state_init  = targetstate, veh_para= veh_para_target, phi_change_not_exceed = phi_change_not_exceed)
        
        """
        #   the boundary line expression. a1 is 1D array of the length len(cg_xs_target)
        #       np.array(cg_xs_target, cg_ys_target, phi_es_target).shape is (3, N)
        #(a1,b1,c1),(a2,b2,c2),(a3,b3,c3),(a4,b4,c4) = self.a_b_c_vehboundary( veh_state = np.array([cg_xs_target, cg_ys_target, phi_es_target]), veh_para= veh_para_target)
        """
        d1s = [permittedtraveldistance]
        d2s = [0]
        #------------------------
        traveldistances_target = np.cumsum(np.sqrt(np.diff(cg_xs_target)**2+np.diff(cg_ys_target)**2))
        for idx in range(len(traveldistances_target)):
            #   indicate whether part of the four trs touch the target vehicle. 
            four_trs_part_touch = False
            #find d1 and d2
            d1 = np.inf
            for tr in list(four_trs)+[(cg_xs_ego,cg_ys_ego)]:
                #xs is an array. 
                xs,ys = tr
                #   np.array([xs,ys]).shape is (2,N)
                #   touch is a bool, dis0 is a float. point of dis0 is just outside the veh boundary
                touch0,dis0 = self.InnerVehBoundary_points(veh_state = (cg_xs_target[idx], cg_ys_target[idx], phi_es_target[idx]) ,points = np.array([xs,ys]), points4cal_dis = np.array([cg_xs_ego,cg_ys_ego]), veh_para = veh_para_target, area_buffer = area_buffer)
                #print(traveldistances_target[idx],len(xs), len(ys), dis0)
                if touch0:
                    
                    #print('sdsdfsdf')
                    four_trs_part_touch = True
                    #
                    d1 = min(d1, dis0)
            
            #if part of the four trs touch with the trajectory
            if four_trs_part_touch:
                d2s.append(traveldistances_target[idx])
                d1s.append(d1)
            else:
                evasioncondition = traveldistances_target[idx]
                evasiondistance  = d1s[-1] + buffer_evasion_after_exacttouch
                #
                d2s.append(traveldistances_target[idx])
                d1s.append(evasiondistance)
                break
                
                #-----------------------OLD VERSION2 BEGIN------------------------------------
                #evasioncondition = traveldistances_target[idx]
                #evasiondistance  = d1s[-1]
                #break
                #-----------------------OLD VERSION2 END------------------------------------
                
                #-----------------------OLD VERSION1 BEGIN------------------------------------
                # #builtins.tmp = (cg_xs_target[idx], cg_ys_target[idx], phi_es_target[idx]), distances_ego, d1s, (cg_xs_ego,cg_ys_ego), four_trs
                # #
                # evasioncondition = traveldistances_target[idx]
                # #builtins.tmp = (distances_ego, d1s, cg_xs_ego,cg_ys_ego)
                # #   np.where(distances_ego>max(d1s))
                # evasiondistance  = min(distances_ego[np.where(distances_ego>max(d1s))[0]])
                # break
                #-----------------------OLD VERSION1 END------------------------------------
        
        #----------------If holds, means the two trajectories overlay
        if (idx==len(traveldistances_target)-1) and four_trs_part_touch:
            #means that target veh always locate at ego veh path. 
            evasioncondition = np.inf
            evasiondistance = np.inf
            
        #-------------------------------------------------PLOT
        if plot:
            #
            fig,ax = plt.subplots()
            #
            polygon = self.shapely_given_state(targetstate)
            self.plot_shapely_in_matplotlib(polygon, ax = ax)
            #
            polygon = self.shapely_given_state(egostate)
            self.plot_shapely_in_matplotlib(polygon, ax = ax)
            #
            ax.plot(cg_xs_ego,cg_ys_ego)
            ax.plot(cg_xs_target,cg_ys_target, 'k')
            #
            polygon = self.shapely_given_state(builtins.tmp[0])
            self.plot_shapely_in_matplotlib(polygon, ax = ax)
            #
            ax.axis('equal')
            ax.set_xlim([-15,15])
            ax.set_ylim([-15,15])
            
            
            pass
        #traveldistances_target,distances_ego, d1s, cg_xs_ego,cg_ys_ego = builtins.tmp
        #builtins.tmp = traveldistances_target,distances_ego, d1s, cg_xs_ego,cg_ys_ego
        return blockornot,d1s,d2s,permittedtraveldistance,evasioncondition,evasiondistance
    
    @classmethod
    def ThreeDistances_and_d1sd2s_improved(self, egostate  = (0, 0, 90*np.pi/180.0), targetstate = (10, 20, 90*np.pi/180.0), deltaf_ego = 10*np.pi/180.0, deltaf_target =  10*np.pi/180.0, v = 4,  veh_para_ego= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, veh_para_target= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, t = 20, N_t = 300, plot = False, phi_change_not_exceed = 100*np.pi/180.0, area_buffer = 1e-2, forward_backward = 1, buffer_evasion_after_exacttouch = .1):#t= 50, deltaf = 1*np.pi/180.0,
        """
        
        Calculate the three distances, i.e. permittedtraveldistance,evasioncondition,evasiondistance, given the states of two vehicles. 
        
        and d1s d2s. 
        
        
        Difference between:
            - self.ThreeDistances_and_d1sd2s_improved()
            - self.ThreeDistances_and_d1sd2s()
            
        The latter use the enumerate method to find the three distance, while the former one use the optimization to find the three distances. 
        
        
        Callback method:
        
            blockornot,d1s,d2s,permittedtraveldistance,evasioncondition,evasiondistance = VM.VehicleKineticSolver.ThreeDistances_and_d1sd2s_improved(egostate  = (0, 0, 90*np.pi/180.0), targetstate = (10, 20, 90*np.pi/180.0), deltaf_ego = 10*np.pi/180.0, deltaf_target =  10*np.pi/180.0, v = 4,  veh_para_ego= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, veh_para_target= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, t= 20, N_t = 300, plot = False, phi_change_not_exceed = np.pi/2, area_buffer = 1e-2)
            
        
        ------------------------------------------
        @input: buffer_evasion_after_exacttouch
            
            Formally is it hard to defind the evasiondistance. 
            
            Thus when the target vehicle doesnot intersect with the ego vehicle trajectory, the evasiondistance is calculated as the last distance (it is travel distance which is blocked) plus a smaller buffer, which is given by buffer_evasion_after_exacttouch
            
        @input: forward_backward
            it is either 1 or -1. 
            
            1 means forward and -1 means backward, 
            
        @input: t N_t
            the default time the vehicle travels.
        
            N_t is the number of discretization of the time.
        
        @input: area_buffer
            used to buffer the colision check using area method. 
        
        @output: blockornot,d1s,d2s,permittedtraveldistance,evasioncondition,evasiondistance
            blockornot = False
            d1s = []
            d2s = []
            permittedtraveldistance = False
            evasioncondition = False
            evasiondistance = False
            
        -------------------------------------------
        @METHOD:
            - 
            - 
            
            
            - 
            
        """
        #-------------returned values. 
        blockornot = False
        d1s = []
        d2s = []
        permittedtraveldistance = False
        evasioncondition = False
        evasiondistance = False
        
        #-------------Determine blockornot.
        #get ego trajectories. 
        #   cg_xs_ego is array. 
        cg_xs_ego,cg_ys_ego,_ = self.CG_until_t(t = t, N_t = N_t, deltaf = deltaf_ego, v = v*forward_backward, veh_state_init  = egostate, veh_para= veh_para_ego, phi_change_not_exceed = phi_change_not_exceed)
        #   distances_ego is a 1D array. distance that ego travel, 
        distances_ego = np.cumsum(np.sqrt(np.diff(cg_xs_ego)**2+ np.diff(cg_ys_ego)**2))
        #   (xsA,ysA),(xsB,ysB),(xsC,ysC),(xsD,ysD)  = four_trs, xsA is an array. 
        four_trs = self.Corners_xsys_Until_t(t = t, N_t = N_t, deltaf = deltaf_ego, v = v*forward_backward, veh_state_init  = egostate, veh_para = veh_para_ego, plot = False, phi_change_not_exceed = phi_change_not_exceed)#t= 50, deltaf = 1*np.pi/180.0,
        #       
        #---------------------Get the touch and permittedtraveldistance
        blockornot  = False#indicate whether the two vehicles is in block relationship. 
        permittedtraveldistance = np.inf
        for tr in list(four_trs)+[(cg_xs_ego,cg_ys_ego)]:
            #xs is an array. 
            xs,ys = tr
            #   np.array([xs,ys]).shape is (2,N)
            #   touch is a bool, dis is a float. 
            touch0,dis0 = self.InnerVehBoundary_points(veh_state = targetstate , points = np.array([xs,ys]), points4cal_dis = np.array([cg_xs_ego,cg_ys_ego]), veh_para= veh_para_target, area_buffer = area_buffer)
            if touch0:
                blockornot = touch0
                permittedtraveldistance = min(permittedtraveldistance, dis0)
        
        #--------------------------
        if not blockornot:
            return blockornot,d1s,d2s,permittedtraveldistance,evasioncondition,evasiondistance
        
        #----------------------target pose----------------
        #   cg_xs_target is an array. 
        cg_xs_target, cg_ys_target, phi_es_target = self.CG_until_t(t = t, N_t = N_t, deltaf = deltaf_target, v = v*forward_backward, veh_state_init  = targetstate, veh_para= veh_para_target, phi_change_not_exceed = phi_change_not_exceed)
        
        """
        #   the boundary line expression. a1 is 1D array of the length len(cg_xs_target)
        #       np.array(cg_xs_target, cg_ys_target, phi_es_target).shape is (3, N)
        #(a1,b1,c1),(a2,b2,c2),(a3,b3,c3),(a4,b4,c4) = self.a_b_c_vehboundary( veh_state = np.array([cg_xs_target, cg_ys_target, phi_es_target]), veh_para= veh_para_target)
        """
        d1s = [permittedtraveldistance]
        d2s = [0]
        #------------------------
        traveldistances_target = np.cumsum(np.sqrt(np.diff(cg_xs_target)**2+np.diff(cg_ys_target)**2))
        for idx in range(len(traveldistances_target)):
            #   indicate whether part of the four trs touch the target vehicle. 
            four_trs_part_touch = False
            #find d1 and d2
            d1 = np.inf
            for tr in list(four_trs)+[(cg_xs_ego,cg_ys_ego)]:
                #xs is an array. 
                xs,ys = tr
                #   np.array([xs,ys]).shape is (2,N)
                #   touch is a bool, dis0 is a float. point of dis0 is just outside the veh boundary
                touch0,dis0 = self.InnerVehBoundary_points(veh_state = (cg_xs_target[idx], cg_ys_target[idx], phi_es_target[idx]) ,points = np.array([xs,ys]), points4cal_dis = np.array([cg_xs_ego,cg_ys_ego]), veh_para = veh_para_target, area_buffer = area_buffer)
                #print(traveldistances_target[idx],len(xs), len(ys), dis0)
                if touch0:
                    
                    #print('sdsdfsdf')
                    four_trs_part_touch = True
                    #
                    d1 = min(d1, dis0)
            
            #if part of the four trs touch with the trajectory
            if four_trs_part_touch:
                d2s.append(traveldistances_target[idx])
                d1s.append(d1)
            else:
                evasioncondition = traveldistances_target[idx]
                evasiondistance  = d1s[-1] + buffer_evasion_after_exacttouch
                #
                d2s.append(traveldistances_target[idx])
                d1s.append(evasiondistance)
                break
                
                #-----------------------OLD VERSION2 BEGIN------------------------------------
                #evasioncondition = traveldistances_target[idx]
                #evasiondistance  = d1s[-1]
                #break
                #-----------------------OLD VERSION2 END------------------------------------
                
                #-----------------------OLD VERSION1 BEGIN------------------------------------
                # #builtins.tmp = (cg_xs_target[idx], cg_ys_target[idx], phi_es_target[idx]), distances_ego, d1s, (cg_xs_ego,cg_ys_ego), four_trs
                # #
                # evasioncondition = traveldistances_target[idx]
                # #builtins.tmp = (distances_ego, d1s, cg_xs_ego,cg_ys_ego)
                # #   np.where(distances_ego>max(d1s))
                # evasiondistance  = min(distances_ego[np.where(distances_ego>max(d1s))[0]])
                # break
                #-----------------------OLD VERSION1 END------------------------------------
        
        #----------------If holds, means the two trajectories overlay
        if (idx==len(traveldistances_target)-1) and four_trs_part_touch:
            #means that target veh always locate at ego veh path. 
            evasioncondition = np.inf
            evasiondistance = np.inf
            
        #-------------------------------------------------PLOT
        if plot:
            #
            fig,ax = plt.subplots()
            #
            polygon = self.shapely_given_state(targetstate)
            self.plot_shapely_in_matplotlib(polygon, ax = ax)
            #
            polygon = self.shapely_given_state(egostate)
            self.plot_shapely_in_matplotlib(polygon, ax = ax)
            #
            ax.plot(cg_xs_ego,cg_ys_ego)
            ax.plot(cg_xs_target,cg_ys_target, 'k')
            #
            polygon = self.shapely_given_state(builtins.tmp[0])
            self.plot_shapely_in_matplotlib(polygon, ax = ax)
            #
            ax.axis('equal')
            ax.set_xlim([-15,15])
            ax.set_ylim([-15,15])
            
            
            pass
        #traveldistances_target,distances_ego, d1s, cg_xs_ego,cg_ys_ego = builtins.tmp
        #builtins.tmp = traveldistances_target,distances_ego, d1s, cg_xs_ego,cg_ys_ego
        return blockornot,d1s,d2s,permittedtraveldistance,evasioncondition,evasiondistance
        
    
    @classmethod
    def Corners_xsys_Until_t(self, t = 20, N_t = 200, deltaf = 10*np.pi/180.0, v = 4, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, plot = False, phi_change_not_exceed = np.pi/2):#t= 50, deltaf = 1*np.pi/180.0,
        """
        Calculate the corners trajectory until t. 
        ----------------------------------
        @input: phi_change_not_exceed
            limit the change of the traejctories. 
        
        @OUTPUT: (xsA,ysA),(xsB,ysB),(xsC,ysC),(xsD,ysD)
            xs* = np.array([x1,x2,x3,,,,]), 1D np.array
        
        """
        ts = np.linspace(0, t, N_t)
        
        x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
        
        #
        lr = veh_para.get('lr', 1.7)
        lf = veh_para.get('lf', 1.7)
        lR = veh_para.get('lR', 2)
        lF = veh_para.get('lF', 2)
        w = veh_para.get('w', 2)
        
        if deltaf==0:
            phi_es  = phi0 + ts*0
            cg_x_es = x0 + ts*v*np.cos(phi0)
            cg_y_es = y0 + ts*v*np.sin(phi0)
            #-------------CornerA
            xsA = cg_x_es + lF*np.cos(phi_es)+w/2.0*np.cos(phi_es - np.pi/2)
            ysA = cg_y_es + lF*np.sin(phi_es)+w/2.0*np.sin(phi_es - np.pi/2)
            
            #-------------CornerB
            xsB = cg_x_es - lR*np.cos(phi_es)+ w/2.0*np.cos(phi_es - np.pi/2)
            ysB = cg_y_es - lR*np.sin(phi_es)+ w/2.0*np.sin(phi_es - np.pi/2)
            
            #-------------CornerC
            xsC = cg_x_es - lR*np.cos(phi_es)+ w/2.0*np.cos(phi_es + np.pi/2)
            ysC = cg_y_es - lR*np.sin(phi_es)+ w/2.0*np.sin(phi_es + np.pi/2)
            
            #-------------CornerD
            xsD = cg_x_es + lF*np.cos(phi_es)+w/2.0*np.cos(phi_es + np.pi/2)
            ysD = cg_y_es + lF*np.sin(phi_es)+w/2.0*np.sin(phi_es + np.pi/2)
            
            
        else:
        
            #
            beta = np.arctan((lr*np.tan(deltaf))/(lf+lr))
            
            #----------------------------
            #phis is a np.array, shape is the same as ts. 
            phi_es  = phi0 + ts*v*np.cos(beta)*np.tan(deltaf)/(lf+lr)
            #-------------------truncate the ts, to make sure that the change of phi should not exceed phi_change_not_exceed.
            if phi_change_not_exceed!=False:
                #------------------------phi_change domain is [0, 2*np.pi]
                phi_change =  np.mod(phi_es -phi0, 2*np.pi)
                #-------------------------invalid_idxs is an array
                invalid_idxs = np.where((phi_change<=2*np.pi-phi_change_not_exceed) & (phi_change>=phi_change_not_exceed))[0]
                
                if len(invalid_idxs)>0:
                    ts = np.linspace(0, ts[invalid_idxs[0]-1], N_t)
                    #ts  = ts[:invalid_idxs[0]]
                    
                    #ts = ts[(phi_change <= phi_change_not_exceed) | (phi_change >= 2*np.pi-phi_change_not_exceed)]
                    phi_es  = phi0 + ts*v*np.cos(beta)*np.tan(deltaf)/(lf+lr)
            
            
            #cg_x_es and cg_y_es
            cg_x_es = x0 + (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.sin(phi0 + ts*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.sin(phi0 + beta))
            #
            cg_y_es =  y0 - (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.cos(phi0 + ts*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.cos(phi0 + beta))
            
            #-------------CornerA
            xsA = cg_x_es + lF*np.cos(phi_es)+w/2.0*np.cos(phi_es - np.pi/2)
            ysA = cg_y_es + lF*np.sin(phi_es)+w/2.0*np.sin(phi_es - np.pi/2)
            
            #-------------CornerB
            xsB = cg_x_es - lR*np.cos(phi_es)+ w/2.0*np.cos(phi_es - np.pi/2)
            ysB = cg_y_es - lR*np.sin(phi_es)+ w/2.0*np.sin(phi_es - np.pi/2)
            
            #-------------CornerC
            xsC = cg_x_es - lR*np.cos(phi_es)+ w/2.0*np.cos(phi_es + np.pi/2)
            ysC = cg_y_es - lR*np.sin(phi_es)+ w/2.0*np.sin(phi_es + np.pi/2)
            
            #-------------CornerD
            xsD = cg_x_es + lF*np.cos(phi_es)+w/2.0*np.cos(phi_es + np.pi/2)
            ysD = cg_y_es + lF*np.sin(phi_es)+w/2.0*np.sin(phi_es + np.pi/2)
            
        if plot:
            fig,ax = plt.subplots()
            ax.plot(xsA,ysA)
            ax.plot(xsB,ysB)
            ax.plot(xsC,ysC)
            ax.plot(xsD,ysD)
            
            ax.axis('equal')
                
        
        return (xsA,ysA),(xsB,ysB),(xsC,ysC),(xsD,ysD)
    

    @classmethod
    def CornerA_at_t(self, t_and_deltaf=(50, 1*np.pi/180.0), v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }):#t= 50, deltaf = 1*np.pi/180.0,
        """
        Calculate the CG (both x and y) at moment t. 
        ----------------------------------
        @input: t_and_deltaf
            t,deltaf = t_and_deltaf[0],t_and_deltaf[1]
            
            the moment and the steering angle of front wheel. 
            
            t unit is sec and 
            
            phi unit is np.pi. 
        
        @input: v
            unit is m/s. 
        
        @input: deltaf
            turn left correspond to positive. 
        
        
        """
        t,deltaf = t_and_deltaf[0],t_and_deltaf[1]
        
        
        x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
        
        #
        lr = veh_para.get('lr', 1.7)
        lf = veh_para.get('lf', 1.7)
        lR = veh_para.get('lR', 2)
        lF = veh_para.get('lF', 2)
        w = veh_para.get('w', 2)
        
        #
        beta = np.arctan((lr*np.tan(deltaf))/(lf+lr))
        
        #phi
        phi = phi0 + t*v*np.cos(beta)*np.tan(deltaf)/(lf+lr)
        
        #CG x and CG y
        cg_x = x0 + (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.sin(phi0 + t*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.sin(phi0 + beta))
        #
        cg_y =  y0 - (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.cos(phi0 + t*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.cos(phi0 + beta))
        
        #
        xA = cg_x + lF*np.cos(phi)+w/2.0*np.cos(phi - np.pi/2)
        yA = cg_y + lF*np.sin(phi)+w/2.0*np.sin(phi - np.pi/2)
        
        
        return (xA, yA)
    

    @classmethod
    def CornerD_at_t(self, t_and_deltaf=(50, 1*np.pi/180.0), v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }):#t= 50, deltaf = 1*np.pi/180.0,
        """
        Calculate the CG (both x and y) at moment t. 
        ----------------------------------
        @input: t_and_deltaf
            t,deltaf = t_and_deltaf[0],t_and_deltaf[1]
            
            the moment and the steering angle of front wheel. 
            
            t unit is sec and 
            
            phi unit is np.pi. 
        
        @input: v
            unit is m/s. 
        
        @input: deltaf
            turn left correspond to positive. 
        
        
        """
        t,deltaf = t_and_deltaf[0],t_and_deltaf[1]
        
        
        x0,y0,phi0 = veh_state_init[0],veh_state_init[1],veh_state_init[2]
        
        #
        lr = veh_para.get('lr', 1.7)
        lf = veh_para.get('lf', 1.7)
        lR = veh_para.get('lR', 2)
        lF = veh_para.get('lF', 2)
        w = veh_para.get('w', 2)
        
        #
        beta = np.arctan((lr*np.tan(deltaf))/(lf+lr))
        
        #phi
        phi = phi0 + t*v*np.cos(beta)*np.tan(deltaf)/(lf+lr)
        
        #CG x and CG y
        cg_x = x0 + (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.sin(phi0 + t*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.sin(phi0 + beta))
        #
        cg_y =  y0 - (lf+lr)/(np.cos(beta) * np.tan(deltaf))*(np.cos(phi0 + t*v*(np.cos(beta) * np.tan(deltaf))/(lf+lr) + beta)-np.cos(phi0 + beta))
        
        #
        xD = cg_x + lF*np.cos(phi)+w/2.0*np.cos(phi + np.pi/2)
        yD = cg_y + lF*np.sin(phi)+w/2.0*np.sin(phi + np.pi/2)
        
        
        return (xD, yD)


    @classmethod
    def CornerA_y_given_x(self, deltaf,veh_state_init  = (0, 0, 90*np.pi/180.0), x = 10, veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }):
        """
        The y coor of corner A given the x. 
        
        There are four corners of a vehicle, ABCD, They are:
            - front-right, rear-right, rear-left, front-left
        
        
        Given vehicle state (x,y,phi), phi unit is np.pi. The coordinate of A is given by:
            
            lR = veh_para['lR']
            lF = veh_para['lF']
            w = veh_para['w']
            #---------------------front right
            x0_right = x + w/2.0*np.cos(phi+270.0/180*np.pi)
            y0_right = y + w/2.0*np.sin(phi+270.0/180*np.pi)
            #
            x0_right_2 = x0_right + lF*np.cos(ang)
            y0_right_2 = y0_right + lF*np.sin(ang)
        -------------------------------------
        
        
        @input: deltaf
            the steering angle of the front wheel. 
            Unit is np.pi.
        
        @input: x
            the x-coordinate, 
        
        @OUTPUT; y
            a float. 
            
            (x,y) will be on the trajectory of corner A. 
        -------------------------------
        
        
        
        """
        
        
        
        
        pass
    
    
    
    @classmethod
    def BlockagePotential(self, ego_state = (0, 0, 90*np.pi/180.0), target_state =(10, 20, 20*np.pi/180.0), egp_veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2}, target_veh_para = {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2}, egosteer = 0, targetsteer = 0, t = 20, N_t = 200, phi_change_not_exceed = np.pi/2, truncate_distance = 15, v = 4, plott = False):
        """
        Callback:
        
             blockageornot, ego2distance,target2distance = VM.VehicleKineticSolver.BlockagePotential(ego_state = (0, 0, 90*np.pi/180.0), \
                                                                                                    target_state =(5, 1, 90*np.pi/180.0), \
                                                                                                    egp_veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2}, \
                                                                                                    target_veh_para = {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2}, \
                                                                                                    egosteer = -.7, targetsteer = .7, t = 20, N_t = 200, \
                                                                                                    phi_change_not_exceed = np.pi/2, truncate_distance = 10, v = 4,\
                                                                                                    plott=True)
                                                                                                
        
        
        Identify the potential blockage, for the purpose of deadlock avoidance. Using the IntersectionCurves package, it is imported as :
            
            from intersect import intersection as IntersectionCurves
            
        It is used as follows:
        
            from intersect import intersection

            a, b = 1, 2
            phi = np.linspace(3, 10, 100)
            x1 = a*phi - b*np.sin(phi)
            y1 = a - b*np.cos(phi)

            x2 = phi
            y2 = np.sin(phi)+2
            x, y = IntersectionCurves(x1, y1, x2, y2)

            plt.plot(x1, y1, c="r")
            plt.plot(x2, y2, c="g")
            plt.plot(x, y, "*k")
            plt.show()
        
        
        x and y are np.array. lengh of x is the number of intersection points. 
        
        It there is no intersection point, then the x and y are length 0. 
        
        ---------------------------------------------------------------------------------
        @input: effective_ego_or_target_mindistance
            
            ths minimal distance (either from ego or target to the confliction point) that we consider the potential risk is efective. 
            
            It ego2distance or target2distance is smaller than this, then the two vehicle may be in blockage, rather than the potantial risk. 
            
        @input: ego_state,target_state
            ego state and target state. 
            
        @input: egp_veh_para,target_veh_para
            two dicts, which express the parameters of the vehicles. 
            
            Can be obtained via:
                -  lflrlFlR = d2d.CarlaDataAnalysis.Read_lflrlFlR(filename = "/home/qhs/Desktop/Deadlock2D/datas/lflrlFlR.csv")
                - veh_para = lflrlFlR[vehicle_type_id]
        @input: egosteer, targetsteer
            
            the steering angle of the ego vehicle and target vehicle. 
            
        @OUTPUT: blockageornot, ego2distance,target2distance
        
            blockageornot is a bool. 
            
            ego2distance is the distance from ego current position to the confloction. 
            
            target2distance is the distane from the target current position to the confliction. 
            
        ----------------------------------------------------------
        @Steps:
            - find the traejctory of the cg. 
            - Check the intersectionornot of the trajectories of the cg
            - If not intersect, return False, np.inf,np.inf
            - If intersect, find the intersectionpoint and calculate the distance, 
            - 
        
        ----------------------------------------------------------
        """
        
        from intersect import intersection as IntersectionCurves
        
        #---------------Step1 find the trajectory of the cg of both ego vehicle and target vehicle.
        #x_rear_ego,y_rear_ego = ego_state[0]-
        cg_xs_ego, cg_ys_ego, phis_ego = self.CG_until_t(t = t, N_t = N_t, deltaf = egosteer, v = v, veh_state_init  = ego_state, veh_para= egp_veh_para, phi_change_not_exceed = phi_change_not_exceed, truncate_distance = truncate_distance)
        
        cg_xs_target, cg_ys_target, phis_target = self.CG_until_t(t = t, N_t = N_t, deltaf = targetsteer, v = v, veh_state_init  = target_state, veh_para= target_veh_para, phi_change_not_exceed = phi_change_not_exceed, truncate_distance = truncate_distance)
        
        if plott:
            
            fig,ax = plt.subplots()
            ax.plot(cg_xs_ego, cg_ys_ego)
            ax.plot(cg_xs_target, cg_ys_target)
            
        #---------------------check the intersection of the trajectory of the cg, 
        #   xs and ys are np.array with the same length. if there is no intersect, then len(x)=0
        xs, ys = IntersectionCurves(cg_xs_ego, cg_ys_ego, cg_xs_target, cg_ys_target)
        
        #-------------------no intersect, just return. 
        if len(xs)==0:
            return False,np.inf,np.inf
        
        #-------------------if there is intersect. we assume that there is only one. 
        x,y = xs[0],ys[0]
        
        #-------------ego2distance
        tmp = list((cg_xs_ego-x)**2)
        idxego = tmp.index(min(tmp))
        ego2distance = sum(np.sqrt(np.diff(cg_xs_ego[:idxego+1])**2+np.diff(cg_ys_ego[:idxego+1])**2))
        
        #--------------target2distance
        tmp = list((cg_xs_target-x)**2)
        idxtarget = tmp.index(min(tmp))
        target2distance = sum(np.sqrt(np.diff(cg_xs_target[:idxtarget+1])**2+np.diff(cg_ys_target[:idxtarget+1])**2))
        
        if ego2distance<egp_veh_para['lF'] or target2distance<target_veh_para['lF']:
            return False,np.inf,np.inf
            
        return True,ego2distance,target2distance

    
    @classmethod
    def BlockageInterval(self, ego_state = (0, 0, 90*np.pi/180.0), target_state =(10, 20, 20*np.pi/180.0), egp_veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2}, target_veh_para = {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2}, v = 2, init_solution0 = (10, 10*np.pi/180), plott = False, savee= False ,filename = 'Tmp.jpg', dpi =  500, pathh = '/home/qhs/Qhs_Files/Program/Python/Deadlock2D/figs/paper_fig/', maxx_wheel_angle = 70*np.pi/180.0, ax = False, forward_backward = 1, buffer_added_2_steer_4safety = 10.0*np.pi/180):
        """
        Given the state and the physical parameters of of the two vehicles, determine the interval of steering angle interval (of ego vehicle, unit is np.pi) that target vehicle will block vehicle 1. 
        
        Callbackmethod:
        
            Forward:
                reload(d2d);reload(VM)
                z = VM.VehicleKineticSolver.BlockageInterval(plott = True)
                        
            Backward:
            
                reload(d2d);reload(VM)
                z = VM.VehicleKineticSolver.BlockageInterval(plott = True, forward_backward = -1,  target_state = (0, 0, 90*np.pi/180.0), \
                                                             ego_state  =(10, 20, 20*np.pi/180.0))
        
        -------------------------------------------------------
        
        @input: buffer_added_2_steer_4safety
            
            a angle. or a bool (False)
            
            float. 
            
            It is the steering angle, which is added for sake of collision avoidance. 
            
        @input: forward_backward
            1 or -1. 
            
            1 means forward
            
            -1 means ego vehicle drive backward 
        @input: ego_state,target_state
            ego state and target state. 
            
        @input: egp_veh_para,target_veh_para
            two dicts, which express the parameters of the vehicles. 
            
            Can be obtained via:
                -  lflrlFlR = d2d.CarlaDataAnalysis.Read_lflrlFlR(filename = "/home/qhs/Desktop/Deadlock2D/datas/lflrlFlR.csv")
                - veh_para = lflrlFlR[vehicle_type_id]
        
        @input: init_solution0
            t0, deltaf0 = init_solution0
            
            The default solution of the time and the steeringangle of front wheel. 
            
            The sign (minus or positive) will be determined within the method. 
        
        @OUTPUT: steer_infos
            
            steer_infos.keys()
            
                dict_keys(['minsteer', 'maxsteer'])
            
            
            steer_infos['minsteer'].keys()
            
                dict_keys(['steer', 'distance', 'xsysD', 'xsysA'])
            
            
            steer_infos['minsteer']['distance'] is the travel distance of CG
            
            xsD,ysD= steer_infos['minsteer']['xsysD'] are trajectories of cornerD. 
            
        ------------------------------------------------
        @STEPS:
            - For each corner of target_veh
                - Find the deltaf of point C D of ego vehicle. 
                - minimum of them is the UP threhold
        
        """
        
        #-----------------find the coor of corner A of target vehicle in pointA 
        x0,y0,phi0 = ego_state[0],ego_state[1],ego_state[2]
        x1,y1,phi1 = target_state[0],target_state[1],target_state[2]
        #   pointA = (x, y)
        target_pointA, target_pointB, target_pointC, target_pointD = self.FourCornersCoor(x=x1, y = y1, ang = phi1, veh_para= target_veh_para)
        
        #-----------------------------minimum steer, using SteerAngleGivenCornerAndPathPoint
        #-----obtain steer_minimum,t_correspond_minimum,distance_correspond_minimum
        steer_minimum = np.inf
        t_correspond_minimum = 0
        distance_correspond_minimum = 0
        #
        for point in (target_pointA, target_pointB, target_pointC, target_pointD):
            #
            #ego vehicle corner C
            #SteerAngleGivenCornerAndPathPoint(self, point = (10, 10),v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, init_solution0 = (10, -10*np.pi/180), corner = 'A'):
            #print('@@@@@', point, v,  '@@@@@', ego_state, '@@@@@', egp_veh_para,  '@@@@@', init_solution0,  'C', '    @@@@@')
            t_cornerC,deltaf_cornerC,distancee_cornerC = self.SteerAngleGivenCornerAndPathPoint(point = point,v = v, veh_state_init  = ego_state, veh_para= egp_veh_para, init_solution0 = init_solution0, corner = 'C', forward_backward = forward_backward)
            
            #
            if deltaf_cornerC<steer_minimum:
                steer_minimum = deltaf_cornerC
                t_correspond_minimum = t_cornerC
                distance_correspond_minimum = distancee_cornerC
            #ego vehicle corner D
            t_cornerD,deltaf_cornerD, distancee_cornerD =self.SteerAngleGivenCornerAndPathPoint(point = point,v = v, veh_state_init  = ego_state, veh_para= egp_veh_para, init_solution0 = init_solution0, corner = 'D', forward_backward = forward_backward)
            if deltaf_cornerD<steer_minimum:
                steer_minimum = deltaf_cornerD
                t_correspond_minimum = t_cornerD
                distance_correspond_minimum = distancee_cornerD
        
        #-----------------------------maximum steer, using SteerAngleGivenCornerAndPathPoint
        #-----obtain steer_maximum,t_correspond_maximum,distance_correspond_maximum
        steer_maximum = -np.inf
        t_correspond_maximum = 0
        distance_correspond_maximum = 0
        for point in (target_pointA, target_pointB, target_pointC, target_pointD):
            
            #ego vehicle corner C
            #SteerAngleGivenCornerAndPathPoint(self, point = (10, 10),v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, init_solution0 = (10, -10*np.pi/180), corner = 'A')
            t_cornerA,deltaf_cornerA,distancee_cornerA =self.SteerAngleGivenCornerAndPathPoint(point = point,v = v, veh_state_init  = ego_state, veh_para= egp_veh_para, init_solution0 = init_solution0, corner = 'A', forward_backward = forward_backward)
            if deltaf_cornerA>steer_maximum:
                steer_maximum = deltaf_cornerA
                t_correspond_maximum = t_cornerA
                distance_correspond_maximum = distancee_cornerA
            #
            t_cornerB,deltaf_cornerB,distancee_cornerB =self.SteerAngleGivenCornerAndPathPoint(point = point,v = v, veh_state_init  = ego_state, veh_para= egp_veh_para, init_solution0 = init_solution0, corner = 'B', forward_backward = forward_backward)
            if deltaf_cornerB>steer_maximum:
                steer_maximum = deltaf_cornerB
                t_correspond_maximum = t_cornerB
                distance_correspond_maximum = distancee_cornerB
        
        # if deltaf_cornerC<deltaf_cornerD:
            # steer_minimum = deltaf_cornerC
            # t_correspond_minimum = t_cornerC
        # else:
            # steer_minimum = deltaf_cornerD
            # t_correspond_minimum = t_cornerD
        
        # #
        # if deltaf_cornerA<deltaf_cornerB:
            # steer_maximum = deltaf_cornerB
            # t_correspond_maximum = t_cornerB
        # else:
            # steer_maximum = deltaf_cornerA
            # t_correspond_maximum = t_cornerA
        
        
        #----------------conpute the trajectory of corner A and D. 
        #   xD, yD = self.CornerD_at_t(self, t_and_deltaf=(50, 1*np.pi/180.0), v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, })
        #cornerD
        xsD_min_steer= [];ysD_min_steer  = []
        for t in np.linspace(0, t_correspond_minimum, 50):
            xD, yD = self.CornerD_at_t(t_and_deltaf=(t,steer_minimum), v = v*forward_backward, veh_state_init  = ego_state, veh_para= egp_veh_para)
            xsD_min_steer.append(xD)
            ysD_min_steer.append(yD)
        xsD_max_steer= [];ysD_max_steer  = []
        for t in np.linspace(0, t_correspond_maximum, 50):
            xD, yD = self.CornerD_at_t(t_and_deltaf=(t,steer_maximum), v = v*forward_backward, veh_state_init  = ego_state, veh_para= egp_veh_para)
            xsD_max_steer.append(xD)
            ysD_max_steer.append(yD)
        #CornerA
        xsA_min_steer= [];ysA_min_steer  = []
        for t in np.linspace(0, t_correspond_minimum, 50):
            xA, yA = self.CornerA_at_t(t_and_deltaf=(t,steer_minimum), v = v*forward_backward, veh_state_init  = ego_state, veh_para= egp_veh_para)
            xsA_min_steer.append(xA)
            ysA_min_steer.append(yA)
        xsA_max_steer= [];ysA_max_steer  = []
        for t in np.linspace(0, t_correspond_maximum, 50):
            xA, yA = self.CornerA_at_t(t_and_deltaf=(t,steer_maximum), v = v*forward_backward, veh_state_init  = ego_state, veh_para= egp_veh_para)
            xsA_max_steer.append(xA)
            ysA_max_steer.append(yA)
        
        if plott:
            if ax==False:
                fig,ax = plt.subplots()
            
            #ego
            ego_polygon = self.shapely_given_state(state = ego_state, veh_para= egp_veh_para)
            ax = self.plot_shapely_in_matplotlib(shapely_polygon = ego_polygon, alpha = .7, ax = ax)
            
            
            #ego touch minimum
            #CG_and_phi_at_t(self, t= 50, deltaf = 1*np.pi/180.0, v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, })
            x,y,phi = self.CG_and_phi_at_t(t= t_correspond_minimum, deltaf = steer_minimum, v = v*forward_backward, veh_state_init  =ego_state, veh_para=egp_veh_para)
            ego_polygon = self.shapely_given_state(state = (x,y,phi), veh_para= egp_veh_para)
            ax = self.plot_shapely_in_matplotlib(shapely_polygon = ego_polygon, ax= ax, alpha = .1)
            
            color = np.random.random((3,))
            ax.plot(xsA_min_steer,ysA_min_steer, color = color)
            ax.plot(xsD_min_steer,ysD_min_steer, color = color)
            
            color = np.random.random((3,))
            ax.plot(xsA_max_steer,ysA_max_steer, color = color)
            ax.plot(xsD_max_steer,ysD_max_steer, color = color)
            
            #ego touch maximum
            #CG_and_phi_at_t(self, t= 50, deltaf = 1*np.pi/180.0, v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, })
            x,y,phi = self.CG_and_phi_at_t(t= t_correspond_maximum, deltaf = steer_maximum, v = v*forward_backward, veh_state_init  =ego_state, veh_para=egp_veh_para)
            ego_polygon = self.shapely_given_state(state = (x,y,phi), veh_para= egp_veh_para)
            ax = self.plot_shapely_in_matplotlib(shapely_polygon = ego_polygon, ax= ax, alpha = .1)
            
            #the bounds
            
            
            #
            polygon = self.shapely_given_state(state = target_state, veh_para= target_veh_para)
            ax = self.plot_shapely_in_matplotlib(shapely_polygon = polygon, ax= ax, alpha = .7)
            
            ax.grid()
            ax.axis('equal')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            if savee:
                plt.savefig(pathh + filename, dpi = dpi)
        
        if not isinstance(buffer_added_2_steer_4safety, bool):
            steer_minimum = max(-maxx_wheel_angle, steer_minimum-buffer_added_2_steer_4safety)
            steer_maximum = min(maxx_wheel_angle, steer_maximum + buffer_added_2_steer_4safety)
        
        return {'minsteer':{'steer':steer_minimum, 'distance':distance_correspond_minimum,'xsysD':(xsD_min_steer, ysD_min_steer), 'xsysA':(xsA_min_steer, ysA_min_steer)}, 'maxsteer':{'steer':steer_maximum, 'distance':distance_correspond_maximum,'xsysD':(xsD_max_steer, ysD_max_steer), 'xsysA':(xsA_max_steer, ysA_max_steer)}}
        

    @classmethod
    def BlockageInterval_BKP_SUCCESS(self, ego_state = (0, 0, 90*np.pi/180.0), target_state =(10, 20, 20*np.pi/180.0), egp_veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2}, target_veh_para = {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2}, v = 2, init_solution = (10, 10*np.pi/180), plott = True):
        """
        Given the state and the physical parameters of of the two vehicles, determine the interval of steering angle interval (of ego vehicle, unit is np.pi) that target vehicle will block vehicle 1. 
        
        
        
        -------------------------------------------------------
        @input: ego_state,target_state
            ego state and target state. 
            
        @input: egp_veh_para,target_veh_para
            two dicts, which express the parameters of the vehicles. 
            
            Can be obtained via:
                -  lflrlFlR = d2d.CarlaDataAnalysis.Read_lflrlFlR(filename = "/home/qhs/Desktop/Deadlock2D/datas/lflrlFlR.csv")
                - veh_para = lflrlFlR[vehicle_type_id]
        
        @OUTPUT: interval
            (steer_LW, steerUP)
            
            steer_LW and steer_UP unit is np.pi.
            
            It meas that when ego vehicle use the steer angle between (steer_LW, steerUP), it will be blocked by target vehicle. 
        ------------------------------------------------
        @STEPS:
            - For each corner of target_veh
                - Find the deltaf of point C D of ego vehicle. 
                - minimum of them is the UP threhold
        
        """
        
        #find the coor of corner A of target vehicle. 
        x0,y0,phi0 = ego_state[0],ego_state[1],ego_state[2]
        x1,y1,phi1 = target_state[0],target_state[1],target_state[2]
        
        #   pointA = (x, y)
        pointA,pointB,pointC,pointD = self.FourCornersCoor(x=x1, y = y1, ang = phi1, veh_para= target_veh_para)
        
        
        #SteerAngleGiven_CornerA_PathPoint(self, point = (10, 10),v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, }, init_solution = (10, 0*np.pi/180))
        
        #deltaf unit is np.pi
        t,deltaf =  self.SteerAngleGiven_CornerA_PathPoint(point = pointA,v = v, veh_state_init = ego_state, veh_para= egp_veh_para, init_solution0 = init_solution)
        
        
        if plott:
            fig,ax = plt.subplots()
            
            #ego
            ego_polygon = self.shapely_given_state(state = ego_state, veh_para= egp_veh_para)
            ax = self.plot_shapely_in_matplotlib(shapely_polygon = ego_polygon, alpha = .1, ax = ax)
            
            
            #ego touch
            #CG_and_phi_at_t(self, t= 50, deltaf = 1*np.pi/180.0, v = 2, veh_state_init  = (0, 0, 90*np.pi/180.0), veh_para= {'lR':3, 'lF':2, 'w':2, 'lf':1.4, 'lr':2, })
            x,y,phi = self.CG_and_phi_at_t(t= t, deltaf = deltaf, v = v, veh_state_init  =ego_state, veh_para=egp_veh_para)
            ego_polygon = self.shapely_given_state(state = (x,y,phi), veh_para= egp_veh_para)
            ax = self.plot_shapely_in_matplotlib(shapely_polygon = ego_polygon, ax= ax, alpha = .6)
            
            #
            polygon = self.shapely_given_state(state = target_state, veh_para= target_veh_para)
            ax = self.plot_shapely_in_matplotlib(shapely_polygon = polygon, ax= ax, alpha = .1)
            
            pass
        
        return t,deltaf
        
        

    @classmethod
    def CheckTR_WithV(self,  T = np.linspace(0, 20, 100), Vs = np.linspace(.1, 60, 40), angle = 30*np.pi/180.0, params = {'params':{}}):
        """
        
        """
        fig,ax = plt.subplots()
        
        veh_model = self.control_model_Rajamani( params = params , model = self.Rajamani_v_as_control, inputs= ('a', 'delta_f','delta_r'), outputs=('x', 'y', 'phi'), states=('x', 'y', 'phi'))
        
        builtins.tmp = veh_model
        
        for v in Vs:
            #
            X0 = [0, 0, 90/180*np.pi]
            inputs = np.array([[v, angle, 0] for i in range(len(T))]).T
            #y.shape = ( , moments)
            t, y = self.TR_solover(veh_model, T, inputs, X0)
            
            #print(y.shape)
            ax.plot(y[0, :], y[1, :], '.-', label = str(v))
            
            
        ax.legend()
        return ax
        
    @classmethod
    def affine_polygon(self, polygonshapely, new_state= (0, 0 , 90*np.pi/180.0)):
        """
        just affine single polygon. 
        """
        
        #phi should be unit in np.pi
        x,y,phi = new_state[0],new_state[1],new_state[2]
        
        
        translated = shapely.affinity.translate(polygonshapely, xoff=x, yoff= y)
        affined = shapely.affinity.rotate(translated, angle = phi*180/np.pi-90, origin=(x,y))
        
        
        return affined
    
    @classmethod
    def Affine2dTR(self, xy_array, new_state= (0, 0 , 90*np.pi/180.0)):
        """
        Move and rotate the expected trajectory, whihc is represented by a list of polygons. 
        Using: 
            - shapely.affinity.translate(geom, xoff=0.0, yoff=0.0)
            - shapely.affinity.rotate(geom, angle, origin='center', use_radians=False)
        
        polygons_list[0] is the start pose of the vehicle. The heading angle of the start pose should be 90degree and the start location (CG location) should be (0, 0). 
        

        polygons_list can be obtained via:
            reload(VM)
            res = VM.VehicleKineticSolver.GetExpectedTrajectory_shapely(steering_angle = 10*np.pi/180.0)
            polygons_list = res['polygons']
        
        -----------------------------------------
        @input: xy_array
            an array. 
            shape is 2*N.
            
            THe 1st row is x and 2nd row is y. 
        
        @input: new_state
            x,y,phi=new_state
            
            THe phi is the heading angle which is defined for the angle between head and x-axis. unit is np.pi. 
        
        @output: 
            affined_polygons
            
            len(affined_polygons) = len(polygons_list)
        -----------------------------------------
            
        """
        #phi should be unit in np.pi
        xy_array_new = copy.deepcopy(xy_array)
        
        #rotated angle, unit is degree. -90 is because that 
        theta_roted = (new_state[2]*180/np.pi-90)*np.pi/180.0
        c,s = np.cos(theta_roted), np.sin(theta_roted)
        rotete_matrix = np.array(((c, -s), (s, c)))
        xy_array_new1 = np.matmul(rotete_matrix, xy_array_new[[0,1],:])
        
        #translate
        xy_array_new[0,:] = xy_array_new1[0,:]+new_state[0]
        xy_array_new[1,:] = xy_array_new1[1,:]+new_state[1]
        xy_array_new[2,:] = copy.deepcopy(xy_array[2]-np.pi/2 + new_state[2])
        
        return xy_array_new
    
    
    @classmethod
    def AffineExpectedTrajectory(self, polygons_list, new_state= (0, 0 , 90*np.pi/180.0)):
        """
        Move and rotate the expected trajectory, whihc is represented by a list of polygons. 
        Using: 
            - shapely.affinity.translate(geom, xoff=0.0, yoff=0.0)
            - shapely.affinity.rotate(geom, angle, origin='center', use_radians=False)
        
        polygons_list[0] is the start pose of the vehicle. The heading angle of the start pose should be 90degree and the start location (CG location) should be (0, 0). 
        

        polygons_list can be obtained via:
            reload(VM)
            res = VM.VehicleKineticSolver.GetExpectedTrajectory_shapely(steering_angle = 10*np.pi/180.0)
            polygons_list = res['polygons']
        
        -----------------------------------------
        @input: polygons_list
            expected trajectory. The start pose should 
        
        @input: new_state
            x,y,phi=new_state
            
            THe phi is the heading angle which is defined for the angle between head and x-axis. unit is np.pi. 
        
        @output: 
            affined_polygons
            
            len(affined_polygons) = len(polygons_list)
        -----------------------------------------
            
        """
        #phi should be unit in np.pi
        x,y,phi = new_state[0],new_state[1],new_state[2]
        
        #
        affined_polygons = []
        
        #
        for p in polygons_list:
            #
            #p1 = shapely.affinity.translate(p, xoff=x, yoff= y)
            
            #
            translated = shapely.affinity.translate(p, xoff=x, yoff= y)
            #-90 is because that the default start pose is is 90 degree. 
            p2 = shapely.affinity.rotate(translated, angle = phi*180/np.pi-90, origin=(x,y))
            
            affined_polygons.append(p2)
        
        return affined_polygons
        
        
    
    @classmethod
    def GetExpectedTrajectorys_shapely(self, max_steering_angle = 70*np.pi/180, N_steering_angles = 300, X0 = [0, 0, 90/180*np.pi],  default_v = 10, delta_t = 0.05, T_horizon = 5, distance_threshold = 30, T_horizon_incremental = 3, veh_para1= {'lR':3, 'lF':2, 'w':2}, mirror = True):
        """
        Generate the expected trajectory (represented by a list of distance and a list of shapely).
        
        
        X0 = [0, 0, 90/180*np.pi] means the heading angle is 90 degree. 
        ---------------
        
        --------------
        
            
        Expected trajectory for specific steering_angle can naively obtained via:
        
            X0 = state_veh1#x,y,phi
            T = np.linspace(0, T_horizon, int(T_horizon/delta_t)+1) 
            inputs = np.array([[default_v, steering_angle1, 0] for i in range(len(T))]).T
            t, y1 = VM.VehicleKineticSolver.TR_solover(veh_model, T, inputs, X0)
            distances_veh1 = [0]+list(np.cumsum(np.sqrt(np.diff(y1[0,:])**2 + np.diff(y1[1,:])**2)))
            #steps2: obtain the shapelys of the polygons in polygons_veh1, a list. 
            #   shapelys_polygons is multipolygon instance. 
            polygons_veh1 = VM.VehicleKineticSolver.shapelys_given_states(TR = y1, veh_para = veh_para1)
            
        -------------------------------------
        @input: T_horizon_incremental
            if the distance is not long enough, the T_horizon will be increased and trajectory will be calculated again by extending the 
            
        @input: X0
            x,y, phi = X0
            
            phi is the heading angle, unit is np.pi
            
        @input: max_steering_angle and N_steering_angles
        
            max_steering_angle is the max steering angle of front wheel for bicycle model.  unit is np.pi. 
            
            N_steering_angles is the number of discretization of the max_steering_angle.
            
        @input: distance_threshold
            the generated trajectory length threshold. 
            
        @input: mirror
            mirror about the y-axis 
            
            Note that when steering angle is positive, then turn left, else turn right. 
            
            
        @OUTPUT: expected_trajectorys_steers
            a dict. expected_trajectorys_steers keys are steer angle. 
            
            expected_trajectorys_steers[steer_angle] = {'distances':distances, 'polygons':polygons_veh, 'states':states_list}
            
            states_list = [X0, X1， X2,...]
            
            distances,polygons_veh both are list. 
            distances is a list of float, represent the travelling distacne of the CG from X0, and 
            polygons_veh1 is a list of shapely polygon. 
            
        """
        
        #returned value. 
        expected_trajectorys_steers =  {}
        
        veh_model = self.control_model_Rajamani_using_v(params = veh_para1)
        
        for steering_angle in np.linspace(0, max_steering_angle, N_steering_angles):
            
            #-------------obatain distances and polygons_veh
            distances_veh1= []
            while len(distances_veh1)==0 or distances_veh1[-1]<distance_threshold:
                if len(distances_veh1)>0:
                    T_horizon = T_horizon + T_horizon_incremental
                
                T = np.linspace(0, T_horizon, int(T_horizon/delta_t)+1)
                #   the last 0 is for steering angle of the rear wheel.  
                inputs = np.array([[default_v, steering_angle, 0] for i in range(len(T))]).T
                #   t and y1 are np.array. 
                #       t.shape is (len(T),) and y1 shape is (3, len(T))
                t, y1 = VehicleKineticSolver.TR_solover(veh_model, T, inputs, X0)
                distances_veh1 = [0]+list(np.cumsum(np.sqrt(np.diff(y1[0,:])**2 + np.diff(y1[1,:])**2)))
                #steps2: obtain the shapelys of the polygons in polygons_veh1, a list. 
                #   shapelys_polygons is multipolygon instance. 
            
            #find the index in distances_veh1 that distances_veh1[idx]-distance_threshold is minimal
            idx = np.where(np.array(distances_veh1)>=distance_threshold)[0][0]
            
            #
            distances = distances_veh1[:idx]
            polygons_veh = self.shapelys_given_states(TR = y1[:,:idx], veh_para = veh_para1)
            #
            expected_trajectorys_steers[steering_angle] = {'distances':distances, 'polygons':polygons_veh}
            
            #mirror the resulf for -steering_angle
            if mirror:
                polygons_veh2 = [shapely.affinity.scale(p, xfact = -1, origin = (0, 0)) for p in polygons_veh]
                expected_trajectorys_steers[-steering_angle] = {'distances':distances, 'polygons':polygons_veh2}
            
            #
            #expected_trajectorys_steers[steering_angle]['states'] = [X0, X1, X2,...]
            expected_trajectorys_steers[steering_angle]['states'] = [list(y1[:,i]) for i in range(idx)]
            
            if mirror:
                #   the state is the same while y is mirrot about y-axis and phi is changed
                expected_trajectorys_steers[-steering_angle]['states'] = [[-y1[0,i],y1[1,i],np.pi-y1[2,i]] for i in range(idx)]
            
        return expected_trajectorys_steers
    
    
    
    
    
    @classmethod
    def GetExpectedTrajectory_shapely(self, steering_angle, X0 = [0, 0, 90/180*np.pi],  default_v = 10, delta_t = 0.05, T_horizon = 5, distance_threshold = 30, T_horizon_incremental = 3, veh_para1= {'lR':3, 'lF':2, 'w':2}):
        """
        Generate the expected trajectory (represented by a list of distance and a list of shapely).
        
        Callback method:
            reload(VM)
            res = VM.VehicleKineticSolver.GetExpectedTrajectory_shapely(steering_angle = 10*np.pi/180.0)
            
            #visualize use:
            shapely.geometry.MultiPolygon(res['polygons'])
            
        Can naively obtained via:
        
            X0 = state_veh1#x,y,phi
            T = np.linspace(0, T_horizon, int(T_horizon/delta_t)+1) 
            inputs = np.array([[default_v, steering_angle1, 0] for i in range(len(T))]).T
            t, y1 = VM.VehicleKineticSolver.TR_solover(veh_model, T, inputs, X0)
            distances_veh1 = [0]+list(np.cumsum(np.sqrt(np.diff(y1[0,:])**2 + np.diff(y1[1,:])**2)))
            #steps2: obtain the shapelys of the polygons in polygons_veh1, a list. 
            #   shapelys_polygons is multipolygon instance. 
            polygons_veh1 = VM.VehicleKineticSolver.shapelys_given_states(TR = y1, veh_para = veh_para1)
            
        -------------------------------------
        @input: T_horizon_incremental
            if the distance is not long enough, the T_horizon will be increased and trajectory will be calculated again by extending the 
            
        @input: X0
            x,y, phi = X0
            
            phi is the heading angle, unit is np.pi
        @input: steering_angle
            the steering angle of front wheel, unit is np.pi. 
        
        @input: distance_threshold
            the generated trajectory length threshold. 
            
        
        @OUTPUT: {'distances':distances, 'polygons':polygons_veh1}
            
            distances,polygons_veh1 
            
            Both are list. 
            distances is a list of float, represent the travelling distacne of the CG from X0, and 
            polygons_veh1 is a list of shapely polygon. 
            
        """
        
        #vehicle model
        veh_model = self.control_model_Rajamani_using_v(params = veh_para1)
        
        
        
        distances_veh1= []
        while len(distances_veh1)==0 or distances_veh1[-1]<distance_threshold:
            if len(distances_veh1)>0:
                T_horizon = T_horizon + T_horizon_incremental
            
            T = np.linspace(0, T_horizon, int(T_horizon/delta_t)+1)
            #   the last 0 is for steering angle of the rear wheel.  
            inputs = np.array([[default_v, steering_angle, 0] for i in range(len(T))]).T
            #   t and y1 are np.array. 
            #       t.shape is (len(T),) and y1 shape is (3, len(T))
            t, y1 = VehicleKineticSolver.TR_solover(veh_model, T, inputs, X0)
            distances_veh1 = [0]+list(np.cumsum(np.sqrt(np.diff(y1[0,:])**2 + np.diff(y1[1,:])**2)))
            #steps2: obtain the shapelys of the polygons in polygons_veh1, a list. 
            #   shapelys_polygons is multipolygon instance. 
        
        #find the index in distances_veh1 that distances_veh1[idx]-distance_threshold is minimal
        idx = np.where(np.array(distances_veh1)>=distance_threshold)[0][0]
        
        #
        distances = distances_veh1[:idx]
        polygons_veh1 = self.shapelys_given_states(TR = y1[:,:idx], veh_para = veh_para1)
        
        return {'distances':distances, 'polygons':polygons_veh1}
    
    
    
    
    
    
    
    @classmethod
    def VehicleFanSolver(self, angle_limit = 30*np.pi/180.0, a = 0, X0 = [0, 0, 90/180*np.pi, 5], T = np.linspace(0, 20, 100), N_angles = 100, params = {'params':{}}):
        """
        - Calculate the vehicle trajectory given the uplimit and downlimit of the angle. 
        
        - 
        #9d09d801d2b54b31f11cb987c80aeed71de85758b6fb969f&token=9d09d801d2b54b31f11cb987c80aeed71de85758b6fb969f
        
        -----------------------------------------------
        @input: X0:
            length is four. 
            [x,y,phi,v]
        """
        fig,ax = plt.subplots()
        
        veh_model = self.control_model_Rajamani( params = params )
        
        for angle in np.linspace(-angle_limit, angle_limit, N_angles):
            #
            inputs = np.array([[a, angle, 0] for i in range(len(T))]).T
            #y.shape = ( , moments)
            t, y = self.TR_solover(veh_model, T, inputs, X0)
            
            #print(y.shape)
            ax.plot(y[0, :], y[1, :], '.-')
            
        
        return ax
        
        
        
        
        pass
    
    
    
    @classmethod
    def VehicleKineticModel_Rajamani4jacobi_cal_without_acce(self, STATES, U, params = {}):
        """
        
        THis method is designed for obtaining the jacobian of Rajamani vehicle model. Note that difference:
        
        Callback method:
            #states variable
            x,y,phi = sympy.symbols('x y phi v')
            #controm variable
            v, delta_f = sympy.symbols('a delta_f delta_r')

            STATES = sympy.Matrix([x,y,phi])
            U =  sympy.Matrix([v, delta_f])

            tmp = VM.VehicleKineticSolver.VehicleKineticModel_Rajamani4jacobi_cal_without_acce(STATES, U)

            tmp.jacobian(U)
            tmp.jacobian(STATES)
        
        If you want to evaluate the jacobi at delta_f=0 then simply
            tmp.subs({'delta_f':0})
        
        ------------------------------------------------
        @input: STATES, U
            sympy Matrix. 
            
            They are constructed as follows:
                # states variable
                x,y,phi,v = sympy.symbols('x y phi v')
                # controm variable
                a, delta_f, delta_r = sympy.symbols('a delta_f delta_r')
                #
                STATES = sympy.Matrix([x,y,phi,v])
                U =  sympy.Matrix([a, delta_f, delta_r ])
            
        """
        import sympy
        
        lr = params.get('lr', 2)
        lf = params.get('lf', 2)
        
        # decouple the STATES and U
        x,y,phi = STATES
        v, delta_f = U
        
        #the derivates of the states variable. 
        x_derivate,y_derivate,phi_derivate = sympy.symbols('x_derivate y_derivate phi_derivate')
        
        #get beta
        beta = sympy.atan((lr*sympy.tan(delta_f))/(lf+lr))
        
        x_derivate  = v*sympy.cos(phi+beta)
        y_derivate = v*sympy.sin(phi+beta)
        phi_derivate  = v*(sympy.tan(delta_f))*sympy.cos(beta)/(lf+lr)
        
        return sympy.Matrix([x_derivate, y_derivate, phi_derivate])
    
    
    
    
    @classmethod
    def VehicleKineticModel_Rajamani4jacobi_cal(self, STATES, U, params = {}):
        """
        
        THis method is designed for obtaining the jacobian of Rajamani vehicle model. 
        
        Callback method:
            #states variable
            x,y,phi,v = sympy.symbols('x y phi v')
            #controm variable
            a, delta_f, delta_r = sympy.symbols('a delta_f delta_r')

            STATES = sympy.Matrix([x,y,phi,v])
            U =  sympy.Matrix([a, delta_f, delta_r ])

            tmp = VM.VehicleKineticSolver.VehicleKineticModel_Rajamani4jacobi_cal(STATES, U)

            tmp.jacobian(U)
            tmp.jacobian(STATES)
        
        If you want to evaluate the jacobi at delta_f=0 then simply
            tmp.subs({'delta_f':0})
        
        ------------------------------------------------
        @input: STATES, U
            sympy Matrix. 
            
            They are constructed as follows:
                # states variable
                x,y,phi,v = sympy.symbols('x y phi v')
                # controm variable
                a, delta_f, delta_r = sympy.symbols('a delta_f delta_r')
                #
                STATES = sympy.Matrix([x,y,phi,v])
                U =  sympy.Matrix([a, delta_f, delta_r ])
            
        """
        import sympy
        
        lr = params.get('lr', 2)
        lf = params.get('lf', 2)
        
        # decouple the STATES and U
        x,y,phi,v = STATES
        a, delta_f, delta_r = U
        
        #the derivates of the states variable. 
        x_derivate,y_derivate,phi_derivate,v_derivate = sympy.symbols('x_derivate y_derivate phi_derivate v_derivate')
        
        #get beta
        beta = sympy.atan((lf*sympy.tan(delta_r)+lr*sympy.tan(delta_f))/(lf+lr))
        
        x_derivate  = v*sympy.cos(phi+beta)
        y_derivate = v*sympy.sin(phi+beta)
        phi_derivate  = v*sympy.cos(beta)/(lf+lr)*(sympy.tan(delta_f)-sympy.tan(delta_r))
        v_derivate = a
        
        return sympy.Matrix([x_derivate, y_derivate, phi_derivate,  v_derivate])
    
    @classmethod
    def envelop_given_TR(self, TR, veh_para= {'lR':3, 'lF':2, 'w':2}):
        """
        Given the trajectory, get the envelop of the trajectory. 
        ------------------------------------------
        @input: TR
            an array. 
            the trajectories. 
            TR shape is (M,N), where N is the number of trajectory points. 
            TR[0,:] is x
            TR[1,:] is y
            TR[2,:] is phi
            
        
        """
        #----------------------------
        #envelop = [(x1,y1), (x2,y2), (x3, y3)...]
        envelop = []
        
        #--------------------------------get four corners along the trajectories. 
        four_corners = []
        for idx in range(TR.shape[1]):
            x = TR[0, idx]
            y = TR[1, idx]
            phi = TR[2, idx]
                
            #the order: front-right, rear-right, rear-left, front-left
            #   frontright = (x,y)
            frontright,rearright,rearleft,frontleft = self.FourCornersCoor(x=x , y = x, ang = phi, veh_para = veh_para)
            four_corners.append((frontright,rearright,rearleft,frontleft))
        
        #------------------------
        
        return four_corners
        
    @classmethod
    def shapelys_given_states(self, TR, veh_para= {'lR':3, 'lF':2, 'w':2}):
        """
        
        ----------------------------------------------
        @input: TR
            an array. 
            the trajectories. 
            TR shape is (M,N), where N is the number of trajectory points. 
            TR[0,:] is x
            TR[1,:] is y
            TR[2,:] is phi
        
        @input: veh_para
            a dict containing the vehicle parameter. 
            
            lR = veh_para['lR']
            LF = veh_para['lF']
            w = veh_para['w']
        -----------------------------------------------
        @
        
        """

        
        #--------------------------------get four corners along the trajectories. 
        shapelys_polygons = []
        for idx in range(TR.shape[1]):
            x = TR[0, idx]
            y = TR[1, idx]
            phi = TR[2, idx]
            
            #the order: front-right, rear-right, rear-left, front-left
            #   frontright = (x,y)
            shapelys_polygons.append(self.shapely_given_state(state = (x,y,phi), veh_para = veh_para))
        
        #------------------------
        return shapelys_polygons
        return shapely.geometry.MultiPolygon(shapelys_polygons)
    
    @classmethod
    def shapely_given_state(self, state, veh_para= {'lR':3, 'lF':2, 'w':2}):
        """
        
        ----------------------------------------------
        @input: state
            state = (x,y,phi)
            
            x and y are the coordinates, and phi is the angle, unit is not degree, rather than np.pi/180*degree.
        
        @input: veh_para
            a dict containing the vehicle parameter. 
            
            lR = veh_para['lR']
            LF = veh_para['lF']
            w = veh_para['w']
        -----------------------------------------------
        @
        
        """
        x,y,phi = state[0],state[1],state[2]
        
        #the order: front-right, rear-right, rear-left, front-left
        #   frontright = (x,y)
        frontright,rearright,rearleft,frontleft = self.FourCornersCoor(x=x , y = y, ang = phi, veh_para = veh_para)
        
        return shapely.geometry.Polygon((frontright,rearright,rearleft,frontleft, frontright))
        
        pass
    
    @classmethod
    def VehicleKineticModel_Rajamani1(self, t, STATES, params):
        """
        Difference:
            - self.VehicleKineticModel_Rajamani1(), the u is combind into the states, 
            - self.VehicleKineticModel_Rajamani(), the u is not combined into the states 
        
        """
        lr = params.get('lr', 2)
        lf = params.get('lf', 2)
        
        #control input
        a = U[0]
        front_steer_angle = U[1]
        rear_steer_angle = U[2]
        
        #states variable. 
        X = STATES[0]
        Y = STATES[1]
        PHI = STATES[2]
        V = STATES[3]
        #
        a = STATES[4]
        front_steer_angle = STATES[5]
        rear_steer_angle = STATES[6]
        
        #beta, the intermediate parameter
        #print(lr,lf, front_steer_angle, np.tan(front_steer_angle))
        tmp = (lr*np.tan(front_steer_angle) + lf*np.tan(rear_steer_angle))/(lr+lf)
        beta = np.arctan(tmp)
        
        #_diff means differential 
        diff_X = V*np.cos( PHI + beta)
        diff_Y = V*np.sin( PHI + beta)
        diff_PHI = V*np.cos(beta)/(lr+lf)*np.tan(front_steer_angle)
        diff_V = a
        return [diff_X,diff_Y,diff_PHI, diff_V]
    
        
        
        pass

    @classmethod
    def Rajamani_v_as_control(self, t, STATES, U, params):
        """
        The speed is taken as control input. 
        
        Return the derivate of the X,Y and PHI. 
        X is the horizontal coordinate
        Y is the vertical coordinate
        PHI is the heading angle. 
        -------------------------------------------
        @input: U
            the control input of the model. 
            a,front_steer,rear_steer
            
            len(U)=3 
            
            front_steer_angle = U[1], domain is [0,2*np.pi]
            
            V = U[1], the speed of the center of gravity of the vehicle, m/s
        
        @input: params
            the parameters of the vehicle. 
        
        
        @input: lf and lr
            unit is meter. 
            lf is the lengh of front. i.e. the distance between front axel to the CG. 
            lr is the rear length, or the ditance betweene rear axel to the CG. 
        
        @OUTPUT: diff_STATES
            len(STATES)=4:
                - X = STATES[0], the X of the CG. X is the horizontal axis. 
                - Y = STATES[1], the Y of the CG
                - PHI = STATES[2], the heading angle. between the vehicle and the X axis. 
        """
        lr = params.get('lr', 2)
        lf = params.get('lf', 2)
        
        #control input
        V = U[0]
        front_steer_angle = U[1]
        rear_steer_angle = U[2]
        
        #states variable. 
        X = STATES[0]
        Y = STATES[1]
        PHI = STATES[2]
        
        #beta, the intermediate parameter
        #print(lr,lf, front_steer_angle, np.tan(front_steer_angle))
        tmp = (lr*np.tan(front_steer_angle) + lf*np.tan(rear_steer_angle))/(lr+lf)
        beta = np.arctan(tmp)
        
        #_diff means differential 
        diff_X = V*np.cos( PHI + beta)
        diff_Y = V*np.sin( PHI + beta)
        diff_PHI = V*np.cos(beta)/(lr+lf)*np.tan(front_steer_angle)
        return [diff_X,diff_Y,diff_PHI]

    @classmethod
    def VehicleKineticModel_Rajamani(self, t, STATES, U, params):
        """
        THE VEHICLE IS FRONT STEER. 
        
        Return the derivate of the X,Y and PHI. 
        X is the horizontal coordinate
        Y is the vertical coordinate
        PHI is the heading angle. 
        V is the speed. 
        -------------------------------------------
        @input: U
            the control input of the model. 
            a,front_steer,rear_steer
            
            len(U)=3 
            
            front_steer_angle = U[1], domain is [0,2*np.pi]
            
            V = U[1], the speed of the center of gravity of the vehicle, m/s
        
        @input: params
            the parameters of the vehicle. 
        
        
        @input: lf and lr
            unit is meter. 
            lf is the lengh of front. i.e. the distance between front axel to the CG. 
            lr is the rear length, or the ditance betweene rear axel to the CG. 
        
        @OUTPUT: diff_STATES
            len(STATES)=4:
                - X = STATES[0], the X of the CG. X is the horizontal axis. 
                - Y = STATES[1], the Y of the CG
                - PHI = STATES[2], the heading angle. between the vehicle and the X axis. 
                - V = STATES[3]
        """
        lr = params.get('lr', 2)
        lf = params.get('lf', 2)
        
        #control input
        a = U[0]
        front_steer_angle = U[1]
        rear_steer_angle = U[2]
        
        #states variable. 
        X = STATES[0]
        Y = STATES[1]
        PHI = STATES[2]
        V = STATES[3]
        
        #beta, the intermediate parameter
        #print(lr,lf, front_steer_angle, np.tan(front_steer_angle))
        tmp = (lr*np.tan(front_steer_angle) + lf*np.tan(rear_steer_angle))/(lr+lf)
        beta = np.arctan(tmp)
        
        #_diff means differential 
        diff_X = V*np.cos( PHI + beta)
        diff_Y = V*np.sin( PHI + beta)
        diff_PHI = V*np.cos(beta)/(lr+lf)*np.tan(front_steer_angle)
        diff_V = a
        return [diff_X,diff_Y,diff_PHI, diff_V]
    
    
    @classmethod
    def VehicleKineticModel_Rajamani_BKP(self, t, STATES, U, params):
        """
        THE VEHICLE IS FRONT STEER. 
        
        Return the derivate of the X,Y and PHI. 
        X is the horizontal coordinate
        Y is the vertical coordinate
        PHI is the heading angle. 
        -------------------------------------------
        @input: U
            the control input of the model. 
            len(U)=2, 
            front_steer_angle = U[0], domain is [0,2*np.pi]
            
            V = U[1], the speed of the center of gravity of the vehicle, m/s
        
        @input: params
            the parameters of the vehicle. 
        
        
        @input: lf and lr
            unit is meter. 
            lf is the lengh of front. i.e. the distance between front axel to the CG. 
            lr is the rear length, or the ditance betweene rear axel to the CG. 
        
        @OUTPUT: STATES
            len(STATES)=3:
                - X = STATES[0], the X of the CG. X is the horizontal axis. 
                - Y = STATES[1], the Y of the CG
                - PHI = STATES[2], the heading angle. between the vehicle and the X axis. 
        """
        lr = params.get('lr', 2)
        lf = params.get('lf', 2)
        
        #control input
        front_steer_angle = U[0]
        V = U[1]
        
        #states variable. 
        X = STATES[0]
        Y = STATES[1]
        PHI = STATES[2]
        
        #beta, the intermediate parameter
        #print(lr,lf, front_steer_angle, np.tan(front_steer_angle))
        tmp = lr*np.tan(front_steer_angle)/(lr+lf)
        beta = np.arctan(tmp)
        
        #_diff means differential 
        diff_X = V*np.cos( PHI + beta)
        diff_Y = V*np.sin( PHI + beta)
        diff_PHI = V*np.cos(beta)/(lr+lf)*np.tan(front_steer_angle)
            
        return [diff_X,diff_Y,diff_PHI]
        
    
    @classmethod
    def ConstantControlInput(self, steer_angle = 0 , v = 0, ts = np.linspace(0, 5, 100)):
        """
        Genrate constant control input. 
        The constant input include the steer_angle and the speed v. 
        
        @input: steer_angle
            the domain is [0, 2*np.pi]
            
        @input: v
            speed, unit is m/s
        @input: ts
            a list constaining the moments. 
            unit is sec
        """
        #inputs.shape = (2, len(ts))
        #   0-th row is steer_angle, and 1-th row is speed; 
        inputs = np.array([[steer_angle,v] for i in range(len(ts))]).T
        
        
        return inputs


    @classmethod
    def TR_solover(self, veh_model, T, inputs, X0):
        """
        @input: T
            T, a list contanint the moments. 
        @input: inputs
            a numpy.ndarray
            shape is (n, len(T))
        @input: X0
            the initial state. 
            list. Each element is a state. 
        
        @input: distance_MIN
            whether set the min distance. 
            When the trajectory is short than this value, than reevaluate to solve the TR. 
        
        """
        #builtins.tmp =veh_model, T, inputs, X0
        t, y = control.input_output_response(veh_model, T, inputs, X0, solve_ivp_kwargs={'method':'DOP853'})
        
        return t,y
    
    @classmethod
    def TR_solover_BKP(self, veh_model, T, inputs, X0, distance_MIN = 100):
        """
        @input: T
            T, a list contanint the moments. 
        @input: inputs
            a numpy.ndarray
            shape is (n, len(T))
        @input: X0
            the initial state. 
            list. Each element is a state. 
        
        @input: distance_MIN
            whether set the min distance. 
            When the trajectory is short than this value, than reevaluate to solve the TR. 
        
        """
        #builtins.tmp =veh_model, T, inputs, X0
        t, y = control.input_output_response(veh_model, T, inputs, X0, solve_ivp_kwargs={'method':'DOP853'})
        
        if not distance_MIN==False:
            #compute the distance in d
            d = sum(np.sqrt(np.diff(y[0,:])**2 + np.diff(y[1,:])**2))
            if d<distance_MIN:
                
                newT = list(T)
                newinputs = [inputs]
                for i in range(int(distance_MIN/d)):
                    newinputs.append(inputs)
                    newT.extend(list(np.array(T)+newT[-1]-T[0]+T[1]-T[0]))
                
                #builtins.tmp = newT
                t, y = control.input_output_response(veh_model, newT, np.concatenate(newinputs,axis = 1), X0, solve_ivp_kwargs={'method':'DOP853'})
                
                pass
        
        return t,y
        
        
        
        
    @classmethod
    def control_model_Rajamani(self, params = {'params':{}}, model = False, inputs= ('a', 'delta_f','delta_r'), outputs=('x', 'y', 'phi', 'v'), states=('x', 'y', 'phi', 'v') ):
        """
        
        ------------------------------------
        @input: params
            the parameters used in self.VehicleKineticModel_Rajamani().
            
        -----------------------------------
        
        """
        if model==False:
            model = self.VehicleKineticModel_Rajamani
        
        #
        veh_model = control.NonlinearIOSystem(
            model, None, inputs= inputs, outputs = outputs,
            states = states, params = params, name='Rajamani')
                
        return veh_model

    @classmethod
    def control_model_Rajamani_using_v(self, params = {'lf':2, 'lr':1.2}, model = False, inputs= ('v', 'delta_f','delta_r'), outputs=('x', 'y', 'phi'), states=('x', 'y', 'phi') ):
        """
        
        ------------------------------------
        @input: params
            the parameters used in self.VehicleKineticModel_Rajamani().
            
        -----------------------------------
        
        """
        if model==False:
            model = self.Rajamani_v_as_control
        
        #
        veh_model = control.NonlinearIOSystem(
            model, None, inputs= inputs, outputs = outputs,
            states = states, params = params, name='Rajamani')
                
        return veh_model

    
    @classmethod
    def plot_trajectory_plotly(self, states, veh_params = {}, N_smaples = 10):
        """
        @input: states
            the shape is (3,n_points).
            
            The 1st row is the x of the CG, unit is m
            the 2nd row is the y of the CG, unit is m
            the 3rd row is the heading angle, unit is [0, 2*np.pi]
        
        @input: N_smaples
            just plot this samples. 
            
        """
        fig = plotly_go.Figure()
        
        #veh parametters. 
        lr = veh_params.get('lr', 2)
        lf = veh_params.get('lf', 3)
        veh_width = veh_params.get('width', 2)
        veh_len = veh_params.get('length', 5)
        
        #-----------------------determine the idx of the ploted samples, in plot_columns_idxs
        #trunctuate total_samples
        total_samples = states.shape[1]
        N_smaples = min([total_samples, N_smaples])
        #   determin plot_columns_idxs
        plot_columns_idxs = list(range(0, total_samples,  int(total_samples/N_smaples)))
        if plot_columns_idxs[-1]!=total_samples-1:
            plot_columns_idxs.append(total_samples-1)
        
        #builtins.tmp = plot_columns_idxs
        
        #
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #
        patches = []
        for idx in plot_columns_idxs:
            #
            x = states[0, idx]
            y = states[1, idx]
            heading = states[2, idx]
            
            #
            rear_right_x,rear_right_y = self.Veicle_Rear_Right(xy_CG = (x,y),w = veh_width, lr =lr, ang= heading)
            
            #print(rear_right_x,rear_right_y)
            #
            rect = matplotlib.patches.Rectangle((rear_right_x,rear_right_y), veh_len, veh_width, heading*180.0/np.pi, facecolor='yellow', edgecolor='violet', linewidth=2.0)
            
            patches.append(rect)
            
        pcs = PatchCollection(patches)
        ax.add_collection(pcs)
        
        tmp_max = max([max(states[0,:])-min(states[0,:]), max(states[1,:])-min(states[1,:])])
        ax.set_xlim([min(states[0,:]), min(states[0,:])+tmp_max])
        ax.set_ylim([min(states[1,:]), min(states[1,:])+tmp_max])
        
        fig.show()
        
        return ax
    
    @classmethod
    def plot_shapelys_in_matplotlib(self, shapely_polygons, figsize = (5,5), ax = False, alpha=0.4):
        """
        shapelys is a list of shapely. 
        
        The method that get single polygon coordinate is:
            points = list(polygon.exterior.coords) = [(x1,y1), (x2, y2), (x3, y3)....]
        """
        import matplotlib
        color = np.random.random((3,1))[:,0]
        #
        if ax==False:
            fig,ax = plt.subplots(figsize = figsize)
        
        patches = []
        for p in shapely_polygons:
            #[(x1,y1), (x2, y2), (x3, y3)....]
            points = list(p.exterior.coords)
            
            #matplotlib.patches.Polygon(xy), xy is a np array and shape is N*2.
            patches.append(matplotlib.patches.Polygon(np.array(points)),)
            
        p = matplotlib.collections.PatchCollection(patches, alpha=alpha,  facecolor = color)
            
        ax.add_collection(p)
        ax.grid()
        ax.axis('equal')
        return ax
    
    
    
    @classmethod
    def plot_veh_given_state(self,  veh_state, ax = False, alpha=0.4, figsize  = (5,5), veh_para= {'lR':3, 'lF':2, 'w':2}):
        """
        @input: veh_state
            
            x,y,phi = veh_state[0],veh_state[1],veh_state[2]
            
            phi unit is np.pi.
        
        """
        import matplotlib
        color = np.random.random((3,1))[:,0]
        #
        if ax==False:
            fig,ax = plt.subplots(figsize = figsize)
        
        polygon = self.shapely_given_state(state  =veh_state, veh_para= {'lR':3, 'lF':2, 'w':2})
        points = list(polygon.exterior.coords)
        patches = [matplotlib.patches.Polygon(np.array(points))]
        
        patches.append(matplotlib.patches.Polygon(np.array(points)),)
        p = matplotlib.collections.PatchCollection(patches, alpha=alpha,  facecolor = color)
        
        ax.add_collection(p)
        
        
        ax.grid()
        ax.axis('equal')
        return ax
        
        
    @classmethod
    def plot_shapely_in_matplotlib(self, shapely_polygon, ax = False, alpha=0.4, figsize  = (5,5),):
        """
        shapelys is a list of shapely. 
        
        #
        points = list(polygon.exterior.coords) = [(x1,y1), (x2, y2), (x3, y3)....]
        """
        import matplotlib
        facecolor= np.random.uniform(size = (3,))
        
        #
        if ax==False:
            fig,ax = plt.subplots(figsize = figsize)
        
        #[(x1,y1), (x2, y2), (x3, y3)....]
        points = list(shapely_polygon.exterior.coords)
        
        #matplotlib.patches.Polygon(xy), xy is a np array and shape is N*2.
        
        
        p = matplotlib.collections.PatchCollection([matplotlib.patches.Polygon(np.array(points))], alpha=alpha, facecolor= facecolor, )
        
        ax.add_collection(p, )
        ax.grid()
        ax.axis('equal')
        return ax
    

    
    @classmethod
    def plot_trajectory_matplotlib(self, states, veh_params = {}, N_smaples = 10, facecolor= [0,0.5,0], ax = False,  alpha = .3, figsize=(6,3)):
        """
        @input: states
            the shape is (4,n_points).
            
            The 1st row is the x of the CG, unit is m
            the 2nd row is the y of the CG, unit is m
            the 3rd row is the heading angle, unit is [0, 2*np.pi]
            the 4rd row is the v. 
            
        @input: N_smaples
            just plot this samples. 
            
        """
        import matplotlib
        
        #veh parametters. 
        lr = veh_params.get('lr', 2)
        lf = veh_params.get('lf', 3)
        veh_width = veh_params.get('width', 2)
        veh_len = veh_params.get('length', 5)
        
        #-----------------------determine the idx of the ploted samples, in plot_columns_idxs
        #trunctuate total_samples
        total_samples = states.shape[1]
        N_smaples = min([total_samples, N_smaples])
        #   determin plot_columns_idxs
        plot_columns_idxs = list(range(0, total_samples,  int(total_samples/N_smaples)))
        if plot_columns_idxs[-1]!=total_samples-1:
            plot_columns_idxs.append(total_samples-1)
        
        #builtins.tmp = plot_columns_idxs
        
        #
        if ax==False:
            fig,ax = plt.subplots(figsize = figsize)
        
        #
        patches = []
        for idx in plot_columns_idxs:
            #
            x = states[0, idx]
            y = states[1, idx]
            heading = states[2, idx]
            
            #
            rear_right_x,rear_right_y = self.Veicle_Rear_Right(xy_CG = (x,y),w = veh_width, lr =lr, ang= heading)
            
            #print(rear_right_x,rear_right_y)
            #
            rect = matplotlib.patches.Rectangle((rear_right_x,rear_right_y), veh_len, veh_width, heading*180.0/np.pi, alpha = alpha, facecolor=facecolor, ec = 'k')#fill=None, facecolor='none'
            
            ax.add_patch(rect)
            
        
        #tmp_max = max([max(states[0,:])-min(states[0,:]), max(states[1,:])-min(states[1,:])])
        #ax.set_xlim([min(states[0,:]), min(states[0,:])+tmp_max])
        #ax.set_ylim([min(states[1,:]), min(states[1,:])+tmp_max])
        
        return ax




    
    @classmethod
    def Veicle_Rear_Left(self, xy_CG = (0,0),w = 2, lr = 2, ang= np.pi):
        """
        COnstruct the shapely of the intersection, 
        
        @input: xy_CG
            the x and y of the center of gravity. 
        @inout: w and lr
            w is the width of the vehilce;
            lr is the distance of CG to rear axis of the vehicle. 
        @input: ang
            the heading angle of the vehicle. 
            unit is degree*np.pi/180. 
        """
        #
        x,y = xy_CG
        #the coordinate of the point of left boundary, CG-this_point is parallel with front and rear bump. 
        x0_left = x + w/2.0*np.cos(ang+90.0/180*np.pi)
        y0_left = y + w/2.0*np.sin(ang+90.0/180*np.pi)
        #
        x0_left_1 = x0_left - lr*np.cos(ang)
        y0_left_1 = y0_left - lr*np.sin(ang)
        
        return (x0_left_1,y0_left_1)
        
    
    @classmethod
    def Veicle_Rear_Right(self, xy_CG = (0,0),w = 2, lr = 2, ang = np.pi):
        """
        COnstruct the shapely of the intersection, 
        
        @input: xy_CG
            the x and y of the center of gravity. 
        @inout: w and lr
            w is the width of the vehilce;
            lr is the distance of CG to rear axis of the vehicle. 
        @input: ang
            the heading angle of the vehicle. 
            unit is degree*np.pi/180. 
        """
        #
        x,y = xy_CG
        #the coordinate of the point of right boundary, CG-this_point is parallel with front and rear bump. 
        x0_right = x + w/2.0*np.cos(ang+270.0/180*np.pi)
        y0_right = y + w/2.0*np.sin(ang+270.0/180*np.pi)
        #
        x0_right_1 = x0_right - lr*np.cos(ang)
        y0_right_1 = y0_right - lr*np.sin(ang)
        
        return (x0_right_1,y0_right_1)
        




