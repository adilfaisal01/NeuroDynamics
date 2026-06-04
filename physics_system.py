import numpy as np
from scipy.integrate import solve_ivp

class ChaoticSystem:
    
    def __init__(self,parameters,n_states):
        self.parameters=parameters
        self.n_states=n_states
    
    def sample_parameters(self):
        return {k: np.random.uniform(*v) for k, v in self.parameters.items()}
    
    def ode(self,t,states,params):
        raise NotImplementedError('insert your physics equations here')
    
    def solve_system(self, init_state=None, time_steps=50000, T=50):
        raise NotImplementedError('implement your solver here')
    def noisy_data_output(self,y,noise_std=0.15):
        return y+np.random.normal(loc=0,scale=noise_std,size=y.shape)





class DoublePendulum(ChaoticSystem):

    "Source for double pendulum: https://web.mit.edu/jorloff/www/chaosTalk/double-pendulum/double-pendulum-en.html"

    def __init__(self, parameters={"m1": [0.25,1.5],
                                   "m2": [0.25,2.5],
                                   "l1":[0.45, 2.5],
                                   "l2":[0.45,2.5]}, n_states=4):
        self.parameters=parameters
        self.n_states=n_states
    
    def ode(self,t,states,params):
        theta1,omega1,theta2,omega2=states
        m1,m2=params["m1"], params["m2"]
        l1, l2=params["l1"],params["l2"]
        g=9.81
        denom1=l1*(2*m1+m2-m2*np.cos(2*theta1-2*theta2))
        denom2=l2*(2*m1+m2-m2*np.cos(2*theta1-2*theta2))
        nume1=-g*(2*m1+m2)*np.sin(theta1)-m2*g*np.sin(theta1-2*theta2)-2*np.sin(theta1-theta2)*m2*(omega2**2*l2+omega1**2*l1*np.cos(theta1-theta2))
        nume2=2*np.sin(theta1-theta2)*((omega1**2*l1*(m1+m2))+g*(m1+m2)*np.cos(theta1)+omega2**2*l2*m2*np.cos(theta1-theta2))
        
        alpha1=nume1/denom1
        alpha2=nume2/denom2

        return np.array([omega1,alpha1,omega2,alpha2])
    
    def solve_system(self, init_state=None, time_steps=5000, T=50):
        
        # defining the initial conditions
        if init_state is None:
            theta1=np.random.uniform(low=-np.pi/3, high=np.pi/3)
            theta2=np.random.uniform(low=-np.pi/4, high=np.pi/4)
            omega1,omega2=0,0
            y0=np.array([theta1,omega1,theta2,omega2])
        else:
            y0=init_state
        
        #choosing a random set of initial parameters
        params=self.sample_parameters()

        #solving the ODE using the RK45 method to generate trajectory
        t_span=[0, T]
        t_eval=np.linspace(0,T,time_steps)
        
        sol=solve_ivp(fun=lambda t,states:self.ode(t,states,params),
                      t_span=t_span,
                      t_eval=t_eval,
                      y0=y0,
                      method='RK45'
                      )
        y=sol.y
        t=sol.t

        return params,t,y
    def noisy_data_output(self,y,noise_std=0.15):
        return y+np.random.normal(loc=0,scale=noise_std,size=y.shape)

class ChuaCircuit(ChaoticSystem):

    "sources: http://www.scholarpedia.org/article/Chua_circuit", "https://link.springer.com/content/pdf/10.1007/s11071-022-08078-y.pdf"

    def __init__(self, parameters={"L":[1e-6,1e-4],
                                   "C1":[1e-12, 1e-10],
                                   "C2":[1e-12, 1e-10],
                                   "R":[1e3,2e3]
                                   }, n_states=3):
        
        self.parameters=parameters
        self.n_states=n_states
        
    def ode(self,t,states,params):
        l,r,c1,c2=params["L"],params["R"],params["C1"],params["C2"]
        alpha=c2/c1
        beta=c2*r**2/l
        x,y,z=states

        xdot=-alpha*(x-y)-alpha*((1/16)*x**3-(1/6)*x)
        ydot=z-(y-x)
        zdot=-beta*y
        return np.array([xdot,ydot,zdot])
    
    def solve_system(self, init_state=None, time_steps=50000, T=50):
        
        #define the initial conditions
        if init_state is None:
            x0,y0,z0=np.random.uniform(0,1,3)
            start_points=np.array([x0,y0,z0])
        else:
            start_points=init_state

        #choosing parameters
        params=self.sample_parameters()

        #solving the ODE using the RK45 method to generate trajectory
        t_span=[0, T]
        t_eval=np.linspace(0,T,time_steps)
        
        sol=solve_ivp(fun=lambda t,states:self.ode(t,states,params),
                      t_span=t_span,
                      t_eval=t_eval,
                      y0=start_points,
                      method='RK45'
                      )
        y=sol.y
        t=sol.t

        return params,t,y     
        



# # ## testing the classes
# setup=ChuaCircuit()
# parameters,time,trajectories=setup.solve_system()

# print(f'parameters:{parameters}')
# print(f'time:{time}')
# print(f'trajectories:{trajectories[0,:]}',flush=True)

# import matplotlib
# # matplotlib.use('Qt5Agg')  # or 'Qt5Agg' if you have Qt installed
# import matplotlib.pyplot as plt

# plt.plot(time,trajectories[0,:],color='r',label='x')
# plt.plot(time,trajectories[1,:],color='b',label='y')
# plt.plot(time,trajectories[2,:],color='y',label='z')

# plt.legend()
# plt.show()


        
        

        





        