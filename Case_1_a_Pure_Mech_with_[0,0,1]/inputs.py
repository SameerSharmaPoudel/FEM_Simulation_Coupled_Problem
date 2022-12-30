### Parameter declarations
# numerical parameters
alphaNM = 1.0 #/ 4.0
betaNM =  1.0 #/ 2.0
residualPrecision = 1E-10

# material parameters
nu = 0.3                                        # Poisson's ratio
emod = 210000.0                                 # Young's modulus
#llambda = emod*nu/((1.0-2.0*nu)*(1.0+nu))       # 1st Lamé constant
#lmu = emod/(2.0 + 2.0*nu)                       # 2nd Lamé constant
#llambda = 76.6
llambda = 76.6e9
lmu = 44.7e9
gamma = -.56e-8                                                #                                                #
beta1 = 4.4                          
beta2 = 4.4                            

#cp = 480000000.0                                # heat capacity, constant pressure
#rho = 0.0000000075                              # initial density
#rT = 0.0                                        # external heat source

# problem instance
#length = 100.0                                  # dimensions of the cuboid 
#width = 80.0          
#height = 140.0
length = 1.0                                  # dimensions of the cuboid 
width = 0.80          
height = 1.40
epotential0 = 0                             # initial temperature
initValues = [0.0, 0.0, 0.0, epotential0]            # initial displacements and temperature
numberOfElements = 8
numberOfNodes = 27
dofPerNode = 4
nodesPerElement = 8
numberOfGaussPoints = 8
simulationTime = 500      # load steps
timeStep = 1              # step size
maxNewtonIterations = 10
#load = -1000000.0
a = [0,0,1]  #polarization vector
#Making a zero to check the mechanical part
#The code works absolutely fine for the mechanical part
### boundary conditions
fixedDofs = {\
        0 : [0, 1, 2,3], \
        1 : [0, 1, 2,3], \
        2 : [0, 1, 2,3], \
        3 : [0, 1, 2,3], \
        4 : [0, 1, 2,3], \
        5 : [0, 1, 2,3], \
        6 : [0, 1, 2,3], \
        7 : [0, 1, 2,3], \
        8 : [0, 1, 2,3]}

    
# forces
# node: [list, of, forces per dof]
#Charge_Density = 1e-7
Charge_Density =0
fz=100
#fz=0
externalForces = {\
        18 : [0, 0, fz , Charge_Density / 4 / 4], \
        19 : [0, 0, fz , Charge_Density / 4 / 2], \
        20 : [0, 0, fz, Charge_Density / 4 / 4], \
        21 : [0, 0, fz, Charge_Density / 4 / 2], \
        22 : [0, 0, fz, Charge_Density/ 4 / 1], \
        23 : [0, 0, fz, Charge_Density / 4 / 2], \
        24 : [0, 0, fz, Charge_Density / 4 / 4], \
        25 : [0, 0, fz, Charge_Density / 4 / 2], \
        26 : [0, 0, fz, Charge_Density / 4 / 4]}