# element class 
from node import Node
from numpy import linspace, array, zeros, ones, matmul, insert, delete, round, count_nonzero, transpose, dot, float64
from numpy.linalg import inv, det
from math import sqrt
from numba import njit


# this is calculated externally because numba cannot completely support nested custom classes currently
@njit
def computeStiffness(fv, fa, stiffness, numberOfGaussPoints, dofPerNode, numberOfNodes, bmatrix, mC, dVol, shapeFunctionValues,  sigmapelectricField,electricDisppEps,electricDisppelectricField):
    for gp in range(0, numberOfGaussPoints):
        for node1 in range(0, numberOfNodes):
            #mechanical part K_uu
            for node2 in range(0, numberOfNodes):
                for i in range(0, 3): 
                    for j in range(0, 3):
                        for k in range(0, 2*3):
                            for l in range(0, 2*3):
                                stiffness[ node1*dofPerNode + i][ node2*dofPerNode + j] \
                                                  += bmatrix[gp][ node1 ][ k ][ i ] * mC[ k][ l ] \
                                                   * bmatrix[gp][ node2 ][ l][ j ] * dVol[gp]

                # electrical part K_phiphi

                for i in range(0, 3):
                    for j in range(0, 3):
                        stiffness[ node1*dofPerNode + 3 ][ node2*dofPerNode + 3] \
                                          += 1* shapeFunctionValues[gp][node1][ i ] * electricDisppelectricField[i][j] \
                                           * shapeFunctionValues[gp][node2][ j ] * dVol[gp]  

                for i in range(0, 3):
                    for j in range(0, 2*3):
                        for k in range(0,3):
                            stiffness[ node1*dofPerNode + i][ node2*dofPerNode + 3] \
                                          += bmatrix[gp][node1][ j ][ i] * sigmapelectricField[j][k] \
                                           * -1 * shapeFunctionValues[gp][node2][k] * dVol[gp]
                        
                # coupling part K_phi_u
                for i in range(0, 3):
                    for j in range(0, 2*3):
                        for k in range(0,3):
                            stiffness[node1*dofPerNode + 3][node2*dofPerNode + k] \
                                          += -1*shapeFunctionValues[gp][ node1][i] * electricDisppEps[i][j] \
                                                                       * bmatrix[gp][node2][ j ][ k ] * dVol[gp]
                                                                        

@njit(parallel=True)
def computeForces(forces, numberOfGaussPoints, dofPerNode, numberOfNodes, bmatrix, sigma, dVol, shapeFunctionValues, electricDisp):
    for gp in range(0, numberOfGaussPoints):
        for node in range(0, numberOfNodes):
            #mechanical part f_u
            for i in range(0, 3):
                for j in range(0, 2*3):
                    forces[node][i] -= bmatrix[gp][node][j][i] * sigma[gp][j] * dVol[gp]


            # electrical part f_phiphi
            for i in range(0, 3):
                forces[node][3] += shapeFunctionValues[gp][node][i] * electricDisp[gp][i] * dVol[gp]

 
class Element:
    def __init__(self, nodesPerElement, dofPerNode, numberOfGaussPoints, mC,  lmu, llambda, \
                  gamma, beta1, beta2,a):
        self.numberOfNodes = nodesPerElement
        self.dofPerNode = dofPerNode
        self.nodes = []
        self.gradU = zeros((numberOfGaussPoints, 3, 3), dtype=float64)
        self.electricPotential = zeros(numberOfGaussPoints, dtype=float64)
        self.electricField = zeros((numberOfGaussPoints, 3), dtype=float64)
        self.productofEpsandElectricfield = zeros((numberOfGaussPoints, 3), dtype=float64)
        self.vonMises = zeros(numberOfGaussPoints, dtype=float64) 
        self.stiffness = zeros((nodesPerElement * dofPerNode, nodesPerElement * dofPerNode), dtype=float64)
        self.forces = zeros((nodesPerElement, dofPerNode), dtype=float64)
        self.numberOfGaussPoints = numberOfGaussPoints
        self.weights = ones(nodesPerElement, dtype=float64) # can be customized
        self.dVol = zeros(nodesPerElement, dtype=float64)
        self.shapeFunctionValues = zeros((numberOfGaussPoints, nodesPerElement, dofPerNode), dtype=float64) 
        self.bmatrix = zeros((numberOfGaussPoints, nodesPerElement, 2*3, 3), dtype=float64) 

        # material parameters
        self.lmu = lmu
        self.llambda = llambda
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2
        self.a = a
        # material fields 
        self.epsilon =     zeros((numberOfGaussPoints, 6), dtype=float64)
        self.traceEps =    zeros((numberOfGaussPoints), dtype=float64)
        self.normofelectricField_a = zeros((numberOfGaussPoints), dtype=float64)
        self.sigma =       zeros((numberOfGaussPoints, 6), dtype=float64)
        self.electricDisp  =  zeros((numberOfGaussPoints, 3), dtype=float64)
        
        #tangent modulus matrix
        self.mC = mC   #elastic tangent modulus, already defined in mesh file
        self.electricDisppEps = zeros((3, 6), dtype=float64) # coupled tangent modulus
        self.sigmapelectricField = zeros((6, 3), dtype=float64)  # coupled tangent modulus
                                                                 
        self.electricDisppelectricField = zeros((3, 3), dtype=float64) # dilelectric tangent modulus
        
        self.nodePositions = array([[-1.0, -1.0, -1.0],
                            [ 1.0, -1.0, -1.0],
                            [ 1.0,  1.0, -1.0],
                            [-1.0,  1.0, -1.0],
                            [-1.0, -1.0,  1.0],
                            [ 1.0, -1.0,  1.0],
                            [ 1.0,  1.0,  1.0],
                            [-1.0,  1.0,  1.0]], dtype=float64)

    
        self.gaussPoints = self.nodePositions / sqrt(3)
        
        
    def addNode(self, node):
        if len(self.nodes) < self.numberOfNodes:
            self.nodes.append(node)
        else:
            raise ValueError("Trying to assign more than 8 nodes to element")


    # position are the isoparametric coordinates of the node, 
    # point are the isoparametric coordinates where you want to evaluate the shape function
    def shapeFunction(self, position, point):
        return 1.0/8.0 * (1.0 + position[0]*point[0]) * (1.0 + position[1]*point[1]) * (1.0 + position[2]*point[2])

    # derivatives of shape function:
    def shapeFunctionX(self, position, point):
        return 1.0/8.0 * position[0] * (1.0 + position[1]*point[1]) * (1.0 + position[2]*point[2])

    def shapeFunctionY(self, position, point):
        return 1.0/8.0 * position[1] * (1.0 + position[2]*point[2]) * (1.0 + position[0]*point[0])

    def shapeFunctionZ(self, position, point):
        return 1.0/8.0 * position[2] * (1.0 + position[0]*point[0]) * (1.0 + position[1]*point[1])
    
    
    def resetStresses(self):
        self.epsilon =     zeros((self.numberOfGaussPoints, 6), dtype=float64)
        self.traceEps =    zeros((self.numberOfGaussPoints), dtype=float64)
        self.sigma =       zeros((self.numberOfGaussPoints, 6 ), dtype=float64)
        self.vonMises =    zeros(self.numberOfGaussPoints, dtype=float64) 
        self.electricDisp =  zeros((self.numberOfGaussPoints, 3), dtype=float64)
        #self.mC = mC
        self.electricDisppEps = zeros((3, 6), dtype=float64) 
        self.sigmapelectricField = zeros((6, 3), dtype=float64)
        self.electricDisppelectricField = zeros((3,3), dtype=float64)


    def resetStiffness(self):
        self.stiffness = zeros((self.numberOfNodes * self.dofPerNode, self.numberOfNodes * self.dofPerNode), dtype=float64)
        
        
    def resetForces(self):
        self.forces = zeros((self.numberOfNodes, self.dofPerNode), dtype=float64)

    #@jit
    def calculateShapeFunctions(self):
        # the following variables are only needed locally, so we don't have to save these in the element
        for gp in range(0, self.numberOfGaussPoints):
            isoDeformationGrad = zeros(( 3, 3), dtype=float64)
            inverseIsoDefoGrad = zeros((3, 3 ), dtype=float64)
            shapeFunctionDerivatives = zeros((self.numberOfNodes, 3), dtype=float64)

            for node in range(0, self.numberOfNodes):
                # we don't need to save these in the element for further calculations
                shapeFunctionDerivatives[node] = array([self.shapeFunctionX(self.nodePositions[node], self.gaussPoints[gp]), \
                                                        self.shapeFunctionY(self.nodePositions[node], self.gaussPoints[gp]), \
                                                        self.shapeFunctionZ(self.nodePositions[node], self.gaussPoints[gp])])

            for i in range(0,3):
                for j in range(0,3):
                    for k in range(0, self.numberOfNodes):
                        isoDeformationGrad[i][j] += self.nodes[k].position[i] * shapeFunctionDerivatives[k][j]   ###refPos?

            self.dVol[gp] = det( isoDeformationGrad ) * self.weights[gp]

            # get inverse Jacobi matrix
            inverseIsoDefoGrad = inv(isoDeformationGrad)

            for k in range(0, self.numberOfNodes):
                # this array saves all nodal shape function values at each Gauss point
                self.shapeFunctionValues[gp][k] = array([inverseIsoDefoGrad[0][0] * shapeFunctionDerivatives[k][0] \
                                                       + inverseIsoDefoGrad[0][1] * shapeFunctionDerivatives[k][1] \
                                                       + inverseIsoDefoGrad[0][2] * shapeFunctionDerivatives[k][2], \
                                                         inverseIsoDefoGrad[1][0] * shapeFunctionDerivatives[k][0] \
                                                       + inverseIsoDefoGrad[1][1] * shapeFunctionDerivatives[k][1] \
                                                       + inverseIsoDefoGrad[1][2] * shapeFunctionDerivatives[k][2], \
                                                         inverseIsoDefoGrad[2][0] * shapeFunctionDerivatives[k][0] \
                                                       + inverseIsoDefoGrad[2][1] * shapeFunctionDerivatives[k][1] \
                                                       + inverseIsoDefoGrad[2][2] * shapeFunctionDerivatives[k][2], \
                                                         self.shapeFunction(self.nodePositions[k], self.gaussPoints[gp])]) 

                self.bmatrix[gp][k] = array([[self.shapeFunctionValues[gp][k][0], 0, 0],\
                                             [0, self.shapeFunctionValues[gp][k][1], 0],\
                                             [0, 0, self.shapeFunctionValues[gp][k][2]],\
                                             [self.shapeFunctionValues[gp][k][1], self.shapeFunctionValues[gp][k][0], 0],\
                                             [0, self.shapeFunctionValues[gp][k][2], self.shapeFunctionValues[gp][k][1]],\
                                             [self.shapeFunctionValues[gp][k][2], 0, self.shapeFunctionValues[gp][k][0]]])
                
                



    # the field variables for Newton-Raphson
    def computeFieldVars(self):
        self.gradU     = zeros((self.numberOfGaussPoints,3,3), dtype=float64)
        self.electricPotential    = zeros(self.numberOfGaussPoints, dtype=float64)
        self.electricField= zeros((self.numberOfGaussPoints, 3 ), dtype=float64)         
        
        for gp in range(0, self.numberOfGaussPoints):
            for k in range(0, self.numberOfNodes):
                for i in range(0, 3):
                    for j in range(0, 3):
                        self.gradU[gp][i][j]    += self.shapeFunctionValues[gp][k][j]  * self.nodes[k].displacement[i]


                    self.electricField[gp][i] -= self.shapeFunctionValues[gp][k][i]  * self.nodes[k].displacement[3]

                self.electricPotential[gp] +=  self.shapeFunctionValues[gp][k][3] * self.nodes[k].displacement[3]



    # "material routine"
    def computeStresses(self):
        for gp in range(0, self.numberOfGaussPoints):
            # linear strain vector
            self.epsilon[gp][0] = self.gradU[gp][0][0]
            self.epsilon[gp][1] = self.gradU[gp][1][1]
            self.epsilon[gp][2] = self.gradU[gp][2][2]
            self.epsilon[gp][3] = self.gradU[gp][0][1] + self.gradU[gp][1][0]
            self.epsilon[gp][4] = self.gradU[gp][1][2] + self.gradU[gp][2][1]
            self.epsilon[gp][5] = self.gradU[gp][2][0] + self.gradU[gp][0][2]


            for i in range(0,3):
                self.traceEps[gp] += self.epsilon[gp][i]
                
                self.normofelectricField_a[gp] = dot(self.electricField[gp][i],self.a[i]) 


            self.sigma[gp][0] = self.llambda * self.traceEps[gp] + 2*self.lmu*self.epsilon[gp][0]\
                              + self.beta1*self.normofelectricField_a[gp]

            
            self.sigma[gp][1] = self.llambda * self.traceEps[gp] + 2*self.lmu*self.epsilon[gp][1]\
                              +self.beta1*self.normofelectricField_a[gp]

            
            self.sigma[gp][2] = self.llambda * self.traceEps[gp] + 2*self.lmu*self.epsilon[gp][2]\
                              + self.beta1*self.normofelectricField_a[gp]

           
            self.sigma[gp][3] = self.lmu*self.epsilon[gp][3]
            self.sigma[gp][4] = self.lmu*self.epsilon[gp][4]
            self.sigma[gp][5] = self.lmu*self.epsilon[gp][5]


            self.electricDisp[gp][0] = -2*self.gamma*self.electricField[gp][0] - self.beta1*self.traceEps[gp]*self.a[0]
            self.electricDisp[gp][1] = -2*self.gamma*self.electricField[gp][1] - self.beta1*self.traceEps[gp]*self.a[1]
            self.electricDisp[gp][2] = -2*self.gamma*self.electricField[gp][2] - self.beta1*self.traceEps[gp]*self.a[2]

                                       
            
        #tangent modulus matrix
        #self.mC = mC   #elastic tangent modulus, already defined in mesh file
            self.electricDisppEps = zeros((3, 6), dtype=float64) # coupled tangent modulus
            self.sigmapelectricField = zeros((6, 3), dtype=float64)  # coupled tangent modulus
                                                                 # these two are symmetric to eachother in total tangent modulus matrix
            self.electricDisppelectricField = zeros((3, 3), dtype=float64) # dilelectric tangent modulus
            
            ## elastic tangent modulus in Voigt notation
            self.mC[0][0] = 2 * self.lmu + self.llambda
            self.mC[0][1] = self.llambda
            self.mC[0][2] = self.llambda
            self.mC[1][0] = self.llambda
            self.mC[1][1] = 2 * self.lmu + self.llambda
            self.mC[1][2] = self.llambda
            self.mC[2][0] = self.llambda
            self.mC[2][1] = self.llambda
            self.mC[2][2] = 2 * self.lmu + self.llambda
            self.mC[3][3] = self.lmu
            self.mC[4][4] = self.lmu
            self.mC[5][5] = self.lmu


            self.sigmapelectricField[0][0] = self.beta1*self.a[0]
            self.sigmapelectricField[0][1] = self.beta1*self.a[1]
            self.sigmapelectricField[0][2] = self.beta1*self.a[2]
            self.sigmapelectricField[1][0] = self.beta1*self.a[0]
            self.sigmapelectricField[1][1] = self.beta1*self.a[1]
            self.sigmapelectricField[1][2] = self.beta1*self.a[2]
            self.sigmapelectricField[2][0] = self.beta1*self.a[0]
            self.sigmapelectricField[2][1] = self.beta1*self.a[1]
            self.sigmapelectricField[2][2] = self.beta1*self.a[2]
            self.sigmapelectricField[3][0] = 0
            self.sigmapelectricField[3][1] = 0
            self.sigmapelectricField[3][2] = 0
            self.sigmapelectricField[4][0] = 0
            self.sigmapelectricField[4][1] = 0
            self.sigmapelectricField[4][2] = 0
            self.sigmapelectricField[5][0] = 0
            self.sigmapelectricField[5][1] = 0
            self.sigmapelectricField[5][2] = 0

            
            self.electricDisppelectricField[0][0] =-2*self.gamma
            self.electricDisppelectricField[0][1] = 0
            self.electricDisppelectricField[0][2] = 0
            self.electricDisppelectricField[1][0] = 0
            self.electricDisppelectricField[1][1] =-2*self.gamma
            self.electricDisppelectricField[1][2] = 0
            self.electricDisppelectricField[2][0] = 0
            self.electricDisppelectricField[2][1] = 0
            self.electricDisppelectricField[2][2] =-2*self.gamma


            self.electricDisppEps[0][0] = -self.beta1*self.a[0]
            self.electricDisppEps[0][1] = -self.beta1*self.a[0]
            self.electricDisppEps[0][2] = -self.beta1*self.a[0]
            self.electricDisppEps[0][3] = 0
            self.electricDisppEps[0][4] = 0
            self.electricDisppEps[0][5] = 0
            self.electricDisppEps[1][0] = -self.beta1*self.a[1]
            self.electricDisppEps[1][1] = -self.beta1*self.a[1]
            self.electricDisppEps[1][2] = -self.beta1*self.a[1]
            self.electricDisppEps[1][3] = 0
            self.electricDisppEps[1][4] = 0
            self.electricDisppEps[1][5] = 0
            self.electricDisppEps[2][0] = -self.beta1*self.a[2]
            self.electricDisppEps[2][1] = -self.beta1*self.a[2]
            self.electricDisppEps[2][2] = -self.beta1*self.a[2]
            self.electricDisppEps[2][3] = 0
            self.electricDisppEps[2][4] = 0
            self.electricDisppEps[2][5] = 0
            
    
            # von Mises stress
            self.vonMises[gp] += (self.sigma[gp][0] - self.sigma[gp][1])**2 + (self.sigma[gp][1] - self.sigma[gp][2])**2 \
                         + (self.sigma[gp][2] - self.sigma[gp][0])**2 \
                         + 6.0 * (self.sigma[gp][3]**2 + self.sigma[gp][4]**2 + self.sigma[gp][5]**2)
            self.vonMises[gp] = 1.0/2.0 * sqrt(self.vonMises[gp])

            # update nodal stresses and von Mises stresses
            for k in range(0, self.numberOfNodes):
                self.nodes[k].weightFactor += self.shapeFunctionValues[gp][k][3]**2 * self.dVol[gp]
                self.nodes[k].vonMises += self.vonMises[gp] * self.shapeFunctionValues[gp][k][3]**2 * self.dVol[gp]
                for i in range(0, 2*3):
                    self.nodes[k].sigma[i] += self.sigma[gp][i] * self.shapeFunctionValues[gp][k][3]**2 * self.dVol[gp]
                    


    def computeStiffness(self):
        fv = self.nodes[0].betaNM/(self.nodes[0].alphaNM * self.nodes[0].timeStep)
        fa = 1.0 / (self.nodes[0].alphaNM * self.nodes[0].timeStep**2)


        computeStiffness(fv, fa, self.stiffness, self.numberOfGaussPoints, self.dofPerNode, self.numberOfNodes, self.bmatrix, self.mC, self.dVol, self.shapeFunctionValues, self.sigmapelectricField, self.electricDisppEps, self.electricDisppelectricField)
      

    def computeForces(self):
        computeForces(self.forces, self.numberOfGaussPoints, self.dofPerNode, self.numberOfNodes, self.bmatrix, self.sigma, self.dVol, self.shapeFunctionValues, self.electricDisp)
    
        # move forces to nodes
        for node in range(0, self.numberOfNodes):
            self.nodes[node].forces += self.forces[node]


    def printNodes(self):
        for node in self.nodes:
            node.printCoordinates()