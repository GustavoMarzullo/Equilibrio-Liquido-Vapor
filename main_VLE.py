#importando módulos
import numpy as np
from math import sqrt, exp, log as ln
from scipy.optimize import newton, minimize, Bounds
import matplotlib.pyplot as plt
R = 8.314472

#Definindo os Group interaction parameters (GIP): (Akl = Alk) and (Bkl = Blk)

#Grupo A
GIP_A = np.zeros([6,6])
#CH3 (grupo 1)
GIP_A[0][1] = 74.81e6 #CH2 (grupo 2)
GIP_A[0][2] = 261.5e6 #CH (grupo 3)
GIP_A[0][3] = 396.7e6 #C (groupo 4)
GIP_A[0][4] = 28.5e6 #32.94e6 #CH4 (grupo 5)
GIP_A[0][5] = 8.579e6 #C2H6 (grupo 6)

#CH2 (grupo 2)
GIP_A[1][2] = 54.47e6 #CH (grupo 3)
GIP_A[1][3] = 88.53e6 #C (groupo 4)
GIP_A[1][4] = 36.72e6 #CH4 (grupo 5)
GIP_A[1][5] = 31.23e6 #C2H6 (grupo 6)

#CH (grupo 3)
GIP_A[2][3] = -305.7e6 #C (groupo 4)
GIP_A[2][4] = 145.2e6 #CH4 (grupo 5)
GIP_A[2][5] = 174.3e6 #C2H6 (grupo 6)

#C (grupo 4)
GIP_A[3][4] = 263.9e6 #CH4 (grupo 5)
GIP_A[3][5] = 333.2e6 #C2H6 (grupo 6)

#CH4 (grupo 5)
GIP_A[4][5] = 13.04e6 #C2H6 (grupo 6)

GIP_A += GIP_A.T #aritifício para criar uma matriz simétria a partir dos dados superiores

#Grupo B
GIP_B = np.zeros([6,6])
#CH3 (grupo 1)
GIP_B[0][1] = 165.7e6 #CH2 (grupo 2)
GIP_B[0][2] = 388.8e6 #CH (grupo 3)
GIP_B[0][3] = 804.3e6 #C (groupo 4)
GIP_B[0][4] = 20.2e6 #-35e6 #CH4 (grupo 5)
GIP_B[0][5] = -29.5e6 #C2H6 (grupo 6)

#CH2 (grupo 2)
GIP_B[1][2] = 79.61e6 #CH (grupo 3)
GIP_B[1][3] = 315e6 #C (groupo 4)
GIP_B[1][4] = 108.4e6 #CH4 (grupo 5)
GIP_B[1][5] = 84.76e6 #C2H6 (grupo 6)

#CH (grupo 3)
GIP_B[2][3] = -250.8e6 #C (groupo 4)
GIP_B[2][4] = 301.6e6 #CH4 (grupo 5)
GIP_B[2][5] = 352.1e6 #C2H6 (grupo 6)

#C (grupo 4)
GIP_B[3][4] = 531.5e6 #CH4 (grupo 5)
GIP_B[3][5] = 203.8e6 #C2H6 (grupo 6)

#CH4 (grupo 5)
GIP_B[4][5] = 6.863e6 #C2H6 (grupo 6)

GIP_B += GIP_B.T #aritifício para criar uma matriz simétria a partir dos dados superiores

def V_PR78(P,T,a,b):
    '''Retorna o volume molar de gás e líquido. O maior valor é a raíz do gás e o menor valor é a raíz do líquido.
    Args:
        P (float): Valor da pressão do sistema (Pascal)
        T (float): Valor da temperatura do sistema (Kelvin)
        a (float): Valor de a da equação de Peng-Robson
        b (float): Valor de b da equação de Peng-Robson
    Returns:
        list: Valor do volume molar de gás(maior valor) e líquido(menor valor).
    '''
    a3 = -P
    a2  = R*T - b*P
    a1 = 3*b**2*P + 2*b*R*T - a
    a0 = a*b - b**3*P - b**2*R*T
    x = np.roots([a3,a2,a1,a0])
    filtered = [i.real for i in x if i.imag == 0]
    return filtered
    
class composto(object):
    
    def __init__(self, _grupos , fator_acentrico, Tc, Pc):
        '''Args:
            grupos (list): Grupos presentes na molécula.
                Grupo 1: CH3
                Grupo 2: CH2
                Grupo 3: CH
                Grupo 4: C
                Grupo 5: CH4
                Grupo 6: C2H6
                Ex.: Propano (CH3CH2CH3) => [2,1,0,0,0]
            fator_acentrico (float): Valor do fator acêntrico do composto
            Tc (float): Temperatura crítica do composto (Kelvin)
            Pc (float):  Pressão crítica do composto (bar)'''
        self._grupos = _grupos
        self.w = fator_acentrico
        self.Tc = Tc
        self.Pc = Pc*1e5
        
    def grupos(self,i):
        return self._grupos[i]

    def alpha(self):
        _alpha = np.zeros(len(self._grupos))
        numero_total_de_grupos = sum(self._grupos)
        for i in range(len(self._grupos)):
            _alpha[i] = self._grupos[i]/numero_total_de_grupos
        return _alpha

    def m(self):
        if self.w <= 0.941:
            return 0.37464 + 1.54226*self.w - 0.2699*self.w**2
        else:
           return  0.379642 + 1.48503*self.w - 0.164423*self.w**2 + 0.016666*self.w**3        

class sistema(object):
    def __init__(self, compostos:list , T):
        '''Args:
            compostos (list): Lista com os compostos presentes no  sistema
            T (float): Temperatura do sistema (Kelvin)'''
        self.compostos = compostos
        self.T = T

    def M(self):
        _m = np.zeros(len(self.compostos))
        i=0
        for composto in (self.compostos):
            _m[i]=composto.m()
            i+=1
        return _m
    
    def Alpha(self):
        _Alpha = []
        for composto in (self.compostos):
            _Alpha.append(composto.alpha())
        return _Alpha

    def a(self,i):
        molecula = self.compostos[i]
        Tc = molecula.Tc
        Pc = molecula.Pc
        m = molecula.m()
        T = self.T
        return 0.457235529 * (R**2*Tc**2)/(Pc) * (1 +  m*(1 - sqrt(T/Tc)))**2
            
    def b(self,i):
        molecula = self.compostos[i]
        return 0.077796073 * (R*molecula.Tc)/(molecula.Pc)

    def K(self,i,j):
        if i==j:
            return 0
        else: 
            DS=0
            alpha = self.Alpha()
            for k in range(6):
                for l in range(6):
                    if k!=l:
                        DS += (alpha[i][k] - alpha[j][k])*(alpha[i][l] - alpha[j][l]) * GIP_A[k][l] * (298.15/self.T) ** (GIP_B[k][l]/GIP_A[k][l] - 1)
                    else:
                        DS += 0
            DS -= (sqrt(self.a(i))/self.b(i) - sqrt(self.a(j))/self.b(j))**2
            D = 2 * sqrt(self.a(i) * self.a(j))/(self.b(i)*self.b(j))
            return (-0.5*DS)/D
        
    def am(self,Z):
        N = len(self.compostos)
        S = 0
        for i in range(N):
            for j in range(N):
                S +=  Z[i]*Z[j] * sqrt(self.a(i)*self.a(j)) * (1-self.K(i,j))
        return S

    def bm(self,Z):
        N = len(self.compostos)
        S = 0
        for i in range(N):
            S += Z[i]*self.b(i)
        return S

    def V(self,P,Z, tipo="G"):
        '''Retorna o volume molar de gás ou do líquido no sistema.
        Args:
            P (float): Valor da pressão do sistema (bar)
            Z (list): Valores de fração molar do composto no sistema
            tipo (string): "G" para retornar volume molar do gás e "L" para retornar o volume molar do líquido.
        Returns:
            float: Valor do volume molar de gás ou do líquido no sistema.'''
        a = self.am(Z)
        b = self.bm(Z)
        T = self.T
        P*=1e5
        x = V_PR78(P,T,a,b)
        if tipo == "L" or tipo == "l":
            res = min(x) #b + 1e-10
        elif tipo == "G" or tipo == "g":
            res = max(x) #(R*T)/P 
        else:
            print("Escreve direito")
        return res
    
    def delta(self, i, Z):
        S = 0
        N = len(self.compostos)
        for j in range(N):
            S += Z[j]*sqrt(self.a(j))*(1-self.K(i,j))
        return 2*sqrt(self.a(i))/self.am(Z) * S

    def phi(self, P, Z, i, tipo="G"):
        '''Retorna o coeficiente de fugacidade do composto i na fase líquida ou gasosa.
        Args:
            P (float): Valor da pressão do sistema (bar)
            Z (list): Valores de fração molar do composto no sistema
            i (int): Índice do composto 
            tipo (string): "G" para retornar volume molar do gás e "L" para retornar o volume molar do líquido.
        Returns:
            float: Valor do coeficiente de fugacidade do composto i na fase líquida ou gasosa.'''
        bi = self.b(i)
        am = self.am(Z)
        bm = self.bm(Z)
        V = self.V(P, Z, tipo = tipo)
        T = self.T
        P*=1e5
        delta = self.delta(i,Z)
        ln_phi = (bi/bm) * ((P*V)/(R*T) -1) - ln((P*(V-bm))/(R*T)) - am/(2*sqrt(2)*R*T*bm) * (delta - (bi/bm)) * ln( (V+bm*(1+sqrt(2)))/(V+bm*(1-sqrt(2))) ) 
        return exp(ln_phi)
    
def estimarxy(P,P1sat,P2sat):
    fobj = lambda y1: (P*y1)/(P1sat) + (P*(1-y1))/P2sat -1
    y1 = newton(fobj,0.5)
    x1 = (y1*P)/P1sat
    return x1,y1

def fobj(X):
    x0, y0 = X
    x1 = 1-x0
    y1 = 1-y0
    phi0L = S.phi(P,[x0,x1], 0, tipo="L")
    phi0G = S.phi(P,[y0,y1], 0, tipo="G")
    phi1L = S.phi(P,[x0,x1], 1, tipo="L")
    phi1G = S.phi(P,[y0,y1], 1, tipo="G")
    return ((x0*phi0L)/(y0*phi0G)-1)**2 + ((x1*phi1L)/(y1*phi1G)-1)**2

def X(P, S, Psat):
    '''Retorna a fração molar de x0 e y0 para um sistema binário.
    Args:
        P (float): Valor da pressão do sistema (bar)
        S (sistema): Sistema binário a ser calculado
        Psat (list): Valores de pressão de saturação do composto na temperatura do sistema (bar)
    Returns:
        list: Valor de x0 e y0'''
    #estimando o chute
    P1sat,P2sat = Psat
    chute = estimarxy(P, P1sat, P2sat)

    #definindo fronteiras e restrições
    bounds = Bounds((1e-15,1e-15),(1-1e-15,1-1e-15))
    def con1(X):
        x0, y0 = X
        x1 = 1-x0
        y1 = 1-y0
        phi0L = S.phi(P,[x0,x1], 0, tipo="L")
        phi0G = S.phi(P,[y0,y1], 0, tipo="G")
        return x0*phi0L-y0*phi0G
    def con2(X):
        x0, y0 = X
        x1 = 1-x0
        y1 = 1-y0
        phi1L = S.phi(P,[x0,x1], 1, tipo="L")
        phi1G = S.phi(P,[y0,y1], 1, tipo="G")
        return x1*phi1L-y1*phi1G
    cons = [{'type':'eq', 'fun': con1},{'type':'eq', 'fun': con2}]
    
    #minimizando
    res = minimize(fobj,x0=chute, bounds = bounds,constraints=cons)
    return res.x
    
if __name__ == "__main__":
    ### PROPANO/N-BUTANO
    # criando o sistema
    P = composto([2,1,0,0,0,0], 0.152, 369.83, 42.48) #propano
    B = composto([2,2,0,0,0,0], 0.200, 425.12, 37.96) #n-butano
    S = sistema([P,B],363.38)
    Psat = [37,13]
    
    #calculando e plotando
    Pressoes = list(np.arange(13,38,1))
    Composiçoes = []
    
    for P in Pressoes:
        Composiçoes.append(X(P,S,Psat=Psat))

        
    x = [i[0] for i in Composiçoes]
    y = [i[1] for i in Composiçoes]

    XYexp = [0.0621583316460822, 0.120719443882027, 0.233002632111763, 0.354666936626847, 0.55342511979483, 0.670290882094891, 0.798764932172504, 0.887399608557737, 0.931781062293311, 0.894290342174529, 0.84057501518526, 0.688668421407842, 0.520395491664979, 0.394182358102179]
    Pexp = [13.8238509819801, 13.6560032395221, 15.3083620166025, 16.7886211783761, 20.6962947965175, 23.6458797327394, 27.8932982385098, 32.4830937436728, 33.9566713909698, 34.2127961125734, 32.3061348451103, 28.2296011338327, 23.8058311399068, 20.5960720793682]
        
    plt.plot(x, Pressoes,color='black',label="Curva de líquido",zorder=1)
    plt.plot(y,Pressoes,color='silver',label="Curva de vapor",zorder=2)
    plt.scatter(XYexp,Pexp, color='black',label="Dados exp. Jaubert et. al. (2004)", marker="x",zorder=3,s=30)
    
    plt.xlabel("$x_1, y_1$",size=12)
    plt.ylabel("P (bar)",size=12)
    plt.title("Sistema Binário Propano(1)/n-Butano (T=363,38 K)")
    plt.xlim(0,1)
    plt.ylim(0,40)
    plt.legend(fontsize = 'small',loc='upper left')
    plt.savefig("Sistema Binário Propano(1);n-Butano (T=363,38 K).jpg", dpi = 300)
    plt.show()
    
    ### DIMETILPENTANO/OCTANO
    # criando o sistema
    DMP = composto([4,1,2,0,0,0], 0.307, 519.73, 27.4) #dimetilpentano
    C8 = composto([2,6,0,0,0,0], 0.393, 569.32, 24.97) #octano
    S = sistema([DMP,C8],313.15)
    Psat = [0.05,0.25]

    #calculando e plotando
    Pressoes = list(np.arange(0.051,0.26,0.01))
    Composiçoes = []
    
    for P in Pressoes:
        Composiçoes.append(X(P,S,Psat))
        
    x = [i[0] for i in Composiçoes]
    y = [i[1] for i in Composiçoes]

    XYexp =[0.421103858182903, 0.515073257458764, 0.622513204510946, 0.714367286554109, 0.795190573212037, 0.835680080979015, 0.862699040969671, 0.88976991058567, 0.919137781123064, 0.932725125556406, 0.955448564049988, 0.982649207729343, 0.875481786210208, 0.792504250100575, 0.693850007137573, 0.653503250840287, 0.568332533059942, 0.510012068987892, 0.460684947506391, 0.395668141765186, 0.296974966583178, 0.220732704361706, 0.148993602138676, 0.108607913622383]
    Pexp = [0.0654541443347132, 0.0752832318932738, 0.0885928598310341, 0.10365573536473, 0.124538977639928, 0.138468925600529, 0.14891833318193, 0.161693291978665, 0.177373892053934, 0.186086922667635, 0.204096967180139, 0.222685804015209, 0.221585319957953, 0.204191702246389, 0.184481617503925, 0.176946935385493, 0.161298778826063, 0.148541988398198, 0.138686946026967, 0.125934048821002, 0.104479800667038, 0.0888264531450744, 0.0749146735533436, 0.0656358280234112]

    plt.plot(x, Pressoes,color='black',label="Curva de líquido",zorder=1)
    plt.plot(y,Pressoes,color='silver',label="Curva de vapor",zorder=2)
    plt.scatter(XYexp,Pexp, color='black',label="Dados exp. Jaubert et. al. (2004)", marker="x",zorder=3,s=30)
   
    plt.xlabel("$x_1, y_1$",size=12)
    plt.ylabel("P (bar)",size=12)
    plt.xlim(0,1)
    plt.ylim(0,0.3)
    plt.legend(fontsize = 'small',loc='upper left')
    plt.title("Sistema Binário 2,4-dimetilpentano(1)/n-octano (T=313,15 K)")
    plt.savefig("Sistema Binário 2,4-dimetilpentano(1);n-octano (T=313,15 K).jpg", dpi = 300)
    plt.show()
    
    ### METANO/ETANO
    # criando o sistema
    C1 = composto([0,0,0,0,1,0], 0.012, 190.6, 45.99) #METANO
    C2 = composto([0,0,0,0,0,1], 0.100, 305.3, 48.72) #ETANO
    S = sistema([C1,C2],172.04)
    Psat = [27,1]

    #calculando e plotando
    Pressoes = list(np.arange(0.5,26,0.5))
    Composiçoes = []
    
    for P in Pressoes:
        Composiçoes.append(X(P,S,Psat))
        
    x = [i[0] for i in Composiçoes]
    y = [i[1] for i in Composiçoes]

    XYexp = [0.77, 0.78, 0.85, 0.92, 0.95, 0.96, 0.98, 0.99, 0.95, 0.92, 0.87, 0.71, 0.57, 0.51, 0.31, 0.26, 0.21, 0.11, 0.07]
    Pexp = [2.37, 2.52, 3.28, 6.15, 8.56, 12.33, 17.44, 20.9, 23.45, 22.69, 20.88, 17.25, 13.92, 12.56, 8.47, 6.96, 5.75, 3.32, 2.42]
        
    plt.plot(x, Pressoes,color='black',label="Curva de líquido",zorder=1)
    plt.plot(y,Pressoes,color='silver',label="Curva de vapor",zorder=2)
    plt.scatter(XYexp,Pexp, color='black',label="Dados exp. Jaubert et. al. (2004)", marker="x",zorder=3,s=30)
    plt.xlabel("$x_1, y_1$",size=12)
    plt.ylabel("P (bar)",size=12)
    plt.xlim(0,1)
    plt.ylim(0,30)
    plt.legend(loc='upper left',fontsize = 'small')
    plt.title("Sistema Binário Metano(1)/Etano (T=172,04 K)")
    plt.savefig("Sistema Binário Metano(1);Etano (T=172,04 K).jpg", dpi = 300)
    plt.show()
    
