#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 18:35:25 2020

@author: josuelg
"""
import numpy as np
import matplotlib.pyplot as plt

def mesh (a , b , Nt ):
    """
    Funciòón que calcula el espaciamiento entre pasos de tiempo

    Parameters
    ----------
    a : integer
        Punto inicial de tiempo
    b : integer
        Punto final de tiempo
    Nt : integer
        Número de pasos de tiempo

    Returns
    -------
    ht : float
        Distancia entre pasos de tiempo

    """
    ht = (b - a ) / Nt
    return ht

def f (t , y ):
    """
    Funcion que calcula el comportamiento de la derivada

    Parameters
    ----------
    t : float
        Valor de tiempo.
    y : float
        Valor espacial

    Returns
    -------
    float
        valor de la función evaluada en los valores (t,y)

    """
    return y - t **2 + 1

def Exacta ( t ):
    """
    Función que calcula la solución analitica

    Parameters
    ----------
    t : float array
        Arreglo de valores de tiempo

    Returns
    -------
    float array 
        Arreglo de valores de la solución analitica

    """
    return ( t +1)**2 - 0.5 * np . exp ( t )

def Euler (f , t , w , ht ):
    """
    Función que calcula la aproximacion de Euler

    Parameters
    ----------
    f : function
        función del comportamiento de la derivada
    t : float array
        Arreglo de valores de tiempo
    w : float array
        Arreglo que almacena los valores de la aproximación
    ht : float
        espaciamiento entre pasos de tiempo

    Returns
    -------
    None.

    """
    for i , val in enumerate ( w [0: -1]):
        w [ i +1] = w [ i ] + ht * f ( t [ i ] , w [ i ])
        t [ i +1] = t [0] + ( i +1) * ht
        
def RK2 (f , t , w , ht ):
    """
    Función que calcula la aproximacion de Runge-Kutta de orden 2

    Parameters
    ----------
    f : function
        función del comportamiento de la derivada
    t : float array
        Arreglo de valores de tiempo
    w : float array
        Arreglo que almacena los valores de la aproximación
    ht : float
        espaciamiento entre pasos de tiempo

    Returns
    -------
    None.

    """
    for i , val in enumerate ( w [0: -1]):
        k1 = ht * f ( t [ i ] , w [ i ])
        w [ i +1] = w [ i ] +ht * f ( t [ i ] + ht * 0.5 ,w [ i ] + k1 * 0.5)
        t [ i +1] = a + ( i +1) * ht
        
def RK3 (f ,t , w , ht ):
    """
    Función que calcula la aproximacion de Runge-Kutta de orden 3

    Parameters
    ----------
    f : function
        función del comportamiento de la derivada
    t : float array
        Arreglo de valores de tiempo
    w : float array
        Arreglo que almacena los valores de la aproximación
    ht : float
        espaciamiento entre pasos de tiempo

    Returns
    -------
    None.

    """
    for i ,val in enumerate ( w [0: -1]):
        k1= ht * f ( t [ i ] , w [ i ])
        k2= ht * f ( t [ i ] + ht /3 ,w [ i ] + k1 / 3)
        k3 = ht * f ( t [ i ] + 2 * ht / 3 ,w [ i ] + 2 * k2 / 3)
        w [ i + 1] = w [ i ] + ( k1 + 3 * k3 ) / 4
        t [ i +1] = a + ( i +1) * ht

def RK4 (f ,t , w , ht ):
    """
    Función que calcula la aproximacion de Runge-Kutta de orden 4

    Parameters
    ----------
    f : function
        función del comportamiento de la derivada
    t : float array
        Arreglo de valores de tiempo
    w : float array
        Arreglo que almacena los valores de la aproximación
    ht : float
        espaciamiento entre pasos de tiempo

    Returns
    -------
    None.

    """
    for i , val in enumerate ( w [0: -1]):
        k1= ht * f ( t [ i ] , w [ i ])
        k2= ht * f ( t [ i ] + ht /2 ,w [ i ] + k1 / 2)
        k3 = ht * f ( t [ i ] + ht /2 ,w [ i ] + k2 / 2)
        k4 = ht * f ( t [ i ] + ht , w [ i ] + k3 )
        w [ i +1] = w [ i ] + ( k1 + 2* k2 +2* k3 + k4 ) / 6
        t [ i +1] = a + ( i +1) * ht

Nt = 32 # 4 , 8 , 16 , 32
a = 0
b = 4
ht = mesh (a , b , Nt )
y0 = 0.5

t = np . linspace (a , b , Nt +1)
y_eul = np . zeros ( Nt +1);
y_rk2 = np . zeros ( Nt +1)
y_rk3 = np . zeros ( Nt +1)
y_rk4 = np . zeros ( Nt +1)

y_eul [0]= y0
y_rk2 [0]= y0
y_rk3 [0]= y0
y_rk4 [0]= y0

Euler (f , t , y_eul , ht )
RK2 (f , t , y_rk2 , ht )
RK3 (f , t , y_rk3 , ht )
RK4 (f , t , y_rk4 , ht )

yp=Exacta(t)
e_eul=np.fabs(yp-y_eul)
e_rk2=np.fabs(yp-y_rk2)
e_rk3=np.fabs(yp-y_rk3)
e_rk4=np.abs(yp-y_rk4)

n_error_eul=np.linalg.norm(e_eul,2)
n_error_rk2=np.linalg.norm(e_rk2,2)
n_error_rk3=np.linalg.norm(e_rk3,2)
n_error_rk4=np.linalg.norm(e_rk4,2)

tl=np.linspace(a,b,100)
yp=Exacta(tl)
Error = 'E EUL = {:10.9f}, E RK2 = {:10.9f}, E RK3 = {:10.9f}, E RK4 = {:10.9f} '. format (n_error_eul,n_error_rk2,n_error_rk3, n_error_rk4)

plt.suptitle('Solucion y aproximacion $N_t$ = {}'. format(Nt), fontsize=14,)
plt.title(Error,fontsize =10 , color = 'blue')
plt.plot(tl,yp,'g-', lw=2, label= 'Sol. Exacta')
plt.plot(t,y_eul,'C1v--', lw=1, label= 'Euler')
plt.plot(t,y_rk2,'C2^--', lw=1, label= 'RK2')
plt.plot(t,y_rk3,'C3o--', lw=1, label= 'RK3')
plt.plot(t,y_rk4,'C4<--', lw=1, label= 'RK4')
plt.xlim(-0.1,t[-1]+0.1)
plt.ylim(-3,7)
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.legend(loc='upper left', ncol=1, framealpha=0.75, fancybox=True, fontsize=10)
plt.grid()
plt.savefig('Comparacion_Metodos_Nt_ {}.pdf'. format ( Nt ))
plt.show ()

plt.suptitle('Errores $N_t$ = {}'. format(Nt), fontsize=14,)
plt.title(Error,fontsize =10 , color = 'blue')
plt.plot(e_eul[1:],'C1o-', lw=1, label= 'Euler')
plt.plot(e_rk2[1:],'C2o-', lw=1, label= 'RK2')
plt.plot(e_rk3[1:],'C3o-', lw=1, label= 'RK3')
plt.plot(e_rk4[1:],'C4o-', lw=1, label= 'RK4')
plt.xlabel('$n$')
plt.ylabel('$Error$')
plt.yscale('log')
plt.legend(loc='upper left', ncol=1, framealpha=0.75, fancybox=True, fontsize=10)
plt.grid()
plt.savefig('Errores_Nt_ {}.pdf'. format ( Nt ))
plt.show ()