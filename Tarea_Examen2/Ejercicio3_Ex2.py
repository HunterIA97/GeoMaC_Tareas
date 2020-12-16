#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:03:57 2020

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

def exactSolution (t , y0 , lam ):
    """
    Función que calcula la solución exacta del modelo

    Parameters
    ----------
    t : float array 
        Arreglo de valores de tiempo
    y0 : float
        condicion inicial
    lam : float
        factor de decaimiento

    Returns
    -------
    float array
        valores de la solución analítica del sistema

    """
    return y0 /(y0+(1-y0) *np . exp ( - lam * t ))

def forwardEuler (y , ht , lam ):
    """
    Formula que resuelve el sistema por 
    medio del método de Euler  hacia adelante

    Parameters
    ----------
    y : float array
        Arreglo que almacena los datos de la solución
    ht : float
        Distancia entre pasos de tiempo
    lam : float
        factor de decaimiento

    Returns
    -------
    y : float array
        Arreglo que almacena los valores de la aproximaciòn

    """
    A = 1+ht * lam
    B=-ht*lam
    for i , val in enumerate ( y [0: -1]):
        y [ i +1] = A* y [ i ]+B*y[i]**2
    return y

#Nt=4
Nt = 4
Tmax = 1
ht = mesh (0 , Tmax , Nt )
y0 = 0.01
lam = 10
t = np . linspace (0 , Tmax , Nt +1)
yf = np . zeros ( Nt +1)
yf [0] =y0
yf= forwardEuler ( yf , ht , lam )
tl = np . linspace (0 , Tmax , 100)
y_exacta = exactSolution ( tl , y0 , lam )
y_exac_p = exactSolution (t , y0 , lam )
error_f = np.linalg.norm( yf - y_exac_p ,2)

Ecuacion = '$y ( t ) = \\frac{y_0}{y_0 +(1 - y_0) e ^{\lambda t}}$'

plt.suptitle('Función Logística: '+Ecuacion, fontsize=14,)
plt.plot( tl , y_exacta , 'g-' , lw =3 , label = ' Sol . Exacta ')
plt.plot(t , yf , 'C7o--' , label = '$N_t$ = {}'. format ( Nt ) +', Error = {:10.9f}'. format (error_f))


#Nt=16
Nt = 16
ht = mesh (0 , Tmax , Nt )
t = np . linspace (0 , Tmax , Nt +1)
yf = np . zeros ( Nt +1)
yf [0] =y0
An= forwardEuler ( yf , ht , lam )
y_exac_p = exactSolution (t , y0 , lam )
error_f = np.linalg.norm( yf - y_exac_p ,2)

plt.plot(t , yf , 'C5<--' , label = '$N_t$ = {}'. format ( Nt ) +', Error = {:10.9f}'. format (error_f))

#Nt=64
Nt = 64
ht = mesh (0 , Tmax , Nt )
t = np . linspace (0 , Tmax , Nt +1)
yf = np . zeros ( Nt +1)
yf [0] =y0
An= forwardEuler ( yf , ht , lam )
y_exac_p = exactSolution (t , y0 , lam )
error_f = np.linalg.norm( yf - y_exac_p ,2)

plt.plot(t , yf , 'C6*--' , label = '$N_t$ = {}'. format ( Nt ) +', Error = {:10.9f}'. format (error_f))

#atributos de la gràfica
plt.xlim(-0.05,t[-1]+0.1)
plt.ylim (-0.05,1.05)
plt.xlabel ( ' $t$ ')
plt.ylabel ( ' $y ( t ) $ ')
plt.legend ( loc = 'upper left' , ncol =1 , framealpha =0.75 , fancybox = True , fontsize =10)
plt.grid ( color = 'w')

nticks = np . arange (1 , Nt +1 ,1)
plt.savefig('Funcion_logistica.pdf')
plt.show ()