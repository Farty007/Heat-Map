## module SOLVING_METHOD_With_PIVOTING


import numpy
import error
import swap
import math
import scipy
def gaussPivot(a,b,tol=1.0e-9):
    '''x = gaussPivot(a,b,tol=1.0e-9).
    Solves [a]{x} = {b} by Gauss elimination with
    scaled row pivoting
    '''
    n = len(b)
# Set up scale factors
    s = numpy.zeros(n)
    for i in range(n):
        s[i] = max(abs(a[i,:]))
    for k in range(0,n-1):
# Row interchange, if needed
        p = numpy.argmax(abs(a[k:n,k])/s[k:n]) + k
        if abs(a[p,k]) < tol:
            error.err('Matrix is singular')
        if p != k:
            swap.swapRows(b,k,p)
            swap.swapRows(s,k,p)
            swap.swapRows(a,k,p)
# Elimination
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i,k+1:n] = a [i,k+1:n] - lam*a[k,k+1:n]
                b[i] = b[i] - lam*b[k]
    if abs(a[n-1,n-1]) < tol:
        error.err('Matrix is singular')
# Back substitution
    for k in range(n-1,-1,-1):
        b[k] = (b[k] - numpy.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
    return b

## module LUdecomp


def LUdecomp(a,tol=1.0e-9):
    '''  a,seq = LUdecomp(a,tol=1.0e-9).
    LU decomposition of matrix [a] using scaled row pivoting.
    The returned matrix [a] = [L/U] contains [U] in the upper
    triangle and the nondiagonal terms of [L] in the lower triangle.
    Note that [L][U] is a row-wise permutation of the original [a];
    the permutations are recorded in the vector {seq}.
    '''

    n = len(a)
    seq = numpy.array(range(n))
    
# Set up scale factors
    s = numpy.zeros((n),dtype=float)
    for i in range(n):
        s[i] = max(abs(a[i,:]))
        
    for k in range(0,n-1):
        
# Row interchange, if needed
        p = int(numpy.argmax(abs(a[k:n,k])/s[k:n])) + k
        if abs(a[p,k]) < tol:
            error.err('Matrix is singular')
        if p != k:
            swap.swapRows(s,k,p)
            swap.swapRows(a,k,p)
            swap.swapRows(seq,k,p)
# Elimination
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                a[i,k] = lam
    return a,seq



def LUsolve(a,b,seq):
    '''x = LUsolve(a,b,seq).
    Solves [L][U]{x} = {b}, where the matrix [a] = [L/U] and the
    permutation vector {seq} are returned from LUdecomp.
    '''
    n = len(a)
    
# Rearrange constant vector; store it in [x]
    x = b.copy()
    for i in range(n):
        x[i] = b[seq[i]]
        
# Solution
    for k in range(1,n):
        x[k] = x[k] - numpy.dot(a[k,0:k],x[0:k])
    for k in range(n-1,-1,-1):
        x[k] = (x[k] - numpy.dot(a[k,k+1:n],x[k+1:n]))/a[k,k]
    return x


## modulde LUdecomp3

## module LUdecomp3
''' c,d,e = LUdecomp3(c,d,e).
LU decomposition of tridiagonal matrix [c\d\e]. On output
{c},{d} and {e} are the diagonals of the decomposed matrix.
x = LUsolve(c,d,e,b).
Solves [c\d\e]{x} = {b}, where {c}, {d} and {e} are the
vectors returned from LUdecomp3.
'''
def LUdecomp3(c,d,e):
    n = len(d)
    for k in range(1,n):
        lam = c[k-1]/d[k-1]
        d[k] = d[k] - lam*e[k-1]
        c[k-1] = lam
    return c,d,e

def LUsolve3(c,d,e,b):
    n = len(d)
    for k in range(1,n):
        b[k] = b[k] - c[k-1]*b[k-1]
    b[n-1] = b[n-1]/d[n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - e[k]*b[k+1])/d[k]
    return b

def cramer(a,b):
    '''x = cramer(a,b).
    Solves for cramer by taking in the values as a matrix in the form of a and b
    where a is a nxm matrix and b is 1xn matrix'''
    x = b.copy()
    deta = numpy.linalg.det(a)
    if deta == 0:
        return error.err('Matrix is Singular')
    for i in range(len(x)):
        ai = swap.swapCramer(a,b,i)
        detai = numpy.linalg.det(ai)
        x[i] = detai/deta
    return x


def accuracy(a,b,x):
    q = numpy.dot(a,x)
    p = q - b
    r = numpy.linalg.norm(p)
    return r

## module gaussSeidel

def gaussSeidel(iterEqs,x,tol = 1.0e-9):
    ''' x,numIter,omega = gaussSeidel(iterEqs,x,tol = 1.0e-9)
    Gauss-Seidel method for solving [A]{x} = {b}.
    The matrix [A] should be sparse. User must supply the
    function iterEqs(x,omega) that returns the improved {x},
    given the current {x} (’omega’ is the relaxation factor).
    '''
    omega = 1.0
    k = 10
    p=1
    for i in range(1,501):
        xOld = x.copy()
        x = iterEqs(x,omega)
        dx = math.sqrt(numpy.dot(x-xOld,x-xOld))
        if dx < tol: 
            return x,i,omega
# Compute of relaxation factor after k+p iterations
        if i == k: 
            dx1 = dx
        if i == k + p:
            dx2 = dx
            omega = 2.0/(1.0 + math.sqrt(1.0 - (dx2/dx1)**(1.0/p)))
    print ('Gauss-Seidel failed to converge')


## module conjGrad


def conjGrad(Av,x,b,tol=1.0e-9):
    ''' x, numIter = conjGrad(Av,x,b,tol=1.0e-9)
    Conjugate gradient method for solving [A]{x} = {b}.
    The matrix [A] should be sparse. User must supply
    the function Av(v) that returns the vector [A]{v}'''
    n = len(b)
    r = b - Av(x)
    s = r.copy()
    for i in range(n):
        u = Av(s)
        alpha = numpy.dot(s,r)/numpy.dot(s,u)
        x = x + alpha*s
        r = b - Av(x)
        if(math.sqrt(numpy.dot(r,r))) < tol:
            break
        else:
            beta = -numpy.dot(r,u)/numpy.dot(s,u)
            s = r + beta*s
    return x,i

# module newtonPoly

## module newtonPoly

''' p = evalPoly(a,xData,x).
Evaluates Newton’s polynomial p at x. The coefficient
vector {a} can be computed by the function ’coeffts’.
a = coeffts(xData,yData).
Computes the coefficients of Newton’s polynomial.
'''

def evalPoly(a,xData,x):
    n = len(xData) - 1 # Degree of polynomial
    p = a[n]
    for k in range(1,n+1):
        p = a[n-k] + (x -xData[n-k])*p
    return p
def coeffts(xData,yData):
    m = len(xData) # Number of data points
    a = yData.copy()
    for k in range(1,m):
        a[k:m] = (a[k:m] - a[k-1])/(xData[k:m] - xData[k-1])
    return a

# rational

## module rational
''' p = rational(xData,yData,x)
Evaluates the diagonal rational function interpolant p(x)
that passes through he data points
'''
def rational(xData,yData,x):
    m = len(xData)
    r = yData.copy()
    rOld = np.zeros(m)
    for k in range(m-1):
        for i in range(m-k-1):
            if abs(x - xData[i+k+1]) < 1.0e-9:
                return yData[i+k+1]
            else:
                c1 = r[i+1] - r[i]
                c2 = r[i+1] - rOld[i+1]
                c3 = (x - xData[i])/(x - xData[i+k+1])
                r[i] = r[i+1] + c1/(c3*(1.0 - c1/c2) - 1.0)
                rOld[i+1] = r[i+1]
    return r[0]

# cubicSpline

## module cubicSpline
''' k = curvatures(xData,yData).
Returns the curvatures of cubic spline at its knots.
y = evalSpline(xData,yData,k,x).
Evaluates cubic spline at x. The curvatures k can be
computed with the function ’curvatures’.
'''
def curvatures(xData,yData):
    n = len(xData) - 1
    c = np.zeros(n)
    d = np.ones(n+1)
    e = np.zeros(n)
    k = np.zeros(n+1)
    c[0:n-1] = xData[0:n-1] - xData[1:n]
    d[1:n] = 2.0*(xData[0:n-1] - xData[2:n+1])
    e[1:n] = xData[1:n] - xData[2:n+1]
    k[1:n] =6.0*(yData[0:n-1] - yData[1:n]) \
            /(xData[0:n-1] - xData[1:n]) \
            -6.0*(yData[1:n] - yData[2:n+1]) \
            /(xData[1:n] - xData[2:n+1])
    LUdecomp3(c,d,e)
    LUsolve3(c,d,e,k)
    return k

def evalSpline(xData,yData,k,x):
    def findSegment(xData,x):
        iLeft = 0
        iRight = len(xData)- 1
        while 1:
            if (iRight-iLeft) <= 1: 
                return iLeft
            i = (iLeft + iRight)//2
            if x < xData[i]: 
                iRight = i
            else: 
                iLeft = i

    i = findSegment(xData,x)
    h = xData[i] - xData[i+1]
    y = ((x - xData[i+1])**3/h - (x - xData[i+1])*h)*k[i]/6.0 \
        - ((x - xData[i])**3/h - (x - xData[i])*h)*k[i+1]/6.0 \
        + (yData[i]*(x - xData[i+1]) \
        - yData[i+1]*(x - xData[i]))/h
    return y

# polyFit

## module polyFit

''' c = polyFit(xData,yData,m).
Returns coefficients of the polynomial
p(x) = c[0] + c[1]x + c[2]xˆ2 +...+ c[m]xˆm
that fits the specified data in the least
squares sense.
sigma = stdDev(c,xData,yData).
Computes the std. deviation between p(x)
and the data.
'''

def polyFit(xData,yData,m):
    a = np.zeros((m+1,m+1))
    b = np.zeros(m+1)
    s = np.zeros(2*m+1)
    for i in range(len(xData)):
        temp = yData[i]
        for j in range(m+1):
            b[j] = b[j] + temp
            temp = temp*xData[i]
        temp = 1.0
        for j in range(2*m+1):
            s[j] = s[j] + temp
            temp = temp*xData[i]
    for i in range(m+1):
        for j in range(m+1):
            a[i,j] = s[i+j]
    return gaussPivot(a,b)

def stdDev(c,xData,yData):
    
    def evalPoly(c,x):
        m = len(c) - 1
        p = c[m]
        for j in range(m):
            p = p*x + c[m-j-1]
        return p
    
    n = len(xData) - 1
    m = len(c) - 1
    sigma = 0.0
    for i in range(n+1):
        p = evalPoly(c,xData[i])
        sigma = sigma + (yData[i] - p)**2
    sigma = math.sqrt(sigma/(n - m))
    return sigma

# plotPoly

'''plotPoly(xData,yData,coeff,xlab='x',ylab='y')
Plots data points and the fitting
polynomial defined by its coefficient
array coeff = [a0, a1. ...]
xlab and ylab are optional axis labels
'''

import numpy as np
import matplotlib.pyplot as plt
def plotPoly(xData,yData,coeff,xlab='x',ylab='y'):
    m = len(coeff)
    x1 = min(xData)
    x2 = max(xData)
    dx = (x2 - x1)/20.0
    x = np.arange(x1,x2 + dx/10.0,dx)
    y = np.zeros((len(x)))*1.0
    for i in range(m):
        y = y + coeff[i]*x**i
    plt.plot(xData,yData,'o',x,y,'-')
    plt.xlabel(xlab); plt.ylabel(ylab)
    plt.grid (True)
    plt.show()
    

## module rootsearch
''' x1,x2 = rootsearch(f,a,b,dx).
Searches the interval (a,b) in increments dx for
the bounds (x1,x2) of the smallest root of f(x).
Returns x1 = x2 = None if no roots were detected.'''

def rootsearch(f,a,b,dx):
    x1 = a; f1 = f(a)
    x2 = a + dx; f2 = f(x2)
    while numpy.sign(f1) == numpy.sign(f2):
        if x1 >= b: return None,None
        x1 = x2; f1 = f2
        x2 = x1 + dx; f2 = f(x2)
    else:
        return x1,x2
    

## module bisection
'''root = bisection(f,x1,x2,switch=0,tol=1.0e-9).
Finds a root of f(x) = 0 by bisection.
The root must be bracketed in (x1,x2).
Setting switch = 1 returns root = None if
f(x) increases upon bisection.
'''

def bisection(f,x1,x2,switch=1,tol=1.0e-9):
    f1 = f(x1)
    if f1 == 0.0: return x1
    f2 = f(x2)
    if f2 == 0.0: return x2
    if numpy.sign(f1) == numpy.sign(f2):
        error.err('Root is not bracketed')
    n = int(math.ceil(math.log(abs(x2 - x1)/tol)/math.log(2.0)))
    for i in range(n):
        x3 = 0.5*(x1 + x2); f3 = f(x3)
        if (switch == 1) and (abs(f3) > abs(f1)) and (abs(f3) > abs(f2)):
            return None
    if f3 == 0.0: return x3
    if numpy.sign(f2)!= numpy.sign(f3): x1 = x3; f1 = f3
    else: x2 = x3; f2 = f3
    return (x1 + x2)/2.0
    
## module newtonRaphson
''' root = newtonRaphson(f,df,a,b,tol=1.0e-9).
Finds a root of f(x) = 0 by combining the Newton-Raphson
method with bisection. The root must be bracketed in (a,b).
Calls user-supplied functions f(x) and its derivative df(x).
'''
def newtonRaphson(f,df,a,b,tol=1.0e-9):
    ''' root = newtonRaphson(f,df,a,b,tol=1.0e-9).
    Finds a root of f(x) = 0 by combining the Newton-Raphson
    method with bisection. The root must be bracketed in (a,b).
    Calls user-supplied functions f(x) and its derivative df(x).
    '''
    fa = f(a)
    if fa == 0.0: return a
    fb = f(b)
    if fb == 0.0: return b
    if numpy.sign(fa) == numpy.sign(fb): error.err('Root is not bracketed')
    x = 0.5*(a + b)
    for i in range(30):
        fx = f(x)
        if fx == 0.0: return x
# Tighten the brackets on the root
        if numpy.sign(fa) != numpy.sign(fx): b = x
        else: a = x
        # Try a Newton-Raphson step
        dfx = df(x)
# If division by zero, push x out of bounds
        try: dx = -fx/dfx
        except ZeroDivisionError: dx = b - a
        x = x + dx
# If the result is outside the brackets, use bisection
        if (b - x)*(x - a) < 0.0:
            dx = 0.5*(b - a)
            x = a + dx
# Check for convergence
        if abs(dx) < tol*max(abs(b),1.0): return x
    print('Too many iterations in Newton-Raphson')    


## module newtonRaphson2
''' soln = newtonRaphson2(f,x,tol=1.0e-9).
Solves the simultaneous equations f(x) = 0 by
the Newton-Raphson method using {x} as the initial
guess. Note that {f} and {x} are vectors.
'''
def newtonRaphson2(f,x,tol=1.0e-9):
    def jacobian(f,x):
        h = 1.0e-4
        n = len(x)
        jac = numpy.zeros((n,n))
        f0 = f(x)
        for i in range(n):
            temp = x[i]
            x[i] = temp + h
            f1 = f(x)
            x[i] = temp
            jac[:,i] = (f1 - f0)/h
        return jac,f0
    for i in range(30):
        jac,f0 = jacobian(f,x)
        if math.sqrt(np.dot(f0,f0)/len(x)) < tol: return x
        dx = gaussPivot(jac,-f0)
        x = x + dx
        if math.sqrt(np.dot(dx,dx)) < tol*max(max(abs(x)),1.0):
            return x
        print('Too many iterations')
     
        
        

def MBT(**kwargs):
    M = kwargs['Mach']
    t = kwargs['theta']
    b = kwargs['beta']

    if (M == None and t != None and b != None):
        print('theta, beta = ', t,b)
        print('Solving for Mach')
        
        t = numpy.radians(t)
        
        b = numpy.radians(b)
        
        x1 = 0
        
        x2 = 21
        
        def f1(x):
            return ((2./numpy.tan(b))*((x**2)*((numpy.sin(b))**2)-1)/((x**2)*(1.4+numpy.cos(2*b))+2.)) - (numpy.tan(t))
        
        def df1(x):
            return (x*((8*(numpy.sin(b))*(numpy.cos(b)))+(((4*numpy.cos(2*b))+(5.6))*(1/(numpy.tan(b))))))/((((x**2)*(numpy.cos(2*b)))+(1.4*(x**2))+(2))**2)
        
        roota = rootsearch(f1, x1, x2, (x1+x2)/1000)
        
        roots = newtonRaphson(f1, df1, roota[0], roota[1])
        
        print('Mach number is: ', roots)
        
        return

    if (M != None and t == None and b != None):
        print('Mach, beta = ', M,b)
        print('Solving for Theta')
        b = numpy.radians(b)
        
        x1 = numpy.radians(0.001)
        x2 = numpy.radians(90)
        
        def f2(x):
            return ((2./numpy.tan(b))*((M**2)*((numpy.sin(b))**2)-1)/((M**2)*(1.4+numpy.cos(2*b))+2.)) - (numpy.tan(x))
        
        def df2(x):
            return (-1/(np.cos(x))**2)
        
        roota = rootsearch(f2, x1, x2, (x1+x2)/1000)
        
        roots = newtonRaphson(f2, df2, roota[0], roota[1])
        
        print('Theta is:', numpy.degrees(roots))
        
        
        return

    if (M != None and t != None and b == None):
        M = float(M)
        t = float(t)
        print('Mach, theta = ', M,t)
        print('Solving for Beta')
        t = numpy.radians(t)
        
        x1 = numpy.radians(0.001)
        x2 = numpy.radians(90)
        
        def f3(x):
            return ((2./numpy.tan(x))*((M**2)*((numpy.sin(x))**2)-1)/((M**2)*(1.4+numpy.cos(2*x))+2.)) - (numpy.tan(t))
        
        def df3(x):
            return ((0.4*(1/((numpy.sin(x))**2)))*(((-1.75*(M**4))*(numpy.cos(4*x)))+(0.75*(M**4))+(2*(M**2))+(((M**4)+(10*(M**2)))*(numpy.cos(2*x)))+(10)))/((((M**2)*(numpy.cos(2*x)))+(1.4*(M**2))+(2))**2)
        
        roota = rootsearch(f3, x1, x2, (x1 + x2)/1000)
        
        roots = newtonRaphson(f3, df3, roota[0], roota[1])
        
        print('1st angle is: ', numpy.degrees(roots))
        
        x1 = roots+0.00001
        
        roota = rootsearch(f3, x1, x2, (x1 + x2)/1000)
        
        roots = newtonRaphson(f3, df3, roota[0], roota[1])
        
        print('2nd angle is:', numpy.degrees(roots))
        
        return
    print('Yo mannnnnn!!!!, you gotta give the correct inputs mannnnnnnn!!!')
    return


## module trapezoid
''' Inew = trapezoid(f,a,b,Iold,k).
Recursive trapezoidal rule:
old = Integral of f(x) from x = a to b computed by
trapezoidal rule with 2ˆ(k-1) panels.
Inew = Same integral computed with 2ˆk panels.
'''

def trapezoid(f,a,b,Iold,k):
    if k == 1:Inew = (f(a) + f(b))*(b - a)/2.0
    else:
        n = 2**(k -2 ) # Number of new points
        h = (b - a)/n # Spacing of new points
        x = a + h/2.0
        sum = 0.0
        for i in range(n):
            sum = sum + f(x)
            x=x+h
        Inew = (Iold + h*sum)/2.0
    return Inew


## module romberg
''' I,nPanels = romberg(f,a,b,tol=1.0e-6).
Romberg integration of f(x) from x = a to b.
Returns the integral and the number of panels used. '''

def romberg(f,a,b,tol=1.0e-6):
    def richardson(r,k):
        for j in range(k-1,0,-1):
            const = 4.0**(k-j)
            r[j] = (const*r[j+1] - r[j])/(const - 1.0)
        return r
    r = np.zeros(21)
    r[1] = trapezoid(f,a,b,0.0,1)
    r_old = r[1]
    for k in range(2,21):
        r[k] = trapezoid(f,a,b,r[k-1],k)
        r = richardson(r,k)
        if abs(r[1]-r_old) < tol*max(abs(r[1]),1.0):
            return r[1],2**(k-1)
        r_old = r[1]
    print("Romberg quadrature did not converge")
    
    
    
def root_derivative_integral(f,tol,lowlim,uplim,x1,x2,x):
    '''This function returns the root, derivative and intergral at the asked points
    The f is the function to be provided, tol is the tolerance in the answer required'''
    def df(x,f=f,h=tol):
        return (((f(x+h))-(f(x-h)))/(2*h))
    derivative = df(x)
    roots = newtonRaphson(f,df,x1,x2,tol=tol)
    integral = scipy.integrate.quad(f,lowlim,uplim)
    return derivative, roots, integral