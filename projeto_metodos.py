from sympy import *
import matplotlib.pyplot as plt

# Euler simples
# y(n + 1) = y(n) + h * f(t(n), y(n)
def euler_t(y0, t0, h, n, f, can_print):
    cnt = 0
    y_cur = y0
    t_cur = t0
    y = [y_cur]
    t = [t_cur] #Listas para retornar os pontos, caso precise
    while(cnt < n):
        cnt += 1
        y_cur = y_cur + h * f(t_cur, y_cur)
        t_cur = t_cur + h 
        y.append(y_cur)
        t.append(t_cur)
    if (can_print):
        file_w.write('Metodo de euler\n')
        file_w.write("y( " + str(t0) + " ) = " + str(y0) + '\n')
        file_w.write("h = " + str(h) + '\n')
        for i in range(n + 1):
            file_w.write(str(i) + " " + str(y[i]) + '\n')
        file_w.write('\n')
    return t, y

# Euler inverso
# y(n + 1) = y(n) + h * f(t(n + 1), y(n + 1))
def euler_inverso(y0, t0, h, n, f, can_print):
    cnt = 0
    y_cur = y0
    t_cur = t0
    y = [y_cur]
    t = [t_cur] #Listas para retornar os pontos, caso precise
    while(cnt < n):
        cnt += 1
        y_cur = y_cur + h * f(t_cur + h, y_cur + h * f(t_cur, y_cur))
        t_cur = t_cur + h 
        y.append(y_cur)
        t.append(t_cur)
    if (can_print):
        file_w.write('Metodo de euler inverso\n')
        file_w.write("y( " + str(t0) + " ) = " + str(y0) + '\n')
        file_w.write("h = " + str(h) + '\n')
        for i in range(n + 1):
            file_w.write(str(i) + " " + str(y[i]) + '\n')
        file_w.write('\n')
    return t, y

# Euler Aprimorado
# y(n + 1) = y(n) + h/2 * (f(t(n), y(n)) + f(t(n + 1), y(n + 1))
def euler_aprimorado(y0, t0, h, n, f, can_print):
    cnt = 0
    y_cur = y0
    t_cur = t0
    y = [y_cur]
    t = [t_cur] #Listas para retornar os pontos, caso precise
    while(cnt < n):
        cnt += 1
        y_cur = y_cur + h / 2 * (f(t_cur, y_cur) + f(t_cur + h, y_cur + h * f(t_cur, y_cur)))
        t_cur = t_cur + h 
        y.append(y_cur)
        t.append(t_cur)
    if (can_print):
        file_w.write('Metodo de euler aprimorado\n')
        file_w.write("y( " + str(t0) + " ) = " + str(y0) + '\n')
        file_w.write("h = " + str(h) + '\n')
        for i in range(n + 1):
            file_w.write(str(i) + " " + str(y[i]) + '\n')
        file_w.write('\n')
    return t, y

def runge_kutta(y0, t0, h, n, f, can_print):
    cnt = 0
    y_cur = y0
    t_cur = t0
    y = [y_cur]
    t = [t_cur] #Listas para retornar os pontos, caso precise
    while(cnt < n):
        cnt += 1
        #Calcula os coeficientes
        k1 = f(t_cur, y_cur)
        k2 = f(t_cur + h / 2, y_cur + h / 2 * k1)
        k3 = f(t_cur + h / 2, y_cur + h / 2 * k2)
        k4 = f(t_cur + h, y_cur + h * k3)
        #Calcula y(n + 1)
        y_cur = y_cur + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t_cur = t_cur + h
        y.append(y_cur)
        t.append(t_cur)
    if (can_print):
        file_w.write('Metodo de Runge-Kutta\n')
        file_w.write("y( " + str(t0) + " ) = " + str(y0) + '\n')
        file_w.write("h = " + str(h) + '\n')
        for i in range(n + 1):
            file_w.write(str(i) + " " + str(y[i]) + '\n')
        file_w.write('\n')
    return t, y

#recebe a lista de pontos iniciais
#cria uma matriz, com os coeficientes de cada grau
#calcula os proximos pontos
def adam_bashforth(y, t0, h, n, f, grau, can_print):   
    t = []
    for i in range(grau):
        t.append(t0 + h * i)
    coef = [[3/2, -1/2],
            [23/12, -4/3, 5/12],
            [55/24, -59/24, 37/24, -3/8],
            [1901/720, -1387/360, 109/30, -637/360, 251/720],
            [4277/1440, -2641/480, 4991/720, -3649/720, 959/480, -95/288],
            [198721/60480, -18637/2520, 235183/20160, -10754/945, 135713/20160, -5603/2520, 19087/60480],
            [16083/4480, -1152169/120960, 242653/13440, -296053/13440, 2102243/120960, -115747/13440, 32863/13440, -5257/17280]] 
    cnt = grau - 1
    while(cnt < n):
        cnt += 1
        y_cur = y[-1]
        t_cur = t[-1] + h
        #calcula y(n + 1)
        for i in range(grau):
            y_cur += h * coef[grau - 2][i] * f(t[-i - 1], y[-i - 1])
        y.append(y_cur)
        t.append(t_cur)
    if (can_print):
        file_w.write('Metodo de Adam-Bashforth\n')
        file_w.write("y( " + str(t0) + " ) = " + str(y[0]) + '\n')
        file_w.write("h = " + str(h) + '\n')
        for i in range(n + 1):
            file_w.write(str(i) + " " + str(y[i]) + '\n')
        file_w.write('\n')
    return t, y

#Cada adam_bashforth_by_*
#Recebe o ponto inicial
#Cria a lista inicial, e passa a lista pro adam_bashforth
#E la o resto dos pontos é calculado
def adam_bashforth_by_euler(y0, t0, h, n, f, grau):
    t, y = euler_t(y0, t0, h, grau - 1, f, 0)
    t, y = adam_bashforth(y, t0, h, n, f, grau, 0)
    file_w.write('Metodo de Adam-Bashforth por Euler\n')
    file_w.write("y( " + str(t0) + " ) = " + str(y[0]) + '\n')
    file_w.write("h = " + str(h) + '\n')
    for i in range(n + 1):
         file_w.write(str(i) + " " + str(y[i]) + '\n')
    file_w.write('\n')
    return t, y

def adam_bashforth_by_euler_inverso(y0, t0, h, n, f, grau):
    t, y = euler_inverso(y0, t0, h, grau - 1, f, 0)
    t, y = adam_bashforth(y, t0, h, n, f, grau, 0)
    file_w.write('Metodo de Adam-Bashforth por Euler Inverso\n')
    file_w.write("y( " + str(t0) + " ) = " + str(y[0]) + '\n')
    file_w.write("h = " + str(h) + '\n')
    for i in range(n + 1):
         file_w.write(str(i) + " " + str(y[i]) + '\n')
    file_w.write('\n')
    return t, y

def adam_bashforth_by_euler_aprimorado(y0, t0, h, n, f, grau):
    t, y = euler_aprimorado(y0, t0, h, grau - 1, f, 0)
    t, y = adam_bashforth(y, t0, h, n, f, grau, 0)
    file_w.write('Metodo de Adam-Bashforth por Euler Aprimorado\n')
    file_w.write("y( " + str(t0) + " ) = " + str(y[0]) + '\n')
    file_w.write("h = " + str(h) + '\n')
    for i in range(n + 1):
         file_w.write(str(i) + " " + str(y[i]) + '\n')
    file_w.write('\n')
    return t, y

def adam_bashforth_by_runge_kutta(y0, t0, h, n, f, grau):
    t, y = runge_kutta(y0, t0, h, grau - 1, f, 0)
    t, y = adam_bashforth(y, t0, h, n, f, grau, 0)
    file_w.write('Metodo de Adam-Bashforth por Runge-Kutta ( ordem = ' + str(grau) + " )\n")
    file_w.write("y( " + str(t0) + " ) = " + str(y[0]) + '\n')
    file_w.write("h = " + str(h) + '\n')
    for i in range(n + 1):
         file_w.write(str(i) + " " + str(y[i]) + '\n')
    file_w.write('\n')
    return t, y

#Calcula por adam_multon utilizando previsão
#com adam_bashforth
def adam_multon(y, t0, h, n, f, grau, can_print): 
    t = []
    for i in range(grau - 1):
        t.append(t0 + h * i)

    coef = [[1/2, 1/2],
            [5/12, 2/3, -1/12],
            [3/8, 19/24, -5/24, 1/24],
            [251/720, 323/360, -11/30, 53/360, -19/720],
            [95/288,  1427/1440, -133/240, 241/720, -173/1440, 3/160],
            [19087/60480, 2713/2520, -15487/20160, 586/945, -6737/20160, 263/2520, -863/60480],
            [5257/17280, 139849/120960, -4511/4480, 123133/120960, -88547/120960, 1537/4480, -11351/120960, 275/24192]] 

    cnt = grau - 2
    while(cnt < n):
        cnt += 1
        t_temp, y_temp = adam_bashforth(y[cnt - grau + 1 : cnt], t[cnt - grau + 1], h, grau - 1, f, grau - 1, 0)
        y.append(y_temp[-1])
        t.append(t_temp[-1])
        y_cur = y[-2]
        for i in range(grau):
            y_cur += h * coef[grau - 2][i] * f(t[-i - 1], y[-i - 1])
        y.pop()
        y.append(y_cur)

    if (can_print):
        file_w.write('Metodo de Adam-Multon\n')
        file_w.write("y( " + str(t0) + " ) = " + str(y[0]) + '\n')
        file_w.write("h = " + str(h) + '\n')
        for i in range(n + 1):
            file_w.write(str(i) + " " + str(y[i]) + '\n')
        file_w.write('\n')
    return t, y

#Cada adam_multon_by_*
#Recebe o ponto inicial
#Cria a lista inicial, e passa a lista pro adam_multon
#E la o resto dos pontos é calculado
def adam_multon_by_euler(y0, t0, h, n, f, grau):
    t, y = euler_t(y0, t0, h, grau - 2, f, 0)
    t, y = adam_multon(y, t0, h, n, f, grau, 0)
    file_w.write('Metodo de Adam-Multon por Euler\n')
    file_w.write("y( " + str(t0) + " ) = " + str(y[0]) + '\n')
    file_w.write("h = " + str(h) + '\n')
    for i in range(n + 1):
        file_w.write(str(i) + " " + str(y[i]) + '\n')
    file_w.write('\n')
    return t, y

def adam_multon_by_euler_inverso(y0, t0, h, n, f, grau):
    t, y = euler_inverso(y0, t0, h, grau - 2, f, 0)
    t, y = adam_multon(y, t0, h, n, f, grau, 0)
    file_w.write('Metodo de Adam-Multon por Euler Inverso\n')
    file_w.write("y( " + str(t0) + " ) = " + str(y[0]) + '\n')
    file_w.write("h = " + str(h) + '\n')
    for i in range(n + 1):
        file_w.write(str(i) + " " + str(y[i]) + '\n')
    file_w.write('\n')
    return t, y

def adam_multon_by_euler_aprimorado(y0, t0, h, n, f, grau):
    t, y = euler_aprimorado(y0, t0, h, grau - 2, f, 0)
    t, y = adam_multon(y, t0, h, n, f, grau, 0)
    file_w.write('Metodo de Adam-Multon por Euler Aprimorado\n')
    file_w.write("y( " + str(t0) + " ) = " + str(y[0]) + '\n')
    file_w.write("h = " + str(h) + '\n')
    for i in range(n + 1):
        file_w.write(str(i) + " " + str(y[i]) + '\n')
    file_w.write('\n')
    return t, y

def adam_multon_by_runge_kutta(y0, t0, h, n, f, grau):
    t, y = runge_kutta(y0, t0, h, grau - 2, f, 0)
    t, y = adam_multon(y, t0, h, n, f, grau, 0)
    file_w.write('Metodo de Adam-Multon por Runge-Kutta ( ordem = ' + str(grau) + " )\n")
    file_w.write("y( " + str(t0) + " ) = " + str(y[0]) + '\n')
    file_w.write("h = " + str(h) + '\n')
    for i in range(n + 1):
        file_w.write(str(i) + " " + str(y[i]) + '\n')
    file_w.write('\n')
    return t, y

#Recebe a lista inicial
#E calcula por formula inversa os proximos pontos
#Utiliza adam_bashforth na previsao
def formula_inversa(y, t0, h, n, f, grau, can_print):
    t = []
    for i in range(grau - 1):
        t.append(t0 + h * i)
    coef = [[1, 1],
            [4/3, -1/3, 2/3],
            [18/11, -9/11, 2/11, 6/11],
            [48/25, -36/25, 16/25, -3/25, 12/25],
            [300/137, -300/137, 200/137, -75/137, 12/137, 60/137],
            [360/147, -450/147, 400/147, -225/147, 72/147, -10/147, 60/147]]
    cnt = grau - 2
    while(cnt < n):
        cnt += 1
        t_temp, y_temp = adam_bashforth(y[cnt - grau + 1 : cnt], t[cnt - grau + 1], h, grau - 1, f, grau - 1, 0)
        y_cur = f(t_temp[-1], y_temp[-1]) * h * coef[grau - 2][-1]
        for i in range(grau - 1):
            y_cur += y[-i -1] * coef[grau - 2][i]
        y.append(y_cur)
        t.append(t_temp[-1])
    if (can_print):
        file_w.write('Metodo Formula Inversa de Diferenciacao\n')
        file_w.write("y( " + str(t0) + " ) = " + str(y[0]) + '\n')
        file_w.write("h = " + str(h) + '\n')
        for i in range(n + 1):
            file_w.write(str(i) + " " + str(y[i]) + '\n')
        file_w.write('\n')
    return t, y

#Cada formula_inversa_by_*
#Recebe o ponto inicial
#Cria a lista inicial, e passa a lista pra formula_inversa
#E la o resto dos pontos é calculado
def formula_inversa_by_euler(y0, t0, h, n, f, grau):
    t, y = euler_t(y0, t0, h, grau - 2, f, 0)
    t, y = formula_inversa(y, t0, h, n, f, grau, 0)
    file_w.write('Metodo Formula Inversa de Diferenciacao por Euler\n')
    file_w.write("y( " + str(t0) + " ) = " + str(y[0]) + '\n')
    file_w.write("h = " + str(h) + '\n')
    for i in range(n + 1):
        file_w.write(str(i) + " " + str(y[i]) + '\n')
    file_w.write('\n')
    return t, y

def formula_inversa_by_euler_inverso(y0, t0, h, n, f, grau):
    t, y = euler_inverso(y0, t0, h, grau - 2, f, 0)
    t, y = formula_inversa(y, t0, h, n, f, grau, 0)
    file_w.write('Metodo Formula Inversa de Diferenciacao por Euler Inverso\n')
    file_w.write("y( " + str(t0) + " ) = " + str(y[0]) + '\n')
    file_w.write("h = " + str(h) + '\n')
    for i in range(n + 1):
        file_w.write(str(i) + " " + str(y[i]) + '\n')
    file_w.write('\n')
    return t, y

def formula_inversa_by_euler_aprimorado(y0, t0, h, n, f, grau):
    t, y = euler_aprimorado(y0, t0, h, grau - 2, f, 0)
    t, y = formula_inversa(y, t0, h, n, f, grau, 0)
    file_w.write('Metodo Formula Inversa de Diferenciacao por Euler Aprimorado\n')
    file_w.write("y( " + str(t0) + " ) = " + str(y[0]) + '\n')
    file_w.write("h = " + str(h) + '\n')
    for i in range(n + 1):
        file_w.write(str(i) + " " + str(y[i]) + '\n')
    file_w.write('\n')
    return t, y

def formula_inversa_by_runge_kutta(y0, t0, h, n, f, grau):
    t, y = runge_kutta(y0, t0, h, grau - 2, f, 0)
    t, y = formula_inversa(y, t0, h, n, f, grau, 0)
    file_w.write('Metodo Formula Inversa de Diferenciacao por Runge-Kutta ( ordem = ' + str(grau) + " )\n")
    file_w.write("y( " + str(t0) + " ) = " + str(y[0]) + '\n')
    file_w.write("h = " + str(h) + '\n')
    for i in range(n + 1):
        file_w.write(str(i) + " " + str(y[i]) + '\n')
    file_w.write('\n')
    return t, y

file = open("entrada.txt", "r")
file_w = open("saida.txt", "w")
for line in file:
    lst = line.split()
    metodo = lst[0]

    if (metodo == "adam_bashforth"):
        y0_list = []
        grau = int(lst[-1])
        for i in range(grau):
            y0_list.append(float(lst[i + 1]))
        t0 = float(lst[grau + 1])
        h = float(lst[grau + 2])
        n = int(lst[grau + 3])
        funcao = lst[grau + 4]
        y, t = symbols("y t")
        funcao = sympify(funcao)
        f = lambdify((t, y), funcao, "numpy")
        t, y = adam_bashforth(y0_list, t0, h, n, f, grau, 1)
    elif (metodo == "adam_multon"):
        y0_list = []
        grau = int(lst[-1])
        for i in range(grau - 1):
            y0_list.append(float(lst[i + 1]))
        t0 = float(lst[grau])
        h = float(lst[grau + 1])
        n = int(lst[grau + 2])
        funcao = lst[grau + 3]
        y, t = symbols("y t")
        funcao = sympify(funcao)
        f = lambdify((t, y), funcao, "numpy")
        t, y = adam_multon(y0_list, t0, h, n, f, grau, 1)
    elif (metodo == "formula_inversa"):
        y0_list = []
        grau = int(lst[-1])
        for i in range(grau - 1):
            y0_list.append(float(lst[i + 1]))
        t0 = float(lst[grau])
        h = float(lst[grau + 1])
        n = int(lst[grau + 2])
        funcao = lst[grau + 3]
        y, t = symbols("y t")
        funcao = sympify(funcao)
        f = lambdify((t, y), funcao, "numpy")
        t, y = formula_inversa(y0_list, t0, h, n, f, grau, 1)
    else:
        y0 = float(lst[1])
        t0 = float(lst[2])
        h = float(lst[3])
        n = int(lst[4])
        funcao = lst[5]
        y, t = symbols("y t")
        funcao = sympify(funcao)
        f = lambdify((t, y), funcao, "numpy")

        if (metodo == "euler"):
            t, y = euler_t(y0, t0, h, n, f, 1)
        elif (metodo == "euler_inverso"):
            t, y = euler_inverso(y0, t0, h, n, f, 1)
        elif (metodo == "euler_aprimorado"):
            t, y = euler_aprimorado(y0, t0, h, n, f, 1)
        elif (metodo == "runge_kutta"):
            t, y = runge_kutta(y0, t0, h, n, f, 1)
        elif (metodo == "adam_bashforth_by_euler"):
            grau = int(lst[6])
            t, y = adam_bashforth_by_euler(y0, t0, h, n, f, grau)
        elif (metodo == "adam_bashforth_by_euler_inverso"):
            grau = int(lst[6])
            t, y = adam_bashforth_by_euler_inverso(y0, t0, h, n, f, grau)
        elif (metodo == "adam_bashforth_by_euler_aprimorado"):
            grau = int(lst[6])
            t, y = adam_bashforth_by_euler_aprimorado(y0, t0, h, n, f, grau)
        elif (metodo == "adam_bashforth_by_runge_kutta"):
            grau = int(lst[6])
            t, y = adam_bashforth_by_runge_kutta(y0, t0, h, n, f, grau)
        elif (metodo == "adam_multon_by_euler"):
            grau = int(lst[6])
            t, y = adam_multon_by_euler(y0, t0, h, n, f, grau)
        elif (metodo == "adam_multon_by_euler_inverso"):
            grau = int(lst[6])
            t, y = adam_multon_by_euler_inverso(y0, t0, h, n, f, grau)
        elif (metodo == "adam_multon_by_euler_aprimorado"):
            grau = int(lst[6])
            t, y = adam_multon_by_euler_aprimorado(y0, t0, h, n, f, grau)
        elif (metodo == "adam_multon_by_runge_kutta"):
            grau = int(lst[6])
            t, y = adam_multon_by_runge_kutta(y0, t0, h, n, f, grau)
        elif (metodo == "formula_inversa_by_euler"):
            grau = int(lst[6])
            t, y = formula_inversa_by_euler(y0, t0, h, n, f, grau)
        elif (metodo == "formula_inversa_by_euler_inverso"):
            grau = int(lst[6])
            t, y = formula_inversa_by_euler_inverso(y0, t0, h, n, f, grau)
        elif (metodo == "formula_inversa_by_euler_aprimorado"):
            grau = int(lst[6])
            t, y = formula_inversa_by_euler_aprimorado(y0, t0, h, n, f, grau)
        elif (metodo == "formula_inversa_by_runge_kutta"):
            grau = int(lst[6])
            t, y = formula_inversa_by_runge_kutta(y0, t0, h, n, f, grau)
        
    #printa o grafico
    #Comente as proximas 3 linhas caso nao queira o grafico
    plt.plot(t, y, 'ro')
    plt.axis([0, t[-1], 0, y[-1]])
    plt.show()