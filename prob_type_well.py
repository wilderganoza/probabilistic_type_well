# Importamos la librerías a utilizar
import streamlit as st # 

# Librerías estándar
import math # Funciones matemáticas básicas
import datetime # Manipulación de fechas
import time # Funciones relacionadas al tiempo

# Librerías de terceros
import numpy as np # Manipulación de arreglos
import pandas as pd # Manipulación de dataframes
import matplotlib.pyplot as plt # Generación de gráficos
import matplotlib.ticker as ticker # Formato de las etiquetas de los ejes de los gráficos
from scipy.optimize import curve_fit # Ajuste de curvas

# Funcion que elimina valores nulos o vacios del dataframe con la produccion historica
def clear_nulls(data):
  # Convertimos a string la columna con los nombres de los pozos
  data = data.astype({"Well": str})

  # Conservamos todos los valores mayores a 0 o que no sean nulos
  data = data[(data['Oil Volume'].notnull()) & (data['Oil Volume'] > 0)]

  # Reinicializamos los indices del dataframe
  data = data.reset_index(drop = True)

  # Retornamos el dataframe filtrado
  return data

# Funcion que obtiene la cantidad de dias de cada mez
def calc_days(data):
  # Creamos una columna donde colocaremos la cantidad de dias del mes
  data['Days'] = data['Date'].dt.days_in_month

  # Retornamos el dataframe aumentado
  return data

# Funcion que calcula el caudal de produccion
def calc_rate(data):
  # Calculamos el caudal promedio de cada mes
  data['Oil Rate'] = round(data['Oil Volume'] / data['Days'], 2)

  # Retornamos el dataframe aumentado
  return data

# Funcion que obtiene un listado de pozos activos
def active_wells(data, last_date, calc_type):
  # Si el usuario selecciona que solo utilizara pozos activos en el analisis
  if calc_type == 'Only Active Wells':
    # Conservamos unicamente los pozos que cuenten con produccion hasta la ultima fecha del dataframe
    data = data[data['Date'] == last_date]

    # Conservamos unicamente la columna con los nombres de los pozos
    data = data[['Well']]

    # Reinicializamos los indices del dataframe
    data = data.reset_index(drop = True)

    # Convertimos el dataframe a un listado
    active_wells = data['Well'].tolist()
  # Si el usuario selecciona que utilizara todos los pozos en el analisis
  else:
    active_wells = data['Well'].unique()

  # Retornamos el listado
  return active_wells

# Función que calcula la distancia entre dos puntos
def calc_distance(x_1, y_1, x_2, y_2):
  # Calculamos la distancia entre dos puntos a partir de sus coordenadas X e Y
  dist = ((x_2 - x_1)**2 + (y_2 - y_1)**2)**0.5
  
  # Retornamos el valor de la distancia
  return dist

# Funcion que filtra solo los pozos que se encuentren dentro del radio de busqueda
def selected_wells(locations, X, Y, R, active_well_list):
  # Creamos un nuevo dataframe donde almacenaremos los pozos que se encuentren dentro del radio de búsqueda
  selected_wells = pd.DataFrame()

  # Convertimos el tipo de dato de la columna con el nombre de los pozos a string
  locations = locations.astype({"Well": str})

  # Bucle que calcula la distancia entre cada pozo y la locación
  for i in range(0, len(locations)):
    # Obtenemos el nombre y ubicacion del pozo  
    well_name = locations.loc[i, "Well"]
    X_neigh = locations.loc[i, "X"]
    Y_neigh = locations.loc[i, "Y"]

    # Calculamos la distancia entre el pozo y la locacion
    dist = calc_distance(X, Y, X_neigh, Y_neigh) 
      
    # Seleccionamos y almacenamos en el dataframe la informacion del pozo solo si se 
    # encuentra dentro del radio de busqueda
    if dist <= R and well_name in active_well_list:
      selected_wells = selected_wells.append(locations.loc[i], ignore_index = True)

  # Convertimos el dataframe anterior a un listado
  selected_wells = selected_wells['Well'].unique()

  # Retornamos el listado
  return selected_wells

# Funcion que crea el dataframe con los pozos sincronizados
def sync_wells(data, selected_well_list):
  # Creamos un dataframe donde almacenaremos la producción de los pozos dentro del radio de búsqueda
  sync_wells_data = pd.DataFrame()

  # Bucle que extrae la producción de cada pozo y la coloca en el dataframe creado
  for well in selected_well_list: 
    # Extraemos la produccion correspondiente al pozo actual 
    prd = data[data.Well == well]

    # Reinicializamos los índices
    prd = prd.reset_index(drop = True) 
    
    # Creamos un dataframe donde almacenaremos temporalmente los caudales del pozo actual
    dummy = pd.DataFrame() 

    # Copiamos los caudales del pozo actual a nuestro dataframe temporal
    dummy['Oil Rate'] = prd['Oil Rate'] 

    # Colocamos el nombre del pozo en el encabezdo de la columna 'Oil Rate'
    dummy = dummy.rename(columns = {'Oil Rate': well}) 

    # Concatenamos la columna al dataframe de pozos sincronizados
    sync_wells_data = pd.concat([sync_wells_data, dummy], axis = 1)

  # Retornamos el dataframe con los pozos sincronizados
  return sync_wells_data

# Funcion que calcula el caudal promedio para cada timestep del dataframe de pozos sincronizados
def calc_avg(data):
  # Creamos un dataframe donde almacenaremos cada percentil
  average = pd.DataFrame()

  # Extraemos los percentiles 10, 50 y 90
  P10 = data.quantile(q = 0.9, axis = 1)
  P50 = data.quantile(q = 0.5, axis = 1)
  P90 = data.quantile(q = 0.1, axis = 1)

  # Añadimos las columnas correspondientes a cada percentil
  average = pd.concat([average, P10], axis = 1)
  average = pd.concat([average, P50], axis = 1)
  average = pd.concat([average, P90], axis = 1)

  # Renombramos las columnas
  average = average.rename(columns={0.9: 'Average P10'})
  average = average.rename(columns={0.5: 'Average P50'})
  average = average.rename(columns={0.1: 'Average P90'})

  # Reinicializamos los indices del dataframe
  average.reset_index(inplace = True)

  # Calculamos el tiempo relativo para cada punto del dataframe
  for i in range(0, len(average)):
    average.loc[i, 'Time'] = average.loc[i, 'index'] * 30

  # Convertimos el tipo de dato de la columna "Time" a entero
  average['Time'] = average['Time'].astype(int)

  # Retornamos el dataframe con los valores promedio
  return average 

# Función que define el modelo de declinación hiperbólica de Arps
def arps_hyperbolic(t, qi, b, Di):
  # Calculamos el caudal de acuerdo a los parametros recibidos
  q = qi / ((1 + b*Di*t)**(1/b))
  
  return q

# Función que define el modelo de declinación exponencial de Arps
def arps_exponential(t, qi, Di):
  # Calculamos el caudal de acuerdo a los parametros recibidos
  q = qi / np.exp(Di*t)

  return q

# Funcion que obtiene los parametros de declinacion a partir de las ecuaciones de Arps
def calc_parameters(data):
  #
  max_rate_P10 = data['Average P10'].max()
  max_rate_P50 = data['Average P50'].max()
  max_rate_P90 = data['Average P90'].max()

  # Realizamos el ajuste para la ecuación hiperbólica
  popt_P10, pcov_P10 = curve_fit(arps_hyperbolic, data['Time'], data['Average P10'], p0=[max_rate_P10, 0.5, 0.01], bounds=([0, 0, 0.000137],[max_rate_P10, 1, 1]))
  popt_P50, pcov_P50 = curve_fit(arps_hyperbolic, data['Time'], data['Average P50'], p0=[max_rate_P50, 0.5, 0.01], bounds=([0, 0, 0.000137],[max_rate_P50, 1, 1]))
  popt_P90, pcov_P90 = curve_fit(arps_hyperbolic, data['Time'], data['Average P90'], p0=[max_rate_P90, 0.5, 0.01], bounds=([0, 0, 0.000137],[max_rate_P90, 1, 1]))

  # Obtenemos los parámetros de declinación
  qi_P10 = round(popt_P10[0], 2)
  b_P10 = round(popt_P10[1], 4)
  Di_P10 = round(popt_P10[2], 6)

  qi_P50 = round(popt_P50[0], 2)
  b_P50 = round(popt_P50[1], 4)
  Di_P50 = round(popt_P50[2], 6)

  qi_P90 = round(popt_P90[0], 2)
  b_P90 = round(popt_P90[1], 4)
  Di_P90 = round(popt_P90[2], 6)

  # Retornamos los parametros de declinacion
  return qi_P10, b_P10, Di_P10, qi_P50, b_P50, Di_P50, qi_P90, b_P90, Di_P90

# Funcion que genera los valores ajustados para el pozo tipo
def calc_forecast(data, qi_P10, b_P10, Di_P10, qi_P50, b_P50, Di_P50, qi_P90, b_P90, Di_P90):
  # Calculamos los valores de la curva de ajuste
  if b_P10 == 0:
    for i in range(0, len(data)):
      data.loc[i, 'Adjusted P10'] = round(arps_exponential(data.loc[i, 'Time'], qi_P10, Di_P10), 4)
  else:
    for i in range(0, len(data)):
      data.loc[i, 'Adjusted P10'] = round(arps_hyperbolic(data.loc[i, 'Time'], qi_P10, b_P10, Di_P10), 4)

  if b_P50 == 0:
    for i in range(0, len(data)):
      data.loc[i, 'Adjusted P50'] = round(arps_exponential(data.loc[i, 'Time'], qi_P50, Di_P50), 4)
  else:
    for i in range(0, len(data)):
      data.loc[i, 'Adjusted P50'] = round(arps_hyperbolic(data.loc[i, 'Time'], qi_P50, b_P50, Di_P50), 4)

  if b_P90 == 0:
    for i in range(0, len(data)):
      data.loc[i, 'Adjusted P90'] = round(arps_exponential(data.loc[i, 'Time'], qi_P90, Di_P90), 4)
  else:
    for i in range(0, len(data)):
      data.loc[i, 'Adjusted P90'] = round(arps_hyperbolic(data.loc[i, 'Time'], qi_P90, b_P90, Di_P90), 4)

  # Redondeamos la columna con los valores del ajuste a 2 decimales
  data['Adjusted P10'] = round(data['Adjusted P10'], 2)
  data['Adjusted P50'] = round(data['Adjusted P50'], 2)
  data['Adjusted P90'] = round(data['Adjusted P90'], 2)

  # Eliminamos la columna con los indices
  data = data.drop('index', axis=1)

  # Reordenamos las columnas del dataframe 
  data = data[['Time', 'Average P10', 'Average P50', 'Average P90', 'Adjusted P10', 'Adjusted P50', 'Adjusted P90']]

  # Convertimos los valores de la columna 'Time' a dias
  data['Time'] = data['Time'] / 30

  # Renombramos la columna 'Time'
  data = data.rename(columns = {'Time': 'Month'})

  # Retornamos el dataframe con la curva ajustada
  return data 

# Funcion para graficar la locacion, los pozos seleccionados y no seleccionados, y el radio de busqueda
def plot_wells(locations, selected_wells, X, Y, R):
  # Creamos la figura y definimos sus dimensiones
  fig = plt.figure(figsize = (16, 9))
  ax = fig.add_subplot(111)

  # Añadimos el radio de busqueda
  plt.gcf().gca().add_artist(plt.Circle((X, Y), R, color = 'g', fill = True, alpha = 0.2))

  # Añadimos la ubicación de los pozos y la locacion
  ax.scatter(locations['X'], locations['Y'], color = 'b', zorder = 1) # Todos los pozos
  ax.scatter(X, Y, marker = '*', color = 'r', s = 150, label = 'Location', zorder = 2) # Locación

  # Añadimos los pozos seleccionados
  dummy = locations[locations['Well'].isin(selected_wells)]
  ax.scatter(dummy['X'], dummy['Y'], marker = 'o', color = 'g', s = 50, label = 'Selected Wells', zorder = 3) 

  # Definimos los incrementos del eje x iguales a los del eje y
  ax.ticklabel_format(style = 'plain')
  ax.set_aspect('equal', adjustable = 'box')

  # Colocamos el nombre de los pozos
  #for i in range(0, len(locations)):
    #y1 = locations.loc[i, "Y"] + 100
    #plt.text(locations.loc[i, "X"], y1, str(locations.loc[i,"Well"]), color = 'black', ha = 'center', va =  'bottom', alpha = 0.8)

  # Configuración secundaria del gráfico
  plt.grid() # Habilitamos la cuadrícula
  plt.legend() # Habilitamos la leyenda
  plt.xlabel("X Coordinate", size = 12, labelpad = 15) # Título del eje x
  plt.ylabel("Y Coordinate", size = 12, labelpad = 15) # Título del eje y
  
  # Configuración de los límites de los ejes
  ax.set_xlim(ax.get_xticks()[0], ax.get_xticks()[-2]) # Límites del eje x
  ax.set_ylim(ax.get_yticks()[0], ax.get_yticks()[-1]) # Límites del eje y

  # Retornamos la figura
  return fig

# Funcion que grafica el promedio de los pozos sincronizados y la curva de ajuste
def plot_type_well(data):
  # Creamos la figura y definimos sus dimensiones
  fig, ax = plt.subplots(figsize = (16, 9))

  # Añadimos las curvas con la producción promedio y el ajuste
  plt.plot(data.index, data['Average P10'], color='green', label='Average P10')
  plt.plot(data.index, data['Adjusted P10'], color='green', linewidth=2, label='Adjusted P10')

  plt.plot(data.index, data['Average P50'], color='red', label='Average P50')
  plt.plot(data.index, data['Adjusted P50'], color='red', linewidth=2, label='Adjusted P50')

  plt.plot(data.index, data['Average P90'], color='blue', label='Average P90')
  plt.plot(data.index, data['Adjusted P90'], color='blue', linewidth=2, label='Adjusted P90')

  # Configuración secundaria del gráfico
  plt.grid(which ='both') # Habilitamos la cuadricula
  plt.title('Type Well', size = 15, y = 1.02) # Titulo del grafico
  plt.legend() # Leyenda del grafico
  plt.xlabel('Month', size = 12, labelpad = 15) # Titulo del eje X
  plt.ylabel('Oil Rate', size = 12, labelpad = 15) # Titulo del eje Y

  # Configuración del eje y
  plt.yscale("log") # Escala del eje y
  ax.set_ylim(ax.get_yticks()[1], ax.get_yticks()[-2]) # Límites del eje y
  ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y))) # Formato de los marcadores del eje y

  # Retornamos el gráfico
  return fig

# Funcion que grafica el pozo tipo, la curva promedio y la produccion de los pozos seleccionados
def plot_all_wells(type_well_data, time_zero_wells):
  # Creamos la figura y definimos sus dimensiones
  fig, ax = plt.subplots(figsize=(16, 9))
  
  # Añadimos las curvas con la producción promedio y el ajuste
  plt.plot(type_well_data.index, type_well_data['Average P10'], color='green', label='Average P10')
  plt.plot(type_well_data.index, type_well_data['Adjusted P10'], color='green', linewidth=2, label='Adjusted P10')

  plt.plot(type_well_data.index, type_well_data['Average P50'], color='red', label='Average P50')
  plt.plot(type_well_data.index, type_well_data['Adjusted P50'], color='red', linewidth=2, label='Adjusted P50')

  plt.plot(type_well_data.index, type_well_data['Average P90'], color='blue', label='Average P90')
  plt.plot(type_well_data.index, type_well_data['Adjusted P90'], color='blue', linewidth=2, label='Adjusted P90')

  # Añadimos una curva para cada pozo escogido
  for column in time_zero_wells:
    #plt.plot(average["Time"], type_well_data[column], marker = '^', linestyle = '', markersize = 4, label = column)
    plt.plot(time_zero_wells.index, time_zero_wells[column], marker = '^', linestyle = '', markersize = 4, label = column)

  # Estimamos la cantidad de columnas para la leyenda
  if len(time_zero_wells.axes[1]) <= 30:
    ncol = 1
  elif len(time_zero_wells.axes[1]) > 30 and len(time_zero_wells.axes[1]) <= 60:
    ncol = 2
  else:
    ncol = 3

  # Configuración secundaria del gráfico
  plt.grid(which = 'both') # Habilidamos la cuadricula
  plt.title('Type Well', size = 15, y = 1.02) # Titulo del grafico
  plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), ncol = ncol) # Habilitamos la leyenda
  plt.xlabel('Month', size = 12, labelpad = 15) # Titulo del eje X
  plt.ylabel('Oil Rate', size = 12, labelpad = 15) # Titulo del eje Y

  # Configuración del eje y
  plt.yscale("log") # Escala del eje y
  ax.set_ylim(ax.get_yticks()[1], ax.get_yticks()[-2]) # Límites del eje y
  ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y))) # Formato de los marcadores del eje y

  # Retornamos el gráfico
  return fig

@st.cache_data
def type_well(production, locations, X, Y, R, calc_type):
  # Eliminamos los valores nulos o vacios de la produccion historica
  production = clear_nulls(production)

  # Calculamos la cantidad de dias de cada mes del dataframe de produccion historica
  production = calc_days(production)

  # Calculamos el caudal de produccion promedio de cada mes
  production = calc_rate(production)
  
  # Obtenemos la ultima fecha del dataframe con la produccion historica
  last_date = production['Date'].max()

  # Obtenemos el listado de pozos activos (opcion por defecto)
  active_well_list = active_wells(production, last_date, calc_type)

  # Seleccionamos solo los pozos que se encuentren dentro del radio de busqueda
  selected_well_list = selected_wells(locations, X, Y, R, active_well_list)
  
  # Sincronizamos los pozos seleccionados a tiempo 0
  sync_wells_data = sync_wells(production, selected_well_list)

  # Calculamos el promedio para cada timestep del dataframe de pozos sincronizados
  type_well_data = calc_avg(sync_wells_data)

  # Obtenemos los parametros de declinacion para la curva promedio
  qi_P10, b_P10, Di_P10, qi_P50, b_P50, Di_P50, qi_P90, b_P90, Di_P90 = calc_parameters(type_well_data)

  # Generamos la curva de ajuste del pozo tipo
  type_well_data = calc_forecast(type_well_data, qi_P10, b_P10, Di_P10, qi_P50, b_P50, Di_P50, qi_P90, b_P90, Di_P90)

  # Retornamos los resultados
  return type_well_data, sync_wells_data, selected_well_list, qi_P10, b_P10, Di_P10, qi_P50, b_P50, Di_P50, qi_P90, b_P90, Di_P90
