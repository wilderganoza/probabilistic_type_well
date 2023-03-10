# Importamos las librerias a utilizar
import streamlit as st #
import pandas as pd # Manipulación de dataframes
from PIL import Image # Manipulacion de imagenes
from io import BytesIO # 
from pyxlsb import open_workbook as open_xlsb # 

# Libreria interna
import prob_type_well # Generacion de pozo tipo probabilistico

# Almacenamos el favicon en una variable
favicon = Image.open("Favicon.png")

# Configuracion de la pagina
st.set_page_config(page_title = "Type Well Analysis", page_icon = favicon,)

# CSS para ocultar menu de Streamlit
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """

# Ocultamos el menu de Streamlit
st.markdown(hide_menu_style, unsafe_allow_html = True)

# Desactivamos los argumentos adicionales para mostrar figuras
st.set_option('deprecation.showPyplotGlobalUse', False)

# Funcion para convertir dataframes a Excel
def to_excel(df):
  output = BytesIO()
  writer = pd.ExcelWriter(output, engine = 'xlsxwriter')
  df.to_excel(writer, index = False, sheet_name = 'Sheet1')
  workbook = writer.book
  worksheet = writer.sheets['Sheet1']
  format1 = workbook.add_format({'num_format': '0.00'}) 
  worksheet.set_column('A:A', None, format1)  
  writer.save()
  processed_data = output.getvalue()

  return processed_data

# Titulo de la pagina
st.write("## Type Well Analysis")

# Descripcion breve
st.write('This app creates a traditional decline analysis for the average of a group of wells within a search radius.')

# Mostramos el widget para carga del archivo con las coordenadas
well_locations = st.file_uploader('Load Well Locations')

# Mostramos el widget para carga del archivo con la produccion
production_data = st.file_uploader('Load Production Data')

# Una vez cargados ambos archivos de Excel ejecutamos el resto del procedimiento
if production_data is not None and well_locations is not None:
  with st.spinner('Wait for it...'):
    # Asignamos el archivo con las coordendas a un dataframe
    locations = pd.read_excel(well_locations)
    
    # Asignamos el archivo con la produccion a un dataframe
    PRD_data = pd.read_excel(production_data)

    # Subtitulo para la seccion de las coordenadas
    st.subheader('Well Locations')

    # Mostramos el dataframe con las coordenadas de todos los pozos
    st.dataframe(locations.style.format({'X': '{:.2f}', 'Y': '{:.2f}'}), 
                 height = 500, use_container_width = True)

    # Subtitulo para la seccion de la produccion historica
    st.subheader('Production Data')
    
    # Mostramos el dataframe con la produccion historica
    st.dataframe(PRD_data.style.format({'Oil Volume': '{:.0f}', 'Date': '{:%d-%m-%Y}'}), 
                 height = 500, use_container_width = True)

  # Calculamos la locacion por defecto del pozo propuesta
  X_default = int((locations['X'].min() + locations['X'].max()) / 2)
  Y_default = int((locations['Y'].min() + locations['Y'].max()) / 2)

  # Calculamos el radio de busqueda por defecto
  x_calc = (locations['X'].max() - locations['X'].min()) / 3
  y_calc = (locations['Y'].max() - locations['Y'].min()) / 3

  if x_calc < y_calc:
    R_default = int(x_calc)
  else:
    R_default = int(y_calc)

  # Subtitulo para la seccion de opciones
  st.subheader('Options')

  # Definimos las columnas para las opciones
  col1, col2 = st.columns(2)

  # Agregamos el menu de opciones
  with col1:
    X = st.number_input('X Coordinate (m):', value = X_default)
    Y = st.number_input('Y Coordinate (m):', value = Y_default)
  with col2:
    R = st.number_input('Search Radius (m):', value = R_default)
    calc_type = st.selectbox('Calculation Type', ('Only Active Wells', 'All Wells'))

  # Generamos el pozo tipo
  results = prob_type_well.type_well(PRD_data, locations, X, Y, R, calc_type)

  # Almacenamos los resultados
  type_well_data = results[0]
  sync_wells_data = results[1]
  selected_wells_list = results[2]

  # Obtenemos los parametros de declinacion
  qi_P10 = results[3]
  b_P10 = results[4]
  Di_P10 = results[5]
  qi_P50 = results[6]
  b_P50 = results[7]
  Di_P50 = results[8]
  qi_P90 = results[9]
  b_P90 = results[10]
  Di_P90 = results[11]

  # Subtitulo para la seccion de pozos escogidos
  st.subheader('Selected Wells') 

  # Asignamos el grafico con la locacion de los pozos a una variable
  plot1 = prob_type_well.plot_wells(locations, selected_wells_list, X, Y, R)

  # Mostramos el grafico con la locacion de los pozos
  st.pyplot(plot1)

  # Mostramos el dataframe los pozos sincronizados
  st.dataframe(sync_wells_data.style.format(precision = 2),
                 height = 500, use_container_width = True)

  # Convertimos el dataframe a Excel
  sync_wells_excel = to_excel(sync_wells_data)

  # Boton de descarga para el archivo con los pozos sincronizados
  st.download_button(label ='📥 Download Sync Wells', data = sync_wells_excel, file_name = 'Results.xlsx')
  
  # Subtitulo para la seccion de resultados
  st.subheader('Results') 

  st.caption('P10:')

  # Definimos las columnas para los parametros de declinacion
  col3, col4, col5 = st.columns(3)

  # Mostramos los parametros de declinacion
  col3.metric("Initial Rate (qi)", '{:.2f}'.format(round(qi_P10, 2)))
  col4.metric("Decline Constant (b)", '{:.4f}'.format(round(b_P10, 4)))
  col5.metric("Decline Rate (Di)", '{:.6f}'.format(round(Di_P10 * 30.48, 6)))
  
  st.caption('P50:')

  # Definimos las columnas para los parametros de declinacion
  col6, col7, col8 = st.columns(3)

  # Mostramos los parametros de declinacion
  col6.metric("Initial Rate (qi)", '{:.2f}'.format(round(qi_P50, 2)))
  col7.metric("Decline Constant (b)", '{:.4f}'.format(round(b_P50, 4)))
  col8.metric("Decline Rate (Di)", '{:.6f}'.format(round(Di_P50 * 30.48, 6)))

  st.caption('P90:')

  # Definimos las columnas para los parametros de declinacion
  col9, col10, col11 = st.columns(3)

  # Mostramos los parametros de declinacion
  col9.metric("Initial Rate (qi)", '{:.2f}'.format(round(qi_P90, 2)))
  col10.metric("Decline Constant (b)", '{:.4f}'.format(round(b_P90, 4)))
  col11.metric("Decline Rate (Di)", '{:.6f}'.format(round(Di_P90 * 30.48, 6)))
  
  # Definimos las columnas para las opciones del grafico del pozo tipo
  col12, col13 = st.columns(2)

  # Agregamos el selector de opciones para el grafico del pozo tipo
  with col12:
    plot_type = st.selectbox('Plot Options', ('Only Type Well', 'Show All Wells'))

  # Dependiendo de la opcion seleccionada, asignamos un grafico a una variable
  if plot_type == 'Only Type Well':
    plot2 = prob_type_well.plot_type_well(type_well_data)
  else:
    plot2 = prob_type_well.plot_all_wells(type_well_data, sync_wells_data) 

  # Mostramos el grafico del pozo tipo
  st.pyplot(plot2)

  # Mostramos el dataframe con los resultados
  st.dataframe(type_well_data.style.format({'Month': '{:.0f}', 'Average P10': '{:.2f}', 'Adjusted P10': '{:.2f}',
                                           'Average P50': '{:.2f}', 'Adjusted P50': '{:.2f}',
                                           'Average P90': '{:.2f}', 'Adjusted P90': '{:.2f}'}), 
                 height = 500, use_container_width = True)

  # Convertimos el dataframe a Excel
  type_well_excel = to_excel(type_well_data)

  # Boton de descarga para los resultados
  st.download_button(label ='📥 Download Type Well', data = type_well_excel, file_name = 'Type Well.xlsx')
