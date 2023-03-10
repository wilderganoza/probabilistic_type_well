# Importamos las librerias a utilizar
import streamlit as st # 
import pandas as pd # Manipulacion de dataframes
from PIL import Image # Manipulacion de imagenes
from io import BytesIO # 
from pyxlsb import open_workbook as open_xlsb # 

# Cargamos el favicon en una variable
favicon = Image.open("Favicon.png")

# Configuracion de la pagina
st.set_page_config(page_title = "Probabilistic Type Well Analysis", page_icon = favicon,)

# CSS para ocultar menu
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """

# Ocultamos el menu de Streamlit
st.markdown(hide_menu_style, unsafe_allow_html = True)

# Cargamos el logo de la empresa a una variable
image = Image.open('Logo.png')

# Mostramos el logo de la empresa
st.image(image)

# Mensaje de bienvenida
st.write("## Welcome to the Type Well Anaylisis App! 👋")

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

# Descripcion de la aplicacion
st.markdown(
  """
  With this app, you can create a traditional decline analysis for the average of a group of wells. 
  
  A type well analysis is performed on the group as a whole — not on individual wells — although the forecast 
  can be copied to the individual wells that make up the group. 
  
  In most cases, you are going to be using analog wells, and type well forecasts are commonly applied to wells 
  with limited or no historical production data.
  """
)

# Mensaje para descargar los formatos de ejemplo
st.write("You can download an example here ⬇️")

# Almacenamos los formatos de ejemplo en dataframes
file1 = pd.read_excel('Coordinates.xlsx')
file2 = pd.read_excel('Production.xlsx')

# Convertimos los dataframe a Excel
df1 = to_excel(file1)
df2 = to_excel(file2)

# Definimos las columnas para los botones de descarga
col1, col2, col3 = st.columns(3)

# Botones de descarga para ejemplos
with col1:
  st.download_button(label = '📥 Download Coordinates', data = df1, file_name = 'Coordinates.xlsx')

with col2:
  st.download_button(label = '📥 Download Production', data = df2, file_name = 'Production.xlsx')
