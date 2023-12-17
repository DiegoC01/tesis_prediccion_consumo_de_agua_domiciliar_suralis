import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import tz
from datetime import datetime, timedelta

def ordenarDatosMensuales(ruta, mes, anio):

    # Open data: MARZO 2023
    datos_crudos_consumo_de_agua_suralis = (ruta)
    df_datos_crudos_consumo_de_agua_suralis = pd.read_csv(datos_crudos_consumo_de_agua_suralis, sep=';', encoding='unicode_escape')


    # Qué columnas tiene?
    print(df_datos_crudos_consumo_de_agua_suralis.columns)


    # Borrar columnas innecesarias
    
    if(mes == 10 and anio == 2023):
        # NUM;NUM_CLIENTE;FECHA_FACTURACION;TARIFA;CATEGORIA;CONCESION;CONSUMO_BASE;CONSUMO_ADIC;CONSUMO_GEN;CONSUMO_TOT;
        # CONSUMO_AJUSTE;SECTOR;CASAS;COMUNA;CUARTEL;U_D;DIAMETRO;APNPP;_SUBSIDIO;IND_FISCAL;LOCALIDAD;COMUNA
        df_datos_crudos_consumo_de_agua_suralis = df_datos_crudos_consumo_de_agua_suralis.drop(columns=[
                    'NUM', 'CONCESION', 'CONSUMO_ADIC', 'CONSUMO_GEN', 'CONSUMO_TOT', 'CONSUMO_AJUSTE', 'CASAS',
                    'CUARTEL', 'U_D', 'DIAMETRO', 'APNPP', 'IND_FISCAL', 'LOCALIDAD', '_SUBSIDIO', 'COMUNA.1'
                    ])
        df_datos_crudos_consumo_de_agua_suralis = df_datos_crudos_consumo_de_agua_suralis.rename(columns={"NUM_CLIENTE": "NUMERO_CLIENTE",
                                                                                                        "FECHA_FACTURACION": "FECHA_FAC"})
    elif(anio == 2023 or (anio == 2022 and mes == 12)):
        # NUMERO_CLIENTE;FECHA_FAC;TARIFA;CATEGORIA;SISTEMA;CONSUMO_BASE;CONSUMO_ADICIONAL;CONSUMO_GENERAL;CONSUMO_TOTAL;
        # CONSUMO_AJUSTE;SECTOR;CANTIDAD_CASAS;COMUNA;CUARTEL;TIENE_UD;DIAMETRO;APNPP;PRESTACION;DESC_TARIFA;DESC_CATEGORIA;
        # DESC_LOCALIDAD;DESC_LOCALIDAD;COD_SISS_II_LOC;COD_SISS_II_COM;SUBSIDIO;;;;;;
        df_datos_crudos_consumo_de_agua_suralis = df_datos_crudos_consumo_de_agua_suralis.drop(columns=['Unnamed: 25',
            'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28', 'Unnamed: 29',
            'Unnamed: 30', 'SISTEMA', 'CONSUMO_ADICIONAL', 'CONSUMO_GENERAL', 'CONSUMO_TOTAL',
            'CONSUMO_AJUSTE', 'CANTIDAD_CASAS', 'CUARTEL',
            'TIENE_UD', 'DIAMETRO', 'APNPP', 'PRESTACION', 'DESC_TARIFA',
            'DESC_CATEGORIA', 'DESC_LOCALIDAD.1',
            'COD_SISS_II_LOC', 'COD_SISS_II_COM', 'SUBSIDIO', 'COMUNA', 'SECTOR'])
    elif(anio == 2022):
        df_datos_crudos_consumo_de_agua_suralis = df_datos_crudos_consumo_de_agua_suralis.drop(columns=[
            'SISTEMA', 'CONSUMO_ADICIONAL', 'CONSUMO_GENERAL', 'CONSUMO_TOTAL',
            'CONSUMO_AJUSTE', 'CANTIDAD_CASAS', 'CUARTEL',
            'TIENE_UD', 'DIAMETRO', 'APNPP', 'PRESTACION', 'DESC_TARIFA',
            'DESC_CATEGORIA', 'DESC_LOCALIDAD.1',
            'COD_SISS_II_LOC', 'COD_SISS_II_COM', 'SUBSIDIO', 'COMUNA', 'SECTOR'])
    elif((anio == 2021) and (mes == 11)):
        df_datos_crudos_consumo_de_agua_suralis = df_datos_crudos_consumo_de_agua_suralis.drop(columns=[
            'SISTEMA', 'CONSUMO_ADICIONAL', 'CONSUMO_GENERAL', 'CONSUMO_TOTAL',
            'CONSUMO_AJUSTE', 'CANTIDAD_CASAS', 'CUARTEL',
            'TIENE_UD', 'DIAMETRO', 'APNPP', 'APNPP.1', 'PRESTACION', 'DESC_TARIFA',
            'DESC_CATEGORIA', 'DESC_LOCALIDAD.1',
            'COD_SISS_II_LOC', 'COD_SISS_II_COM', 'SUBSIDIO', 'COMUNA', 'SECTOR'])
    else:           
        df_datos_crudos_consumo_de_agua_suralis = df_datos_crudos_consumo_de_agua_suralis.drop(columns=[
            'SISTEMA', 'CONSUMO_ADICIONAL', 'CONSUMO_GENERAL', 'CONSUMO_TOTAL',
            'CONSUMO_AJUSTE', 'CANTIDAD_CASAS', 'CUARTEL',
            'TIENE_UD', 'DIAMETRO', 'APNPP', 'APNPP.1', 'PRESTACION', 'DESC_TARIFA',
            'DESC_CATEGORIA', 'DESC_LOCALIDAD.1',
            'COD_SISS_II_LOC', 'COD_SISS_II_COM', 'COMUNA', 'SECTOR'])

    # Corroborar columnas nuevas
    #print(df_datos_crudos_consumo_de_agua_suralis.columns)

    # Corroborar que hayan fechas solamentes del mes de octubre del 2023
    df_dates = df_datos_crudos_consumo_de_agua_suralis['FECHA_FAC']
    df_dates = df_dates.drop_duplicates()
    #print("Fechas registradas")
    #print(df_dates)

    # Corroborar como lucen los datos
    #print(df_datos_crudos_consumo_de_agua_suralis)

    # De la corroboración de fechas y de los datos, se vió que hay muchas filas con datos nullos. Se eliminan
    df_datos_crudos_consumo_de_agua_suralis = df_datos_crudos_consumo_de_agua_suralis.dropna(axis=0)

    # De la fecha de facturación, solo interesa el mes y el año, y como se tienen solo datos de Enero del 2023, se agregan en una nueva columna
    df_datos_crudos_consumo_de_agua_suralis.insert(1, "MES", mes)
    df_datos_crudos_consumo_de_agua_suralis.insert(2, "AÑO", anio)

    # Solo interesan los datos residenciales (CATEOGORÍA) y los que sean de 1 sola casa (TARIFA)
    df_datos_crudos_consumo_de_agua_suralis = df_datos_crudos_consumo_de_agua_suralis[df_datos_crudos_consumo_de_agua_suralis['CATEGORIA'] == 1]
    df_datos_crudos_consumo_de_agua_suralis = df_datos_crudos_consumo_de_agua_suralis[df_datos_crudos_consumo_de_agua_suralis['TARIFA'] == 11]
    df_datos_crudos_consumo_de_agua_suralis = df_datos_crudos_consumo_de_agua_suralis[(df_datos_crudos_consumo_de_agua_suralis['DESC_LOCALIDAD'] == 'OSORNO') | (df_datos_crudos_consumo_de_agua_suralis['DESC_LOCALIDAD'] == 'PUERTO MONTT')]

    # Se eliminan FECHA_FAC, TARIFA y CATEGORIA ya que no son relevantes para las predicciones
    df_datos_crudos_consumo_de_agua_suralis = df_datos_crudos_consumo_de_agua_suralis.drop(columns=['FECHA_FAC'])
    df_datos_crudos_consumo_de_agua_suralis = df_datos_crudos_consumo_de_agua_suralis.drop(columns=['TARIFA'])
    df_datos_crudos_consumo_de_agua_suralis = df_datos_crudos_consumo_de_agua_suralis.drop(columns=['CATEGORIA'])
    df_datos_crudos_consumo_de_agua_suralis = df_datos_crudos_consumo_de_agua_suralis.drop(columns=['DESC_LOCALIDAD'])


    # Se guardan los datos de enero del 2023, ordenados por número de cliente
    df_dato_mes = df_datos_crudos_consumo_de_agua_suralis.sort_values(by=['NUMERO_CLIENTE'])


    # Plotting data
    #df_02_2023.plot(y=["CONSUMO_BASE"])
    #plt.title("Historical water consumption")
    #plt.show()

    # Se buscan valores vacíos
    print("Buscando NaN")
    print(df_dato_mes['NUMERO_CLIENTE'].isna()[df_dato_mes['NUMERO_CLIENTE'].isna() == True])
    print(df_dato_mes['CONSUMO_BASE'].isna()[df_dato_mes['CONSUMO_BASE'].isna() == True])
    #print(df_dato_mes['SECTOR'].isna()[df_dato_mes['SECTOR'].isna() == True])
    #print(df_dato_mes['COMUNA'].isna()[df_dato_mes['COMUNA'].isna() == True])

    # Se corrobora como lucen
    #print(df_dato_mes)
    #nueva_columna_consumo = "CONSUMO_BASE_"+str(mes)+"_"+str(anio)
    #df_dato_mes = df_dato_mes.rename(columns={"CONSUMO_BASE": nueva_columna_consumo})

    print("\n\n")
    print(df_dato_mes)
    print("\n\n")

    return df_dato_mes


def ordenarSecuenciaDeMeses(df, mesesFuturos=1):

  for i in range(1, mesesFuturos + 1):
      if(i <= mesesFuturos - 1):
        df[f'CONSUMO_BASE-{i}'] = df.groupby('NUMERO_CLIENTE')['CONSUMO_BASE'].shift(i)
        df[f'MES-{i}'] = df.groupby('NUMERO_CLIENTE')['MES'].shift(i)
        df[f'AÑO-{i}'] = df.groupby('NUMERO_CLIENTE')['AÑO'].shift(i)
        #df[f'SECTOR-{i}'] = df.groupby('NUMERO_CLIENTE')['SECTOR'].shift(i)
        #df[f'COMUNA-{i}'] = df.groupby('NUMERO_CLIENTE')['COMUNA'].shift(i)

      df[f'CONSUMO_BASE+{i}'] = df.groupby('NUMERO_CLIENTE')['CONSUMO_BASE'].shift(-i)

  column_order = ['NUMERO_CLIENTE']

  for i in range(1, mesesFuturos):
      #column_order.extend([f'CONSUMO_BASE-{mesesFuturos-i}', f'MES-{mesesFuturos-i}', f'AÑO-{mesesFuturos-i}',
                          #f'SECTOR-{mesesFuturos-i}', f'COMUNA-{mesesFuturos-i}'])
      column_order.extend([f'CONSUMO_BASE-{mesesFuturos-i}', f'MES-{mesesFuturos-i}', f'AÑO-{mesesFuturos-i}'])

  #column_order.extend(['CONSUMO_BASE', 'MES', 'AÑO', 'SECTOR', 'COMUNA'])
  column_order.extend(['CONSUMO_BASE', 'MES', 'AÑO'])

  for i in range(1, mesesFuturos + 1):
      column_order.extend([f'CONSUMO_BASE+{i}'])

  df = df[column_order]
  # Mostrar el resultado

  df = (df.dropna()).reset_index(drop=True)

  return df

def calculatingBoundWithIQR(df):
    # IQR
    Q1 = np.percentile(df['CONSUMO_BASE'], 25)
    Q3 = np.percentile(df['CONSUMO_BASE'], 75)
    IQR = Q3 - Q1

    # Above Upper bound
    upper=Q3+1.5*IQR
    upper_array=np.array(df['CONSUMO_BASE']>=upper)
    
    #Below Lower bound
    lower=Q1-1.5*IQR
    lower_array=np.array(df['CONSUMO_BASE']<=lower)

    return (upper, lower)

def deletingOutliers(df):
    
    # Identifying outliers with the iqr function
    (upper, lower) = calculatingBoundWithIQR(df)

    # Deleting outliers
    df['CONSUMO_BASE'] = np.where(df['CONSUMO_BASE'] > upper, None, df['CONSUMO_BASE'])
    df['CONSUMO_BASE'] = np.where(df['CONSUMO_BASE'] < lower, None, df['CONSUMO_BASE'])

    # applying the method
    count_nan = df['CONSUMO_BASE'].isnull().sum()
    
    # printing the number of values present
    # in the column
    print('Number of NaN values present: ' + str(count_nan))

    promedio_valores = df['CONSUMO_BASE'].mean()

    df['CONSUMO_BASE'].fillna(promedio_valores, inplace=True)
    print(df)
    # applying the method
    count_nan = df['CONSUMO_BASE'].isnull().sum()
    
    # printing the number of values present
    # in the column
    print('Number of NaN values present: ' + str(count_nan))
    return df

# Open data: ENERO 2023
datos_crudos_consumo_de_agua_suralis_2023_01 = ('src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-01_2023.csv')
df_datos_crudos_consumo_de_agua_suralis = pd.read_csv(datos_crudos_consumo_de_agua_suralis_2023_01, sep=';')

df_01_2023 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-01_2023.csv', mes=1, anio=2023)
df_02_2023 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-02_2023.csv', mes=2, anio=2023)
df_03_2023 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-03_2023.csv', mes=3, anio=2023)
df_04_2023 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-04_2023.csv', mes=4, anio=2023)
df_05_2023 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-05_2023.csv', mes=5, anio=2023)
df_06_2023 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-06_2023.csv', mes=6, anio=2023)
#df_07_2023 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-07_2023.csv', mes=7, anio=2023)
#df_08_2023 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-08_2023.csv', mes=8, anio=2023)
#df_09_2023 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-09_2023.csv', mes=9, anio=2023)
#df_10_2023 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-10_2023.csv', mes=10, anio=2023)


df_01_2022 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-01_2022.csv', mes=1, anio=2022)
df_02_2022 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-02_2022.csv', mes=2, anio=2022)
df_03_2022 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-03_2022.csv', mes=3, anio=2022)
df_04_2022 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-04_2022.csv', mes=4, anio=2022)
df_05_2022 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-05_2022.csv', mes=5, anio=2022)
df_06_2022 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-06_2022.csv', mes=6, anio=2022)
df_07_2022 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-07_2022.csv', mes=7, anio=2022)
df_08_2022 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-08_2022.csv', mes=8, anio=2022)
df_09_2022 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-09_2022.csv', mes=9, anio=2022)
df_10_2022 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-10_2022.csv', mes=10, anio=2022)
df_11_2022 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-11_2022.csv', mes=11, anio=2022)
df_12_2022 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-12_2022.csv', mes=12, anio=2022)


df_01_2021 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-01_2020.csv', mes=1, anio=2021)
df_02_2021 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-02_2020.csv', mes=2, anio=2021)
df_03_2021 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-03_2020.csv', mes=3, anio=2021)
df_04_2021 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-04_2020.csv', mes=4, anio=2021)
df_05_2021 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-05_2020.csv', mes=5, anio=2021)
df_06_2021 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-06_2020.csv', mes=6, anio=2021)
df_07_2021 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-07_2020.csv', mes=7, anio=2021)
df_08_2021 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-08_2020.csv', mes=8, anio=2021)
df_09_2021 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-09_2020.csv', mes=9, anio=2021)
df_10_2021 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-10_2020.csv', mes=10, anio=2021)
df_11_2021 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-11_2020.csv', mes=11, anio=2021)
df_12_2021 = ordenarDatosMensuales(ruta='src/datos/datos_crudos/csv/consumo_de_agua/archivos_separados/Suralis-consumo_de_agua-12_2020.csv', mes=12, anio=2021)

print(df_01_2023)
print(df_02_2023)
print(df_03_2023)
print(df_04_2023)
print(df_05_2023)
print(df_06_2023)
#print(df_07_2023)
#print(df_08_2023)
#print(df_09_2023)
#print(df_10_2023)

print(df_01_2022)
print(df_02_2022)
print(df_03_2022)
print(df_04_2022)
print(df_05_2022)
print(df_06_2022)
print(df_07_2022)
print(df_08_2022)
print(df_09_2022)
print(df_10_2022)
print(df_11_2022)
print(df_12_2022)

print(df_01_2021)
print(df_02_2021)
print(df_03_2021)
print(df_04_2021)
print(df_05_2021)
print(df_06_2021)
print(df_07_2021)
print(df_08_2021)
print(df_09_2021)
print(df_10_2021)
print(df_11_2021)
print(df_12_2021)

result = pd.concat([ 
                        #df_09_2023,
                        #df_08_2023,
                        #df_07_2023,
                        df_06_2023,
                        df_05_2023,
                        df_04_2023,
                        df_03_2023,
                        df_02_2023,
                        df_01_2023,
                        df_12_2022,
                        df_11_2022,
                        df_10_2022,
                        df_09_2022,
                        df_08_2022,
                        df_07_2022,
                        df_06_2022,
                        df_05_2022,
                        df_04_2022,
                        df_03_2022,
                        df_02_2022,
                        df_01_2022,
                        df_12_2021,
                        df_11_2021,
                        df_10_2021,
                        df_09_2021,
                        df_08_2021,
                        df_07_2021,
                        df_06_2021,
                        df_05_2021,
                        df_04_2021,
                        df_03_2021,
                        df_02_2021,
                        df_01_2021,
                    ]
                   , axis=0)
result = (result.sort_values(by=['NUMERO_CLIENTE', 'AÑO', 'MES']))
result = result.reset_index(drop=True)
print(result)

counts = result['NUMERO_CLIENTE'].value_counts()

# Imprime las cuentas de cada NUMERO_CLIENTE
print("\nCounts:")
print(counts)

# Filtra solo los NUMERO_CLIENTE que tienen exactamente 10 registros
result = result[result['NUMERO_CLIENTE'].isin(counts[counts == 30].index)]

result['NUMERO_CLIENTE'] = pd.to_numeric(result['NUMERO_CLIENTE'])
result['MES'] = pd.to_numeric(result['MES'])
result['AÑO'] = pd.to_numeric(result['AÑO'])
result['CONSUMO_BASE'] = pd.to_numeric(result['CONSUMO_BASE'])
#result['SECTOR'] = pd.to_numeric(result['SECTOR'])
#result['COMUNA'] = pd.to_numeric(result['COMUNA'])

result = deletingOutliers(result)

result.to_csv("src/datos/datos_preprocesados/datos_suralis-2021-2023.csv")

# HASTA ACÁ YA ESTÁ LIMPIO, AHORA SIGUE EL ORDEN

#result['CONSUMO_BASE_SIGUIENTE'] = (result.groupby('NUMERO_CLIENTE')['CONSUMO_BASE'].shift(-1))
#result['CONSUMO_BASE_SIGUIENTE'] = result['CONSUMO_BASE'].shift(-1)
#result['CONSUMO_BASE_SIGUIENTE'] = result.groupby('NUMERO_CLIENTE')['CONSUMO_BASE'].apply(lambda x: x.shift(-1))
"""
result['CONSUMO_BASE_SIGUIENTE'] = result.groupby('NUMERO_CLIENTE')['CONSUMO_BASE'].shift(-1)

print(result.sort_values(by=['NUMERO_CLIENTE', 'MES', 'AÑO']).iloc[:,:])

result = result.dropna()

print(result.sort_values(by=['NUMERO_CLIENTE', 'MES', 'AÑO']).iloc[:,:])

result['NUMERO_CLIENTE'] = pd.to_numeric(result['NUMERO_CLIENTE'])
result['MES'] = pd.to_numeric(result['MES'])
result['AÑO'] = pd.to_numeric(result['AÑO'])
result['CONSUMO_BASE'] = pd.to_numeric(result['CONSUMO_BASE'])
result['SECTOR'] = pd.to_numeric(result['SECTOR'])
result['COMUNA'] = pd.to_numeric(result['COMUNA'])
result['CONSUMO_BASE_SIGUIENTE'] = pd.to_numeric(result['CONSUMO_BASE_SIGUIENTE'])

X = result[['MES', 'AÑO', 'CONSUMO_BASE', 'SECTOR', 'COMUNA']].values
y = (result['CONSUMO_BASE_SIGUIENTE'].values).reshape(-1, 1)

print(X)
print(y)
"""

# CÓDIGO DE PRUEBA
"""
# Tu DataFrame original
data = {'NUMERO_CLIENTE': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        'MES': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'AÑO': [2023]*20,
        'MEDIDA': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        'TEMPERATURA': [3.0]*20,
        'HUMEDAD': [1.0]*20}

df = pd.DataFrame(data)


meses_futuros = 1
for i in range(1, meses_futuros + 1):
    if(i == meses_futuros - 1):
      df[f'MEDIDA-{i}'] = df.groupby('NUMERO_CLIENTE')['MEDIDA'].shift(i)
      df[f'MES-{i}'] = df.groupby('NUMERO_CLIENTE')['MES'].shift(i)
      df[f'AÑO-{i}'] = df.groupby('NUMERO_CLIENTE')['AÑO'].shift(i)
      df[f'TEMPERATURA-{i}'] = df.groupby('NUMERO_CLIENTE')['TEMPERATURA'].shift(i)
      df[f'HUMEDAD-{i}'] = df.groupby('NUMERO_CLIENTE')['HUMEDAD'].shift(i)

    df[f'MEDIDA+{i}'] = df.groupby('NUMERO_CLIENTE')['MEDIDA'].shift(-i)

column_order = ['NUMERO_CLIENTE']

for i in range(1, meses_futuros):
    column_order.extend([f'MEDIDA-{i}', f'MES-{i}', f'AÑO-{i}',
                         f'TEMPERATURA-{i}', f'HUMEDAD-{i}'])

column_order.extend(['MEDIDA', 'MES', 'AÑO', 'TEMPERATURA', 'HUMEDAD'])

for i in range(1, meses_futuros + 1):
    column_order.extend([f'MEDIDA+{i}'])

df = df[column_order]
# Mostrar el resultado
print(df)
"""
"""
meses_futuros = 2
for i in range(1, meses_futuros + 1):
    if(i == meses_futuros - 1):
      result[f'CONSUMO_BASE-{i}'] = result.groupby('NUMERO_CLIENTE')['CONSUMO_BASE'].shift(i)
      result[f'MES-{i}'] = result.groupby('NUMERO_CLIENTE')['MES'].shift(i)
      result[f'AÑO-{i}'] = result.groupby('NUMERO_CLIENTE')['AÑO'].shift(i)
      result[f'SECTOR-{i}'] = result.groupby('NUMERO_CLIENTE')['SECTOR'].shift(i)
      result[f'COMUNA-{i}'] = result.groupby('NUMERO_CLIENTE')['COMUNA'].shift(i)

    result[f'CONSUMO_BASE+{i}'] = result.groupby('NUMERO_CLIENTE')['CONSUMO_BASE'].shift(-i)

column_order = ['NUMERO_CLIENTE']

for i in range(1, meses_futuros):
    column_order.extend([f'CONSUMO_BASE-{i}', f'MES-{i}', f'AÑO-{i}',
                         f'SECTOR-{i}', f'COMUNA-{i}'])

column_order.extend(['CONSUMO_BASE', 'MES', 'AÑO', 'SECTOR', 'COMUNA'])

for i in range(1, meses_futuros + 1):
    column_order.extend([f'CONSUMO_BASE+{i}'])

result = result[column_order]
# Mostrar el resultado
print(result.iloc[:50, :])
"""

HORIZONTE_MESES = 12
FEATURES_CANTIDAD = 3
df_preprocesado = ordenarSecuenciaDeMeses(result, HORIZONTE_MESES)

print(df_preprocesado)



#X = (df_preprocesado[[
                      #'CONSUMO_BASE-3', 'MES-3', 'AÑO-3', 'SECTOR-3', 'COMUNA-3',
                      #'CONSUMO_BASE-2', 'MES-2', 'AÑO-2', 'SECTOR-2', 'COMUNA-2',
                      #'CONSUMO_BASE-1', 'MES-1', 'AÑO-1', 'SECTOR-1', 'COMUNA-1',
                      #'CONSUMO_BASE', 'MES', 'AÑO', 'SECTOR', 'COMUNA'
                      #]].values)
X = (df_preprocesado[[  
                      'CONSUMO_BASE-11', 'MES-11', 'AÑO-11',
                      'CONSUMO_BASE-10', 'MES-10', 'AÑO-10',
                      'CONSUMO_BASE-9', 'MES-9', 'AÑO-9',
                      'CONSUMO_BASE-8', 'MES-8', 'AÑO-8',
                      'CONSUMO_BASE-7', 'MES-7', 'AÑO-7',
                      'CONSUMO_BASE-6', 'MES-6', 'AÑO-6',
                      'CONSUMO_BASE-5', 'MES-5', 'AÑO-5',
                      'CONSUMO_BASE-4', 'MES-4', 'AÑO-4',
                      'CONSUMO_BASE-3', 'MES-3', 'AÑO-3',
                      'CONSUMO_BASE-2', 'MES-2', 'AÑO-2',
                      'CONSUMO_BASE-1', 'MES-1', 'AÑO-1',
                      'CONSUMO_BASE', 'MES', 'AÑO'
                      ]].values)

#inner_shape = (1, cantidadFeatures)

# Reshape
X = X.reshape(X.shape[0], -1, FEATURES_CANTIDAD)

y = (df_preprocesado[[
                      'CONSUMO_BASE+1',
                      'CONSUMO_BASE+2', 
                      'CONSUMO_BASE+3', 
                      'CONSUMO_BASE+4', 
                      'CONSUMO_BASE+5',
                      'CONSUMO_BASE+6',
                      'CONSUMO_BASE+7',
                      'CONSUMO_BASE+8', 
                      'CONSUMO_BASE+9', 
                      'CONSUMO_BASE+10', 
                      'CONSUMO_BASE+11',
                      'CONSUMO_BASE+12',
                    ]].values).reshape(-1, HORIZONTE_MESES)


print("X")
print(X)
print("y")
print(y)
"""
Quiero, usando pandas y dataframes, pasar de esto:
       NUMERO_CLIENTE  MES   AÑO MEDIDA  TEMPERATURA  HUMEDAD   
            1           1  2023          0.0     3.0     1.0    
            1           2  2023          1.0     3.0     1.0    
            1           3  2023          2.0     3.0     1.0    
            2           1  2023          3.0     3.0     1.0
            2           2  2023          4.0     3.0     1.0
            2           3  2023          5.0     3.0     1.0
            3           1  2023          6.0     3.0     1.0    
            3           2  2023          7.0     3.0     1.0    
            3           3  2023          8.0     3.0     1.0    
            4           1  2023          9.0     3.0     1.0     
            4           2  2023          10.0     3.0     1.0   
            4           3  2023          11.0     3.0     1.0    


a esto:
    NUMERO_CLIENTE  MES   AÑO  MEDIDA  TEMPERATURA  HUMEDAD  MEDIDA_siguiente_1_mes  MEDIDA_siguiente_2_meses  MEDIDA_siguiente_3_meses
                1    1  2023     0.0          3.0      1.0                     1.0                       2.0                       3.0
                1    2  2023     1.0          3.0      1.0                     2.0                       3.0                       4.0
                1    3  2023     2.0          3.0      1.0                     3.0                       4.0                       5.0
                2    1  2023     3.0          3.0      1.0                     4.0                       5.0                       6.0
                2    2  2023     4.0          3.0      1.0                     5.0                       6.0                       7.0
                2    3  2023     5.0          3.0      1.0                     6.0                       7.0                       8.0
                3    1  2023     6.0          3.0      1.0                     7.0                       8.0                       9.0
                3    2  2023     7.0          3.0      1.0                     8.0                       9.0                       10.0
                3    3  2023     8.0          3.0      1.0                     9.0                      10.0                      11.0
                4    1  2023     9.0          3.0      1.0                    10.0                      11.0                      NaN
                4    2  2023    10.0          3.0      1.0                    11.0                       NaN                     NaN
                4    3  2023    11.0          3.0      1.0                     NaN                      NaN                      NaN

"""





























#from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta

import sklearn
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet


import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.losses import *
from keras.metrics import RootMeanSquaredError, F1Score, R2Score
from keras.optimizers import *



def predict_with_CNN_GRU(X_train, y_train, X_val, y_val, epochs=100):
    

  print("Prediction using CNN-GRU is being made...")
  print(X_train.shape)
  print(y_train.shape)
  number_of_inputs = int(X_train.shape[1])
  number_of_outputs = int(y_train.shape[1])

  # Model creation
  lstm_model = Sequential([
    Conv1D(32, (number_of_inputs), activation='relu', input_shape=(number_of_inputs, 1)),
    #Flatten(),
    GRU(32),
    Dense(number_of_outputs)]
  )

  # Model summary
  lstm_model.summary()

   # Early Stopping Callback
  early_stopping_monitor = EarlyStopping(
    monitor='val_root_mean_squared_error',
    patience=25,         
    verbose=1,           
    restore_best_weights=True 
  )

  # Training model
  lstm_model.compile(loss=MeanAbsoluteError(), optimizer=RMSprop(), metrics=[RootMeanSquaredError()])
  #lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
  lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[early_stopping_monitor])
  loss, accuracy = lstm_model.evaluate(X_test, y_test)
  # Imprimir las métricas
  print("Loss:", loss)
  print("Accuracy:", accuracy)



  # Predicting and saving results

  lstm_model_results = lstm_model.predict(X_val)

  print(lstm_model_results.shape)
  print(y_val.shape)
  print("Predicho: "+str(lstm_model_results[0]))
  print("Real: "+str(y_val[0]))
  lstm_train_results = pd.DataFrame(data={'Predicted Values':lstm_model_results[0], 'Real Values':y_val[0]})

  return (lstm_train_results, lstm_model_results, y_val)

def predict_with_GRU(X_train, y_train, X_val, y_val, epochs=100):
  # Message of indentification
  print("Prediction using GRU is being made...")

  number_of_inputs = int(X_train.shape[1])
  number_of_outputs = int(y_train.shape[1])
  number_of_features = int(X_train[0].shape[1])
  print(number_of_inputs, X_train[0].shape[1], X_train[0].shape[0])

  # Model creation
  lstm_model = Sequential([
    LSTM(32, input_shape=(number_of_inputs, number_of_features), dropout=0.1, recurrent_dropout=0.5, return_sequences=True),
    LSTM(64, activation='relu', dropout=0.1, recurrent_dropout=0.5),
    Dense(number_of_outputs)
    ]
  )

  # Model summary
  lstm_model.summary()

 # Early Stopping Callback
  early_stopping_monitor = EarlyStopping(
    monitor='val_root_mean_squared_error',
    patience=25,         
    verbose=1,           
    restore_best_weights=True 
  )

  # Training model
  lstm_model.compile(loss=MeanAbsoluteError(), optimizer=RMSprop(), metrics=[RootMeanSquaredError()])
  #lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
  fit_model = lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[early_stopping_monitor])

  # Graficar la pérdida (loss) durante el entrenamiento
  #plt.figure(figsize=(12, 6))
  #plt.plot(fit_model.history['root_mean_squared_error'], label='Validation Loss')
  #plt.xlabel('Epochs')
  #plt.ylabel('Loss')
  #plt.legend()
  #plt.show()

  # Predicting and saving results

  lstm_model_results = lstm_model.predict(X_val)

  print(lstm_model_results.shape)
  print(y_val.shape)
  print("Predicho: "+str(np.sum((lstm_model_results[0]))))
  print("Real: "+str(np.sum((y_val[0]))))
  lstm_train_results = pd.DataFrame(data={'Predicted Values':lstm_model_results[0], 'Real Values':y_val[0]})

  return (lstm_train_results, lstm_model_results, y_val)

# Dividing data
q_64 = int(len(y) * .64)
q_80 = int(len(y) * .8)

X_train, y_train = X[:q_64], y[:q_64]
X_val, y_val = X[q_64:q_80], y[q_64:q_80]
X_test, y_test = X[q_80:], y[q_80:]

epochs = 1

#GRU
(cnn_gru_results, pred_val_cnn_gru, real_val_cnn_gru) = predict_with_GRU(X_train, y_train, X_val, y_val, epochs=epochs)


def get_performance(predicted_values, real_values, model_name):
  rmse = sklearn.metrics.mean_squared_error(real_values, predicted_values,squared=False)
  mae = sklearn.metrics.mean_absolute_error(real_values, predicted_values)
  mse = sklearn.metrics.mean_squared_error(real_values, predicted_values)
  r2 = sklearn.metrics.r2_score(real_values, predicted_values, multioutput='variance_weighted')

  print("\tMetrics of "+str(model_name))
  print("RMSE: "+str(rmse))
  print("MAE: "+str(mae))
  print("MSE: "+str(mse))
  print("R^2: "+str(r2))
  print()

get_performance(pred_val_cnn_gru, real_val_cnn_gru, "GRU")

cnn_gru_results.plot(y=['Real Values', 'Predicted Values'])
plt.title('GRU results')
plt.show()
