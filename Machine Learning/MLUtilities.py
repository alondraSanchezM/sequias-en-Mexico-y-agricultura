from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

def particionar(entradas, salidas, porcentaje_entrenamiento, porcentaje_validacion, porcentaje_prueba):
    temp_size = porcentaje_validacion + porcentaje_prueba
    x_train, x_temp, y_train, y_temp = train_test_split(entradas, salidas, test_size =temp_size)
    if(porcentaje_validacion > 0):
        test_size = porcentaje_prueba/temp_size
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size = test_size)
    else:
        return [x_train, None, x_temp, y_train, None, y_temp]
    return [x_train, x_val, x_test, y_train, y_val, y_test]


def K_fold (datos,k, semilla, aleatorio ):
  if (k==1):
      kfold = KFold(len(datos), shuffle = aleatorio, random_state = semilla)
  else:
      kfold = KFold(k, shuffle = aleatorio, random_state = semilla)
  
  ciclo = 1
  for indices_train, indices_test in kfold.split(datos):
      print("Ciclo: "+str(ciclo))
      print("\t Datos para Entrenamiento:"+str(datos[indices_train]))
      print("\t Datos para prueba:"+str(datos[indices_test]))
      ciclo+=1


def matriz_confusion( datos_esperados, datos_predichos) :
  resultado = confusion_matrix(datos_esperados, datos_predichos)
  (TN, FP, FN, TP) = resultado.ravel()
  print("True positives: "+str(TP))
  print("True negatives: "+str(TN))
  print("False positives: "+str(FP))
  print("False negative: "+str(FN))
  resultados = {'TN':TN, 'FP':FP, 'FN':FN, 'TP':TP}
  return resultados


def metricas ( datos_esperados, datos_predichos) :
  resultado = confusion_matrix(datos_esperados, datos_predichos)
  (TN, FP, FN, TP) = resultado.ravel()
  exactitud = ((TP + TN) / (TP + TN + FP + FN)) * 100
  sensibilidad = (TP / (TP + FN) )   * 100
  especificidad = ( TN / (TN + FP))  * 100
  precision=(TP / (TP + FP) )   * 100
  print("Exactitud: "+str(exactitud) + '%')
  print("Sensibilidad: "+str(sensibilidad) +'%')
  print("Especificidad: "+str(especificidad)+'%')
  print("Precision: "+str(precision)+'%')
  resultados= {'exactitud':exactitud, 'sensibilidad':sensibilidad, 'especificidad':especificidad, 'precision':precision}
  return resultados

def comparador(esperados, clasificador1, clasificador2):
    resultado_1 = confusion_matrix(esperados, clasificador1)
    (TN_1, FP_1, FN_1, TP_1) = resultado_1.ravel()
    sensibilidad_1 = (TP_1 / (TP_1 + FN_1) )   * 100
    especificidad_1 = ( TN_1 / (TN_1 + FP_1))  * 100
    precision_1=(TP_1 + TN_1) / (TP_1 + TN_1 + FP_1 + FN_1)  * 100

    resultado_2 = confusion_matrix(esperados, clasificador2)
    (TN_2, FP_2, FN_2, TP_2) = resultado_2.ravel()
    sensibilidad_2 = (TP_2 / (TP_2 + FN_2) )   * 100
    especificidad_2 = ( TN_2 / (TN_2 + FP_2))  * 100
    precision_2=(TP_2 + TN_2) / (TP_2 + TN_2 + FP_2 + FN_2)  * 100

    if precision_1 > precision_2:
        print('El clasificador 1 tiene mejor precisión con: ' +str(precision_1) + '%')
    elif precision_1 == precision_2:
        print('Los clasificadores tienen la misma precisión ' + str(precision_1) + '%')
    else:
        print('El clasificador 2 tiene mejor precisión con: ' +str(precision_2) + '%')
        
    if sensibilidad_1 > sensibilidad_2:
        print('El clasificador 1 tiene mejor sensibilidad con: ' +str(sensibilidad_1) + '%')
    elif sensibilidad_1 == sensibilidad_2:
        print('Los clasificadores tienen la misma sensibilidad '+ str(sensibilidad_1) + '%')
    else:
        print('El clasificador 2 tiene mejor sensibilidad con: ' +str(sensibilidad_2) + '%')
        
    if especificidad_1 > especificidad_2:
        print('El clasificador 1 tiene mejor especificidad con: ' +str(especificidad_1) + '%')
    elif especificidad_1 == especificidad_2:
        print('Los clasificadores tienen la misma especificidad con: ' + str(especificidad_1) + '%')
    else:
        print('El clasificador 2 tiene mejor especificidad con: ' +str(especificidad_2) + '%')
    
    
    resultados_1= {'sensibilidad':sensibilidad_1, 'especificidad':especificidad_1, 'precision':precision_1}
    resultados_2= {'sensibilidad':sensibilidad_2, 'especificidad':especificidad_2, 'precision':precision_2}
    
    return resultados_1, resultados_2
        
def metricas_multiclase( datos_esperados, datos_predichos):
  resultado = confusion_matrix(datos_esperados, datos_predichos)
  FP = resultado.sum(axis=0) - np.diag(resultado) 
  FN = resultado.sum(axis=1) - np.diag(resultado)
  TP = np.diag(resultado)
  TN = resultado.sum() - (FP + FN + TP)

  exactitud = ((TP + TN) / (TP + TN + FP + FN)) * 100
  sensibilidad = (TP / (TP + FN) )   * 100
  especificidad = ( TN / (TN + FP))  * 100
  precision=(TP / (TP + FP) )   * 100
  resultados= {'exactitud':exactitud, 'sensibilidad':sensibilidad, 'especificidad':especificidad, 'precision':precision}
  return resultados
