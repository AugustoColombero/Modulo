"""
Módulo de Python con clases para análisis de datos, generación de datos
y resolución de problemas de regresión lineal y logística.

Clases incluidas:
- AnalisisDescriptivo: Análisis descriptivo y estimación de densidad.
- GeneradoraDeDatos: Generación de muestras aleatorias de distintas distribuciones.
- Regresion: Clase base para modelos de regresión.
- RegresionLineal: Implementa regresión lineal con statsmodels.
- RegresionLogistica: Implementa regresión logística con statsmodels.
"""
import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
class AnalisisDescriptivo:
    """
    Clase para realizar análisis descriptivo de datos, incluyendo histogramas
    y estimaciones de densidad con distintos núcleos.
    """
    def __init__(self,datos):
      """
      Inicializa el objeto con un conjunto de datos.

      Parámetros:
      datos: Lista o arreglo de datos numéricos.
      """
      self.datos = np.array(datos)

    def genera_histograma(self, h):
          """
          Genera el histograma de frecuencias absolutas para el h dado.

          Parámetros:
          h: Ancho de clase del histograma.

          Retorna:
          frec: Frecuencias absolutas por clase.
          bins: Bordes de los intervalos del histograma.
          """
          frec, bins = np.histogram(self.datos, bins=np.arange(min(self.datos), max(self.datos) + h, h))
          return frec, bins

    def evalua_histograma(self, h, x):
          """
          Evalúa la estimación de densidad por histograma en los puntos dados.

          Parámetros:
          h: Ancho de clase del histograma.
          x: Puntos en los que se desea estimar la densidad.

          Retorna:
          estim_hist: Valores estimados de densidad para cada punto de x.
          """
          frec, bins = self.genera_histograma(h)
          total_datos = len(self.datos)
          frec_relativas = frec / total_datos
          frec_densidad = frec_relativas / h
          estim_hist = np.zeros(len(x))
          for i, xi in enumerate(x):
              idx = np.digitize(xi, bins) - 1
              if 0 <= idx < len(frec_densidad):
                  estim_hist[i] = frec_densidad[idx]
          return estim_hist

    def kernel_gaussiano(self, x):
          """
          Aplica el kernel gaussiano a los valores dados.

          Parámetros:
          x: Valores sobre los cuales aplicar el kernel.

          Retorna:
              Resultado del kernel gaussiano.
          """
          return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

    def kernel_uniforme(self, x):
          """
          Aplica el kernel uniforme a los valores dados.

          Parámetros:
          x: Valores sobre los cuales aplicar el kernel.

          Retorna:
              Resultado del kernel uniforme.
          """
          return np.where((-1/2 < x) & (x <= 1/2), 1, 0)

    def kernel_cuadratico(self, x):
        """
        Aplica el kernel cuadratico a los valores dados.

        Parámetros:
        x: Valores sobre los cuales aplicar el kernel.

        Retorna:
            Resultado del kernel cuadratico.
        """
        valor_kernel_cuadratico = np.where(abs(x) <= 1, 3/4 * (1 - x**2), 0)
        return valor_kernel_cuadratico

    def kernel_triangular(self, x):
        """
        Aplica el kernel triangular a los valores dados.

        Parámetros:
        x: Valores sobre los cuales aplicar el kernel.

        Retorna:
            Resultado del kernel triangular.
        """
        valor_kernel_triangular = np.where(abs(x) <= 1, 1 - abs(x), 0)
        return valor_kernel_triangular

    def mi_densidad(self, x, h, kernel):
          """
          Estima la densidad de probabilidad en los puntos dados utilizando un método de núcleos.

          Parámetros:
          x: Puntos en los que se desea estimar la densidad.
          h: Ancho de clase del kernel.
          kernel: Tipo de kernel a utilizar ('uniforme', 'gaussiano', 'cuadratico' o 'triangular').

          Retorna:
          density: Estimación de la densidad para cada punto de x.
          """
          n = len(self.datos)
          density = np.zeros_like(x, dtype=float)

          if kernel == 'uniforme':
              for i in range(n):
                  density += self.kernel_uniforme((x - self.datos[i]) / h)
          elif kernel == 'gaussiano':
              for i in range(n):
                  density += self.kernel_gaussiano((x - self.datos[i]) / h)
          elif kernel == 'cuadratico':
              for i in range(n):
                  density += self.kernel_cuadratico((x - self.datos[i]) / h)
          elif kernel == 'triangular':
              for i in range(n):
                  density += self.kernel_triangular((x - self.datos[i]) / h)
          else:
              raise ValueError("El kernel debe ser 'uniforme', 'gaussiano', 'cuadratico' o 'triangular'")

          density /= (n * h)
          return density

class GeneradoraDeDatos:
    """
    Clase para generar muestras aleatorias a partir de diferentes distribuciones
    y obtener la densidad teórica correspondiente.
    """
    def __init__(self,n_datos):
      """
      Inicializa la clase con la cantidad de datos a generar.

      Parámetros:
      n_datos: Cantidad de datos a generar.
      """
      self.n_datos = n_datos

    def r_BS(self):
        """
        Genera una muestra de una distribución BS

        Retorna:
        y: Muestra generada.
        f_bs: Densidad teórica evaluada en 1000 puntos equiespaciados.
        """
        u = np.random.uniform(size=(self.n_datos,))
        y = u.copy()
        ind = np.where(u > 0.5)[0]
        y[ind] = np.random.normal(0, 1, size=len(ind))
        for j in range(5):
            ind = np.where((u > j * 0.1) & (u <= (j+1) * 0.1))[0]
            y[ind] = np.random.normal(j/2 - 1, 1/10, size=len(ind))
        self.datos = y
        x = np.linspace(min(y), max(y), 1000)
        f_bs = 1/2*norm.pdf(x,0,1)+1/10*sum([norm.pdf(x,j/2-1,1/10) for j in range(5)])
        return y, f_bs

    def norm(self,mu,sigma):
        """
        Genera una muestra de una distribución normal N(mu, sigma^2).

        Parámetros:
        mu: Media de la distribución.
        sigma: Desvío estándar de la distribución.

        Retorna:
        y: Muestra generada.
        f_x: Densidad teórica evaluada en 1000 puntos equiespaciados.
        """
        y = np.random.normal(mu, sigma, self.n_datos)
        self.datos = y
        x = np.linspace(min(y), max(y), 1000)
        f_x = norm.pdf(x, mu, sigma)
        return y, f_x

    def uniform(self,a,b):
        """
        Genera una muestra de una distribución uniforme U(a, b).

        Parámetros:
        a: Límite inferior.
        b: Límite superior.

        Retorna:
        y: Muestra generada.
        f_u: Densidad teórica evaluada en 1000 puntos equiespaciados.
        """
        y = np.random.uniform(a, b, self.n_datos)
        self.datos = y
        x = np.linspace(min(y), max(y), 1000)
        f_u = uniform.pdf(x, loc=a, scale=b-a)
        return y, f_u


class Regresion:
    """
    Clase base para modelos de regresión.
    Guarda las variables predictoras y respuesta,
    y permite mostrar el resumen del modelo ajustado.
    """
    def __init__(self, x, y):
        """
        Inicializa el objeto con las variables x e y.

        Parámetros:
        - x: de tipo array, variables predictoras.
        - y: de tipo array, variable respuesta.
        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.result = None

    def modelo_ajustado(self):
        """
        Imprime el resumen del modelo ajustado.
        Si el modelo aún no fue ajustado, lo ajusta automáticamente.
        """
        if self.result is None:
            self.ajustar_modelo()
        print(self.result.summary())

class RegresionLineal(Regresion):
    """
    Clase para realizar regresión lineal utilizando statsmodels.
    """
    def __init__(self, x, y):
        """
        Inicializa la instancia de regresión lineal con los datos x e y.

        Parámetros:
        x : de tipo array
            Datos de las variables independientes.
        y : de tipo array
            Datos de la variable dependiente.
        """
        super().__init__(x, y)

    def ajustar_modelo(self):
        """
        Ajusta el modelo usando mínimos cuadrados
        y guarda el resultado en el atributo 'result'.
        """
        X = sm.add_constant(self.x)
        modelo = sm.OLS(self.y, X)
        resultado = modelo.fit()
        self.result = resultado

    def graficar_recta_ajustada(self):
        """
        Grafica la recta ajustada.

        Para regresión simple (una variable predictora) muestra un solo gráfico.
        Para regresión múltiple, genera un gráfico por cada variable predictora.
        """
        if self.result is None:
            self.ajustar_modelo()

        betas = self.betas()
        b0 = betas[0]
        coefs = betas[1:]
        if self.x.ndim == 1 or (self.x.ndim == 2 and self.x.shape[1] == 1):
            x_vals = self.x if self.x.ndim == 1 else self.x[:, 0]
            y_optima = b0 + coefs[0] * x_vals
            plt.figure()
            plt.scatter(x_vals, self.y, marker="o", facecolors="none", edgecolors="blue", label="Datos")
            plt.plot(x_vals, y_optima, color="red", label="Recta ajustada")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("Regresión Lineal Simple")
            plt.grid(True)
            plt.legend()
            plt.show()
        else:
            n_variables = self.x.shape[1]
            for i in range(n_variables):
                x_vals = self.x[:, i]
                y_optima = b0 + coefs[i] * x_vals
                plt.figure()
                plt.scatter(x_vals, self.y, marker="o", edgecolors="blue", label="Datos")
                plt.plot(x_vals, y_optima, color="red", label=f"Recta ajustada X{i+1}")
                plt.xlabel(f"X{i+1}")
                plt.ylabel("Y")
                plt.title(f"Regresión Lineal - Variable Predictora {i+1}")
                plt.grid(True)
                plt.legend()
                plt.show()

    def betas(self):
        """
        Obtiene los coeficientes estimados del modelo.

        Retorna:
            Coeficientes del modelo.
        """
        if self.result is None:
            self.ajustar_modelo()
        betas = self.result.params
        return betas

    def errores_estandar(self):
        """
        Obtiene los errores estándar de los coeficientes estimados.

        Retorna:
            Errores estándar de los coeficientes.
        """
        if self.result is None:
            self.ajustar_modelo()
        return self.result.bse

    def t_observados(self):
        """
        Obtiene los valores t observados de cada coeficiente.

        Retorna:
            Valores t de cada coeficiente.
        """
        if self.result is None:
            self.ajustar_modelo()
        return self.result.tvalues

    def p_valores(self):
        """
        Obtiene los P-valores asociados a los coeficientes.
        Retorna:
            Valores P-valores de cada coeficiente.
        """
        if self.result is None:
            self.ajustar_modelo()
        return self.result.pvalues

    def predecir(self, x_nuevo):
        """
        Realiza predicciones para nuevos datos de las variables independientes.

        Parámetros:
        x_nuevo : de tipo array
            Nuevos datos de variables independientes para predecir.

        Retorna:
            Valores predichos para la variable dependiente.
        """
        if self.result is None:
            self.ajustar_modelo()
        X_nuevo = sm.add_constant(x_nuevo)
        y_pred = self.result.predict(X_nuevo)
        return y_pred

    def calcular_correlacion(self):
        """
        Calcula el coeficiente de correlación entre cada variable independiente y la variable dependiente.

        Retorna:
            Arreglo con los coeficientes de correlación para cada variable predictora.
        """
        correlaciones = []
        for i in range(self.x.shape[1]):
            r = np.corrcoef(self.x[:, i], self.y)[0,1]
            correlaciones.append(r)
        coef_correlacion = np.array(correlaciones)
        return coef_correlacion

    def analisis_residuos(self):
        """
        Realiza un análisis gráfico de los residuos ajustados del modelo.

        Genera un QQ-plot para verificar normalidad de residuos y un gráfico de residuos vs valores predichos.

        Retorna:
            Valores de los residuos y valores predichos.
        """
        if self.result is None:
            self.ajustar_modelo()
        residuos = self.result.resid
        predichos = self.result.fittedvalues
        plt.figure(figsize = (15,8))
        sm.qqplot(residuos, line='s')
        plt.title('QQ plot residuos')
        plt.grid(True)
        plt.figure()
        plt.scatter(predichos, residuos)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Valores predichos')
        plt.ylabel('Residuos')
        plt.title('Gráfico de dispersión de residuos vs. valores predichos')
        plt.grid(True)
        plt.show()
        return residuos, predichos

    def intervalo_confianza(self, x_nuevo, alfa=0.05):
        """
        Calcula el intervalo de confianza para la media de la variable respuesta en nuevas observaciones.

        Parámetros:
        x_nuevo : de tipo array
            Nuevas observaciones de variables independientes.
        alfa : de tipo float, opcional
            Nivel de significancia (por defecto 0.05 para un 95% de confianza).

        Retorna:
            Intervalos de confianza para las predicciones de la media.
        """
        if self.result is None:
            self.ajustar_modelo()
        X_nuevo = sm.add_constant(x_nuevo)
        pred = self.result.get_prediction(X_nuevo)
        intervalo = pred.conf_int(alpha=alfa)
        return intervalo

    def intervalo_prediccion(self, x_nuevo, alfa=0.05):
        """
        Calcula el intervalo de predicción para nuevos valores de las variable dependientes.

        Parámetros:
            Nuevas observaciones de variables independientes.
        alfa : de tipo float, opcional
            Nivel de significancia (por defecto 0.05 para un 95% de confianza).

        Retorna:
            Intervalos de predicción para nuevos valores observados.
        """
        if self.result is None:
            self.ajustar_modelo()
        X_nuevo = sm.add_constant(x_nuevo)
        pred = self.result.get_prediction(X_nuevo)
        intervalo = pred.conf_int(obs=True, alpha=alfa)
        return intervalo


    def coeficiente_determinacion(self):
        """
        Calcula el coeficiente de determinación (R^2) del modelo ajustado.

        Retorna:
            Coeficiente de determinación (R^2).
        """
        if self.result is None:
            self.ajustar_modelo()
        coeficiente_determinacion = self.result.rsquared
        return coeficiente_determinacion

    def r_cuadrado_ajustado(self):
        """
        Calcula el coeficiente de determinación ajustado (R^2 ajustado) del modelo ajustado.

        Retorna:
            Coeficiente de determinación ajustado (R^2 ajustado).
        """
        if self.result is None:
            self.ajustar_modelo()
        r_cuadrado_ajustado = self.result.rsquared_adj
        return r_cuadrado_ajustado

class RegresionLogistica(Regresion):
    """
    Clase que implementa regresión logística utilizando statsmodels.
    """
    def __init__(self, x, y):
        """
        Inicializa la clase con las variables predictoras y respuesta.

        Parámetros:
        - x: de tipo array, variables predictoras.
        - y: de tipo array, variable respuesta.
        """
        super().__init__(x, y)
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def ajustar_modelo(self, train_size=0.8, semilla=None):
        """
        Ajusta el modelo de regresión logística.

        Parámetros:
        - train_size: proporción del conjunto de entrenamiento (entre 0 y 1).
        - semilla: valor opcional para fijar la semilla aleatoria.

        Si train_size es None o 1, se usa todo el conjunto como entrenamiento.
        """
        if train_size is None:
            self.x_train = self.x
            self.y_train = self.y
            self.x_test = None
            self.y_test = None

        else:
            if semilla is not None:
                np.random.seed(semilla)

            indices = np.arange(len(self.x))
            np.random.shuffle(indices)

            n_train = int(len(indices) * train_size)
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]

            self.x_train = self.x[train_indices]
            self.y_train = self.y[train_indices]
            self.x_test = self.x[test_indices]
            self.y_test = self.y[test_indices]

        X_train = sm.add_constant(self.x_train)
        model = sm.Logit(self.y_train, X_train)
        self.result = model.fit()

    def betas(self):
        """
        Devuelve los coeficientes estimados del modelo.
        """
        return self.result.params

    def errores_estandar(self):
        """
        Devuelve los errores estándar de los coeficientes.
        """
        return self.result.bse

    def t_observados(self):
        """
        Devuelve los estadísticos t (z en modelos logísticos) observados.
        """
        return self.result.tvalues

    def p_valores(self):
        """
        Devuelve los valores p de cada coeficiente estimado.
        """
        return self.result.pvalues

    def predecir(self, x_nuevo, umbral=0.5):
        """
        Predice para nuevos datos a partir del modelo ajustado.

        Parámetros:
        - x_nuevo: de tipo array, nuevos datos para predecir.
        - umbral: valor de corte para convertir probabilidades.

        Retorna:
        - array de 0 y 1 según el umbral.
        """
        if self.result is None:
            raise ValueError("El modelo no fue ajustado")

        X_nuevo = sm.add_constant(np.array(x_nuevo))
        probs = self.result.predict(X_nuevo)
        return (probs >= umbral).astype(int)

    def evaluar_test(self, umbral=0.5):
        """
        Evalúa el modelo en el conjunto de test usando un umbral dado.

        Parámetros:
        - umbral: valor de corte para clasificar.

        Retorna:
        - Diccionario con matriz de confusión, error total, sensibilidad y especificidad.
        """
        if self.x_test is None or self.y_test is None:
            raise ValueError("No hay datos de test. Ajustá con train_size < 1")

        y_pred = self.predecir(self.x_test, umbral=umbral)
        y_test = self.y_test

        tn = np.sum((y_test == 0) & (y_pred == 0))
        tp = np.sum((y_test == 1) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        fp = np.sum((y_test == 0) & (y_pred == 1))

        error_total = (fp + fn) / len(y_test)
        sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
        especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0

        matriz_confusion = np.array([[tn, fp],
                                    [fn, tp]])

        return {
            "matriz_confusion": matriz_confusion,
            "error_total": error_total,
            "sensibilidad": sensibilidad,
            "especificidad": especificidad
        }

    def evaluar_roc(self):
        """
        Calcula y grafica la curva ROC con los datos de test.

        Marca el punto óptimo según el índice de Youden.
        Retorna:
        - AUC (área bajo la curva ROC),
        - Umbral óptimo,
        - Clasificación cualitativa del AUC.
        """
        if self.x_test is None or self.y_test is None:
            raise ValueError("No hay datos de test. Ajustá con train_size < 1 o setéalos manualmente.")

        X_test_const = sm.add_constant(self.x_test)
        probs = self.result.predict(X_test_const)

        fpr, tpr, thresholds = roc_curve(self.y_test, probs)
        roc_auc = auc(fpr, tpr)

        J = tpr - fpr
        ix = np.argmax(J)
        umbral_optimo = thresholds[ix]
        tpr_opt = tpr[ix]
        fpr_opt = fpr[ix]

        plt.figure()
        plt.plot(fpr, tpr, color='orange', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.scatter(fpr_opt, tpr_opt, color='red', label=f'Índice de Youden\numbral={umbral_optimo:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

        if roc_auc >= 0.9:
            clasificacion = "Excelente"
        elif roc_auc >= 0.8:
            clasificacion = "Bueno"
        elif roc_auc >= 0.7:
            clasificacion = "Regular"
        elif roc_auc >= 0.6:
            clasificacion = "Pobre"
        else:
            clasificacion = "Fallido"

        return {
            "auc": roc_auc,
            "umbral_optimo": umbral_optimo,
            "clasificacion": clasificacion
        }