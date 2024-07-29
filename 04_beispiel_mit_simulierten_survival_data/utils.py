import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Funktion zur Erstellung der simulierten Daten
def create_surv_data(shape_weibull=1.5, scale_weibull_base=50, rate_censoring=0.02, n=1000, 
                     b_bloodp=-0.405, b_diab=0.4, b_age=0.05, b_bmi=0.01, b_kreat=0.2, seed=42):

    # Parameter für Weibull-Verteilung und Censoring
    shape_weibull = shape_weibull
    scale_weibull_base = scale_weibull_base
    rate_censoring = rate_censoring
    n = n

    # Generierung der Kovariaten
    np.random.seed(seed)
    bmi = np.random.normal(25, 5, n)  
    blood_pressure = np.random.binomial(1, 0.3, n)  
    kreatinkinase = np.random.lognormal(mean=5, sigma=1, size=n)
    kreatinkinase = np.clip(kreatinkinase, 30, 8000)
    diabetes = np.random.binomial(1, 0.2, n)  
    age = np.random.normal(50, 10, n)  #
    #plt.boxplot(kreatinkinase)
    #plt.show()

    # Parameter für Weibull-Verteilung
    lambda_weibull = scale_weibull_base * np.exp(
        b_bloodp * blood_pressure +         # Linearer Einfluss von hohem Blutdruck
        b_diab * diabetes +                 # Linearer Einfluss von Diabetes
        b_age * age +                       # Linearer Einfluss des Alters
        b_bmi * (bmi - 25) ** 2  +          # Quadratischer Einfluss des BMI
        b_kreat * np.log(kreatinkinase )    # Exponentieller Einfluss der Kreatinkinase
    )

    # Generierung der Ereigniszeiten basierend auf der Weibull-Verteilung
    event_times = np.random.weibull(shape_weibull, n) * lambda_weibull
    censoring_times = np.random.exponential(1 / rate_censoring, n)
    observed_times = np.minimum(event_times, censoring_times)
    event_occurred = event_times <= censoring_times

    # Erstellung des Datensatzes ohne die nicht-linearen Transformationen
    data = pd.DataFrame({
        'bmi': bmi,
        'blood_pressure': blood_pressure.astype(int), 
        'kreatinkinase': kreatinkinase,
        'diabetes': diabetes.astype(int),
        'age': age,
        't': observed_times,
        'event': event_occurred.astype(int)
    })
    
    print('Data shape:', data.shape)
    print(f'{(data["event"] ==1).sum()/n  * 100} % of the data has an event')
    
    return pd.DataFrame(data)


## Funktion zur Erstellung des neuen Datensatzes in abhängigkeit eines zeitpunktes t
def create_new_dataset(data, t):
    new_data = data.copy()

    new_data.loc[(data['t'] <= t) & (data['event'] == 1), 'survived'] = int(0)
    new_data.loc[(data['t'] >= t) & (data['event'] == 0), 'survived'] = int(1)
    new_data.loc[(data['t'] >= t) & (data['event'] == 1), 'survived'] = int(1)
    new_data.loc[(data['t'] <= t) & (data['event'] == 0), 'survived'] = int(999)

    new_data['survived'] = new_data['survived'].astype(int)

    return pd.DataFrame(new_data)