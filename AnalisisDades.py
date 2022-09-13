from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.sql.functions import concat, lit



'''
Entrenem i evaluem el model d'arbre de decisió utilitzant la matriu amb l'etiqueta disenyada anteriorment.
Per l'entrenament utilitzarem el 80% de les dades mentre que per l'avaluació el 20% restant.
Per entrenar el model fem les següents modificacions a la matriu:
    1) Posem totes les variables predictores en una de sola en format VectorUDT
    2) Afegim una columna etiqueta_idx que es correpongui amb l'etiqueta però en format indx (0 = no-maintenance vs 1 = maintenance)
    3) Afegim una columna amb totes les variables predictores en format índex
A continuació separem els conjunts d'entrenament i de test de forma aleatòria,
    entrenem el model i calculem les mesures d'avaluació mitjançant la matriu de confusió
'''

def AnalisiDades(spark):
    BD = spark.read \
    .format("csv")\
    .option("inferSchema", "true")\
    .option("header", "true")\
    .option("delimiter",";")\
    .load("./MatriuEtiqueta/*.csv")

#Fem les modificacions explicades per poder entrenar el model sense errors
    predictors =["flighthours", "flightcycles", "delayedminutes", "avg(value)"]
    va = VectorAssembler(inputCols = predictors, outputCol='predictors')
    BD = va.transform(BD)

    idx1 = StringIndexer(inputCol="etiqueta", outputCol="etiqueta_idx").fit(BD)
    BD = idx1.transform(BD)

    idx2 = VectorIndexer(inputCol='predictors', outputCol="predictors_idx").fit(BD)
    BD = idx2.transform(BD)

    BD = BD.select(['dia', 'aircraftid','predictors', 'predictors_idx','etiqueta', 'etiqueta_idx'])

#Separem les dade en test i train
    train, test = BD.randomSplit([0.8, 0.2])


    model = DecisionTreeClassifier(featuresCol="predictors_idx", labelCol="etiqueta_idx").fit(train)
    prediccions = model.transform(test)

    #calculem accuracy i recall per evaluar el model a partir de la matriu de confusio
    tp = prediccions[(prediccions.etiqueta_idx == 1) & (prediccions.prediction == 1)].count()
    tn = prediccions[(prediccions.etiqueta_idx == 0) & (prediccions.prediction == 0)].count()
    fp = prediccions[(prediccions.etiqueta_idx == 0) & (prediccions.prediction == 1)].count()
    fn = prediccions[(prediccions.etiqueta_idx == 1) & (prediccions.prediction == 0)].count()

    a = float(tn)/(tp+tn+fp+fn)
    r = float(tp)/(tp + fn)

    print('Recall:', r)
    print('Accuracy:',a)

    #Finalment guardem el model:
    model.write().overwrite().save("model")
