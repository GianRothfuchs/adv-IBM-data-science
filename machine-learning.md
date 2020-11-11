# Machine Learning

## Machine Learning Pipelines
Pipelines are a way to perform data (pre-)processing in the machine learning flow. In particular the preprocessing like string indexing (mapping string to numerical value), normalization, and one hot encoding is performed. A pipeline adds some structure to that process. The pipeline approach supports the data supply of different model specifications.

### Pipelines in practice
The first step is to define the DataFrame schema by using the `StructType()` which is essentially a collection of `StructFields()` that defines the columname and datatype and whether it is nullable. Example:
```python
from pyspark.sql.types import StructType,StructField, IntegerType

schema = StructType([
    StructField("x",IntegerType(),True),
    StructField("y",IntegerType(),True),
    StructField("z",IntegerType(),True)])
```
The follwing section shows an example how to load [csv data](https://github.com/wchill/HMP_Dataset.git) into a DataFrame. The resulting DataFrame can the be [stored](https://www.ibm.com/support/pages/how-do-you-save-dataframe-watson-studio-project-notebook-asset-cloud-object-storage) to cloud object store (COS).

```python
for category in file_list_filtered:
    data_files = os.listdir('HMP_Dataset/' + category)
    
    for data_file in data_files:
        print(data_file)
        #creating a temporary dataframe from file
        temp_df = spark.read.option("header","false").option("delimiter"," ").csv('HMP_Dataset/' + category + '/' + data_file, schema=schema)
        # adding source file and contextual info about the data file: df.withColumn() used to create rename or change column
        temp_df = temp_df.withColumn('class',lit(category))
        temp_df = temp_df.withColumn('source',lit(data_file))
        
        if df is None:
            df = temp_df
        else:
            # df.union appends data vertically
            df = df.union(temp_df)
```

String indexing is the transformation of strings into numerical value representation. The `StringIndexer()` is a fitting function as needs to remember the state of the df. Therefore a fit function is called before transformation.

```python
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="class", outputCol="classIndex")
# this job runns in parallel, more data means just more nodes
# first method fits the indexer, second applies it to df
indexed = indexer.fit(df).transform(df)
```

One hot encoding is pure transformation operation and therefore there is no fit function involved prior to transformation.

```python
from pyspark.ml.feature import OneHotEncoder
encoder = OneHotEncoder(inputCol='classIndex', outputCol='categoryVec')
encoded = encoder.transform(indexed)
```

The pyspark ml library needs a data vector object to work with, which saves the data as ApacheSpark data vector object to the df:
```python
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols=["x","y","z"], outputCol = "features")
features_vectorized = vectorAssembler.transform(encoded)
```

The DataFrame can be normalized by using the built-in `Normalizer()` function.
```python
from pyspark.ml.feature import Normalizer
# arguments to nromalizer are input, output, norm applied
normalizer = Normalizer(inputCol="features", outputCol="features_norm",p=1.0)
normalized_data = normalizer.transform(features_vectorized)
```

Finally the Pipeline collects all the stages from above and applies the in one go:

from pyspark.ml import Pipeline
```python
pipeline = Pipeline(stages=[indexer,encoder,vectorAssembler,normalizer])
model = pipeline.fit(df)
prediction = model.transform(df)
prediction.show()
```

A working version of the code can be found in the following [notebook](/notebooks/ApacheSparkML_Pipeline.ipynb).
 
## SystemML
SystemML is R-like language to to implement machine learning algorithms in a parallel architecture as ApacheSpark.

## SparkML Linear Regression

DataFrame operations to create label (dependent variable).

```python
df.createOrReplaceTempView("df")
df_join = spark.sql("""

select * from df inner join (select sqrt(sum(x*x)+sum(y*y)+sum(z*z)) as label, class from df group by class) df_energy on df.class=df_energy.class

""").show()
```

Prepare Pipeline elements.
```python
from pyspark.ml.feature import VectorAssembler, Normalizer
from pyspark.ml.regression import LinearRegression

vectorAssembler = VectorAssembler(inputCols=["x","y","z"],outputCol=" features")
normalizer = Normalizer(inputCol="features", outputCol="features_norm",p=1.0)
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
```

Run ElasticNet Regression.
```python
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[vectorAssembler,normalizer,lr])
model = pipeline.fit(df_join)
predictions = model.transform(df_join)
model.stages[2].summary.r2
```

## Refresher on Bayesian Inference

Bayesian method of inference is where the probability of a hypothesis (H) is updated as new evidence (E) becomes available. The process follows the following steps:
1. Begin with prior distribution $p(H)$
2. Collect new data/evidence $E$
3. Calcualte the likelihood - how compatible the new evidence (E) is with the hypothesis (H): $p(E|H)$
4. Optain the posterior - the prob of ou the hypothesis given the evidence: $p(H|E) = \frac{p(E|H) p(H)}{p(E)}$


