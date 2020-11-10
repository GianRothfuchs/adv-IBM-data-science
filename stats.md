# Scaling Stats to ApacheSpark
´´´
from math import sqrt
rdd = sc.parallelize(range(100))
mean = rdd.sum() / rdd.count()
print (mean)
# Sums of Squares: 
# (the map method applies an anonymous fun to each item)
sd = sqrt(rdd.map(lambda x : pow(x - mean,2)).sum()/rdd.count())

# More than one col
rddX = sc.parallelize(range(100))
rddY = sc.parallelize(range(100))
rddXY = rddX.zip(rddY)

covXY = rddXY.map(lambda (x,y) : (x-meanX)*(y-meanY)).sum()/rddXY.count()
´´´

Help on [ApacheSparkSQL](https://spark.apache.org/docs/2.3.0/api/sql/)

´´´
result = spark.sql("SELECT voltage,ts FROM washing WHERE voltage is not null order by ts asc")

#result_rdd = result.rdd.sample(False,0.1).map(lambda row : (row.ts,row.voltage)).collect()
result_rdd = result.rdd.sample(False,0.1).map(lambda row : (row.ts,row.voltage))
result_array_voltage = result_rdd.map(lambda rw: rw[1]).collect()
result_array_ts = result_rdd.map(lambda rw: rw[0]).collect()
´´´

´´´
select temperature,ts from washing where temperature is not null order by ts asc
""").sample(False,0.1).rdd.map(lambda row : (row.ts,row.temperature))
result_array_ts = double_tuple_rdd.map(lambda ts_temperature: ts_temperature[0]).collect()
result_array_temperature = double_tuple_rdd.map(lambda ts_temperature: ts_temperature[1]).collect()
return (result_array_ts,result_array_temperature)
´´´ 