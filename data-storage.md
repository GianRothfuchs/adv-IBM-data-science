# W2: Scalabe Data Science

## Data Storage
The most common data storage options are SQL, NoSQL, and ObjectStorage. Each with its own pros and cons.

SQL Database
| Pros                        | Cons                                |
|:---------------------------|:-------------------------------------|
| Well known and established  | Cannot handle well changing schemas |
| High integrity              | hard to scale                       |
| High data normalization     | High storage cost                   |
| Fast indexed access         |                  					|
| Open standard               |                  					|

NoSQL Database
| Pros               | Cons                  |
|:------------------ |:--------------------- |
| Dynamic schema     | No data normalization |
| Low storage cost   | No data integrity     |
| Linear scalability | High storage cost     |
|          			 | Less established      |
|           	     | slower than SQL       |

ObjectStorage
| Pros               | Cons                  |
|:------------------ |:--------------------- |
| Low storage cost   | Not much established  |
| Linear scalability |                       |
| Schema-less        |                       |

The final choice is driven by the following considerations:
1. Storage Cost
2. Change/Variety of data schemas
3. Query performance
4. Special data types like images, audio, video
5. Scalability

## ApacheSpark Setup

### Java Virtual Machine

ApacheSpark is a JVM, where the underlying execution engine is written in Scala. Scala is a JVM compatible programming language. The JVM is limited to one node and hence to the ressources one node can provide. This is the limitation ApacheSpark addresses by parallelizing accross JVMs. In an ApacheSpark cluster there is a driver node that coordinates a number of worker nodes.

### Data Storage

There are essentially two ways how the data is attached to the cluster. The first apporach is where storage is off-node and attached via a fast network connection. A network technology called switching fabric provides high I/O bandwith. In a second approach hard drives are directly attached to worker nodes. This typology is called JBOD approach or directly attached storage. Disk are combined to a large data pool by using sofware like Hadoop Distributed File System (HDFS). The standard is not POSIX compatible and thus it cannot be mounted into a file system tree. However, it provides a RESTful API for interaction. While a file is split into junks that are saved on individual disks the HDFS is aware of the localities and provides a full file view to instruct the local worker nodes.

[Efficent Data Storage](https://www.youtube.com/watch?v=MZNjmfx4LMc)

### Resilient Distributed Dataset (RDD)

The [RDD](https://spark.apache.org/docs/latest/rdd-programming-guide.html#basics) is one of the enterpieces in ApacheSpark. An RDD is a distributed, immutable collection of data, created from different sources (like HDFS, files, SQL, and so on). When an RDD is created it resides in the momeory of the worker nodes, but may spill do attached disks. RDD is lazy in the sense that data is only read form the underlying storage when it is really needed.

### Programming languages support

ApacheSpark support four languages: Java, Scala, Python, and R. Where Scala typically runs fastest. Implementations for Scala and Java include all ApacheSpark APIs. For R only some APIs are available. However, R is one of the slowest languages, this is problematic when local and distributed computation is mixed.


## ApacheSpark Programming in Python

### Setup
Installation and addition to namespace:
´´´
!pip install pyspark==2.4.5

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()
	
´´´


## ApacheSparkSQL Programming














