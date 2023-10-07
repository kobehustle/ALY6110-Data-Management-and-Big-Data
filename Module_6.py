import websocket, json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, window,expr
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, IntegerType

# Create a SparkSession
spark = SparkSession.builder.master('local').appName("stock_data").getOrCreate()

# Define the schema for the JSON data
schema = StructType([
    StructField("T", StringType(), True),
    StructField("S", StringType(), True),
    StructField("o", DoubleType(), True),
    StructField("h", DoubleType(), True),
    StructField("l", DoubleType(), True),
    StructField("c", DoubleType(), True),
    StructField("v", LongType(), True),
    StructField("t", StringType(), True),
    StructField("n", IntegerType(), True),
    StructField("vw",DoubleType(), True)
])

def on_open(ws):
    print("opened")
    auth_data = {
        "action": "auth", "key": "PKCK05T48DVOWBTL9GK8", "secret": "QxUGV8oFnfyaxEv6pGe1RqnKy6I6GLkOmp353zZG"
    }

    ws.send(json.dumps(auth_data))
    listen_message = {"action": "subscribe", "bars": ["*"],"statuses":["*"]}

    ws.send(json.dumps(listen_message))

empty_df = spark.createDataFrame(spark.sparkContext.emptyRDD(), schema)

# Function to process the JSON data
def calculate_top_n_stocks(json_data, n):
    # Convert the JSON data to DataFrame
    json_df = spark.read.schema(schema).json(spark.sparkContext.parallelize([json_data]))

    print("json df show")
    json_df.show()

    # Apply the desired transformations and actions
    result_df = json_df.groupBy("S") \
        .agg({"c": "max"}) \
        .orderBy(col("max(c)").desc()) \
        .limit(n) \
        .select(col("S"), col("max(c)").alias("close price"))

    print("result")
    result_df.show()

    # Output the top N stocks with highest closing prices to console
    query = result_df.writeStream.outputMode("complete").format("console").start()
    query.awaitTermination()

def calculate_stock_with_largest_variation(json_data):
    # Convert the JSON data to DataFrame
    json_df = spark.read.schema(schema).json(spark.sparkContext.parallelize([json_data]))

    print("json df show")
    json_df.show()

    # Calculate the price variation for each stock
    variation_df = json_df.groupBy("S") \
        .agg({"c": "max", "o": "min"}) \
        .withColumn("variation", col("max(c)") - col("min(o)")) \
        .select(col("S"), col("max(c)").alias("close"), col("min(o)").alias("open"),
                col("variation").alias("price variation"))

    # Find the stock with the largest price variation
    result_df = variation_df.orderBy("variation", ascending=False).limit(1)

    print("result")
    result_df.show()

    # Output the stock with the largest price variation to console
    query = result_df.writeStream.outputMode("complete").format("console").start()
    query.awaitTermination()

def on_message(ws, message):
    print("received a message")
    print(message)

    # Process the received JSON data
    calculate_stock_with_largest_variation(message)
    calculate_top_n_stocks(message,5)
    #detect_price_change(message)

def on_close(ws):
    print("closed connection")

socket = "wss://stream.data.alpaca.markets/v2/iex"
ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_close=on_close)
ws.run_forever()
