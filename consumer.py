from confluent_kafka import Consumer

conf = {
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'my-consumer-group',
    'auto.offset.reset': 'earliest'
}

consumer = Consumer(conf)
consumer.subscribe(['topic-new-data'])

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print("Consumer error: {}".format(msg.error()))
            continue
        print('Received message: {}'.format(msg.value().decode('utf-8')))
except KeyboardInterrupt:
    pass
finally:
    consumer.close()
