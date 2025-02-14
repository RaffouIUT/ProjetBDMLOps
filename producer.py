from confluent_kafka import Producer

conf = {'bootstrap.servers': 'kafka:9092'}

producer = Producer(conf)

def delivery_report(err, msg):
    if err is not None:
        print('Message delivery failed:', err)
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))

for data in ['message 1', 'message 2', 'message 3']:
    producer.produce('test-topic', data.encode('utf-8'), callback=delivery_report)

producer.flush()
