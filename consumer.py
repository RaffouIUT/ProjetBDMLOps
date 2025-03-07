from confluent_kafka import Consumer

conf = {
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'my-consumer-group',
    'auto.offset.reset': 'earliest'
}

consumer = Consumer(conf)
consumer.subscribe(['topic-new-data'])

def consume_messages():

    messages = []
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                print("No message received")
                continue
            if msg.error():
                print("Consumer error: {}".format(msg.error()))
                continue
            print(f"Received message: {msg.value().decode('utf-8')}")
            messages.append(msg.value().decode('utf-8'))
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

    print(f"Messages consommés: {messages}")  # Afficher les messages consommés
    return messages