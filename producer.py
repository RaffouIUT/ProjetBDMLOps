from confluent_kafka import Producer
import json
from datetime import datetime

conf = {'bootstrap.servers': 'kafka:9092'}

producer = Producer(conf)

def json_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()  # Convertit datetime en chaîne ISO
    raise TypeError("Type non sérialisable")

def delivery_report(err, msg):
    if err is not None:
        print('Message delivery failed:', err)
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))

def send_message(data):
    print(data)
    # /!\ NOM DU TOPIC
    producer.produce('topic-new-data', json.dumps(data, default=json_serializer).encode('utf-8'), callback=delivery_report)


#print("Début de l'exécution")
#send_message({"message": "Hello, World!", "timestamp": datetime.now()})
#print("Message envoyé à Kafka")
producer.flush()