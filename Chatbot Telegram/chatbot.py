import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from telegram.ext import Updater, MessageHandler, Filters
from sklearn.metrics.pairwise import cosine_similarity


lemmatizer = WordNetLemmatizer()

# Importamos los archivos generados en el código anterior
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'), encoding='utf-8')
classes = pickle.load(open('classes.pkl', 'rb'), encoding='utf-8')
model = load_model('chatbot_model.h5')


# Pasamos las palabras de la oración a su forma raíz
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Convertimos la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    print(bag)
    return np.array(bag)

# Predecimos la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category

# Obtenemos una respuesta aleatoria
def get_response(tag, intents_json, user_sentence):
    list_of_intents = intents_json['intents']
    best_similarity = -1
    best_response = ""

    for intent in list_of_intents:
        if intent['tag'] == tag:
            responses = intent['responses']
            patterns = intent['patterns']
            for pattern in patterns:
                similarity = \
                cosine_similarity(bag_of_words(user_sentence).reshape(1, -1), bag_of_words(pattern).reshape(1, -1))[0][
                    0]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_response = random.choice(responses)
            break

    return best_response


# Handler para manejar los mensajes entrantes
def handle_message(update, context):
    message = update.message.text
    ints = predict_class(message)
    res = get_response(ints, intents, message)  # Pasar 'message' como argumento
    update.message.reply_text(res, parse_mode='HTML')


if __name__ == "__main__":
    # Crear el objeto Updater y pasarle el token del bot
    updater = Updater(token="6056565447:AAGY8KpUaMvQo_dG9HGtSexKVlJ40p3NuPw", use_context=True)

    # Obtener el despachador para registrar los manejadores
    dispatcher = updater.dispatcher

    # Registrar el manejador para los mensajes de texto
    message_handler = MessageHandler(Filters.text, handle_message)
    dispatcher.add_handler(message_handler)

    # Iniciar el bot
    updater.start_polling()
