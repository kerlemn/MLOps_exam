from flask import Flask, request, jsonify
from backend.predictor import add_feedback

app = Flask(__name__)

@app.route('/salva_dati', methods=['POST'])
def salva_dati():
    data = request.json  # Ricevi i dati inviati dal frontend

    # Estrai i dati ricevuti (titolo, preferenza, utente) per salvarli nel database
    title = data.get('title')
    preference = data.get('preference')
    user = data.get('user')

    # Esegui l'operazione di salvataggio nel database usando le funzioni del tuo codice Python per interagire con il database
    add_feedback(title, preference, user)  # Aggiungi i dati nel tuo database

    # Restituisci una risposta al frontend
    return jsonify({"message": "Dati salvati con successo nel database!"})

if __name__ == '__main__':
    app.run(debug=True)