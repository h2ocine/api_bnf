from django.shortcuts import render
from django.http import JsonResponse
import requests
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

#script = "Vous êtes un assistant de chat dans une bibliothèque. Votre objectif est d'aider l'utilisateur à trouver les documents en posant des questions de clarification si nécessaire. Une requête peut être claire sans nécessiter de clarification, tandis qu'une autre peut être ambiguë et il serait préférable d'avoir une conversation avec l'utilisateur pour mieux comprendre son intention. Si oui, donnant une réponse a la requête. Si non, générez 5 questions de clarification pour mieux comprendre l'intention de l'utilisateur. "
script = """
Génère directement la réponse sans afficher mon prompt. Je vais te fournir une requête ; si elle est claire et ne nécessite pas de clarification, génère uniquement la phrase 'La requête est claire et ne nécessite pas de clarification.' et rien de plus. Si la requête n'est pas claire, génère exactement 5 questions de clarification et rien de plus. 

Instructions supplémentaires :
- Ne génère pas de texte supplémentaire.
- Ne génère pas de multiples réponses pour une seule requête.

Exemples :
1. Requête : 'Quelle est la superficie de la France ?' Réponse : 'La requête est claire et ne nécessite pas de clarification.'
2. Requête : 'La France' Réponse : '1. Voulez-vous connaître la superficie de la France ? 2. Voulez-vous connaître la population de la France ? 3. Voulez-vous des informations historiques sur la France ? 4. Voulez-vous connaître la capitale de la France ? 5. Voulez-vous des informations touristiques sur la France ?'

Voici la requête :
"""

def home(request):
    return render(request, 'bnf/home.html')

# Initialiser le modèle et le tokenizer globalement
model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'

# Charger le tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id

# Configurer le modèle avec BitsAndBytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Charger le modèle
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Créer un pipeline de génération de texte
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    batch_size=1
)

def chat(request):
    print('debut chat')
    if request.method == 'POST':
        user_message = request.POST.get('message')
        
        # Préparer le prompt
        prompt = f"{script} {user_message}"
        print(f"user_message : {user_message}")
        try:
            print('debut generation de la réponse')
            # Générer la réponse du modèle
            output = text_generator(
                prompt,
                max_new_tokens=100,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_k=10,
            )
            
            # Extraire le texte généré
            bot_message = output[0]['generated_text']
            full_bot_message = f"BNF CHATBOT : {bot_message.strip()}"

            print(full_bot_message)
            return JsonResponse({'message': full_bot_message})

        except Exception as e:
            print(f"Erreur lors de la génération de texte : {e}")
            return JsonResponse({'message': 'Erreur lors de la génération de texte'}, status=500)

        print(f"full_bot_message : {full_bot_message}")
        return JsonResponse({'message': full_bot_message})
    return JsonResponse({'message': 'Méthode non autorisée'}, status=405)


"""
def chat(request):
    if request.method == 'POST':
        user_message = request.POST.get('message')
        # Appel au modèle LLaMA 
        response = requests.post('URL_DE_LA_API_DU_MODELE_LLAMA', json={'input': user_message})
        bot_message = response.json().get('response', 'Erreur')
        #bot_message_plus = f"{bot_message} helloworld"  
        return JsonResponse({'message': 'bot : ' + bot_message})
    return JsonResponse({'message': 'Méthode non autorisée'}, status=405)
"""

