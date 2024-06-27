from django.shortcuts import render
from django.http import JsonResponse
import requests
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch


def home(request):
    return render(request, 'bnf/home.html')

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
    if request.method == 'POST':
        user_message = request.POST.get('message')
        
        # Préparer le prompt
        prompt = f"{user_message}"
        
        # Générer la réponse du modèle
        output = pipeline(
            prompt,
            max_new_tokens=2000,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_k=10,
        )

        # Extraire le texte généré
        bot_message = output[0]['generated_text']
        full_bot_message = f"bot : {bot_message.strip()}"  

        print(full_bot_message)
        return JsonResponse({'message': full_bot_message})
    return JsonResponse({'message': 'Méthode non autorisée'}, status=405)
