from django.shortcuts import render
from django.http import JsonResponse
import requests
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import time
import os
from bnf.scripts import *
from bnf.utils import *

def home(request):
    return render(request, 'bnf/home.html')

#------------------------------------------------------------------------#
#------------------------------------------------------------------------#
# LLAMA Model 
save_directory = './bnf/model'
model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'

def load_model():
    global tokenizer, model, text_generator
    if os.path.exists(save_directory):
        print('Chargement du modèle et du tokenizer depuis le disque...')
        tokenizer = AutoTokenizer.from_pretrained(save_directory, padding_side="left")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(save_directory, device_map="auto")
    else:
        print('Chargement et sauvegarde du modèle et du tokenizer depuis le hub...')
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)

    return model, tokenizer

model,tokenizer = load_model()

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    batch_size=1
)

terminators = [
    tokenizer.eos_token_id,
    #tokenizer.convert_tokens_to_ids("")
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

#------------------------------------------------------------------------#
def chat(request, conversation=False):
    print('Lancement analyse de la requête')
    if request.method == 'POST':
        if conversation:
            user_message = request.POST.get('conversation')
        else:
            user_message = request.POST.get('message')
        print(f"\nrequête : {user_message} ")

        # Préparer le prompt
        if conversation:
            prompt_type= "conv-AT-CoT-few-shot"
        else:
            prompt_type = "AT-CoT-few-shot"
        prompt = load_prompt(user_message, prompt_type)

        try:
            print('\ngénération de la requête en cours ..................................................................')
            start_time = time.time()
            output = text_generator(
                prompt,
                max_new_tokens=2500,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_k=10,
            )
            end_time = time.time()
            response_time = round(end_time - start_time, 2)
            print(f"\ntemps d'exécution : {response_time}")

            bot_response = output[0]['generated_text']
            parsed_bot_response = parse(bot_response)
            #print("prompt : ",prompt)
            print(f"\nréponse du modèle (sans parsing) : \n{output[0]['generated_text']}")

            if parsed_bot_response == "no response error":
                print('\n\nERREUR : PAS DE RÉPONSE GÉNÉRÉE\n\n')
                return JsonResponse({'status': 'error', 'message': 'Erreur : pas de réponses générées.'})
            elif parsed_bot_response == ["[unambiguous]"]:
                return JsonResponse({'status': 'unambiguous', 'message': 'requête non ambigu.', 'response_time': response_time})
            else:
                return JsonResponse({'status': 'success', 'questions': parsed_bot_response, 'response_time': response_time})

        except Exception as e:
            print(f"Erreur lors de la génération de texte : {e}")
            output_message = f"<span style='color: red;'>Erreur lors de la génération de texte</span>"
            return JsonResponse({'message': output_message})

    return JsonResponse({'message': f"<span style='color: red;'>Méthode non autorisée</span>"}, status=405)

def chat_conversation(request):
    if request.method == 'POST':
        user_message = request.POST.get('conversation')
        return chat(request, conversation=True)
    return JsonResponse({'status': 'error', 'message': 'Méthode non autorisée'}, status=405)


#------------------------------------------------------------------------#
#------------------------------------------------------------------------#
#------------------------------------------------------------------------#
"""
functions for home_one_question.html 
#------------------------------------------------------------------------#
def response(request, temps_execution_cum):
    print('Lancement Réponse a la requete *************************************')
    if request.method == 'POST':
        user_message = request.POST.get('message')
        print(f"\nrequete : {user_message} ")

        # Préparer le prompt
        query = f"[REQUETE] {user_message} [/REQUETE]"
        script = script_response  # Choisir un script spécifique
        balise_de_fin = "[PROMPT_END]"
        prompt = f"{script} {query} {balise_de_fin}"

        try:
            start_time = time.time()

            print('\ngeneration de la réponse en cours............................')

            output = text_generator(
                prompt,
                max_new_tokens=300,
                eos_token_id=terminators,
                do_sample=False,
                temperature=0.7
            )

            end_time = time.time()
            response_time = temps_execution_cum + end_time - start_time

            output_message = output[0]['generated_text']
            if output_message and balise_de_fin in output_message:
                output_message = output_message.split(balise_de_fin)[-1]

            print(f"\ntemps d'execution : {round(response_time,2)}")
            print(f"\nreponse du modele : \n{output[0]['generated_text']}")

            output_message += f"\n(temps de réponse : {round(response_time,2)} s)"
            output_message = f"<span style='color: #a7a211;'>{output_message}</span>"
            return JsonResponse({'message': output_message})

        except Exception as e:
            print(f"Erreur lors de la génération de texte : {e}")
            return JsonResponse({'message': f"<span style='color: red;'>Erreur lors de la génération de texte</span>"}, status=500)

    return JsonResponse({'message': f"<span style='color: red;'>Méthode non autorisée</span>"}, status=405)

#------------------------------------------------------------------------#
def chat(request, conversation=False):
    print('Lancement analyse de la requete *************************************')
    if request.method == 'POST':
        user_message = request.POST.get('message')
        
        # Préparer le prompt
        balise_de_fin = "[PROMPT_END][\PROMPT_END]"
        if conversation: # if it's a conversation with clarifications we use the conversation script
            script= script_conversation
            prompt = f"{script} {user_message} {balise_de_fin}"
        else: # First response
            query = f"[REQUETE] {user_message} [/REQUETE]"
            script = oq_atcotfs_script  # Choose a script for the first response
            prompt = f"{script} {query} {balise_de_fin}"
        print(f"\nrequete : {user_message} ")
        try:
            start_time = time.time()

            print('\ngeneration de la requete en cours............................')
            output = text_generator(
                prompt,
                max_new_tokens=300,
                eos_token_id=terminators,
                do_sample=False,
                temperature=0.7
            )

            end_time = time.time()
            response_time = end_time - start_time
            bot_response = output[0]['generated_text']
            if bot_response and balise_de_fin in bot_response:
                bot_response = bot_response.split(balise_de_fin)[-1]
        
            parsed_bot_response = parse_oq(bot_response)
            print(f"\ntemps d'execution : {round(response_time,2)}")
            print(f"\nreponse du modele : \n{output[0]['generated_text']}")

            if parsed_bot_response.strip() == "no response error":
                print('\n\nERREUR : PAS DE REPONSE GENERE\n\n')
                output_message = f"<span style='color: red;'>Erreur pas de réponse</span>"
                return JsonResponse({'message': output_message})
            elif parsed_bot_response.strip() == "unambiguous":
                output_message = 'unambiguous'
                return response(request, response_time)
            else:
                output_message = f"<span style='color: blue;'>{parsed_bot_response.strip()}</span>"
                conversation = f"[REQUETE]{user_message}[\REQUETE] [ANSWER]{parsed_bot_response.strip()}[\ANSWER]"
                return JsonResponse({'message': output_message + f"\n(temps de réponse : {round(response_time,2)} s)", 'show_button': True, 'user_message': conversation})

        except Exception as e:
            print(f"Erreur lors de la génération de texte : {e}")
            return JsonResponse({'message': f"<span style='color: red;'>Erreur lors de la génération de texte</span>"}, status=500)

    return JsonResponse({'message': f"<span style='color: red;'>Méthode non autorisée</span>"}, status=405)

#------------------------------------------------------------------------#
def chat_conversation(request):
    if request.method == 'POST':
        return chat(request, conversation=True)

    return JsonResponse({'message': f"<span style='color: red;'>Méthode non autorisée</span>"}, status=405)
"""