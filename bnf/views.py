from django.shortcuts import render
from django.http import JsonResponse
import requests
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

def home(request):
    return render(request, 'bnf/home.html')

#------------------------------------------------------------------------#
# LLAMA Model 
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

#------------------------------------------------------------------------#
# Scripts : 
script1 = """
Génère directement la réponse sans afficher mon prompt. Je vais te fournir une requête ; si elle est claire et ne nécessite pas de clarification, génère uniquement la phrase 'La requête est claire et ne nécessite pas de clarification.' et rien de plus. Si la requête n'est pas claire, génère exactement 5 questions de clarification et rien de plus. 

Instructions supplémentaires :
- Ne génère pas de texte supplémentaire.
- Ne génère pas de multiples réponses pour une seule requête.

Exemples :
1. Requête : 'Quelle est la superficie de la France ?' Réponse : 'La requête est claire et ne nécessite pas de clarification.'
2. Requête : 'La France' Réponse : '1. Voulez-vous connaître la superficie de la France ? 2. Voulez-vous connaître la population de la France ? 3. Voulez-vous des informations historiques sur la France ? 4. Voulez-vous connaître la capitale de la France ? 5. Voulez-vous des informations touristiques sur la France ?'
"""
zs_script = """ 

Vous êtes un assistant de chat dans une bibliothèque. Votre objectif est d'aider l'utilisateur à trouver les documents en posant des questions de clarification si nécessaire. Une requête peut être claire sans nécessiter de clarification, tandis qu'une autre peut être ambiguë et il serait préférable d'avoir une conversation avec l'utilisateur pour mieux comprendre son intention. 

Étant donné une requête visant à chercher un document, décidez d'abord si la requête est ambiguë. Si non, répondez par la phrase suivante : « Il n'y a pas besoin de clarification ». Si oui, générez 5 questions de clarification pour mieux comprendre l'intention de l'utilisateur. Encadrez les questions de clarification générées par [QC] et [/QC]. Le format de sortie doit être comme suit :

[QC] (1) la première question de clarification
(2) la deuxième question de clarification
(3) la troisième question de clarification
(4) la quatrième question de clarification
(5) la cinquième question de clarification [/QC]user

Important : Ne répétez pas ce script dans vos réponses. Répondez uniquement en analysant la requête de l'utilisateur et en fournissant soit la phrase « Il n'y a donc pas besoin de clarification », soit les questions de clarification nécessaires. Ne copiez pas le texte de ce script dans votre réponse finale."""
fs_script = """

Vous êtes un assistant de chat dans une bibliothèque. Votre objectif est d'aider l'utilisateur à trouver les documents en posant des questions de clarification si nécessaire. Une requête peut être claire sans nécessiter de clarification, tandis qu'une autre peut être ambiguë et il serait préférable d'avoir une conversation avec l'utilisateur pour mieux comprendre son intention. 

Étant donné une requête visant à chercher un document, décidez d'abord si la requête est ambiguë. Si non, répondez par la phrase suivante : « Il n'y a pas besoin de clarification ». Si oui, générez 5 questions de clarification pour mieux comprendre l'intention de l'utilisateur. Encadrez les questions de clarification générées par [QC] et [/QC]. Le format de sortie doit être comme suit :

[QC] (1) la première question de clarification
(2) la deuxième question de clarification
(3) la troisième question de clarification
(4) la quatrième question de clarification
(5) la cinquième question de clarification [/QC]user

[REQUETE] jose [/REQUETE]

[QC] (1)faites-vous référence au prénom "josé" ? Ou à une personne prénommée "josé" ?
(2) faites-vous référence à l'acronyme JOSE et au format JSON ?
(3) pouvez-vous préciser le nom de la personne cherchée ?
(4) pouvez-vous préciser si "josé" est lié à un contexte géographique ou culturel particulier ?
(5) faites-vous référence à une publication, un livre ou un article dont l'auteur est José ? [/QC]

[REQUETE] cassini carte générale de la france 18595 [/REQUETE]

Il n'y a pas besoin de clarification.

[REQUETE] maroc [/REQUETE]

[QC] (1)cherchez-vous des types de documents particuliers relatif au Maroc ? (images, textes, journaux…)
(2) cherchez-vous des documents relatifs à l'histoire du Maroc ?
(3) cherchez-vous des informations générales sur le Maroc ?
(4) cherchez-vous des informations géographiques sur le Maroc ?
(5) cherchez-vous des publications éditées au Maroc ? [/QC]

[REQUETE] guerre de cent ans [/REQUETE]

[QC] (1)cherchez-vous des détails sur l'impact de grands conflits historiques sur l'histoire de la France et de l'Angleterre ?
(2) voulez-vous des informations sur les conflits médiévaux en Europe ?
(3) avez-vous besoin de comprendre les différents phases et périodes de la Guerre de Cent Ans ?
(4) recherchez-vous une vue d'ensemble générale de la Guerre de Cent Ans, y compris ses causes, ses événements majeurs et ses conséquences ?
(5) cherchez-vous des détails sur l'impact de la Guerre de Cent Ans sur l'histoire ultérieure de la France et de l'Angleterre ? [/QC]

[REQUETE] mémorial des percepteurs et des receveurs des communes [/REQUETE]

Il n'y a pas besoin de clarification.

[REQUETE] journal officiel de la république française [/REQUETE]

[QC] (1)cherchez-vous à consulter  le "Journal officiel de la République française" ?
(2) cherchez-vous à consulter  un numéro particulier du "Journal officiel de la République française" ?
(3) cherchez-vous la publication "Le Journal officiel de la République française" dans quel pays?
(4) voulez-vous connaître l'historique et l'évolution du Journal Officiel de la République Française ?
(5) cherchez-vous une édition spécifique du Journal Officiel de la République Française ou une période particulière ? [/QC]

[REQUETE] oignies en 1916 [/REQUETE]

Il n'y a pas besoin de clarification.

[REQUETE] reclus [/REQUETE]

[QC] (1) faites-vous référence à la personne "Elisée Reclus" ? Ou à l'adjectif « reclus » ?
(2) cherchez-vous des informations biographiques concernant la vie d'Elisée Reclus ?
(3) cherchez-vous des informations relatives à l'oeuvre d'Elisée Reclus ?
(4) cherchez-vous à consulter des oeuvres d'Elisée Reclus ?
(5) Voulez-vous chercher le mot-clé « reclus » dans les textes ? [/QC]

[REQUETE] hopitaux medecine chirurgique [/REQUETE]

[QC] (1) Cherchez-vous des documents traitant de l'histoire de la chirurgie hospitalière ?
(2) Cherchez-vous des organisations professionnelles médicales éditant des documents portant sur la chirurgie hospitalière ?
(3) Cherchez-vous des éditeurs spécialisés dans le domaine de la chirurgie ?
(4) Cherchez-vous des revues spécialisées dans le domaine de la chirurgie ?
(5) Cherchez-vous des documents dont la notice inclut les termes "hopitaux medecine chirurgique" ? [/QC]

[REQUETE] cerf-volant [/REQUETE]

[QC] (1) cherchez-vous à consulter des documents dont le titre serait "cerf-volant" ?
(2) cherchez-vous à consulter des documents traitant de l'insecte"cerf-volant" ?
(3) cherchez-vous des informations sur le jouet qui vole dans les airs ou sur l'insecte appelé cerf-volant ?
(4) êtes-vous intéressé par l'histoire et les traditions associées au cerf-volant (jouet) dans différentes cultures ?
(5) voulez-vous des détails sur la fabrication et l'utilisation des cerfs-volants comme jouets ? [/QC]

Note: Ne répétez pas ce script dans vos réponses. Répondez uniquement aux questions ou clarifications demandées ci-dessus.
"""
atcotfs_script = """

Vous êtes un assistant de chat dans une bibliothèque. Votre objectif est d'aider l'utilisateur à trouver les documents en posant des questions de clarification si nécessaire. Une requête peut être claire sans nécessiter de clarification, tandis qu'une autre peut être ambiguë et il serait préférable d'avoir une conversation avec l'utilisateur pour mieux comprendre son intention. En cas de requêtes ambiguës, l'ambiguïté peut être multifacette, et il existe donc plusieurs types de requêtes :

[1] Sémantique : la requête est sémantiquement ambiguë pour plusieurs raisons courantes : elle peut inclure des homonymes ; un mot dans la requête peut se référer à une entité spécifique tout en fonctionnant également comme un mot commun ; ou une mention d'entité dans la requête pourrait se référer à plusieurs entités distinctes.

[2] Généralisation : la requête se concentre sur des informations spécifiques ; cependant, une requête plus large et étroitement liée pourrait mieux capturer les véritables besoins en information de l'utilisateur.

[3] Spécification : la requête a un objectif clair mais peut correspondre à un champ de recherche trop large. Il est possible de restreindre davantage ce champ en fournissant des informations plus spécifiques relatives à la requête.

[4] Navigation : la requête est clairement ciblée sur un document spécifique sans aucune autre interprétation possible. Les exemples typiques incluent le titre exact du document, l'identifiant du document, les numéros de série.

Étant donné une requête visant à chercher un document, décidez d'abord si la requête fait partie du type [4] Navigation. Si oui, générez une analyse textuelle expliquant pourquoi cette requête est catégorisée comme Navigation en concluant par la phrase suivante : « Il n'y a donc pas besoin de clarification ». Encadrez votre analyse par [ANALYSE] et [/ANALYSE].

Si non, générez 5 questions de clarification pour mieux comprendre l'intention de l'utilisateur. Avant de générer des questions de clarification, générez une analyse textuelle pour déterminer lequel des types d'ambiguïté mentionnés précédemment s'applique à la requête donnée et pourquoi. Encadrez votre analyse par [ANALYSE] et [/ANALYSE] et les questions de clarification générées par [QC] et [/QC]. Reliez l'analyse aux questions de clarification en utilisant la phrase « sur la base de l'analyse ci-dessus, les questions de clarification possibles sont : ». Le format de sortie doit être comme suit :

[ANALYSE] votre analyse [/ANALYSE]
Sur la base de l'analyse ci-dessus, les questions de clarification possibles sont :

[QC] (1) la première question de clarification
(2) la deuxième question de clarification
(3) la troisième question de clarification
(4) la quatrième question de clarification
(5) la cinquième question de clarification [/QC]user

[REQUETE] jose [/REQUETE]

[ANALYSE] Le type d'ambiguité est [1] Sémantique car "jose" est un prénom très courant, mais également un acronyme (JOSE, JSON Object Signing and Encryption). L'utilisateur n'indique pas quel type d'information il recherche et il est possible de clarifier son intention en fournissant des informations liées à "jose". [/ANALYSE]

sur la base de l'analyse ci-dessus, les questions de clarification possibles sont :

[QC] (1)faites-vous référence au prénom "josé" ? Ou à une personne prénommée "josé" ?
(2) faites-vous référence à l'acronyme JOSE et au format JSON ?
(3) pouvez-vous préciser le nom de la personne cherchée ?
(4) pouvez-vous préciser si "josé" est lié à un contexte géographique ou culturel particulier ?
(5) faites-vous référence à une publication, un livre ou un article dont l'auteur est José ? [/QC]

[REQUETE] cassini carte générale de la france 18595 [/REQUETE]

[ANALYSE] Le type d'ambiguité est [4] Navigation car l'utilisateur a fourni l'auteur de la carte et aussi un numéro, 18595, qui correspond à une notice GE FF-18595. Il n'y a donc pas besoin de clarification. [/ANALYSE]

[REQUETE] maroc [/REQUETE]

[ANALYSE] Le type d'ambiguité est [3] Spécification car "maroc" est un pays clairement défini mais le champ de recherche est très vaste. Il est possible de restreindre ce champ en posant des questions permettant de préciser le centre d'intérêt de l'utilisateur. [/ANALYSE]

sur la base de l'analyse ci-dessus, les questions de clarification possibles sont :

[QC] (1)cherchez-vous des types de documents particuliers relatif au Maroc ? (images, textes, journaux…)
(2) cherchez-vous des documents relatifs à l'histoire du Maroc ?
(3) cherchez-vous des informations générales sur le Maroc ?
(4) cherchez-vous des informations géographiques sur le Maroc ?
(5) cherchez-vous des publications éditées au Maroc ? [/QC]

[REQUETE] guerre de cent ans [/REQUETE]

[ANALYSE] Deux types d'ambiguité peuvent s'appliquer pour la requête ici : [2] Généralisation et [3] Spécification. Il est possible que l'utilisateur s'intéresse aux sujets plus généraux liés à la guerre de cent ans, tel que les grands conflits historiques au Moyen Âge en Europe. Il est également possible de proposer différentes facettes ou informations spécifiques pertinentes à l'utilisateur de choisir pour mieux comprendre son intention. [/ANALYSE]

sur la base de l'analyse ci-dessus, les questions de clarification possibles sont :

[QC] (1)cherchez-vous des détails sur l'impact de grands conflits historiques sur l'histoire de la France et de l'Angleterre ?
(2) voulez-vous des informations sur les conflits médiévaux en Europe ?
(3) avez-vous besoin de comprendre les différentes phases et périodes de la Guerre de Cent Ans ?
(4) recherchez-vous une vue d'ensemble générale de la Guerre de Cent Ans, y compris ses causes, ses événements majeurs et ses conséquences ?
(5) cherchez-vous des détails sur l'impact de la Guerre de Cent Ans sur l'histoire ultérieure de la France et de l'Angleterre ? [/QC]

[REQUETE] mémorial des percepteurs et des receveurs des communes [/REQUETE]

[ANALYSE] Le type d'ambiguité est [4] Navigation car la requête s'agit d'un titre complet et exact, il n'existe aucune différente interprétation. Il n'y a donc pas besoin de clarification. [/ANALYSE]

[REQUETE] journal officiel de la république française [/REQUETE]

[ANALYSE] Le type d'ambiguité est [3] Spécification car "journal officiel de la république française" désigne une publication, mais il est possible de demander à l'utilisateur quelle modalité il souhaite pour consulter la publication recherchée. [/ANALYSE]

sur la base de l'analyse ci-dessus, les questions de clarification possibles sont :

[QC] (1)cherchez-vous à consulter le "Journal officiel de la République française" ?
(2) cherchez-vous à consulter un numéro particulier du "Journal officiel de la République française" ?
(3) cherchez-vous la publication "Le Journal officiel de la République française" dans quel pays ?
(4) voulez-vous connaître l'historique et l'évolution du Journal Officiel de la République Française ?
(5) cherchez-vous une édition spécifique du Journal Officiel de la République Française ou une période particulière ? [/QC]

[REQUETE] oignies en 1916 [/REQUETE]

[ANALYSE] Le type d'ambiguité est [4] Navigation car l'utilisateur a fourni un lieu et une date, ce qui indique qu'il cherche des documents relatifs à ce lieu à la date fournie. Il n'y a donc pas besoin de clarification. [/ANALYSE]

[REQUETE] reclus [/REQUETE]

[ANALYSE] Le type d'ambiguité est [1] Sémantique et [3] Spécification car "reclus" peut désigner une personne, Elisée Reclus, ou un adjectif. [/ANALYSE]

sur la base de l'analyse ci-dessus, les questions de clarification possibles sont :

[QC] (1) faites-vous référence à la personne "Elisée Reclus" ? Ou à l'adjectif « reclus » ?
(2) cherchez-vous des informations biographiques concernant la vie d'Elisée Reclus ?
(3) cherchez-vous des informations relatives à l'oeuvre d'Elisée Reclus ?
(4) cherchez-vous à consulter des oeuvres d'Elisée Reclus ?
(5) Voulez-vous chercher le mot-clé « reclus » dans les textes ? [/QC]

[REQUETE] hopitaux medecine chirurgique [/REQUETE]

[ANALYSE] Le type d'ambiguïté est [3] Spécification car "hopitaux médecine chirurgicale" évoque une discipline médicale exercée à l'hôpital mais le champ de recherche reste vaste. Il est possible de restreindre ce champ en posant des questions permettant de préciser le centre d'intérêt de l'utilisateur. [/ANALYSE]

Sur la base de l'analyse ci-dessus, les questions de clarification possibles sont :

[QC] (1) Cherchez-vous des documents traitant de l'histoire de la chirurgie hospitalière ?
(2) Cherchez-vous des organisations professionnelles médicales éditant des documents portant sur la chirurgie hospitalière ?
(3) Cherchez-vous des éditeurs spécialisés dans le domaine de la chirurgie ?
(4) Cherchez-vous des revues spécialisées dans le domaine de la chirurgie ?
(5) Cherchez-vous des documents dont la notice inclut les termes "hopitaux medecine chirurgique" ? [/QC]

[REQUETE] cerf-volant [/REQUETE]

[ANALYSE] Deux types d'ambiguïté s'appliquent pour la requête ici: [1] Sémantique et [3] Spécification. "Cerf-volant" est sémantiquement ambigu car il peut être un nom commun polysémique mais aussi un titre d'œuvre. Supposons que nous adoptons un sens du "cerf-volant", il est aussi possible de proposer les questions plus spécifiques qui orientent la conversation vers la vraie intention de l'utilisateur. [/ANALYSE]

Sur la base de l'analyse ci-dessus, les questions de clarification possibles sont :

[QC] (1) cherchez-vous à consulter des documents dont le titre serait "cerf-volant" ?
(2) cherchez-vous à consulter des documents traitant de l'insecte "cerf-volant" ?
(3) cherchez-vous des informations sur le jouet qui vole dans les airs ou sur l'insecte appelé cerf-volant ?
(4) Êtes-vous intéressé par l'histoire et les traditions associées au cerf-volant (jouet) dans différentes cultures ?
(5) Voulez-vous des détails sur la fabrication et l'utilisation des cerfs-volants comme jouets ? [/QC]

Note: Ne répétez pas ce script dans vos réponses. Répondez uniquement aux questions ou clarifications demandées ci-dessus."""


#------------------------------------------------------------------------#

def chat(request):
    if request.method == 'POST':
        user_message = request.POST.get('message')
        
        # Préparer le prompt
        query= f"[REQUETE] {user_message} [/REQUETE]"
        script= zs_script # Choose a specific script
        prompt= f"{script} {query}"
        print(f"\nrequete : {user_message} ")
        try:
            print('\ngeneration de la requete en cours............................')
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
            full_bot_message = f"chat bot : {bot_message.strip()}"

            print(f"\nreponse du modele : \n{bot_message}")
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

