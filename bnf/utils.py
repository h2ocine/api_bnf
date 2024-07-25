import re

def parse_oq(text):
    if "l n'y a donc pas besoin de clarification" in text or "he type of ambiguity is [4]" in text or "here is no need for clarification" in text or "here is no need for further clarification" in text:
        return 'unambiguous'
    
    questions = re.findall(r'\[\s*Q\s*\](.*?)\[\s*/\s*Q\s*\]', text, re.DOTALL)
    if questions:
        return questions[-1].strip()
    
    return 'no response error'

def parse(text):
    def extract_indexes(s):
        matches = re.findall(r"\((.*?)\)", s)
        indexes = []
        for match in matches:
            if match.isdigit():
                indexes.append(int(match))
        if len(indexes) != 5:
            print("WARNING: # CQ != 5.")
        return sorted(indexes)

    def extract_cqs(s, indexes):
        cqs = []
        for i in range(len(indexes) - 1):
            cqs.append(re.search(rf"\({indexes[i]}\)(.*?)\({indexes[i+1]}\)", s).group(1).strip('\n').strip())
        last_ph = f"({indexes[-1]})"
        cqs.append(s[s.index(last_ph) + len(last_ph):].strip('\n').strip())
        return cqs

    def clean_text(s):
        last_qc_index = s.rfind("<|end_header_id|>")
        if last_qc_index != -1:
            s = s[last_qc_index+len("<|end_header_id|>"):]
        cqs = []
        if s.count("[QC]") == 2 and s.count("[/QC]") == 2:
            s = re.findall(r"\[QC\](.*?)\[\/QC\]", s.replace('\n', ''))[0]
            indexes = extract_indexes(s)
            if indexes:
                return extract_cqs(s, indexes)
        elif "[QC]" in s and "[/QC]" in s:
            # if multiple marker pairs exist, hypothesize that LLM repeats some input and the last occurence is the valid output.
            s = re.findall(r"\[QC\](.*?)\[\/QC\]", s.replace('\n', ''))[-1]
            indexes = extract_indexes(s)
            if indexes:
                cqs = extract_cqs(s, indexes)
        elif "il n'y a pas besoin de clarification" in s.lower() or "il n'y a donc pas besoin de clarification" in s.lower() or "pas besoin de clarification" in s.lower():
            return ["[unambiguous]"]
        else:
            print("ERROR: Markers not found or no unambiguous sign")
        return cqs

    return clean_text(text)


#test
text1="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|> 

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
(5) la cinquième question de clarification [/QC]<|eot_id|><|start_header_id|>user<|end_header_id|>

[REQUETE] vocabulaire de la psychanalyse jean laplanche et j -b  pontalis [/REQUETE]<|eot_id|><|start_header_id|>assistant<|end_header_id|>

[ANALYSE] La requête "vocabulaire de la psychanalyse jean laplanche et j-b pontalis" est claire et précise. Elle se concentre sur un livre spécifique, "Vocabulaire de la psychanalyse" écrit par Jean Laplanche et J.B. Pontalis. La requête est catégorisée comme Navigation car elle cible un document unique et bien défini. [/ANALYSE]

Il n'y a donc pas besoin de clarification.
"""

text2="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|> 

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
(5) la cinquième question de clarification [/QC]<|eot_id|><|start_header_id|>user<|end_header_id|>

[REQUETE] jose [/REQUETE]<|eot_id|><|start_header_id|>assistant<|end_header_id|>

[ANALYSE] La requête "jose" est une requête ambiguë du type [1] Sémantique. Le mot "jose" est un nom propre qui peut se référer à plusieurs personnes ou entités différentes, ce qui rend difficile une interprétation unique de la requête. [/ANALYSE]

Sur la base de l'analyse ci-dessus, les questions de clarification possibles sont :

[QC] (1) Pouvez-vous être plus spécifique sur la personne ou l'entité que vous cherchez, José?
(2) Est-ce un livre, un article ou un document que vous cherchez sur José?
(3) José est-il un auteur, un personnage ou un thème que vous cherchez?
(4) Pouvez-vous me donner un contexte ou un mot-clé supplémentaire pour comprendre votre requête?
(5) José est-il lié à une période spécifique ou un événement particulier que vous cherchez?

[/QC]
"""

#print(parse(text2))