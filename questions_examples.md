# Exemples de Questions Test pour le RAG (Code Pénal Algérien)

Voici quelques exemples de questions réalistes que vous pouvez poser à l'assistant juridique pour tester ses capacités de récupération et de génération de réponses en se basant sur les données du Code Pénal Algérien.

## 1. Questions sur les principes généraux
* "Est-ce que la loi pénale peut être rétroactive ?"
* "La loi pénale algérienne s'applique-t-elle aux infractions commises hors du territoire ?"
* "Qu'est-ce qui est stipulé dans l'article 1er du code pénal ?"

## 2. Questions concernant les mineurs et les peines
* "Quelle est la peine applicable pour un mineur de 15 ans qui commet une simple contravention ?"
* "À partir de quel âge une personne assume-t-elle une pleine responsabilité pénale selon le code ?"

## 3. Questions sur les amendes et sanctions
* "Un juge peut-il prononcer uniquement une peine d'amende dans le cas d'un délit classé comme un crime ?"
* "Quel est le seuil minimum et maximum des amendes en matière délictuelle ?"

## 4. Mises en situation (scénarios complexes)
* "Un individu fournit des moyens à une puissance étrangère pour ébranler la fidélité de l'armée de terre, comment cette infraction est-elle classifiée ?"
* "Est-ce que le délit d'abus de confiance bénéficie des mêmes immunités que celles prévues aux articles 368 et 369 ?"

## Comment utiliser ces questions ?
Vous pouvez lancer le script de chat interactif :
```bash
python scripts/run_legal_rag.py
```
Et poser ces questions directement.

Ou via une seule commande (Single Query Mode) :
```bash
python scripts/run_legal_rag.py --query "Quelle est la peine applicable pour un mineur de 15 ans qui commet une simple contravention ?"
```