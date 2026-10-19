"""Small capability checks with verifiable answers (the idea of ds4-eval): not a leaderboard,
a tripwire. If a quantization breaks arithmetic, facts or code, the pass count drops."""

import json
import re
import subprocess
import sys


def _num(expected):
    def check(ans):
        nums = re.findall(r"-?\d+(?:[.,]\d+)?", ans.replace(".", "") if expected >= 1000 else ans)
        return any(abs(float(n.replace(",", ".")) - expected) < 1e-6 for n in nums)
    return check


def _has(*words):
    return lambda ans: all(re.search(w, ans, re.I) for w in words)


def _code(tests):
    def check(ans):
        m = re.findall(r"```(?:python|py)?\n(.*?)```", ans, re.S)
        code = "\n".join(m) if m else ans
        try:
            r = subprocess.run([sys.executable, "-I", "-c", code + "\n" + tests],
                               capture_output=True, timeout=10)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    return check


def _json_keys(**expected):
    def check(ans):
        m = re.search(r"\{.*\}", ans, re.S)
        try:
            d = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            return False
        return isinstance(d, dict) and all(str(d.get(k)).lower() == str(v).lower() for k, v in expected.items())
    return check


TASKS = [
    ("Quanto fa 17 × 23? Rispondi solo con il numero.", _num(391)),
    ("Un treno percorre 180 km in 2 ore e 15 minuti. Qual è la velocità media in km/h? Rispondi solo con il numero.", _num(80)),
    ("Converti 3,5 ore in minuti. Rispondi solo con il numero.", _num(210)),
    ("Qual è il 15% di 240? Rispondi solo con il numero.", _num(36)),
    ("Calcola la media di 4, 8, 15, 16, 23, 42. Rispondi solo con il numero.", _num(18)),
    ("Quanti secondi ci sono in un giorno? Rispondi solo con il numero.", _num(86400)),
    ("Qual è la capitale dell'Australia? Rispondi con una parola.", _has(r"canberra")),
    ("In che anno è caduto il muro di Berlino? Rispondi solo con l'anno.", _num(1989)),
    ("Chi ha scritto 'I promessi sposi'? Rispondi con nome e cognome.", _has(r"manzoni")),
    ("Qual è il simbolo chimico dell'oro?", _has(r"\bAu\b")),
    ("Quale pianeta è il più grande del sistema solare? Una parola.", _has(r"giove")),
    ("Se oggi è mercoledì, che giorno della settimana sarà tra 10 giorni? Una parola.", _has(r"sabato")),
    ("Ordina alfabeticamente: pera, mela, banana, kiwi. Rispondi solo con la lista separata da virgole.",
     _has(r"banana\W+kiwi\W+mela\W+pera")),
    ("Elenca i primi 6 numeri primi separati da virgole, senza altro testo.", _has(r"2\W+3\W+5\W+7\W+11\W+13")),
    ("Traduci in inglese: 'Il gatto dorme sul divano.'", _has(r"cat", r"sleep", r"sofa|couch")),
    ("Riscrivi al passato remoto, prima persona: 'Io vado a Roma.' Solo la frase.", _has(r"andai")),
    ("Rispondi solo con un oggetto JSON con le chiavi \"nome\" e \"eta\" (numero) per: Mario ha 34 anni.",
     _json_keys(nome="Mario", eta=34)),
    ("Scrivi una funzione Python `is_palindrome(s)` che ignora spazi, punteggiatura e maiuscole. Solo il codice.",
     _code("assert is_palindrome('I topi non avevano nipoti')\nassert is_palindrome('A man, a plan, a canal: Panama')\n"
           "assert not is_palindrome('Ciao mondo')")),
    ("Scrivi una funzione Python `fib(n)` che restituisce l'n-esimo numero di Fibonacci, con fib(0)=0 e fib(1)=1. Solo il codice.",
     _code("assert [fib(i) for i in range(10)] == [0,1,1,2,3,5,8,13,21,34]\nassert fib(50) == 12586269025")),
    ("Scrivi una funzione Python `merge_intervals(intervals)` che riceve una lista di coppie (inizio, fine) e restituisce "
     "la lista ordinata degli intervalli sovrapposti uniti, come lista di tuple. Solo il codice.",
     _code("r = merge_intervals([(1,3),(2,6),(8,10),(15,18)])\nassert [tuple(x) for x in r] == [(1,6),(8,10),(15,18)]\n"
           "assert [tuple(x) for x in merge_intervals([(1,4),(4,5)])] == [(1,5)]")),
    ("Scrivi una funzione Python `flatten(lst)` che appiattisce liste annidate a qualsiasi profondità. Solo il codice.",
     _code("assert flatten([1,[2,[3,[4]],5]]) == [1,2,3,4,5]\nassert flatten([]) == []")),
    ("Scrivi una funzione Python `roman(n)` che converte un intero da 1 a 3999 in numero romano. Solo il codice.",
     _code("assert roman(1994) == 'MCMXCIV'\nassert roman(3999) == 'MMMCMXCIX'\nassert roman(4) == 'IV'")),
    ("Scrivi una funzione Python `word_count(text)` che restituisce un dizionario parola -> occorrenze, "
     "in minuscolo e ignorando la punteggiatura. Solo il codice.",
     _code("assert word_count('Ciao, ciao! Mondo.') == {'ciao': 2, 'mondo': 1}")),
    ("Scrivi una funzione Python `second_largest(nums)` che restituisce il secondo valore distinto più grande, "
     "o None se non esiste. Solo il codice.",
     _code("assert second_largest([3,1,4,4,2]) == 3\nassert second_largest([5,5]) is None\nassert second_largest([]) is None")),
]
