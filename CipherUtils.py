import functools
import re
from copy import deepcopy
from itertools import chain, permutations
from math import sqrt, ceil, log10, exp
from random import choice, sample, randrange, random
from string import ascii_lowercase as lc, ascii_uppercase as uc

# with open(('\\'.join(__file__.split("\\")[:-1])+r"\words_longer.txt").lstrip('\\')) as f:
#     lines = f.readlines()
#     english_words_longer = [x.strip("\n") for x in lines]

with open(('\\'.join(__file__.split("\\")[:-1])+r"\words_list.txt").lstrip('\\')) as f:
    lines = f.readlines()
    english_words = [x.strip("\n") for x in lines]
    english_words_longer = list(filter(lambda x: len(x) > 2, english_words))

with open(('\\'.join(__file__.split("\\")[:-1]) + r"\words_list_plurals.txt").lstrip('\\')) as f:
    lines = f.readlines()
    english_words_plural = [x.strip("\n .,'").lower() for x in lines]

with open(('\\'.join(__file__.split("\\")[:-1])+r"\trigrams.txt").lstrip('\\')) as f:
    lines = f.readlines()
    english_trigrams_count = {}
    for line in lines:
        l = line.strip("\n").split(" ")
        english_trigrams_count[l[0]] = int(l[1])
    n = sum(english_trigrams_count.values())
    english_trigrams = {k.lower(): log10(float(v)/n) for k, v in english_trigrams_count.items()}
    del english_trigrams_count
    floor = log10(0.01 / n)


class Utils:
    @staticmethod
    def get_stats(cipher: str):
        print("Stats of", cipher)
        print("Length: %d Factors:" % len(cipher), Utils.get_factors(len(cipher)))
        print("Letter Frequency:", Utils.sorted_dict_str(Utils.frequency(cipher)))
        unformatted = Utils.unformat(cipher)
        print("\nUnformatted:", unformatted)
        print("Length: %d Factors:" % len(unformatted), sorted(Utils.get_factors(len(unformatted))))

        print("Letter Frequency:", Utils.sorted_dict_str(Utils.frequency_count_letters(unformatted)))
        print("IoC: %.7f Chi²: %.2f" % (Utils.index_of_coincidence(unformatted), Utils.chi_squared(unformatted)))
        print()
        print("Ngram Frequency:", Utils.sorted_dict_str(Utils.frequency_count_ngrams(cipher)))
        print("Trigram Similarity: %.2f" % Utils.ngram_score(unformatted))
        print()

    @staticmethod
    def unformat(formatted: str, spaces=False) -> str:
        return ''.join(x for x in formatted.lower() if (x in lc) or (spaces and x == " "))

    @staticmethod
    def reformat(unformatted: str, format_pattern: str) -> str:
        result = ''
        queue = list(unformatted)
        for f in format_pattern:
            if f in lc: result += queue.pop(0).lower()
            elif f in uc: result += queue.pop(0).upper()
            else: result += f
        return result


    @staticmethod
    def check_english_percent(plain: str, in_words=True) -> float:
        included, excluded = 0, 0
        if in_words:
            for word in plain.split(' '):
                if Utils.unformat(word) in english_words_longer: included += 1
            return float(included)/len(plain.split(' '))
        else:
            return Utils.chi_squared(plain)

    @staticmethod
    def is_english(plain: str, in_words=True) -> bool:
        if in_words: return Utils.check_english_percent(plain, in_words) > 0.2
        else: return Utils.check_english_percent(plain, in_words) > 0.7 * (11 * pow(len(plain), 2))

    @staticmethod
    def chi_squared(plain_f) -> float:
        actual = {"a": 0.08167, "b": 0.01492, "c": 0.02782, "d": 0.04253, "e": 0.12702, "f": 0.02228, "g": 0.02015, "h": 0.06094, "i": 0.06966, "j": 0.00153, "k": 0.00772, "l": 0.04025, "m": 0.02406, "n": 0.06749, "o": 0.07507, "p": 0.01929, "q": 0.00095, "r": 0.05987, "s": 0.06327, "t": 0.09056, "u": 0.02758, "v": 0.00978, "w": 0.0236, "x": 0.0015, "y": 0.01974, "z": 0.00074}
        plain = Utils.unformat(plain_f)
        freq = Utils.frequency_count_letters(plain)
        return sum(((freq[c] - actual[c]*len(plain)) ** 2) / actual[c]*len(plain) for c in lc)

    @staticmethod
    def index_of_coincidence(text: str) -> float:
        num = den = 0.0
        for val in Utils.frequency_count_letters(text).values():
            i = val
            num += i * (i - 1)
            den += i
        if den == 0.0: return 0.0
        else: return num / (den * (den - 1))

    @staticmethod
    def ngram_score(plain_f, l=3):
        plain = Utils.unformat(plain_f)
        score = 0
        for i in range(len(plain)-l+1):
            if plain[i:i+l] in english_trigrams: score += english_trigrams[plain[i:i+l]]
            else: score += floor
        return score


    @staticmethod
    def frequency(text) -> dict:
        freq = {}
        for c in text:
            if freq.get(c): freq[c] += 1
            else: freq[c] = 1
        return freq

    @staticmethod
    def frequency_count_letters(cipher: str) -> dict:
        cipher = Utils.unformat(cipher)
        freq = dict((c, 0) for c in lc)
        for c in cipher: freq[c] += 1
        return freq

    @staticmethod
    def frequency_count_words(cipher: str) -> dict:
        cipher = Utils.unformat(cipher, spaces=True).split(" ")
        return Utils.frequency(cipher)

    @staticmethod
    def frequency_count_ngrams(cipher_f: str, min_length: int = 3, max_length: int = 5) -> dict:
        cipher = Utils.unformat(cipher_f)
        ngrams = {}
        for i in range(min_length, max_length + 1):
            igrams = [cipher[j:j + i] for j in range(len(cipher) - i + 1)]
            igrams_count = {igram: igrams.count(igram) for igram in set(igrams)}
            del igrams
            ngrams.update({k: v for k, v in igrams_count.items() if v > 2})
            del igrams_count
        return ngrams


    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_factors(numb: int) -> list:
        return list(chain(*[(i, int(numb // i)) for i in range(1, int(ceil(sqrt(numb)))) if not numb % i]))

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_useful_factors(numb: int, min_key_len=3, max_key_len=16) -> list:
        return list(chain(*[(i, int(numb // i)) for i in range(max(2, min_key_len),  min(int(ceil(sqrt(numb))), max_key_len)) if not numb % i]))


    @staticmethod
    def sorted_dict_str(dictionary: dict, sorting=lambda x: -x[1]) -> str:
        return "Length: " + str(len(dictionary.keys())) + "    " + ', '.join('%s: %s' % (k, v) for k, v in sorted(dictionary.items(), key=sorting))

    @staticmethod
    def insert_spaces(cipher: str) -> str:
        longest_words = sorted(english_words_plural, key=lambda x: -len(x))
        # print(longest_words)

        cipher = list(cipher)
        result = ""
        while len(cipher) > 0:
            for word in longest_words:
                w = ''.join(cipher[:len(word)])
                if w.lower() == word:
                    result += w + " "
                    del cipher[:len(word)]
                    break
            else:
                result += cipher.pop()
        return result

    @staticmethod
    def get_str_as_vals(text: str) -> list:
        return [lc.index(x) for x in text]

    @staticmethod
    def get_str_from_vals(numbs: list) -> str:
        return ''.join(lc[x] for x in numbs)

    @staticmethod
    def word_match(pattern: str):
        pattern = pattern.lower()+'$'
        for word in english_words_longer:
            if re.match(pattern, word.lower()): print(word)


class Caesar:
    @staticmethod
    def encode(plain: str, shift: int) -> str:
        return Utils.reformat(Caesar.encode_plain(plain, shift), plain)

    @staticmethod
    def encode_plain(plain: str, shift: int) -> str:
        return ''.join(lc[(lc.index(x)+shift) % len(lc)] for x in Utils.unformat(plain))

    @staticmethod
    def decode(cipher: str, shift: int) -> str:
        return Utils.reformat(Caesar.decode_plain(cipher, shift), cipher)

    @staticmethod
    def decode_plain(cipher: str, shift: int) -> str:
        return ''.join(lc[(lc.index(x)-shift) % len(lc)] for x in Utils.unformat(cipher))

    @staticmethod
    def auto_decode(cipher: str) -> (int, str):
        tests = sorted([(s, Caesar.decode_plain(Utils.unformat(cipher), s)) for s in range(len(lc))], key=lambda x: Utils.chi_squared(x[1]))
        return tests[0][0], Utils.reformat(tests[0][1], cipher)


class Affine:
    @staticmethod
    def encode(plain: str, a: int, b: int) -> str:
        return Utils.reformat(Affine.encode_plain(plain, a, b), plain)

    @staticmethod
    def encode_plain(plain: str, a: int, b: int) -> str:
        return ''.join(lc[(a*lc.index(x)+b) % len(lc)] for x in Utils.unformat(plain))

    @staticmethod
    def decode(cipher: str, a: int, b: int) -> str:
        if Affine.gcd(a, len(lc)) != 1: return "Impossible, a and m are not coprime"
        return Utils.reformat(Affine.decode_plain(Utils.unformat(cipher), a, b), cipher)

    @staticmethod
    def decode_plain(cipher: str, a: int, b: int) -> str:
        a_ = Affine.mod_mult_inverse(a, len(lc))
        return ''.join(lc[a_*(lc.index(x)-b) % len(lc)] for x in cipher)

    @staticmethod
    def auto_decode(cipher_f: str) -> ((int, int), str):
        for a in [x for x in range(len(lc)) if Affine.gcd(x, len(lc)) == 1]:
            for b in [x for x in range(len(lc))]:
                p = Affine.decode(cipher_f, a, b)
                if Utils.is_english(p, in_words=False): return (a, b), p
        return "None"

    @staticmethod
    def mod_mult_inverse(a: int, m: int) -> int:
        a = a % m
        for x in range(1, m):
            if (a*x) % m == 1: return x
        return 1

    @staticmethod
    def gcd(a: int, b: int) -> int:
        if b == 0: return a
        return Affine.gcd(b, a % b)

class Keyword:
    @staticmethod
    def encode(plain: str, key: str) -> str:
        return Utils.reformat(Keyword.encode_plain(plain, key), plain)

    @staticmethod
    def encode_plain(plain: str, key: str) -> str:
        encoding = []
        for c in key+lc:
            if c not in encoding: encoding.append(c)
        return Substitution.encode_plain(plain, {k: v for k, v in zip(list(lc), encoding)})

    @staticmethod
    def decode(cipher: str, key: str) -> str:
        return Utils.reformat(Keyword.decode_plain(cipher, key), cipher)

    @staticmethod
    def decode_plain(cipher: str, key: str) -> str:
        encoding = []
        for c in key + lc:
            if c not in encoding: encoding.append(c)
        return Substitution.decode_plain(cipher, {k: v for k, v in zip(list(lc), encoding)})

    @staticmethod
    def auto_decode(cipher: str) -> (str, str):
        return Substitution.auto_decode(cipher)


english_letter_order = "etaoinshrdlcumwfgypbvkjxqz"
cipher_letter_order = ""

class Substitution:
    @staticmethod
    def encode(plain: str, mapping: dict) -> str:
        return Utils.reformat(Substitution.encode_plain(plain, mapping), plain)

    @staticmethod
    def encode_plain(plain: str, mapping: dict) -> str:
        return ''.join(mapping.get(x, x) for x in Utils.unformat(plain))

    @staticmethod
    def encode_from_str(cipher: str, mapping: str) -> str:
        mapping = {lc[i].lower(): mapping[i].lower() for i in range(len(lc))}
        return Utils.reformat(Substitution.encode_plain(cipher, mapping), cipher)

    @staticmethod
    def decode(cipher: str, mapping: dict) -> str:
        return Utils.reformat(Substitution.decode_plain(cipher, mapping), cipher)

    @staticmethod
    def decode_from_str(cipher: str, mapping: str) -> str:
        mapping = {lc[i].lower(): mapping[i].lower() for i in range(len(lc))}
        return Utils.reformat(Substitution.decode_plain(cipher, mapping), cipher)

    @staticmethod
    def decode_plain(cipher: str, mapping: dict) -> str:
        mapping_inverted = {v: k for k, v in mapping.items()}
        return ''.join(mapping_inverted[x] for x in Utils.unformat(cipher))

    @staticmethod
    def auto_decode(cipher_f: str, fitness=Utils.ngram_score, base_depth: int=500) -> (str, str):
        global cipher_letter_order
        cipher = Utils.unformat(cipher_f)

        cipher_letter_order = ''.join(x[0] for x in sorted(Utils.frequency_count_letters(cipher).items(), key=lambda x: -x[1]))
        pkey = {k: v for k, v in zip(cipher_letter_order, english_letter_order)}

        key, plain = Substitution.hillclimb(cipher, pkey, fitness, base_depth)
        return Substitution.get_inp(key, plain, cipher_f, fitness)

    @staticmethod
    def hillclimb(cipher: str, pkey: dict, fitness, base_depth=500) -> (dict, str):
        global cipher_letter_order
        pplain = Substitution.decode_plain(cipher, pkey)
        pscore = fitness(pplain)
        count = 0
        while count < base_depth:
            ckey = deepcopy(pkey)
            c, c2 = choice(cipher_letter_order), choice(cipher_letter_order)
            ckey[c], ckey[c2] = ckey[c2], ckey[c]

            cplain = Substitution.decode_plain(cipher, ckey)
            cscore = fitness(cplain)
            if cscore > pscore:
                pscore = cscore
                pkey = deepcopy(ckey)
                pplain = cplain
                count = 0
            count += 1
        return pkey, pplain

    @staticmethod
    def get_inp(key, plain, cipher_f, fitness):
        Utils.get_stats(plain)
        cipher = Utils.unformat(cipher_f)
        plain = Utils.reformat(plain, cipher_f)
        inp = inp2 = " "
        while inp not in ("q", ""):
            print('Current Best:', plain, Substitution.key_as_str(key))
            inp = input('q to end, s to swap some letters or r to retry with a random original key\n  >>')
            if len(inp) == 0: continue
            if inp[0].lower() == "s":
                while inp2 != "q":
                    inp2 = input("Swap what (a b), q to return\n >>")
                    if inp2 == "q": break
                    a, b = inp2.strip().split(" ")
                    key[a], key[b] = key[b], key[a]
                    new = Substitution.decode(cipher_f, key)
                    print('New Best:', new, Substitution.key_as_str(key))
                    inp3 = input("Keep? y/n")
                    if inp3[0].lower() == "y":
                        plain = new
                    else:
                        print('Current Best:', plain, Substitution.key_as_str(key))

            if inp[0].lower() == "r":
                pkey = {k: v for k, v in zip(lc, sample(list(lc), len(lc)))}
                key, plain = Substitution.hillclimb(cipher, pkey, fitness)
                plain = Utils.reformat(plain, cipher_f)
        return key, plain

    @staticmethod
    def key_as_str(key: dict) -> str:
        return ''.join(v.upper() for k, v in sorted(key.items(), key=lambda x: ord(x[0])))


class Vigenere:
    @staticmethod
    def encode(plain: str, key: str) -> str:
        return Utils.reformat(Vigenere.encode_plain(plain, key), plain)

    @staticmethod
    def encode_plain(plain: str, key: str) -> str:
        return ''.join(lc[(lc.index(p)+lc.index(k)) % len(lc)] for p, k in zip(Utils.unformat(plain), key * int((len(plain) + 1) / len(key))))

    @staticmethod
    def decode(cipher: str, key: str) -> str:
        return Utils.reformat(Vigenere.decode_plain(cipher, key), cipher)

    @staticmethod
    def decode_plain(cipher: str, key: str) -> str:
        return ''.join(lc[(lc.index(p)-lc.index(k)) % len(lc)] for p, k in zip(Utils.unformat(cipher), key * int((len(cipher) + 1) / len(key))))

    @staticmethod
    def auto_decode(cipher_f: str, min_key_len=3, max_key_len=16) -> (str, str):
        cipher = Utils.unformat(cipher_f)
        spacings = Vigenere.get_dist_between_repeats(cipher, Utils.frequency_count_ngrams(cipher))   # Get spacings between repeated ngrams
        factors = Utils.frequency(y for k in [Utils.get_factors(x) for l in spacings.values() for x in l] for y in k if max_key_len > y > min_key_len)
        print(factors)
        result_f, key = "", ""
        for key_len in sorted(factors.keys(), key=lambda x: -factors.get(x)):   # For most common factor first
            print(key_len)
            sections = ["" for _ in range(key_len)]
            for i in range(len(cipher)):
                sections[i % len(sections)] += cipher[i]   # Split into sections

            key, solved_sections = zip(*[Caesar.auto_decode(sections[i]) for i in range(key_len)])  # Try AutoDecode on each

            result = ""
            for c_i in range(len(solved_sections[0])):  # Reconstruct string
                for section in solved_sections:
                    try: result += section[c_i]
                    except IndexError: break

            result_f = Utils.reformat(result, cipher_f)

            print(result_f)
            if Utils.is_english(result, False): break
            # if Utils.ngram_score(result) < 200: break        # Stop if english, otherwise try next factor

        return Utils.get_str_from_vals(key), result_f

    @staticmethod
    def get_dist_between_repeats(cipher_f: str, ngrams_to_check: dict) -> dict:
        cipher = Utils.unformat(cipher_f)
        distances = {}
        for ngram, count in ngrams_to_check.items():
            ngram = Utils.unformat(ngram)
            distances[ngram] = []
            last_pos = -1
            for i in range(len(cipher) - len(ngram) + 1):
                if cipher[i:i + len(ngram)] == ngram:
                    if last_pos != -1: distances[ngram].append(i - last_pos)
                    last_pos = i
        return distances


# noinspection PyShadowingNames
class Beaufort(Vigenere):
    @staticmethod
    def encode(plain: str, key: str) -> str:
        return Utils.reformat(Beaufort.encode_plain(plain, key), plain)

    @staticmethod
    def encode_plain(plain: str, key: str) -> str:
        return ''.join(lc[(lc.index(k) - lc.index(p)) % len(lc)] for p, k in zip(Utils.unformat(plain), key * int((len(plain) + 1) / len(key))))

    @staticmethod
    def decode(cipher: str, key: str) -> str:
        return Utils.reformat(Beaufort.decode_plain(cipher, key), cipher)

    @staticmethod
    def decode_plain(cipher: str, key: str) -> str:
        return ''.join(lc[(lc.index(k) - lc.index(p)) % len(lc)] for p, k in zip(Utils.unformat(cipher), key * int((len(cipher) + 1) / len(key))))

    @staticmethod
    def auto_decode(cipher_f: str, min_key_len=3, max_key_len=16):
        cipher = Utils.unformat(cipher_f)
        spacings = Vigenere.get_dist_between_repeats(cipher, Utils.frequency_count_ngrams(
            cipher))  # Get spacings between repeated ngrams
        factors = Utils.frequency(y for k in [Utils.get_factors(x) for l in spacings.values() for x in l] for y in k if
                                  max_key_len > y > min_key_len)

        result_f, key = "", ""
        for key_len in sorted(factors.keys(), key=lambda x: -factors.get(x)):  # For most common factor first
            sections = ["" for _ in range(key_len)]
            for i in range(len(cipher)):
                sections[i % len(sections)] += cipher[i]  # Split into sections

            key, solved_sections = zip(
                *[Beaufort.auto_trials(sections[i]) for i in range(key_len)])  # Try AutoDecode on each

            result = ""
            for c_i in range(len(solved_sections[0])):  # Reconstruct string
                for section in solved_sections:
                    try:
                        result += section[c_i]
                    except IndexError:
                        break

            result_f = Utils.reformat(result, cipher_f)
            if Utils.is_english(result_f): break  # Stop if english, otherwise try next factor

        return Utils.get_str_from_vals(key), result_f

    @staticmethod
    def auto_trials(cipher):
        tests = sorted([(l, Beaufort.decode_plain(Utils.unformat(cipher), lc[l])) for l in range(len(lc))], key=lambda x: Utils.chi_squared(x[1]))
        return tests[0][0], Utils.reformat(tests[0][1], cipher)


class VariantBeaufort(Vigenere):
    @staticmethod
    def encode(plain: str, key: str) -> str:
        return Utils.reformat(VariantBeaufort.encode_plain(plain, key), plain)

    @staticmethod
    def encode_plain(plain: str, key: str) -> str:
        return ''.join(lc[(lc.index(p) - lc.index(k)) % len(lc)] for p, k in zip(Utils.unformat(plain), key * int((len(plain) + 1) / len(key))))

    @staticmethod
    def decode(cipher: str, key: str) -> str:
        return Utils.reformat(VariantBeaufort.decode_plain(cipher, key), cipher)

    @staticmethod
    def decode_plain(cipher: str, key: str) -> str:
        return ''.join(lc[(lc.index(p) + lc.index(k)) % len(lc)] for p, k in zip(Utils.unformat(cipher), key * int((len(cipher) + 1) / len(key))))

    @staticmethod
    def auto_decode(cipher_f: str, min_key_len=3, max_key_len=16):
        cipher = Utils.unformat(cipher_f)
        spacings = Vigenere.get_dist_between_repeats(cipher, Utils.frequency_count_ngrams(
            cipher))  # Get spacings between repeated ngrams
        factors = Utils.frequency(y for k in [Utils.get_factors(x) for l in spacings.values() for x in l] for y in k if
                                  max_key_len > y > min_key_len)

        result_f, key = "", ""
        for key_len in sorted(factors.keys(), key=lambda x: -factors.get(x)):  # For most common factor first
            sections = ["" for _ in range(key_len)]
            for i in range(len(cipher)):
                sections[i % len(sections)] += cipher[i]  # Split into sections

            key, solved_sections = zip(
                *[VariantBeaufort.auto_trials(sections[i]) for i in range(key_len)])  # Try AutoDecode on each

            result = ""
            for c_i in range(len(solved_sections[0])):  # Reconstruct string
                for section in solved_sections:
                    try:
                        result += section[c_i]
                    except IndexError:
                        break

            result_f = Utils.reformat(result, cipher_f)
            if Utils.is_english(result_f): break  # Stop if english, otherwise try next factor

        return Utils.get_str_from_vals(key), result_f

    @staticmethod
    def auto_trials(cipher):
        tests = sorted([(l, VariantBeaufort.decode_plain(Utils.unformat(cipher), lc[l])) for l in range(len(lc))], key=lambda x: Utils.chi_squared(x[1]))
        return tests[0][0], Utils.reformat(tests[0][1], cipher)


class ColumnarTransposition:
    @staticmethod
    def encode(plain: str, key: list) -> str:
        return ColumnarTransposition.encode_plain(Utils.unformat(plain), key)

    @staticmethod
    def encode_plain(plain: str, key: list) -> str:
        return ''.join([''.join(plain[bs+key.index(i)] if bs+key.index(i) < len(plain) else "X" for i in range(0, len(key))) for bs in range(0, len(plain), len(key))]).upper()

    @staticmethod
    def decode(cipher: str, key: list) -> str:
        return ColumnarTransposition.decode_plain(cipher, key)

    @staticmethod
    def decode_plain(cipher: str, key: list) -> str:
        return ''.join([''.join(cipher[bs+key[i]] if bs+key[i] < len(cipher) else "X" for i in range(0, len(key))) for bs in range(0, len(cipher), len(key))]).upper()

    @staticmethod
    def auto_decode(cipher: str, min_len=3, max_len=7) -> (list, str):
        best = [i for i in range(min_len)]
        bestscore = Utils.ngram_score(ColumnarTransposition.decode(cipher, best))

        for i in range(min_len, max_len+1):
            key, plain = ColumnarTransposition.try_length(cipher, i)
            # print("%d: (%s) %s" % (i, ','.join(str(x) for x in key), plain))
            if Utils.ngram_score(plain) > bestscore:
                best, bestscore = key, Utils.ngram_score(plain)
        return best, ColumnarTransposition.decode(cipher, best)

    @staticmethod
    def try_length(cipher: str, length: int) -> (list, str):
        best = [0 for _ in range(length)]
        bestscore = Utils.ngram_score(ColumnarTransposition.decode(cipher, best))

        for perm in permutations(list(range(length))):
            plain = ColumnarTransposition.decode(cipher, perm)
            if Utils.ngram_score(plain) > bestscore:
                best, bestscore = perm, Utils.ngram_score(plain)
        return best, ColumnarTransposition.decode(cipher, best)

    @staticmethod
    def alternate_decode(cipher: str, key: list) -> (list, str):
        l = len(key)
        l2 = ceil(len(cipher) / l)
        columns = [cipher[i:i + l2] for i in range(0, len(cipher), l2)]

        rows = []
        for i in range(len(columns[0])):
            try:
                rows.append(''.join(columns[j][i] for j in range(l)))
            except IndexError:
                pass

        return ColumnarTransposition.decode(''.join(rows), key)

    @staticmethod
    def alternate_auto_decode(cipher: str, ls: str) -> (list, str):
        results = []
        for l in ls:
            l2 = ceil(len(cipher) / l)
            columns = [cipher[i:i + l2] for i in range(0, len(cipher), l2)]

            rows = []
            for i in range(len(columns[0])):
                try:
                    rows.append(''.join(columns[j][i] for j in range(l)))
                except IndexError:
                    pass

            results.append(ColumnarTransposition.auto_decode(''.join(rows)))
        return sorted(results, key=lambda x: Utils.ngram_score(x))[0]


class Playfair:
    @staticmethod
    def encode(plain: str, key: str, sub=("j", "")) -> str:
        return Playfair.encode_plain(plain, key, sub)

    @staticmethod
    def encode_plain(plain: str, key: str, sub=("j", "")) -> str:
        keyalpha = Utils.unformat(key) + lc.replace(sub[0], "")
        keyfiltered = ''.join(keyalpha[i] for i in range(len(keyalpha)) if i == keyalpha.index(keyalpha[i]))
        key_grid = [[keyfiltered[i*5+j] for j in range(5)] for i in range(5)]

        plain = re.sub(*sub, Utils.unformat(plain))

        q, output = list(plain), ""
        while len(q) > 0:
            if len(q) == 1: q += "x"
            if q[0] == q[1]: output += Playfair.encrypt_pair(q.pop(0), "x", key_grid)
            else: output += Playfair.encrypt_pair(q.pop(0), q.pop(0), key_grid)

        return output

    @staticmethod
    def encrypt_pair(c1, c2, key_grid):
        def get_coord(c: str, grid) -> (int, int):
            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    if grid[i][j] == c: return i, j

        p1, p2 = get_coord(c1, key_grid), get_coord(c2, key_grid)
        if p1[0] == p2[0]: v = key_grid[p1[0]][(p1[1] + 1) % 5] + key_grid[p2[0]][(p2[1] + 1) % 5]
        elif p1[1] == p2[1]: v = key_grid[(p1[0] + 1) % 5][p1[1]] + key_grid[(p2[0] + 1) % 5][p2[1]]
        else: v = key_grid[p1[0]][p2[1]] + key_grid[p2[0]][p1[1]]
        return v

    @staticmethod
    def decode(plain: str, key: str, sub=("j", "")) -> str:
        return Playfair.decode_plain(plain, key, sub)

    @staticmethod
    def decode_plain(plain: str, key: str, sub=("j", "")) -> str:
        keyalpha = Utils.unformat(key) + lc.replace(sub[0], "")
        keyfiltered = ''.join(keyalpha[i] for i in range(len(keyalpha)) if i == keyalpha.index(keyalpha[i]))
        key_grid = [[keyfiltered[i * 5 + j] for j in range(5)] for i in range(5)]

        plain = re.sub(*sub, Utils.unformat(plain))

        q, output = list(plain), ""
        while len(q) > 0:
            if len(q) == 1: q += "x"
            if q[0] == q[1]:
                output += Playfair.decrypt_pair(q.pop(0), "x", key_grid)
            else:
                output += Playfair.decrypt_pair(q.pop(0), q.pop(0), key_grid)

        return output

    @staticmethod
    def decrypt_pair(c1, c2, key_grid):
        def get_coord(c: str, grid) -> (int, int):
            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    if grid[i][j] == c: return i, j

        p1, p2 = get_coord(c1, key_grid), get_coord(c2, key_grid)
        if p1[0] == p2[0]:
            v = key_grid[p1[0]][(p1[1] - 1) % 5] + key_grid[p2[0]][(p2[1] - 1) % 5]
        elif p1[1] == p2[1]:
            v = key_grid[(p1[0] - 1) % 5][p1[1]] + key_grid[(p2[0] - 1) % 5][p2[1]]
        else:
            v = key_grid[p1[0]][p2[1]] + key_grid[p2[0]][p1[1]]
        return v

    @staticmethod
    def auto_decode(cipher_f: str) -> (str, str):
        cipher = Utils.unformat(cipher_f)
        # bigrams = Utils.frequency("".join(x) for x in zip(cipher, cipher[1:]))
        # print(', '.join('%s: %s' % (k, v) for k, v in sorted(bigrams.items(), key=lambda x: -x[1])))
        # print(', '.join('%s: %s' % (Playfair.decode(k, "enabling"), v) for k, v in sorted(bigrams.items(), key=lambda x: -x[1])[:100]))
        k = sample(lc, len(lc))
        k.remove("j")
        key, plain = Playfair.simulated_annealing(cipher.replace("w", ""), k)
        return key, plain

    @staticmethod
    def simulated_annealing(cipher: str, pkey: list):
        T, STEP, COUNT = 20.0, 0.2, 1000

        def modify_key(key: list) -> list:

            key_grid = [[key[i * 5 + j] for j in range(5)] for i in range(5)]
            method = randrange(10)
            a, b = randrange(5), randrange(5)
            if   method == 0: key_grid[a], key_grid[b] = key_grid[b], key_grid[a] # Swap row
            elif method == 1:
                for i in range(5):
                    key_grid[i][a], key_grid[i][b] = key_grid[i][b], key_grid[i][a] # Swap column
            elif method == 2:           # Reverse all
                return list(key)[::-1]
            elif method == 3:           # Reverse
                key_grid = [key_grid[5-i] for i in range(5)] # Reverse rows
            elif method == 4:
                key_grid = [[key_grid[j][5 - i] for i in range(5)] for j in range(5)]  # Reverse columns
            else:
                a, b = randrange(25), randrange(25)
                key[a], key[b] = key[b], key[a]
                return key
            return key_grid

        pplain = Playfair.decode_plain(cipher, ''.join(pkey))
        pscore = Utils.ngram_score(pplain)
        temp = T
        while temp >= 0:
            for c in range(COUNT):
                ckey = modify_key(pkey)
                cplain = Playfair.decode_plain(cipher, ''.join(ckey))
                cscore = Utils.ngram_score(cplain)
                dF = cscore - pscore
                if dF > 0 or exp(dF/T) > random(): pplain, pkey, pscore = cplain, ckey, cscore

            temp -= STEP
        return pkey, pplain

    # @staticmethod
    # def hillclimb(cipher: str, pkey: list) -> (dict, str):
    #     print(pkey)
    #     pplain = Playfair.decode_plain(cipher, ''.join(pkey))
    #     pscore = Utils.ngram_score(pplain)
    #     count = 0
    #     while count < base_depth:
    #         ckey = pkey
    #         a, b = randrange(len(ckey)), randrange(len(ckey))
    #         ckey[a], ckey[b] = ckey[b], ckey[a]
    #
    #         cplain = Playfair.decode_plain(cipher, ''.join(ckey))
    #         print(cplain)
    #         cscore = Utils.ngram_score(cplain)
    #         if cscore > pscore:
    #             pscore = cscore
    #             pkey = ckey
    #             pplain = cplain
    #             count = 0
    #         count += 1
    #     return pkey, pplain


class Scytale: # NOT FINSIHED
    @staticmethod
    def encode(plain: str, key: int) -> str:
        return Scytale.encode_plain(Utils.unformat(plain), key)

    @staticmethod
    def encode_plain(plain: str, key: int) -> str:
        sections = ["" for _ in range(key)]
        for i in range(len(plain)):
            sections[i % key] += plain[i]
        return ''.join(sections)

    @staticmethod
    def decode(cipher: str, key: int) -> str:
        return Scytale.decode_plain(cipher, key)

    @staticmethod
    def decode_plain(cipher: str, key: int) -> str:
        res = ""
        for offset in range(0, key-1):
            for i in range(offset, len(cipher), key-1):
                res += cipher[i]
                print(offset, i, cipher[i])

        return res

    @staticmethod
    def auto_decode(cipher: str) -> (object, str):
        pass



def try_all(cipher: str):
    Utils.get_stats(cipher)
    print("Trying all for", cipher)
    print('Caesar AutoDecode: Decoded Shift: %d, Plaintext: %s' % Caesar.auto_decode(cipher))
    print('Substitution AutoDecode:', Substitution.auto_decode(cipher))
    # print('Affine AutoDecode:', Affine.auto_decode(cipher))
    print('Vigenere AutoDecode: Key: %s Plaintext: %s' % Vigenere.auto_decode(cipher))
    print('Beaufort AutoDecode:', Beaufort.auto_decode(cipher))
    print('Variant Beaufort AutoDecode:', VariantBeaufort.auto_decode(cipher))
    print('Columnar Transposition AutoDecode:', ColumnarTransposition.auto_decode(cipher))
    print('')

"""
class Cipher:
    @staticmethod
    def encode(plain: str, key: object) -> str:
        return Utils.reformat(Cipher.encode_plain(plain, key), plain)
    
    @staticmethod
    def encode_plain(plain: str, key: object) -> str:
        return plain

    @staticmethod
    def decode(cipher: str, key: object) -> str:
        return Utils.reformat(Cipher.decode_plain(cipher, key), cipher)

    @staticmethod
    def decode_plain(cipher: str, key: object) -> str:
        return cipher

    @staticmethod
    def auto_decode(cipher: str) -> (object, str):
        pass
"""