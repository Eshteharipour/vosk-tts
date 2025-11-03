import re

# Define dictionaries for English number words to digits
units = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

tens = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

scales = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1000000,
    "billion": 1000000000,
    "trillion": 1000000000000,
}

all_words = {**units, **tens, **scales}


def pre_replace(match):
    word = match.group(0)
    parts = word.split("-")
    if len(parts) < 2:
        return word
    if len(parts) > 2:
        return " ".join(parts)
    p1, p2 = parts
    if p1 not in all_words or p2 not in all_words:
        return word
    n1 = all_words[p1]
    n2 = all_words[p2]
    if p1 in tens and p2 in units:
        return p1 + " " + p2
    elif n1 < n2 and n1 % 10 == 0 and n2 % 10 == 0:
        return str(n1) + "-" + str(n2)
    else:
        return word


def is_number(x):
    if type(x) == str:
        x = x.replace(",", "")
    try:
        float(x)
    except:
        return False
    return True


def text2int(textnum, numwords={}):
    units = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    scales = ["hundred", "thousand", "million", "billion", "trillion"]
    ordinal_words = {"first": 1, "second": 2, "third": 3, "fifth": 5, "eighth": 8, "ninth": 9, "twelfth": 12}
    ordinal_endings = [("ieth", "y"), ("th", "")]

    # No replace('-', ' ') here to avoid affecting pre-processed ranges

    current = result = 0
    curstring = ""
    onnumber = False
    lastunit = False
    lastscale = False

    def is_numword(x):
        if is_number(x):
            return True
        if x in numwords:
            return True
        return False

    def from_numword(x):
        if is_number(x):
            scale = 0
            increment = int(x.replace(",", ""))
            return scale, increment
        return numwords[x]

    if not numwords:
        numwords["and"] = (1, 0)
        for idx, word in enumerate(units):
            numwords[word] = (1, idx)
        for idx, word in enumerate(tens):
            numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales):
            numwords[word] = (10 ** (idx * 3 or 2), 0)

    for word in textnum.split():
        if word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
            onnumber = True
            lastunit = False
            lastscale = False
        else:
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = "%s%s" % (word[: -len(ending)], replacement)

            if (not is_numword(word)) or (word == "and" and not lastscale):
                if onnumber:
                    curstring += repr(result + current) + " "
                curstring += word + " "
                result = current = 0
                onnumber = False
                lastunit = False
                lastscale = False
            else:
                scale, increment = from_numword(word)
                onnumber = True

                if lastunit and (word not in scales):
                    curstring += repr(result + current)
                    result = current = 0

                if scale > 1:
                    current = max(1, current)

                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0

                lastscale = False
                lastunit = False
                if word in scales:
                    lastscale = True
                elif word in units:
                    lastunit = True

    if onnumber:
        curstring += repr(result + current)

    return curstring


def numbertowords_russian(number: int) -> str:
    if number == 0:
        return "ноль"

    units_ru = [
        "ноль",
        "один",
        "два",
        "три",
        "четыре",
        "пять",
        "шесть",
        "семь",
        "восемь",
        "девять",
        "десять",
        "одиннадцать",
        "двенадцать",
        "тринадцать",
        "четырнадцать",
        "пятнадцать",
        "шестнадцать",
        "семнадцать",
        "восемнадцать",
        "девятнадцать",
    ]

    tens_ru = [
        "",
        "",
        "двадцать",
        "тридцать",
        "сорок",
        "пятьдесят",
        "шестьдесят",
        "семьдесят",
        "восемьдесят",
        "девяносто",
    ]

    hundreds_ru = [
        "",
        "сто",
        "двести",
        "триста",
        "четыреста",
        "пятьсот",
        "шестьсот",
        "семьсот",
        "восемьсот",
        "девятьсот",
    ]

    big_numbers_ru = [
        "",
        "тысяча",
        "миллион",
        "миллиард",
        "триллион",
        "квадриллион",
        "квинтиллион",
        "секстиллион",
        "септиллион",
        "октиллион",
        "нониллион",
        "дециллион",
        "ундециллион",
        "додециллион",
        "тредециллион",
        "кваттуордециллион",
        "квиндециллион",
        "сексдециллион",
        "септемдециллион",
        "октодециллион",
        "новемдециллион",
        "вигинтиллион",
    ]

    words = []
    number_str = str(number)[::-1]
    for i in range(0, len(number_str), 3):
        group = number_str[i : i + 3]
        group_int = int(group[::-1])
        if group_int != 0:
            group_words = []
            if group_int // 100 > 0:
                group_words.append(hundreds_ru[group_int // 100])
            if group_int % 100 < 20:
                if group_int % 100 != 0:
                    group_words.append(units_ru[group_int % 100])
            else:
                group_words.append(tens_ru[(group_int % 100) // 10])
                if group_int % 10 != 0:
                    group_words.append(units_ru[group_int % 10])
            # Special handling for scale == 1 (thousand)
            if i // 3 == 1:
                if group_words and group_words[-1] == "один":
                    group_words[-1] = "одна"
                elif group_words and group_words[-1] == "два":
                    group_words[-1] = "две"
            # Omit "один" for exact 1 before scale
            if group_int == 1 and i // 3 > 0 and len(group_words) == 1 and group_words[0] == "один":
                group_words = []
            words.extend(group_words)
            if i // 3 > 0:
                words.append(big_numbers_ru[i // 3])
    return " ".join(words)


_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    return m.group(1).replace(".", " точка ")


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " долларов"
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        if dollars % 10 == 1 and dollars % 100 != 11:
            dollar_unit = "доллар"
        elif 2 <= dollars % 10 <= 4 and (dollars % 100 < 10 or dollars % 100 >= 20):
            dollar_unit = "доллара"
        else:
            dollar_unit = "долларов"
        if cents % 10 == 1 and cents % 100 != 11:
            cent_unit = "цент"
        elif 2 <= cents % 10 <= 4 and (cents % 100 < 10 or cents % 100 >= 20):
            cent_unit = "цента"
        else:
            cent_unit = "центов"
        return f"{numbertowords_russian(dollars)} {dollar_unit}, {numbertowords_russian(cents)} {cent_unit}"
    elif dollars:
        if dollars % 10 == 1 and dollars % 100 != 11:
            dollar_unit = "доллар"
        elif 2 <= dollars % 10 <= 4 and (dollars % 100 < 10 or dollars % 100 >= 20):
            dollar_unit = "доллара"
        else:
            dollar_unit = "долларов"
        return f"{numbertowords_russian(dollars)} {dollar_unit}"
    elif cents:
        if cents % 10 == 1 and cents % 100 != 11:
            cent_unit = "цент"
        elif 2 <= cents % 10 <= 4 and (cents % 100 < 10 or cents % 100 >= 20):
            cent_unit = "цента"
        else:
            cent_unit = "центов"
        return f"{numbertowords_russian(cents)} {cent_unit}"
    else:
        return "ноль долларов"


def _expand_number(m):
    num = int(m.group(0))
    return numbertowords_russian(num)


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 фунтов", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    # Comment out ordinal if not needed
    # text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


def clean_text_for_phonemizer(text: str) -> str:
    text = text.lower()

    text = re.sub(r"([a-zа-я])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([a-zа-я])", r"\1 \2", text)

    text = re.sub(r"\b([a-z]+(?:-[a-z]+)+)\b", pre_replace, text)

    text = re.sub(r"\s+", " ", text).strip()

    text = text2int(text)

    text = re.sub(r"(\d+)[—–-](\d+)", r"\1 , \2", text)
    text = re.sub(r"\s*[-—–]\s+", ", ", text)

    text = text.replace("–", ",").replace("—", ",")
    text = text.replace("«", '"').replace("»", '"')
    text = text.replace("%", " процентов ")

    text = re.sub(r"\s+", " ", text).strip()

    text = normalize_numbers(text)

    return text
