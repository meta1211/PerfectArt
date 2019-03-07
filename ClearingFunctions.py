patterns = [
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[\w+]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    r'\d+', # numbers
    r"['-]\w+", # words with - and '
    r"[:;=%x][o0\-^_]?[ds\\\[\]\(\)/i|><]+", # smiles
    r"[-+*\\/]+"
]



def ClearFromPatterns(str, patterns):
    result = str
    for pattern in patterns:
        result = re.sub(pattern, '', result)
    return result

def Split(text):
    splitPatter = r"[!?.,:\( \)\\/\"\'*;\[\=+]"
    return re.split(text, splitPatter)
